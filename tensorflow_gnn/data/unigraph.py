# Copyright 2021 The TensorFlow GNN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Universal graph format library.

This is a simple library that supports reading graphs from universal graph
format into a Beam pipeline. See go/universal-graph-format for details.

# File Formats

Supported file formats include:
- 'csv': A CSV file with rows of features for each node or edge.
- 'tfrecord': A binary container of tf.Example protocol buffer instances.
# Placeholder for Google-internal file support docstring
"""

import csv
import hashlib
import os
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Text, Tuple, Union
from absl import logging
import apache_beam as beam
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.proto import graph_schema_pb2

# Placeholder for Google-internal record file format pipeline import
# Placeholder for Google-internal sorted string file format pipeline import


# Special constant column names required to be present in the node and edge
# tabular files. As BigQuery doesn't support the `#` character in column names,
# the special constant column names for BigQuery tables are the same without
# the `#` character. The resulting serizlied tf.train.Example protocol messages
# will contain the `#` character.
NODE_ID = "#id"
SOURCE_ID = "#source"
TARGET_ID = "#target"

# Some tables aren't expected to start with the special '#' feature
# name qualifier.
# TODO(tfgnn): Contemplate simplifying by removing '#'.
_TRANSLATIONS = {
    "id": "#id",
    "source": "#source",
    "target": "#target",
}

gfile = tf.io.gfile
NodeId = bytes
Example = tf.train.Example
PCollection = beam.pvalue.PCollection
FeatureSet = Mapping[Text, tfgnn.Feature]

# A value converter function and dict.
Converter = Callable[[tf.train.Feature, Any], None]
Converters = Dict[str, Converter]


def guess_file_format(filename: str) -> str:
  """Guess the file format from the filename."""
  if re.search(r"[_.-]tfrecords?|\btfr\b", filename):
    return "tfrecord"
  elif re.search(r"[_.-]csv\b", filename):
    return "csv"
  # Placeholder for guessing Google-internal file extensions
  else:
    raise ValueError("Could not guess file format for: {}".format(filename))


def get_base_filename(file_pattern: str) -> str:
  """Return the file pattern without the sharding suffixes."""
  match = re.fullmatch(r"(.*)(@\d+|-[0-9\?]{5}-of-\d{5})", file_pattern)
  return match.group(1) if match else file_pattern


def expand_sharded_pattern(file_pattern: str) -> str:
  """Expand shards in the given pattern for filenames like base@shards.

  For example, '/path/to/basename@3' to expand to
    '/path/to/basename-00000-of-00003'
    '/path/to/basename-00001-of-00003'
    '/path/to/basename-00002-of-00003'

  Args:
    file_pattern: A filename, possibly with a @N shard suffix.
  Returns:
    A globbing pattern that will match the sharded files.
  """
  match = re.fullmatch(r"(.*)@(\d+)", file_pattern)
  if not match:
    return file_pattern
  num_shards = int(match.group(2))
  return "{}-?????-of-{:05d}".format(match.group(1), num_shards)


def get_sharded_pattern_args(pattern: str) -> Dict[str, Any]:
  """Reduce the filename pattern to avoid a single shard if not specified.

  The default for Beam writers is to produce a single shard, unless specified
  with an empty template. We accept a single filename. This routine infers
  whether the filename pattern is sharded or not and produces the correct
  arguments for the Beam sinks.

  Args:
    pattern: A filename pattern, potentially sharded.
  Returns:
    A dict of arguments for filename, num shards, and template.
  """
  # TODO(blais): Eventually support file suffixes.
  match = re.fullmatch(r"(.*)@(\d+)", pattern)
  if match:
    # Beam would unpack automatically for internal containers, but we need this
    # for the open source implementation of the Beam runner.
    return dict(file_path_prefix=match.group(1),
                num_shards=int(match.group(2)),
                shard_name_template="-SSSSS-of-NNNNN")
  else:
    match = re.fullmatch(r"(.*)-(?:\d|\?){5}-of-(\d{5})", pattern)
    if match:
      return dict(file_path_prefix=match.group(1),
                  num_shards=int(match.group(2)),
                  shard_name_template="-SSSSS-of-NNNNN")
    else:
      return dict(file_path_prefix=pattern,
                  num_shards=None, shard_name_template="")


def find_schema_filename(file_or_dir: str) -> str:
  """If the input file is a directory, attempt to find a schema file in it."""
  if gfile.isdir(file_or_dir):
    pbtxts = [filename for filename in gfile.listdir(file_or_dir)
              if filename.endswith(".pbtxt")]
    if len(pbtxts) != 1:
      raise ValueError("Could not find schema pbtxt file in '{}': {}".format(
          file_or_dir, pbtxts))
    file_or_dir = os.path.join(file_or_dir, pbtxts[0])
  return file_or_dir


def read_graph_and_schema(
    file_or_dir: str,
    rcoll: PCollection) -> Tuple[tfgnn.GraphSchema, Dict[str, PCollection]]:
  """Read a universal graph given its schema filename or directory."""

  # Read the schema.
  filename = find_schema_filename(file_or_dir)
  schema = tfgnn.read_schema(filename)

  # Read the graph.
  colls_dict = read_graph(schema, os.path.dirname(filename), rcoll)

  return schema, colls_dict


def _stage_suffix(string: str) -> str:
  """Compute a unique stage name."""
  hsh = hashlib.sha256()
  hsh.update(string.encode("utf8"))
  return hsh.hexdigest()


def get_node_ids(example: Example) -> Tuple[NodeId, Example]:
  """Extract the node id from the input example."""
  feature = example.features.feature[NODE_ID]
  return (feature.bytes_list.value[0], example)


def get_edge_ids(example: Example,
                 edge_reversed=False) -> Tuple[NodeId, NodeId, Example]:
  """Extract the source and target node ids from the input example."""
  if edge_reversed:
    source = example.features.feature[TARGET_ID]
    target = example.features.feature[SOURCE_ID]
  else:
    source = example.features.feature[SOURCE_ID]
    target = example.features.feature[TARGET_ID]
  return (source.bytes_list.value[0], target.bytes_list.value[0], example)


def read_node_set(pcoll: PCollection,
                  filename: str,
                  set_name: tfgnn.NodeSetName,
                  converters: Optional[Converters] = None) -> PCollection:
  sfx = _stage_suffix(filename)
  return (pcoll
          | f"ReadNodes.{set_name}.{sfx}" >> ReadTable(
              filename, converters=converters)
          | f"GetNodeIds.{set_name}.{sfx}" >> beam.Map(get_node_ids))


def read_edge_set(pcoll: PCollection,
                  filename: str,
                  set_name: tfgnn.EdgeSetName,
                  converters: Optional[Converters] = None,
                  edge_reversed=False) -> PCollection:
  sfx = _stage_suffix(filename)
  return (pcoll
          | f"ReadEdges.{set_name}.{sfx}" >> ReadTable(
              filename, converters=converters)
          | f"GetEdgeIds.{set_name}.{sfx}" >> beam.Map(
              get_edge_ids, edge_reversed=edge_reversed))


# TODO(b/245730844): Reconsider arguments as context sets do not return a name
# during iteration.
def read_context_set(pcoll: PCollection,
                     filename: str,
                     set_name: str,
                     converters: Optional[Converters] = None) -> PCollection:
  sfx = _stage_suffix(filename)
  return (pcoll
          | f"ReadContext.{set_name}.{sfx}" >> ReadTable(
              filename, converters=converters))


def float_converter(feature: tf.train.Feature, value: bytes):
  feature.float_list.value.append(float(value))


def int64_converter(feature: tf.train.Feature, value: bytes):
  feature.int64_list.value.append(int(value))


def build_converter_from_schema(features: FeatureSet) -> Converters:
  """Build a converters map from a GraphSchema's features schema of a set."""
  converters = {}
  for fname, feature in features.items():
    if feature.HasField("dtype"):
      if feature.dtype == tf.float32.as_datatype_enum:
        converters[fname] = float_converter
      elif feature.dtype == tf.int64.as_datatype_enum:
        converters[fname] = int64_converter
  return converters


def read_graph(
    schema: tfgnn.GraphSchema,
    graph_dir: str,
    rcoll: PCollection,
    gcs_location: Optional[str] = None,
    bigquery_reader: Callable[..., beam.PCollection[Dict[
        str, Any]]] = beam.io.ReadFromBigQuery
) -> Dict[str, Dict[str, PCollection]]:
  """Read a universal graph given a schema.

  Args:
    schema: An instance of GraphSchema to read the graph of.
    graph_dir: The name of the directory to look for the files from.
    rcoll: The root collection for the reading stages.
    gcs_location: An optional GCS temporary location used by BigQuery EXPORT
      read methods.
    bigquery_reader: **DO NOT SET. ONLY USED FOR UNIIT-TESTS!**
  Returns:
    A dict set type to set name to PCollection of tf.Example of the features.
    Node sets have items of type (node-id, Example).
    Edge sets have items of type (source-id, target-id, Example).
  """
  pcoll_dict = {}
  for set_type, set_name, fset in tfgnn.iter_sets(schema):

    if fset.metadata.HasField("filename"):
      pcoll = (
          rcoll
          | f"ReadFile/{set_name}" >> ReadUnigraphPieceFromFile(
              set_type, set_name, fset, graph_dir))
    elif fset.metadata.HasField("bigquery"):
      pcoll = (
          rcoll | f"ReadBigQuery/{set_name}" >> ReadUnigraphPieceFromBigQuery(
              set_name,
              fset,
              gcs_location=gcs_location,
              bigquery_reader=bigquery_reader))

    # Save the collection for output.
    set_dict = pcoll_dict.setdefault(set_type, {})
    set_dict[set_name] = pcoll

  return pcoll_dict


class ReadUnigraphPieceFromFile(beam.PTransform):
  """Read a unigraph node/edge/context component from a file.

  Returns a PCollection object representing the Unigraph component.
  """

  def __init__(self, fset_type: str, fset_name: str,
               fset: Union[graph_schema_pb2.NodeSet,
                           graph_schema_pb2.EdgeSet], graph_dir: Optional[str]):
    """Constructor for ReadUnigraphPieceFromFile PTransform.

    Args:
      fset_type: String typename for the component.
      fset_name: The string name of the node/edge/context set.
      fset: A NodeSet or EdgeSet protocol buffer message.
      graph_dir: Optional string root graph directory.

    Raises:
      ValueError if initialization fails.
    """
    super().__init__()
    self.fset_type = fset_type
    self.fset_name = fset_name
    self.fset = fset
    self.graph_dir = graph_dir

    if not self.fset.HasField("metadata") and not self.fset.metadata.HasField(
        "filename"):
      raise ValueError(f"{fset_name} does not specify a file: {fset}")

    self.filename = fset.metadata.filename
    if not os.path.isabs(self.filename):
      if not self.graph_dir:
        raise ValueError(f"{self.filename} does not specify a full path "
                         "and graph_dir is None.")
      self.filename = os.path.join(self.graph_dir, self.filename)

    self.converters = build_converter_from_schema(self.fset.features)

    self.reversed = False
    if isinstance(fset, graph_schema_pb2.EdgeSet):
      for kv in fset.metadata.extra:
        if kv.key == "edge_type" and kv.value == "reversed":
          self.edge_reversed = True

  def expand(self, pcoll: beam.PCollection) -> beam.PCollection:
    if self.fset_type == tfgnn.NODES:
      logging.info("Reading NodeSet %s from file: %s", self.fset_name,
                   self.filename)
      return read_node_set(pcoll, self.filename, self.fset_name,
                           self.converters)
    elif self.fset_type == tfgnn.EDGES:
      logging.info("Reading EdgeSet %s (reversed=%s) from file: %s ",
                   self.fset_name, self.reversed, self.filename)
      return read_edge_set(pcoll, self.filename, self.fset_name,
                           self.converters, self.reversed)
    elif self.fset_type == tfgnn.CONTEXT:
      logging.info("Reading Context %s from file: %s", self.fset_name,
                   self.filename)
      assert not self.fset_name, "Context pieces should not have a name."
      return read_context_set(pcoll, self.filename, self.fset_name,
                              self.converters)
    else:
      raise ValueError(
          f"Unknown Unigraph component {self.fset_type}, {self.fset_name}.")


class ReadUnigraphPieceFromBigQuery(beam.PTransform):
  """Read a unigraph node/edge/context component from a BigQuery table.

  Yeilds tf.Example protos of the features from the table.
  """
  _SUPPORTED_DTYPES = [tf.dtypes.float32, tf.dtypes.int64, tf.dtypes.string]
  _ID_COLUMN = "id"
  _SOURCE_COLUMN = "source"
  _TARGET_COLUMN = "target"

  def __init__(
      self,
      fset_name: str,
      fset: Union[graph_schema_pb2.NodeSet, graph_schema_pb2.EdgeSet],
      gcs_location: Optional[str] = None,
      bigquery_reader: Callable[..., beam.PCollection[Dict[
          str, Any]]] = beam.io.ReadFromBigQuery,
  ):
    """Constructor for PTransform for reading a NodeSet from BigQuery.

    Args:
      fset_name: The string name of the node/edge/context set.
      fset: Either a graph_schema_pb2.NodeSet or graph_schema_pb2.EdgeSet
        message.
      gcs_location: An optional string specifying a google storage location
        used if the EXPORT BigQuery method is specified.
      bigquery_reader: Callable, **ONLY USED FOR UNIT-TESTING**.
    """
    super().__init__()

    self.fset_name = fset_name
    self.fset = fset
    self.gcs_location = gcs_location
    self._bigquery_reader = bigquery_reader  # ONLY use for testing

    if not fset.HasField("metadata"):
      raise ValueError(
          "Must specify metadata to read BigQuery graph component.")

    if not fset.metadata.HasField("bigquery"):
      raise ValueError("NodeSet does not specify a BigQuery table.")

    # Only used for edge sets
    self.edge_reversed = False
    if isinstance(self.fset, graph_schema_pb2.EdgeSet):
      for kv in fset.metadata.extra:
        if kv.key == "edge_type" and kv.value == "reversed":
          self.edge_reversed = True

    self.bq = fset.metadata.bigquery
    self.sfx = self.stage_name_suffix(self.fset_name, self.fset)
    self.bq_args = self.bigquery_args_from_proto(self.bq)
    if self.gcs_location:
      self.bq_args["gcs_location"] = self.gcs_location

    for feature_name, feature in self.fset.features.items():
      if feature.dtype not in self._SUPPORTED_DTYPES:
        raise ValueError(
            f"{feature_name}: Only {self._SUPPORTED_DTYPES} feature types are supported."
        )

      # TODO(b/244415126): Add support for array columns in BigQuery.
      if feature.HasField("shape"):
        err = f"{feature_name}: Only scalar value columns are currently supported."
        if len(feature.shape.dim) > 1:
          raise ValueError(err)
        if len(feature.shape.dim) and feature.shape.dim[0].size > 0:
          raise ValueError(err)

  @staticmethod
  def bigquery_args_from_proto(bq: graph_schema_pb2.BigQuery) -> Dict[str, Any]:
    """Parse a tensorflow_gnn.BigQuery message and return BigQuery source args.

    Args:
      bq: A graph_schema_pb2.BigQuery message.

    Returns:
      Dict[str, Any] Dictionary that can be used as arguments to
        beam.io.ReadFromBigQuery function.

    Raises:
      ValueError if unable to parse input message.
    """
    bq_args = {}

    if bq.HasField("table_spec"):
      if not bq.table_spec.dataset:
        raise ValueError("Must provide a big query source dataset string.")

      if not bq.table_spec.table:
        raise ValueError("Must provide a big query source table name.")

      bq_args["table"] = ""
      if bq.table_spec.project:
        bq_args["table"] = f"{bq.table_spec.project}:"
      bq_args["table"] += f"{bq.table_spec.dataset}.{bq.table_spec.table}"

    elif bq.HasField("sql"):
      if not bq.sql:
        raise ValueError("Must provide non-empty SQL query.")
      bq_args["query"] = bq.sql
    else:
      raise ValueError("Must provide BigQuerySource table_spec or query.")

    if bq.read_method == graph_schema_pb2.BigQuery.EXPORT:
      bq_args["method"] = beam.io.ReadFromBigQuery.Method.EXPORT
    elif bq.read_method == graph_schema_pb2.BigQuery.DIRECT_READ:
      bq_args["method"] = beam.io.ReadFromBigQuery.Method.DIRECT_READ

    return bq_args

  @staticmethod
  def stage_name_suffix(
      fset_name: str, fset: Union[graph_schema_pb2.NodeSet,
                                  graph_schema_pb2.EdgeSet]) -> str:
    """Return a stage name suffix from a BigQuery proto message.

    Args:
      fset_name: Name of the feature set.
      fset: A feature set (node or edge).

    Returns:
      The string stage suffix indicating a table_spec or query.

    Raises:
      ValueError if error generating the suffix.
    """
    if not fset.metadata.HasField("bigquery"):
      raise ValueError("Feature set does not specify a BigQuery source.")

    bq = fset.metadata.bigquery
    sfx = "ReadFromBigQuery"
    if isinstance(fset, graph_schema_pb2.NodeSet):
      sfx += "/NodeSet/"
    elif isinstance(fset, graph_schema_pb2.EdgeSet):
      sfx += "/EdgeSet/"
    else:
      raise ValueError("Must specify a Node or Edge set.")

    sfx += f"{fset_name}/"
    if bq.HasField("table_spec"):
      if bq.table_spec.project:
        sfx += f"{bq.table_spec.project}:"

      if not bq.table_spec.dataset:
        raise ValueError("Must provide a big query source dataset string.")

      if not bq.table_spec.table:
        raise ValueError("Must provide a big query source table name.")
      sfx += f"{bq.table_spec.dataset}.{bq.table_spec.table}"
    elif bq.HasField("sql"):
      sfx += "query"
    else:
      raise ValueError("Must provide BigQuerySource table_spec or query.")

    return sfx

  def row_to_keyed_example(self, row: Mapping[str, Any]) -> Any:
    """Convert a single row from a BigQuery result to tf.Example.

    Args:
      row: Dict[str, Any] result of a BigQuery read.

    Returns:
      If the input fset is a graph_schema_pb2.NodeSet, returns
        Tuple(id: str, example: tf.train.Example).
      If the input fset is a graph_schema_pb2.Edgeset, returns
        Tuple(source: str, target: str, example: tf.train.Example)

    Raises:
      ValueError if a field name is not found in the BQ row.
    """
    ret_key = None
    example = Example()

    if isinstance(self.fset, graph_schema_pb2.NodeSet):
      if self._ID_COLUMN not in row.keys():
        raise ValueError(
            f"Query result must have a column named {self._ID_COLUMN}")

      node_id = row[self._ID_COLUMN]
      node_id_feature_name = _TRANSLATIONS[self._ID_COLUMN]
      example.features.feature[node_id_feature_name].bytes_list.value.append(
          node_id.encode("utf-8"))
      ret_key = [node_id]

    elif isinstance(self.fset, graph_schema_pb2.EdgeSet):
      if self._SOURCE_COLUMN not in row.keys():
        raise ValueError(
            f"Query result must have a column named {self._SOURCE_COLUMN}")
      if self._TARGET_COLUMN not in row.keys():
        raise ValueError(
            f"Query result must have a column named {self._TARGET_COLUMN}")

      if self.edge_reversed:
        source_id = row[self._TARGET_COLUMN]
        target_id = row[self._SOURCE_COLUMN]
      else:
        source_id = row[self._SOURCE_COLUMN]
        target_id = row[self._TARGET_COLUMN]

      source_feature_name = _TRANSLATIONS[self._SOURCE_COLUMN]
      target_feature_name = _TRANSLATIONS[self._TARGET_COLUMN]

      example.features.feature[source_feature_name].bytes_list.value.append(
          source_id.encode("utf-8"))
      example.features.feature[target_feature_name].bytes_list.value.append(
          target_id.encode("utf-8"))

      ret_key = [source_id, target_id]
    else:
      raise ValueError("Row must represent at Node or Edge set.")

    id_is_set = False
    for feature_name, feature in self.fset.features.items():
      # In case client encodes `id`, `source` or `target` explicitly in the
      # features specification.
      tf_feature_name = _TRANSLATIONS.get(feature_name, feature_name)

      if feature_name not in row:
        raise ValueError(
            f"Could not find {feature_name} in query result: {row}")

      # If the ID is already set, skip the explicit feature enumeration.
      # This can happen if a table or query already contains a row named `id`.
      if (tf_feature_name == "#id" or tf_feature_name == "id") and id_is_set:
        continue
      else:
        id_is_set = True

      example_feature = example.features.feature[tf_feature_name]
      if feature.dtype == tf.float32.as_datatype_enum:
        example_feature.float_list.value.append(row[feature_name])
      if feature.dtype == tf.int64.as_datatype_enum:
        example_feature.int64_list.value.append(row[feature_name])
      if feature.dtype == tf.string.as_datatype_enum:
        example_feature.bytes_list.value.append(
            row.get(feature_name, "").encode("utf-8"))

    if len(ret_key) == 1:
      return ret_key[0].encode("utf-8"), example
    else:
      return ret_key[0].encode("utf-8"), ret_key[1].encode("utf-8"), example

  def expand(self, pcoll: beam.PCollection) -> beam.PCollection:
    result = (
        pcoll
        | f"{self.sfx}" >> self._bigquery_reader(**self.bq_args)
        |
        f"RowToKeyedExamples/{self.sfx}" >> beam.Map(self.row_to_keyed_example))
    if self.bq.reshuffle:
      result = result | f"Reshuffle/{self.sfx}" >> beam.Reshuffle()
    return result


class ReadTable(beam.PTransform):
  """Read a table of data, dispatch between formats.

  The collection produced yields tf.Example protos of the features from the
  file.

  Attributes:
    file_pattern: Pattern of filenames to read.
    file_format: File format of container. See module docstring.
      If not specified, it is inferred from the filename.
    converters: An optional dict of feature-name to a value Converter function.
      If this is provided, this is used to convert types in formats that don't
      already have a typed schema, e.g. CSV files.
  """

  def __init__(self, file_pattern: str, file_format: Optional[str] = None,
               converters: Optional[Converters] = None):
    super().__init__()
    self.file_pattern = file_pattern
    self.file_format = (file_format
                        if file_format
                        else guess_file_format(file_pattern))
    self.converters = converters

  def expand(self, pcoll: beam.PCollection) -> beam.PCollection:
    coder = beam.coders.ProtoCoder(Example)
    glob_pattern = expand_sharded_pattern(self.file_pattern)
    if self.file_format == "tfrecord":
      return (pcoll
              | beam.io.tfrecordio.ReadFromTFRecord(glob_pattern, coder=coder))
    # Placeholder for Google-internal file reads
    elif self.file_format == "csv":
      # We have to sniff out the schema of those files in order to create a
      # converter. Unfortunately we do this imperatively here.
      #
      # NOTE(blais): tfx_bsl has a nice 'csv_decoder' library for Beam that
      # works entirely in deferred mode that we could leverage eventually.
      # Consider adding a dependency.
      filenames = gfile.glob(self.file_pattern)
      first_filename = sorted(filenames)[0]
      with gfile.GFile(first_filename) as infile:
        header = next(iter(csv.reader(infile)))
      return (pcoll
              | beam.io.ReadFromText(glob_pattern, skip_header_lines=1)
              | beam.Map(csv_line_to_example, header,
                         converters=self.converters))
    else:
      raise NotImplementedError(
          "Format not supported: {}".format(self.file_format))


class WriteTable(beam.PTransform):
  """Write a table of data, dispatch between formats.

  The collection produced consumes tf.Example protos of the features from the
  file and writes them out in one of the supported graph tensor container
  formats.

  Attributes:
    file_pattern: Pattern of filenames to write.
    file_format: File format of container. See module docstring.
      If not specified, it is inferred from the filename.
    coder: The beam.coders.ProtoCoder to use to encode the protos.
  """

  def __init__(self,
               file_pattern: str,
               file_format: Optional[str] = None,
               coder=beam.coders.ProtoCoder(Example)):
    super().__init__()
    self.file_pattern = file_pattern
    self.coder = coder
    # Default to TFRecords if we have to guess and we cannot guess the file
    # format.
    if file_format:
      self.file_format = file_format
    else:
      try:
        self.file_format = guess_file_format(file_pattern)
      except ValueError:
        self.file_format = "tfrecord"

  def expand(self, pcoll: beam.PCollection) -> beam.PCollection:
    kwargs = get_sharded_pattern_args(self.file_pattern)
    if self.file_format == "tfrecord":
      return (pcoll
              | beam.io.tfrecordio.WriteToTFRecord(coder=self.coder, **kwargs))
  # Placeholder for Google-internal file writes
    else:
      raise NotImplementedError(
          "Format not supported: {}".format(self.file_format))


def csv_line_to_example(line: str,
                        header: List[str],
                        check_row_length: bool = True,
                        converters: Optional[Converters] = None) -> Example:
  """Convert a single CSV line row to a Example proto."""

  # Reuse the CSV module to handle quotations properly. Unfortunately csv reader
  # objects aren't pickleable, so this is less than ideal.
  row = next(iter(csv.reader([line])))
  if check_row_length:
    if len(row) != len(header):
      raise ValueError("Invalid row length: {} != {} from header: '{}'".format(
          len(row), len(header), header))

  # Convert to an example.
  example = Example()
  for field_name, value in zip(header, row):
    field_name = _TRANSLATIONS.get(field_name, field_name)
    feature = example.features.feature[field_name]

    # Run a custom converter on the value if provided.
    converter = None
    if converters is not None:
      converter = converters.get(field_name, None)
    if converter is not None:
      value = converter(feature, value)
    else:
      feature.bytes_list.value.append(value.encode("utf8"))
  return example
