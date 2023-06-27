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
import functools
import hashlib
import os
import queue
import re
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, Text, Tuple, Union, Iterable

from absl import logging
import apache_beam as beam
import pyarrow
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.proto import graph_schema_pb2
# Placeholder for optional Google-internal record file format pipeline import
# Placeholder for optional Google-internal record file format import
# Placeholder for optional Google-internal sorted string file format utils
try:
  # pylint: disable-next=g-import-not-at-top
  import google.cloud.bigquery_storage_v1 as bq_storage  # pytype: disable=import-error
except ImportError:
  bq_storage = None


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

    # Save the collection for output.
    set_dict = pcoll_dict.setdefault(set_type, {})
    if fset.metadata.HasField("filename"):
      pcoll = (
          rcoll
          | f"ReadFile/{set_name}" >> ReadUnigraphPieceFromFile(
              set_type, set_name, fset, graph_dir))
      set_dict[set_name] = pcoll
    elif fset.metadata.HasField("bigquery"):
      pcoll = (
          rcoll | f"ReadBigQuery/{set_name}" >> ReadUnigraphPieceFromBigQuery(
              set_name,
              fset,
              gcs_location=gcs_location,
              bigquery_reader=bigquery_reader))
      set_dict[set_name] = pcoll

  return pcoll_dict


def is_edge_reversed(schema_edge_set: graph_schema_pb2.EdgeSet):
  for kv in schema_edge_set.metadata.extra:
    if kv.key == "edge_type" and kv.value == "reversed":
      return True
  return False


# Needed for cloud storage paths to be treated like absolute paths
# instead of being spuriously joined with a directory.
# TODO(b/287083322): Identify if there's a library to check cloud URLs.
def _is_complete_path(path: str) -> bool:
  return (os.path.isabs(path)
          or path.startswith("gs://")
          or path.startswith("s3://"))


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
    if not _is_complete_path(self.filename):
      if not self.graph_dir:
        raise ValueError(f"{self.filename} does not specify a full path "
                         "and graph_dir is None.")
      self.filename = os.path.join(self.graph_dir, self.filename)

    self.converters = build_converter_from_schema(self.fset.features)

    if isinstance(fset, graph_schema_pb2.EdgeSet):
      self.reversed = is_edge_reversed(fset)

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


def _to_bytes(x: Union[str, bytes, pyarrow.StringScalar]) -> bytes:
  if isinstance(x, bytes):
    return x
  elif isinstance(x, str):
    return x.encode("utf-8")
  elif isinstance(x, pyarrow.StringScalar):
    return _to_bytes(x.as_py())
  else:
    raise TypeError("_to_bytes cannot handle type %s" % str(type(x)))


class ReadUnigraphPieceFromBigQuery(beam.PTransform):
  """Read a unigraph node/edge/context component from a BigQuery table.

  Yeilds tf.Example protos of the features from the table.

  **NOTE**(b/252789408): only scalar features (bool, float, int and string) are
    currently supported when using a BQ source.
  """
  _SUPPORTED_DTYPES = [
      tf.dtypes.float32, tf.dtypes.int64, tf.dtypes.string, tf.dtypes.bool
  ]
  ID_COLUMN = "id"
  SOURCE_COLUMN = "source"
  TARGET_COLUMN = "target"

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
    self.edge_reversed = (isinstance(self.fset, graph_schema_pb2.EdgeSet) and
                          is_edge_reversed(self.fset))

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
      bq_args["method"] = "EXPORT"
    elif bq.read_method == graph_schema_pb2.BigQuery.DIRECT_READ:
      bq_args["method"] = "DIRECT_READ"

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

  def _row_to_keyed_example(self, row: Mapping[str, Any]) -> Any:
    return ReadUnigraphPieceFromBigQuery.row_to_keyed_example(
        row, self.fset, edge_reversed=self.edge_reversed)

  @staticmethod
  def row_to_keyed_example(
      row: Mapping[str, Any],
      fset: Union[graph_schema_pb2.NodeSet, graph_schema_pb2.EdgeSet],
      edge_reversed=False, output_bq_row=False) -> Any:
    """Convert a single row from a BigQuery result to tf.Example.

    Will extract values from the retrieved BigQuery row according to the
    features specified by the GraphSchema. This is to support discarding
    entries in the BQ row that are not relevant to the TFGNN model.

    For node sets, it is expected that the BQ row have a column named `id`.
    Any feature with key: 'id' in the input GraphSchema will be ignored.

    For edge sets, the returned BQ row must have columns named `source` and
    `target`. Any feature specified in the GraphSchema with key: `source` or
    key: `target` will be ignored.

    **NOTE**(b/252789408): only scalar features (float, int and string) are
    currently supported when using a BQ source.

    Args:
      row: Dict[str, Any] result of a BigQuery read.
      fset: Schema for NodeSet or EdgeSet.
      edge_reversed: Applicable if `isinstance(fset, graph_schema_pb2.EdgeSet)`.
        If set, edges would be reversed. Specifically, the return would be
        tuple (target, source, example).
      output_bq_row: If set, then output tuple[-1] would be BigQuery row.
        Otherwise (default), then output tuple[-1] will be `tf.Example`.

    Returns:
      If the input fset is a graph_schema_pb2.NodeSet, returns
        Tuple(id: str, example: tf.train.Example).
      If the input fset is a graph_schema_pb2.Edgeset, returns
        Tuple(source: str, target: str, example: tf.train.Example)

    Raises:
      ValueError: If a field name is not found in the BQ row.
      ValueError: If an input feature has an unspported data type
        (see GraphSchema and BigQuery documentation for valid data type
        specifications).
    """
    ret_key = None
    cls = ReadUnigraphPieceFromBigQuery  # for short

    if isinstance(fset, graph_schema_pb2.NodeSet):
      if cls.ID_COLUMN not in row.keys():
        raise ValueError(
            f"Query result must have a column named {cls.ID_COLUMN}")

      node_id = row[cls.ID_COLUMN]
      ret_key = [node_id]

    elif isinstance(fset, graph_schema_pb2.EdgeSet):
      if cls.SOURCE_COLUMN not in row.keys():
        raise ValueError(
            f"Query result must have a column named {cls.SOURCE_COLUMN}")
      if cls.TARGET_COLUMN not in row.keys():
        raise ValueError(
            f"Query result must have a column named {cls.TARGET_COLUMN}")

      if edge_reversed:
        source_id = row[cls.TARGET_COLUMN]
        target_id = row[cls.SOURCE_COLUMN]
      else:
        source_id = row[cls.SOURCE_COLUMN]
        target_id = row[cls.TARGET_COLUMN]

      ret_key = [source_id, target_id]
    else:
      raise ValueError("Row must represent at Node or Edge set.")

    if output_bq_row:
      output_record = row
    else:
      output_record = example = Example()
      if isinstance(fset, graph_schema_pb2.NodeSet):
        example.features.feature[NODE_ID].bytes_list.value.append(
            _to_bytes(node_id))
      elif isinstance(fset, graph_schema_pb2.EdgeSet):
        source_feature_name = _TRANSLATIONS[cls.SOURCE_COLUMN]
        target_feature_name = _TRANSLATIONS[cls.TARGET_COLUMN]

        example.features.feature[source_feature_name].bytes_list.value.append(
            _to_bytes(source_id))
        example.features.feature[target_feature_name].bytes_list.value.append(
            _to_bytes(target_id))

      for feature_name, feature in fset.features.items():
        # In case client encodes `id`, `source` or `target` explicitly in the
        # features specification.
        tf_feature_name = _TRANSLATIONS.get(feature_name, feature_name)

        # The `id`, `source` or `target` fields should already be set, skip any
        # user-defined feature in the input GraphSchema.
        if tf_feature_name in {NODE_ID, SOURCE_ID, TARGET_ID}:
          continue

        if feature_name not in row.keys():
          raise ValueError(
              f"Could not find {feature_name} in query result: {row}")

        feature_value = row.get(feature_name)
        example_feature = example.features.feature[tf_feature_name]
        try:
          if feature.dtype == tf.float32.as_datatype_enum:
            example_feature.float_list.value.append(float(feature_value))
          elif feature.dtype == tf.bool.as_datatype_enum:
            example_feature.int64_list.value.append(int(feature_value))
          elif feature.dtype == tf.int64.as_datatype_enum:
            example_feature.int64_list.value.append(int(feature_value))
          elif feature.dtype == tf.string.as_datatype_enum:
            example_feature.bytes_list.value.append(
                str(row.get(feature_name, "")).encode("utf-8"))
          else:
            raise ValueError(f"Unknown feature.dtype: {feature.dtype}")
        except Exception as exc:
          raise TypeError(f"Feature {feature_name} registered as dtype "
                          f"{feature} but receieved {feature_value}") from exc

    if len(ret_key) == 1:
      return _to_bytes(ret_key[0]), output_record
    else:
      return _to_bytes(ret_key[0]), _to_bytes(ret_key[1]), output_record

  def expand(self, pcoll: beam.PCollection) -> beam.PCollection:
    result = (
        pcoll
        | f"{self.sfx}" >> self._bigquery_reader(**self.bq_args)
        | f"RowToKeyedExamples/{self.sfx}"
        >> beam.Map(self._row_to_keyed_example))
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


def _csv_fields_to_example(
    fields_and_values: Iterable[Tuple[str, Any]],
    converters: Optional[Converters] = None) -> Example:
  """Converts CSV fields and values, `to tf.Example`."""
  # Convert to an example.
  example = Example()
  for field_name, value in fields_and_values:
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


def csv_line_to_example(line: str,
                        header: List[str],
                        check_row_length: bool = True,
                        converters: Optional[Converters] = None) -> Example:
  """Converts CSV fields and values, `to tf.Example`."""
  # Reuse the CSV module to handle quotations properly. Unfortunately csv reader
  # objects aren't pickleable, so this is less than ideal.
  row = next(iter(csv.reader([line])))
  if check_row_length:
    if len(row) != len(header):
      raise ValueError("Invalid row length: {} != {} from header: '{}'".format(
          len(row), len(header), header))
  return _csv_fields_to_example(zip(header, row), converters)


def read_schema(schema_file: str) -> tfgnn.GraphSchema:
  graph_dir = os.path.dirname(schema_file)
  graph_schema = tfgnn.read_schema(schema_file)
  for unused_type, unused_set_name, feats in tfgnn.iter_sets(graph_schema):
    if feats.HasField("metadata") and feats.metadata.HasField("filename"):
      feats.metadata.filename = os.path.join(graph_dir, feats.metadata.filename)
  return graph_schema


_BQ_CLIENT_SINGLETON = None


def _get_bq_singleton_client():
  """Returns singleton instance of `bq_storage.BigQueryReadClient`."""
  if bq_storage is None:
    raise ImportError("Could not `import google.cloud.bigquery_storage_v1`. "
                      "To use BigQuery, make sure it is available and/or "
                      "linked into your binary.")
  global _BQ_CLIENT_SINGLETON
  if _BQ_CLIENT_SINGLETON is None:
    _BQ_CLIENT_SINGLETON = bq_storage.BigQueryReadClient()
  return _BQ_CLIENT_SINGLETON


class DictStreams:
  """Provide methods for streaming Unigraph artifacts (nodes and edges).

  All `Iterable` instances stream read and map records on-the-fly. E.g., using
  files (tfrecord file, CSV file), or BigQuery tables.

  High-level functions are `iter_edges_via_*`, `iter_nodes_via_*`, and
  `iter_graph_via_*`. Input can be path to text proto of `GraphSchema` using,
  e.g., `read_nodes_via_path(path_to_pbtxt)`, or via instance of `GraphSchema`,
  e.g., `read_nodes_via_schema(unigraph.read_schema(path_to_pbtxt))`.

    * read_nodes_via_*: return dict, with each node-set-name being a key, and
      values being streams of (node ID, tf.Example containing features).
    * read_edges_via_*: return dict, with each edge-set-name being a key, and
      values being streams of (src ID, tgt ID, tf.Example containing features).
    * read_graph_via_*: Invokes above two methods and combines them to return
      `{'tfgnn.NODES': read_nodes_via_*() , 'tfgnn.NODES': read_edges_via_*()}`.

  `GraphSchema` is expected to configure the data source through the `metadata`
  attribute of `node_sets` and `edge_sets`. For example, `metadata` attribute
  can have `filename` attribute populated (with path to .csv, .tfrecord, etc),
  or can have `bigquery` attribute populated (e.g., populating `table_spec`).
  """

  @staticmethod
  def iter_tfrecord_examples(
      file_path: str,
      unused_fset: Optional[
          Union[graph_schema_pb2.NodeSet, graph_schema_pb2.EdgeSet]] = None
      ) -> Iterable[Example]:
    """Yields `tf.Example` from tfrecord file."""
    for example in tf.data.TFRecordDataset(file_path):
      yield Example.FromString(example.numpy())

  @staticmethod
  def iter_csv_examples(
      file_path: str,
      fset: Optional[
          Union[graph_schema_pb2.NodeSet, graph_schema_pb2.EdgeSet]] = None
      ) -> Iterable[Example]:
    """Yields `tf.Example` from CSV files."""
    if fset is not None:
      converters = build_converter_from_schema(fset.features)
    else:
      converters = None
    csv_records = csv.DictReader(gfile.GFile(file_path, "r"))
    for csv_record in csv_records:
      yield _csv_fields_to_example(csv_record.items(), converters=converters)

  @staticmethod
  def fn_iter_from_file(file_format) -> Callable[  # pylint: disable=missing-function-docstring
      [str, Union[None, graph_schema_pb2.NodeSet, graph_schema_pb2.EdgeSet]],
      Iterable[Example]]:
    lookup = {
        "csv": DictStreams.iter_csv_examples,
        "tfrecord": DictStreams.iter_tfrecord_examples,
        # "capacitor": lambda path, fset: raise ValueError("Not implemented")
        # "recordio": lambda path, fset: raise ValueError("Not implemented")
    }
        # Iterator for Google-internal data file type.
    return lookup[file_format]

  @staticmethod
  def iter_records_from_filepattern(
      filepattern: str,
      fset: Optional[
          Union[graph_schema_pb2.NodeSet, graph_schema_pb2.EdgeSet]] = None
      ) -> Iterable[Example]:
    """Yields records from SSTables and other data sources."""
    file_format = guess_file_format(filepattern)
    filepattern = expand_sharded_pattern(filepattern)
    files = gfile.glob(filepattern)

    if not files:
      error_str = "No files found for pattern: (%s)." % filepattern
      if not filepattern.startswith("/"):
        error_str += (" You can read GraphSchema using unigraph.read_schema(), "
                      "which converts relative paths to absolute paths.")
      raise ValueError(error_str)

    for filename in files:
      iterator = DictStreams.fn_iter_from_file(file_format)
      for record in iterator(filename, fset):
        yield record

  @staticmethod
  def iter_records_from_bigquery(
      bq_schema: tfgnn.proto.graph_schema_pb2.BigQuery) -> Iterable[
          Dict[str, pyarrow.Scalar]]:
    """Yields records from BigQuery."""
    # NOTE: Does not work if .sql is used -- proto uses "oneof".
    table_spec = bq_schema.table_spec
    if bq_schema.HasField("sql"):
      raise NotImplementedError(
          "Currently, we do not accept SQL statements.")
      # NOTE: this can be implemented as:
      #     from google.cloud import bigquery
      #     client = bigquery.Client(project=table_spec.project)
      #     query_iterator = client.query(sql_query)
      #     records = query_iterator.result(page_size=100_000)

    assert table_spec.project
    assert table_spec.dataset and table_spec.table

    client = _get_bq_singleton_client()

    session = client.create_read_session(
        parent="projects/" + table_spec.project,
        read_session=bq_storage.types.ReadSession(
            data_format=bq_storage.types.DataFormat.ARROW,
            table="/".join((
                "projects", table_spec.project, "datasets", table_spec.dataset,
                "tables", table_spec.table))),
        max_stream_count=10)

    row_iterators = [client.read_rows(stream.name).rows(session)
                     for stream in session.streams]
    records = ParallelMergingIterator(row_iterators)
    return records.iter_all()

  @staticmethod
  def iter_nodes_via_path(schema_file_or_dir: str) ->  Dict[
      str, Iterable[Tuple[bytes, Example]]]:
    return DictStreams.iter_nodes_via_schema(
        read_schema(find_schema_filename(schema_file_or_dir)))

  @staticmethod
  def iter_nodes_via_schema(schema: tfgnn.GraphSchema) -> Dict[
      str, Iterable[Tuple[bytes, Example]]]:
    """Dict of node-set-name to iterator (node ID, `tf.Example`).

    Args:
      schema (tfgnn.GraphSchema): Every `schema.node_sets`, with attribute
        `metadata`, will appear in output. The stream of `tf.Example` will
        contain features, as configured in node set schema.

    Returns:
      dict with keys=node-set names; values=stream of node IDs to `tf.Example`.
    """
    dict_streams = {}
    for node_set_name, node_schema in schema.node_sets.items():
      if not node_schema.HasField("metadata"):
        continue
      if node_schema.metadata.HasField("filename"):
        records = DictStreams.iter_records_from_filepattern(
            node_schema.metadata.filename, node_schema)
        records = map(get_node_ids, records)
      elif node_schema.metadata.HasField("bigquery"):
        records = DictStreams.iter_records_from_bigquery(
            node_schema.metadata.bigquery)
        extract_example_fn = functools.partial(
            ReadUnigraphPieceFromBigQuery.row_to_keyed_example,
            fset=node_schema, output_bq_row=True)
        records = map(extract_example_fn, records)

      dict_streams[node_set_name] = records

    return dict_streams

  @staticmethod
  def iter_edges_via_path(schema_file_or_dir: str) ->  Dict[
      str, Iterable[Tuple[bytes, bytes, Example]]]:
    return DictStreams.iter_edges_via_schema(
        read_schema(find_schema_filename(schema_file_or_dir)))

  @staticmethod
  def iter_edges_via_schema(schema: tfgnn.GraphSchema) -> Dict[
      str, Iterable[Tuple[bytes, bytes, Example]]]:
    """EdgeSetName to iterator of tuple(source ID, target ID, `tf.Example`)."""
    dict_streams = {}
    for edge_set_name, edge_schema in schema.edge_sets.items():
      if not edge_schema.HasField("metadata"):
        continue
      is_reversed = is_edge_reversed(edge_schema)
      if edge_schema.metadata.HasField("filename"):
        records = DictStreams.iter_records_from_filepattern(
            edge_schema.metadata.filename, edge_schema)
        records = map(
            functools.partial(get_edge_ids, edge_reversed=is_reversed), records)
      elif edge_schema.metadata.HasField("bigquery"):
        records = DictStreams.iter_records_from_bigquery(
            edge_schema.metadata.bigquery)
        extract_example_fn = functools.partial(
            ReadUnigraphPieceFromBigQuery.row_to_keyed_example,
            fset=edge_schema, edge_reversed=is_reversed, output_bq_row=True)
        records = map(extract_example_fn, records)

      dict_streams[edge_set_name] = records
    return dict_streams

  @staticmethod
  def iter_graph_via_path(schema_file_or_dir) -> Dict[
      str, Dict[
          str, Union[
              Iterable[Tuple[bytes, bytes, Example]],
              Iterable[Tuple[bytes, Example]]]]]:
    return DictStreams.iter_graph_via_schema(
        read_schema(find_schema_filename(schema_file_or_dir)))

  @staticmethod
  def iter_graph_via_schema(schema: tfgnn.GraphSchema) -> Dict[
      str, Dict[
          str, Union[
              Iterable[Tuple[bytes, bytes, Example]],
              Iterable[Tuple[bytes, Example]]]]]:
    return {
        tfgnn.NODES: DictStreams.iter_nodes_via_schema(schema),
        tfgnn.EDGES: DictStreams.iter_edges_via_schema(schema),
    }


class ParallelMergingIterator:
  """Combines multiple `ReadRowsIterable` iterators into one stream.

  It uses multi-threading. Each iterator will be read in a different thread. The
  method `iter_all()` runs in main thread, pooling information from the various
  threads. Threads write to synchronized `queue.Queue`, form which, main thread
  reads.
  """

  def __init__(self, iterators: List[Any]):
    """Initializes thread-safe queue without starting threads.

    Method `iter_all()` kicks-off the threads.

    Args:
      iterators (List[bq_storage.reader.ReadRowsIterable]): list of iteratbles
        that will be read, each on a different thread.
    """
    self._iterators = iterators
    self.threads_started = False
    self.queue = queue.Queue(len(iterators))  # Thread-safe.
    self.threads_started = False
    self._cur_page = None

  def iter_all(self) -> Iterable[Dict[str, pyarrow.Scalar]]:
    """Yields all rows in all constructor `iterators`, in arbitrary order."""
    self._maybe_start_threads()
    finished_threads = set()
    while len(finished_threads) < len(self._iterators):
      valid_page, page = self.queue.get()
      if not valid_page:
        finished_threads.add(page)
        continue

      for record in page:
        yield record

  def _maybe_start_threads(self):
    if self.threads_started:
      return
    self.finished_threads = set()
    for i in range(len(self._iterators)):
      threading.Thread(
          target=functools.partial(self.populate_queue, i),
          daemon=True).start()
    self.threads_started = True

  def populate_queue(self, iterator_index: int):
    iterator = self._iterators[iterator_index]
    for page in iterator.pages:
      self.queue.put((True, page))
    self.queue.put((False, iterator_index))
