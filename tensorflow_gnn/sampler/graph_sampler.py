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
"""Beam-based graph sampler for homogeneous and heterogeneous graphs.

Samples multiple hops of a graph stored as node entries with adjacency list and
node features (TFRecords file of tf.train.Example protos). Outputs graph tensor
encoded tensorflow.Example protos in a TFRecord wrapper, one per seed node.

The input is in Universal Graph Format (see documentation of unigraph.py
module). The graph may be homogeneous or heterogeneous. The container format is
inferred from the filename fields in the schema (e.g. "TFRecord").
"""

import enum
import functools
import os
from typing import Any, Dict, Iterable, Iterator, Optional, Set, Tuple

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import PipelineOptions
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.data import unigraph
from tensorflow_gnn.sampler import sampling_lib
from tensorflow_gnn.sampler import sampling_spec_pb2
from tensorflow_gnn.sampler import subgraph

from google.protobuf import text_format

_DIRECT_RUNNER = "DirectRunner"
_DATAFLOW_RUNNER = "DataflowRunner"

PCollection = beam.pvalue.PCollection
Example = tf.train.Example
Features = tf.train.Features
Node = sampling_lib.Node
NodeFeatures = sampling_lib.NodeFeatures
NodeId = sampling_lib.NodeId
SampleId = sampling_lib.SampleId
SampledEdges = sampling_lib.SampledEdges


@enum.unique
class EdgeAggregationMethod(enum.Enum):
  """Subgraph generation method.

  NODE: node-centric edge aggregation method. Accumulate all edges in subgraphs
    from sampled nodes regardless if the edges was traversed during sampling.
    This is called `node` based edge aggregation because the adjacency list
    of each node of the subgraph is filtered on nodes included in the subgraph
    sample and all remaining edges are added to the subgraph sample, regardless
    of the sampling traversal path.

  EDGE: For each subgraph sample, only include edges that have been traversed
    while sampling.


  **NOTE** Currently, only `node` based sampling is supported. `edge` based
  sampling will be added shortly.

  Clarifying Example:

  Assume the simple graph with nodes N = {A, B, C} and edges
  E = {A -> B, A -> C, B -> C}. If we sampled by exploring edges
  S = {A -> B, A -> C}; `node` subgraph generation **would** include the
  edge  B->C in the output S_n = {A -> B, A -> C, B -> C}. If `edge` node
  subgraph generation is used, B -> C **would not** be included in the output:
  S_e = {A -> B, A -> C}. 'edge' vs. 'node' subgraph generation trades off
  time and space complexity for local topology representation.
  """
  EDGE = "edge"
  NODE = "node"


def augment_schema_with_sample_features(schema: tfgnn.GraphSchema):
  """Add the `sample_id` and `seed_id` features to the schema."""
  features = schema.context.features
  if "seed_id" not in schema.context.features:
    features["seed_id"].dtype = tf.string.as_datatype_enum
  if "sample_id" not in schema.context.features:
    features["sample_id"].dtype = tf.string.as_datatype_enum


def augment_schema_with_node_ids(schema: tfgnn.GraphSchema,
                                 node_id_feature: str):
  """Add node id feature to schema. Mutate `schema` in place."""
  for node_set in schema.node_sets.values():
    if node_id_feature not in node_set.features:
      node_set.features[node_id_feature].dtype = tf.string.as_datatype_enum


def validate_schema(schema: tfgnn.GraphSchema):
  """Validates assumptions of the graph schema."""

  # Ensure that the filenames are set on the universal graph format.
  for set_type, set_name, set_ in tfgnn.iter_sets(schema):
    if not set_.HasField("metadata"):
      raise ValueError(
          f"{set_type}/{set_name} does not specify a metadata field")

    if not (set_.metadata.HasField("filename") or
            set_.metadata.HasField("bigquery")):
      raise ValueError(f"{set_type}/{set_name} does not specify a "
                       "filename or bigquery source.")

  # Check for raggedness, which we don't support in this sampler (the format
  # supports it).
  for set_type, set_name, set_, feature in tfgnn.iter_features(schema):
    if (feature.HasField("shape") and
        any(dim.size == -1 for dim in feature.shape.dim)):
      raise ValueError("Ragged features aren't supported on {}: {}".format(
          (set_type, set_name), feature))


def create_beam_runner(
    runner_name: Optional[str]) -> beam.runners.PipelineRunner:
  """Creates appropriate runner."""
  if runner_name == _DIRECT_RUNNER:
    runner = beam.runners.DirectRunner()
  elif runner_name == _DATAFLOW_RUNNER:
    runner = beam.runners.DataflowRunner()
  # Placeholder for creating Google-internal pipeline runner
  else:
    runner = None
  return runner


def _validate_sampling_spec(sampling_spec: sampling_spec_pb2.SamplingSpec,
                            schema: tfgnn.GraphSchema):
  """Performs static validation over the given sampling specification.

  Args:
    sampling_spec: The specification to validate.
    schema: The associated GraphSchema to use for validating the sampling spec.

  Raises:
    ValueError if the sampling specification can be statically determined to
    be invalid, e.g. if the operation names are not unique.
  """
  if not isinstance(sampling_spec, sampling_spec_pb2.SamplingSpec):
    raise ValueError(f"Invalid sampling_spec configuration: {sampling_spec}")

  # Validate the seed operation
  if not sampling_spec.seed_op:
    raise ValueError("Sampling spec must be defined.")
  if not sampling_spec.seed_op.op_name:
    raise ValueError("Sampling spec seed operation name must be defined.")
  node_set_name = sampling_spec.seed_op.node_set_name
  if node_set_name not in schema.node_sets:
    raise ValueError(f"Sampling spec seed node set name '{node_set_name}' does "
                     f"not exist in graph node sets.")

  # Validate the sampling operations.
  sampling_ops: Set[str] = {sampling_spec.seed_op.op_name}
  for sampling_op in sampling_spec.sampling_ops:
    if sampling_op.edge_set_name not in schema.edge_sets:
      raise ValueError(
          f"Sampling spec edge set name '{sampling_op.edge_set_name}' does not "
          f"exist in graph edge sets.")
    if sampling_op.strategy not in (
        sampling_spec_pb2.SamplingStrategy.TOP_K,
        sampling_spec_pb2.SamplingStrategy.RANDOM_UNIFORM,
        sampling_spec_pb2.SamplingStrategy.RANDOM_WEIGHTED):
      raise ValueError(
          f"Unsupported sampling strategy {sampling_op.strategy} for operation "
          f"{sampling_op.op_name}, only "
          "TOP_K, RANDOM_UNIFORM and RANDOM_WEIGHTED "
          "are supported at this time.")
    if sampling_op.sample_size <= 0:
      raise ValueError(
          f"Unsupported sample size {sampling_op.sample_size} for operation "
          f"{sampling_op.op_name}.")
    for input_op_name in sampling_op.input_op_names:
      if input_op_name not in sampling_ops:
        raise ValueError(
            f"Invalid execution graph: Input operation '{input_op_name}' "
            f"from operation '{sampling_op.op_name}' not previously defined.")
    sampling_ops.add(sampling_op.op_name)


def _clean_metadata(metadata):
  """Clears the cardinality and filename fields of `metadata` inline."""
  metadata.ClearField("cardinality")
  metadata.ClearField("filename")


def convert_samples_to_examples(
    schema: tfgnn.GraphSchema,
    seeds: Dict[tfgnn.NodeSetName, PCollection[Tuple[SampleId, NodeId]]],
    edges: Dict[tfgnn.EdgeSetName, SampledEdges],
    nodes: Dict[tfgnn.NodeSetName, NodeFeatures]) -> PCollection[Example]:
  """Converts sampled nodes and edges to Tensorflow Example."""

  def filter_by_prefix(prefix: str, group: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key[len(prefix):]: value
        for key, value in group.items()
        if key.startswith(prefix)
    }

  def convert_to_tf_example(schema: tfgnn.GraphSchema, sample_id: SampleId,
                            group: Dict[str, Any]) -> Example:
    import tensorflow as tf  # pylint: disable=reimported,g-import-not-at-top,redefined-outer-name
    from tensorflow_gnn.sampler import subgraph  # pylint: disable=reimported,g-import-not-at-top,redefined-outer-name
    seeds: Dict[tfgnn.NodeSetName, Iterable[NodeId]] = (
        filter_by_prefix("seeds/", group))
    node_sets: Dict[tfgnn.NodeSetName, Dict[NodeId, Features]] = {
        set_name: dict(features)
        for set_name, features in filter_by_prefix("nodes/", group).items()
    }
    edge_sets: Dict[tfgnn.EdgeSetName, Iterable[Node]] = (
        filter_by_prefix("edges/", group))

    if len(seeds) != 1:
      raise ValueError("Sampling from multiple seed node sets is not supported,"
                       f" seed node sets {sorted(seeds.keys())}.")
    seed_ids: Iterable[bytes] = next(iter(seeds.values()))
    context = tf.train.Features()
    assert "sample_id" in schema.context.features, schema
    context.feature["sample_id"].bytes_list.value.append(sample_id)
    assert "seed_id" in schema.context.features, schema
    context.feature["seed_id"].bytes_list.value.extend(seed_ids)

    return subgraph.encode_subgraph_pieces_to_example(
        schema,
        seeds,
        context=context,
        node_sets=node_sets,
        edge_sets=edge_sets)

  group = {}
  for node_set_name, values in seeds.items():
    group[f"seeds/{node_set_name}"] = values
  for node_set_name, values in nodes.items():
    group[f"nodes/{node_set_name}"] = values
  for edge_set_name, values in edges.items():
    group[f"edges/{edge_set_name}"] = values

  return (group
          | "CreateGraphTensors/CoGroupBySampleId" >> beam.CoGroupByKey()
          | "CreateGraphTensors/ConvertToTfExample" >> beam.MapTuple(
              functools.partial(convert_to_tf_example, schema)))


def run_sample_graph_pipeline(
    schema: unigraph.graph_schema_pb2.GraphSchema,
    sampling_spec: sampling_spec_pb2.SamplingSpec,
    edge_aggregation_method: EdgeAggregationMethod,
    output_pattern: str,
    graph_root_path: Optional[str] = None,
    seeds_filename: Optional[str] = None,
    runner_name: Optional[str] = None,
    pipeline_options: Optional[PipelineOptions] = None) -> beam.pvalue.PDone:
  """Runs the pipeline on a graph, which may be homogeneous or heterogeneous.

  Args:
    schema: A unigraph.graph_Schema_pb2.GraphSchema protobuf message describing
      heterogeneous topology and data locations.
    sampling_spec: A sampling_spec_pb2.SamplingSpec protobuf message describing
      the desired sampling tragectory.
    edge_aggregation_method: An enum instance of EdgeAggregationMethod, either
      NODE or EDGE. See enum doc string for more information.
    output_pattern: A (potentially sharded) file pattern for graph sample
      output. A GCS bucket is common.
    graph_root_path: Optional string root directory file-backed unigraph. This
      field is only necessary if defining unigraph components with files on
      persistent storage and with filename metadata fields that specify relative
      locations. `graph_root_path` is specified, the final filename of any graph
      schema component will be the path concatenation of this and the relative
      filename in the filename metadata, e.g., os.path.join(graph_root_path,
      <filename>).
    seeds_filename: An optional filename to a CSV file of seed ids. If a
      `seeds_filename` is not provided, the seed_op in the sampling
      specification will use all the ids of the specified node set.
    runner_name: Optional string specifying a beam runner, e.g., 'DirectRunner'
      (default) or 'DataflowRunner'.
    pipeline_options: Additional beam pipeline options provided to the pipeline
      runner.

  Returns:
    beam.PDone object.
  """

  # Validate configuration.
  _validate_sampling_spec(sampling_spec, schema)
  logging.info("Sampling specification validated. Pipeline commencing...")
  # Produce the sample schema to output.
  sampled_schema = tfgnn.GraphSchema()
  sampled_schema.CopyFrom(schema)

  # Add missing fields to schema
  augment_schema_with_sample_features(sampled_schema)

  # Remove any metadata fields.
  _clean_metadata(sampled_schema.context.metadata)
  for node_set in sampled_schema.node_sets.values():
    _clean_metadata(node_set.metadata)
  for edge_set in sampled_schema.edge_sets.values():
    _clean_metadata(edge_set.metadata)

  with beam.Pipeline(
      runner=create_beam_runner(runner_name), options=pipeline_options) as root:
    # Read the graph.
    gcs_location = None
    if pipeline_options:
      gcs_location = pipeline_options.view_as(GoogleCloudOptions).temp_location
    graph_dict = unigraph.read_graph(
        schema, graph_root_path, root, gcs_location=gcs_location)

    # Read the seeds, or use the node ids as the seeds.
    if seeds_filename:
      seed_nodes = unigraph.read_node_set(root, seeds_filename, "seeds")
    else:
      seed_nodes = graph_dict["nodes"][sampling_spec.seed_op.node_set_name]

    # Currently sample id is generated from source node ids.
    def create_sample_id(node_id: NodeId,
                         count: int) -> Iterator[Tuple[SampleId, NodeId]]:
      yield (node_id, node_id)
      for index in range(1, count):
        yield (node_id + b":" + str.encode(str(index)), node_id)

    seeds: Dict[tfgnn.NodeSetName, PCollection[Tuple[SampleId, NodeId]]] = {
        sampling_spec.seed_op.node_set_name:
            (seed_nodes | beam.Keys()
             | "Seeds/CountUnique" >> beam.combiners.Count.PerElement()
             | "Seeds/CreateSampleId" >> beam.FlatMapTuple(create_sample_id))
    }

    adj_lists = sampling_lib.create_adjacency_lists(
        graph_dict, sort_edges=False)

    sampled_edges = sampling_lib.sample_edges(sampling_spec, seeds, adj_lists)

    logging.info("sampled_schema: %s", sampled_schema)
    node_ids = sampling_lib.create_unique_node_ids(sampled_schema, seeds,
                                                   sampled_edges)
    node_features = sampling_lib.lookup_node_features(node_ids,
                                                      graph_dict["nodes"])

    if edge_aggregation_method == EdgeAggregationMethod.NODE:
      sampled_edges = sampling_lib.find_connecting_edges(
          schema, node_ids, adj_lists)
    else:
      assert edge_aggregation_method == EdgeAggregationMethod.EDGE

    graph_tensors = convert_samples_to_examples(sampled_schema, seeds,
                                                sampled_edges, node_features)

    done = (
        graph_tensors
        | "GraphTensors/Reshuffle" >> beam.Reshuffle()
        | "GraphTensors/Write" >> unigraph.WriteTable(output_pattern))

  logging.info("Pipeline complete, writing output...")
  # Produce output schema of the tensors.
  output_schema_filename = f"{os.path.dirname(output_pattern)}/schema.pbtxt"
  tfgnn.write_schema(sampled_schema, output_schema_filename)

  logging.info("Sampling complete.")
  return done


def define_flags():
  """Creates commandline flags."""

  flags.DEFINE_string(
      "graph_schema", None,
      "Path to a text-formatted GraphSchema proto file or directory "
      "containing one for a graph in Universal Graph Format. This "
      "defines the input graph to be sampled.")

  flags.DEFINE_string(
      "input_seeds", None,
      "Path to an input file with the seed node ids to restrict sampling over. "
      "The file can be in any of the supported unigraph table formats, and as "
      "for node sets, the 'id' column will be used. If the seeds aren't "
      "specified, the full set of nodes from the graph will be used "
      "(optional).")

  flags.DEFINE_string(
      "sampling_spec", None,
      "An input file with a text-formatted SamplingSpec proto to use. This is "
      "a required input and to some extent may mirror some of the schema's "
      "structure. See `sampling_spec.proto` for details on the configuration.")

  flags.DEFINE_enum_class(
      "edge_aggregation_method", EdgeAggregationMethod.EDGE,
      EdgeAggregationMethod,
      "Given a subgraph sampling, specifies a method for retaining edges "
      "between nodes in the subgraph. `node` sampling includes all edges "
      "between the nodes in the sampled subgraph in the output.  `edge` "
      "aggregation only includes edges in the subgraph if the edge was "
      "traversed during sampling.")

  flags.DEFINE_string(
      "output_samples", None,
      "Output file with serialized graph tensor Example protos.")

  runner_choices = [_DIRECT_RUNNER, _DATAFLOW_RUNNER]
  # Placeholder for Google-internal pipeline runner
  flags.DEFINE_enum(
      "runner", None, runner_choices,
      "The underlying runner; if not specified, use the default runner.")

  flags.mark_flags_as_required(
      ["graph_schema", "sampling_spec", "output_samples"])


def app_main(argv):
  """Main sampler entrypoint.

  Args:
    argv: List of arguments passed by flags parser.
  """
  FLAGS = flags.FLAGS  # pylint: disable=invalid-name
  pipeline_args = argv[1:]
  logging.info("output_samples: %s", FLAGS.output_samples)
  logging.info("Additional pipeline args: %s", pipeline_args)
  pipeline_options = PipelineOptions(pipeline_args)

  with tf.io.gfile.GFile(FLAGS.sampling_spec) as spec_file:
    spec = text_format.Parse(spec_file.read(), sampling_spec_pb2.SamplingSpec())
  logging.info("Sampling Specification:\n %s", spec)

  # Read the schema and validate it.
  graph_schema_filename = unigraph.find_schema_filename(FLAGS.graph_schema)
  graph_root_path = os.path.dirname(graph_schema_filename)
  logging.info("Reading schema from: %s", graph_schema_filename)
  schema = tfgnn.read_schema(graph_schema_filename)
  logging.info("Graph Schema:\n %s", schema)
  logging.info("Validating schema...")
  validate_schema(schema)
  logging.info("Schema read and validated.")

  run_sample_graph_pipeline(
      schema,
      spec,
      FLAGS.edge_aggregation_method,
      output_pattern=FLAGS.output_samples,
      graph_root_path=graph_root_path,
      seeds_filename=FLAGS.input_seeds,
      runner_name=FLAGS.runner,
      pipeline_options=pipeline_options)


def main():
  define_flags()
  app.run(
      app_main, flags_parser=lambda argv: flags.FLAGS(argv, known_only=True))


if __name__ == "__main__":
  main()
