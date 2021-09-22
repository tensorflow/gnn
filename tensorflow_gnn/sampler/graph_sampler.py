"""Beam-based graph sampler for homogeneous and heterogeneous graphs.

Samples multiple hops of a graph stored as node entries with adjacency list and
node features (TFRecords file of tf.Example protos). Outputs graph tensor
encoded tensorflow.Example protos in a TFRecords wrapper, one per seed node.

The input is in Universal Graph Format (see documentation of unigraph.py
module). The graph may be homogeneous or heterogeneous. The container format is
inferred from the filename fields in the schema (e.g. "tfrecords").
"""

import collections
import copy
import functools
import hashlib
from os import path
from typing import Any, Callable, Iterable, Iterator, Dict, Optional, Tuple, Set, Sequence

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from apache_beam.runners.direct import direct_runner
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.data import unigraph
from tensorflow_gnn.sampler import sampling_spec_pb2
from tensorflow_gnn.sampler import subgraph
from tensorflow_gnn.sampler import subgraph_pb2

from google.protobuf import text_format

PCollection = beam.pvalue.PCollection
NodeId = bytes
NodeSetName = str
EdgeSetName = str
SampleId = bytes
Example = tf.train.Example
Features = tf.train.Features
Node = subgraph_pb2.Node
Edge = subgraph_pb2.Node.Edge
Subgraph = subgraph_pb2.Subgraph


def create_adjacency_list(
    nodes: PCollection[Tuple[NodeSetName, NodeId, Example]],
    edges: PCollection[Tuple[EdgeSetName, NodeId, NodeId, Example]]
) -> PCollection[Tuple[NodeId, Node]]:
  """Creates Node objects, join in the given nodes with their outgoing edges.

  This function accepts collections of node and edges, as defined by tables with
  a special node id and (source, target) ids and aggregates all the features and
  outgoing edges over the node ids, producing `Subgraph` instances with all the
  aggregated information.

  Args:
    nodes: The nodes to join in the adjacency list.
    edges: All the edges to join with the above nodes.

  Returns:
    A collection of Nodes with their outgoing edges populated, keyed by NodeId.

  """
  source_keyed_nodes = (
      nodes | beam.Map(lambda item: (item[1], (item[0], item[2]))))
  source_keyed_edges = (
      edges | beam.Map(lambda item: (item[1], (item[0], item[2], item[3]))))

  def join_nodes_edges(item) -> Iterator[Tuple[NodeId, Node]]:
    node_id, groups = item
    node_examples = groups["node"]

    # Ignore source-id without no nodes. (An alternative could be to synthesize
    # them).
    if not node_examples:
      return
    assert len(node_examples) == 1, "Node id not unique: {}".format(node_id)
    node_set_name, node_example = node_examples[0]

    # Create node to produce.
    node = Node()
    node.id = node_id
    node.node_set_name = node_set_name
    # pylint: disable=g-explicit-length-test
    if len(node_example.features.feature) > 0:
      node.features.CopyFrom(node_example.features)

    # Join its edges.
    edges = groups["edge"]
    if edges:
      for edge_set_name, target_id, edge_example in edges:
        edge = node.outgoing_edges.add()
        edge.neighbor_id = target_id
        edge.edge_set_name = edge_set_name

        # Copy the edge features, possibly including weight.
        # pylint: disable=g-explicit-length-test
        if len(edge_example.features.feature) > 0:
          edge.features.CopyFrom(edge_example.features)

    yield (node.id, node)

  return ({
      "node": source_keyed_nodes,
      "edge": source_keyed_edges
  }
          | "JoinNodesEdges" >> beam.CoGroupByKey()
          | beam.FlatMap(join_nodes_edges))


def sample_graph(
    nodes_map: PCollection[Tuple[NodeId, Node]], seeds: PCollection[NodeId],
    sampling_spec: sampling_spec_pb2.SamplingSpec
) -> PCollection[Tuple[SampleId, Subgraph]]:
  """Samples a graph and return a collection of graph tensors.

  Currently, these sampling operations are cumulative. That is, taking an
  operation as input will take the new nodes added by that operation as well
  as nodes that operation operated on.

  Args:
    nodes_map: All the nodes in the graph.
    seeds: All the seed node IDs.
    sampling_spec: The specification that defines how we will perform sampling.

  Returns:
    A tuple of a PCollection of Subgraph protos.

  Raises:
    ValueError if a given seed is not present in the given `nodes_map`.
  """
  # Create an initial set of singleton subgraphs, one per seed.
  node_subgraph_empty = seeds | beam.Map(
      functools.partial(create_initial_set, sampling_spec.insert_sample_ids))
  subgraphs = ({
      "subgraph": node_subgraph_empty,
      "node": nodes_map
  }
               | "JoinInitial" >> beam.CoGroupByKey()
               | beam.FlatMap(join_initial))

  # Sample new nodes from the boundary of each subgraph and join them into
  # existing subgraphs.
  def node_combiner(sg: Subgraph, nodes: Iterable[Node]):
    sg = copy.copy(sg)
    for node in nodes:
      sg.nodes.add().CopyFrom(node)
    return sg

  # Keep track of the subgraphs generated after each operation.
  # Note that the execution graph has already been validated previously.
  op_to_subgraph: Dict[str, PCollection[Tuple[SampleId, Subgraph]]] = {
      sampling_spec.seed_op.op_name: subgraphs
  }

  for sampling_op in sampling_spec.sampling_ops:
    input_subgraphs = [
        op_to_subgraph[input_op] for input_op in sampling_op.input_op_names
    ] | f"FlattenInputs{sampling_op.op_name}" >> beam.Flatten()
    subgraphs = inner_join_protos(
        input_subgraphs, nodes_map,
        functools.partial(sample, sampling_op, get_weight_feature),
        node_combiner, f"Sample{sampling_op.op_name}")
    op_to_subgraph[sampling_op.op_name] = subgraphs

  # Combine all subgraphs that share the same seed node.
  def combine_subgraphs(subgraphs: Sequence[Subgraph]) -> Subgraph:
    """Combine all Subgraphs that share the same seed node."""
    subgraphs_list = list(iter(subgraphs))
    if not subgraphs_list:
      # Should never happen.
      raise ValueError("Combining subgraphs received empty list.")
    combined_subgraph = copy.copy(subgraphs_list[0])
    del combined_subgraph.nodes[:]  # Will be populated next.

    # Add all the nodes to the Subgraph.
    node_id_to_nodes = {}
    for sg in subgraphs_list:
      for node in sg.nodes:
        node_id_to_nodes[node.id] = node
    for node in node_id_to_nodes.values():
      combined_subgraph.nodes.add().CopyFrom(node)

    return combined_subgraph

  subgraphs = (
      op_to_subgraph.values() | "FlattenOps" >> beam.Flatten()
      | beam.GroupByKey()
      | "CombineSubgraphs" >> beam.CombineValues(combine_subgraphs))

  # Remove dangling node references.
  return subgraphs | "CleanEdges" >> beam.Map(clean_edges)


def copy_context_features(sg: Subgraph, context: Example):
  """Copies the given context features to the subgraph."""
  for feature_name, feature in context.features.feature.items():
    if feature_name in sg.features.feature:
      raise ValueError(
          f"Context feature named {feature_name} already exists in subgraph.")
    sg.features.feature[feature_name].CopyFrom(feature)
  return sg


def augment_schema_with_sample_features(schema: tfgnn.GraphSchema):
  """Add the `sample_id` and `seed_id` features to the schema."""
  if "sample_id" in schema.context.features:
    raise ValueError("Feature `sample_id` is already in the schema.")
  if "seed_id" in schema.context.features:
    raise ValueError("Feature `sample_id` is already in the schema.")
  features = schema.context.features
  features["sample_id"].dtype = tf.string.as_datatype_enum
  features["seed_id"].dtype = tf.string.as_datatype_enum


def augment_schema_with_node_ids(schema: tfgnn.GraphSchema,
                                 node_id_feature: str):
  """Add node id feature to schema. Mutate `schema` in place."""
  for node_set in schema.node_sets.values():
    if node_id_feature in node_set.features:
      raise ValueError(f"Feature `{node_id_feature}` is already in the schema.")
    node_set.features[node_id_feature].dtype = tf.string.as_datatype_enum


def create_initial_set(insert_sample_ids: bool,
                       seed_node_id: bytes) -> Tuple[NodeId, Subgraph]:
  """Creates initial single-node subgraphs."""
  sg = Subgraph()
  sg.seed_node_id = seed_node_id

  # Generate a short hash. The idea is that if we shard by this hashed sample id
  # the nodes will be distributed UAR into the shards.
  generator = hashlib.shake_128()
  generator.update(seed_node_id)
  # pylint: disable=too-many-function-args
  sample_id = generator.hexdigest(8).encode("ascii")
  sg.sample_id = sample_id

  if insert_sample_ids:
    sg.features.feature["sample_id"].bytes_list.value.append(sample_id)
    sg.features.feature["seed_id"].bytes_list.value.append(seed_node_id)

  return seed_node_id, sg


def join_initial(
    node_info: Tuple[NodeId, Any]) -> Iterable[Tuple[SampleId, Subgraph]]:
  """Initial join between empty subgraph and its node."""
  (node_id, info) = node_info
  subgraphs = info["subgraph"]
  nodes = info["node"]
  if not subgraphs:
    # This just means that the node was not part of the seeds.
    return
  if not nodes:
    # This means a seed was passed in that didn't correspond to a node. This is
    # fairly common, so don't raise an exception, but instead log a warning.
    logging.warning("Seed node with ID %s does not exist in graph.", node_id)
    return
  assert len(subgraphs) == 1
  assert len(nodes) == 1
  sg = copy.copy(subgraphs[0])
  new_node = sg.nodes.add()
  new_node.CopyFrom(nodes[0])
  yield sg.sample_id, sg


def inner_join_protos(left_table: PCollection, right_table: PCollection,
                      enumerator_fn: Callable[[Any], Iterable[bytes]],
                      combiner_fn: Callable[[Any, Iterable[Any]],
                                            Any], name: str) -> PCollection:
  """Runs an inner join between two tables of (id, proto).

  Args:
    left_table: A collection of (key1, proto1), where the proto can enumerate a
      list of `key2` ids to join with. Must be a unique mapping, not a multimap.
    right_table: A collection (key2, proto2) to join to the left table. Must be
      a unique mapping, not a multimap.
    enumerator_fn: A function that given a `proto1` enumerates a list of `key2`
      to join against.
    combiner_fn: A function that accepts a `proto1` and a list of `proto2`
      corresponding to the `key2` ids.
    name: A unique prefix string for the name of the stage.

  Returns:
    A collection with the same type as `left_table`, where the values have
    been processed by `combiner_fn`.
  """

  def enumerator(item):
    left_id, proto = item
    for right_id in enumerator_fn(proto):
      yield (right_id, left_id)

  right_ids = (left_table | f"{name}_Enumerate" >> beam.FlatMap(enumerator))

  def join_right(right_info: Tuple[bytes, Any]) -> Iterable[Tuple[bytes, Any]]:
    (_, info) = right_info
    ids = info["ids"]
    protos = info["protos"]
    if protos:
      assert len(protos) == 1
      proto = protos[0]
      for iid in ids:
        yield iid, proto

  right = ({
      "ids": right_ids,
      "protos": right_table
  }
           | f"{name}_GroupRightByKey" >> beam.CoGroupByKey()
           | f"{name}_JoinRightNodes" >> beam.FlatMap(join_right))

  def join_right_to_left(
      left_info: Tuple[bytes, Any]) -> Iterable[Tuple[bytes, Any]]:
    """Initial new nodes into existing subgraphs."""
    (left_id, info) = left_info
    left_protos = info["left"]
    right_protos = info["right"]
    for l in left_protos:
      left_proto = combiner_fn(l, right_protos)
      yield left_id, left_proto

  return ({
      "left": left_table,
      "right": right
  }
          | f"{name}_GroupLeftByKey" >> beam.CoGroupByKey()
          | f"{name}_JoinLeftNodes" >> beam.FlatMap(join_right_to_left))


WeightFunc = Callable[[Edge], float]


def get_weight_feature(edge: Edge) -> float:
  """Return the weight feature or a default value, if not present."""
  feature = edge.features.feature.get("weight", None)
  if feature is None:
    return 1.0
  value = feature.float_list.value
  return value[0] if value else 1.0


def sample(sampling_op: sampling_spec_pb2.SamplingOp,
           weight_func: Optional[WeightFunc], sg: Subgraph) -> Iterable[NodeId]:
  """Samples top-weighed nodes from the boundary of a subgraph.

  Args:
    sampling_op: The specification for this sample.
    weight_func: A function that produces the weight. If 'None' is provided,
      weights are all 1.0.
    sg: A Subgraph instance.

  Yields:
    Selected (sampled) node-id ids at the boundary (outside) of the subgraph.
    Those should be joined in the subgraph by further operations.
  """
  if weight_func is None:
    weight_func = lambda _: 1.0

  nodeids = {node.id for node in sg.nodes}
  weights = collections.defaultdict(float)
  for node in sg.nodes:
    for edge in node.outgoing_edges:
      if edge.edge_set_name == sampling_op.edge_set_name and edge.neighbor_id not in nodeids:
        weights[edge.neighbor_id] += weight_func(edge)
  sorted_weights = sorted(
      weights.items(), key=lambda item: item[1], reverse=True)
  for node_id, _ in sorted_weights[:sampling_op.sample_size]:
    yield node_id


def clean_edges(
    id_subgraph: Tuple[SampleId, Subgraph]) -> Tuple[SampleId, Subgraph]:
  """Removes references to edges which aren't in the subgraph."""
  sid, sg = id_subgraph
  sg = copy.copy(sg)
  sg_node_ids = {node.id for node in sg.nodes}
  for node in sg.nodes:
    edges = [
        edge for edge in node.outgoing_edges if edge.neighbor_id in sg_node_ids
    ]
    node.ClearField("outgoing_edges")
    node.outgoing_edges.extend(edges)
  return sid, sg


def validate_schema(schema: tfgnn.GraphSchema):
  """Validates assumptions of the graph schema."""

  # Ensure that the filenames are set on the universal graph format.
  for set_type, set_name, set_ in tfgnn.iter_sets(schema):
    if not (set_.HasField("metadata") and set_.metadata.HasField("filename")):
      raise ValueError("Filename is not set on schema's {}/{} set.".format(
          set_type, set_name))

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
  if runner_name == "direct":
    runner = direct_runner.DirectRunner()
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
    if sampling_op.strategy != sampling_spec_pb2.SamplingStrategy.TOP_K:
      raise ValueError(
          f"Unsupported sampling strategy {sampling_op.strategy} for operation "
          f"{sampling_op.op_name}, only TOP_K is supported at this time.")
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


def _create_nodes_map(
    graph_dict: Dict[str, Dict[str, PCollection]]
) -> PCollection[Tuple[NodeId, Node]]:
  """Create a map from Node ID to Node from the given `graph_dict`.

  Args:
    graph_dict: The map returned from a `unigraph` graph reading.

  Returns:
    A collection of all the nodes, keyed by their unique Node IDs.
  """

  # First, add the node and edge set names into their respective collections.
  def prepend_set_name(set_name, item):
    return (set_name,) + item

  node_sets = []
  for node_set_name, node_set in graph_dict["nodes"].items():
    node_sets.append(node_set | f"AddNodeSetName{node_set_name}" >> beam.Map(
        functools.partial(prepend_set_name, node_set_name)))

  edge_sets = []
  for edge_set_name, edge_set in graph_dict["edges"].items():
    edge_sets.append(edge_set | f"AddEdgeSetName{edge_set_name}" >> beam.Map(
        functools.partial(prepend_set_name, edge_set_name)))

  # Join the nodes and their outgoing edges to an adjacency list.
  # 'nodes': A PCollection of (node-set-name, node-id, Example).
  # 'edges': A PCollection of (edge-set-name, source-id, target-id, Example
  nodes = node_sets | "CollectNodes" >> beam.Flatten()
  edges = edge_sets | "CollectEdges" >> beam.Flatten()
  return create_adjacency_list(nodes, edges)


def _clean_metadata(metadata):
  """Clears the cardinality and filename fields of `metadata` inline."""
  metadata.ClearField("cardinality")
  metadata.ClearField("filename")


def run_sample_graph_pipeline(schema_filename: str,
                              sampling_spec: sampling_spec_pb2.SamplingSpec,
                              output_pattern: str,
                              seeds_filename: Optional[str] = None,
                              runner_name: Optional[str] = None):
  """Runs the pipeline on a graph, which may be homogeneous or heterogeneous."""

  # Read the schema and validate it.
  filename = unigraph.find_schema_filename(schema_filename)
  logging.info("Reading schema...")
  schema = tfgnn.read_schema(filename)
  validate_schema(schema)
  logging.info("Schema read and validated.")

  # Validate configuration.
  _validate_sampling_spec(sampling_spec, schema)
  logging.info("Sampling specification validated. Pipeline commencing...")

  with beam.Pipeline(runner=create_beam_runner(runner_name)) as root:
    # Read the graph.
    graph_dict = unigraph.read_graph(schema, path.dirname(filename), root)

    # Read the seeds, or use the node ids as the seeds.
    if seeds_filename:
      seed_nodes = unigraph.read_node_set(root, seeds_filename)
    else:
      seed_nodes = graph_dict["nodes"][sampling_spec.seed_op.node_set_name]
    seeds = seed_nodes | beam.Keys()

    # Create a collection of (node_id, Node) pairs.
    nodes_map = _create_nodes_map(graph_dict)

    # Sample the graph.
    subgraphs = sample_graph(nodes_map, seeds,
                             sampling_spec) | "FilledValues" >> beam.Values()

    # Copy the context features to the subgraphs.
    if "context" in graph_dict:
      context = graph_dict["context"]
      if len(context) != 1 or "" not in context:
        raise ValueError(
            f"Invalid context, expected one empty key, got {context}")

      subgraphs = subgraphs | "CopyContextFeatures" >> beam.Map(
          copy_context_features, context=beam.pvalue.AsSingleton(context[""]))

    # Produce the sample schema to output.
    sampled_schema = copy.copy(schema)
    if sampling_spec.insert_sample_ids:
      augment_schema_with_sample_features(sampled_schema)
    if sampling_spec.node_id_feature:
      augment_schema_with_node_ids(sampled_schema,
                                   sampling_spec.node_id_feature)
    # Remove any metadata fields.
    _clean_metadata(sampled_schema.context.metadata)
    for node_set in sampled_schema.node_sets.values():
      _clean_metadata(node_set.metadata)
    for edge_set in sampled_schema.edge_sets.values():
      _clean_metadata(edge_set.metadata)

    # Encode the subgraphs to graph tensors.
    encoder = functools.partial(subgraph.encode_subgraph_to_example,
                                sampled_schema)
    graph_tensors = (subgraphs | "Encode" >> beam.Map(encoder))

    # Write out the results in a file.
    done = (
        graph_tensors
        | "WriteGraphTensors" >> unigraph.WriteTable(output_pattern))

    # Produce global count subgraph to tf.Example examples.
    _ = (
        subgraphs
        | beam.combiners.Count.Globally()
        | beam.Map(lambda size: logging.info("Produced %s subgraphs.", size)))

  logging.info("Pipeline complete, writing output...")
  # Produce output schema of the tensors.
  output_schema_filename = f"{path.dirname(output_pattern)}/schema.pbtxt"
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

  flags.DEFINE_string(
      "output_samples", None,
      "Output file with serialized graph tensor Example protos.")

  flags.DEFINE_bool("direct", False,
                    "Use the direct runner that will only work locally.")

  runner_choices = ["direct"]
  flags.DEFINE_enum(
      "runner", None, runner_choices,
      "The underlying runner; if not specified, use the default runner.")

  flags.mark_flags_as_required(
      ["graph_schema", "sampling_spec", "output_samples"])


def app_main(unused_argv):
  FLAGS = flags.FLAGS  # pylint: disable=invalid-name
  with tf.io.gfile.GFile(FLAGS.sampling_spec) as spec_file:
    spec = text_format.Parse(spec_file.read(), sampling_spec_pb2.SamplingSpec())
  run_sample_graph_pipeline(FLAGS.graph_schema, spec, FLAGS.output_samples,
                            FLAGS.input_seeds, FLAGS.runner)


def main():
  define_flags()
  app.run(app_main)


if __name__ == "__main__":
  main()
