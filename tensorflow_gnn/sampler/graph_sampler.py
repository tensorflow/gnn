"""Beam-based graph sampler for homogeneous and heterogeneous graphs.

Samples multiple hops of a graph stored as node entries with adjacency list and
node features (TFRecords file of tf.train.Example protos). Outputs graph tensor
encoded tensorflow.Example protos in a TFRecord wrapper, one per seed node.

The input is in Universal Graph Format (see documentation of unigraph.py
module). The graph may be homogeneous or heterogeneous. The container format is
inferred from the filename fields in the schema (e.g. "TFRecord").
"""

import collections
import copy
import enum
import functools
import hashlib
from os import path
import random
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Set, Tuple

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.data import unigraph
from tensorflow_gnn.sampler import sampling_spec_pb2
from tensorflow_gnn.sampler import subgraph
from tensorflow_gnn.sampler import subgraph_pb2

from google.protobuf import text_format

PCollection = beam.pvalue.PCollection
PTransform = beam.PTransform
NodeId = bytes
NodeSetName = str
EdgeSetName = str
SampleId = bytes
Example = tf.train.Example
Features = tf.train.Features
Node = subgraph_pb2.Node
Edge = subgraph_pb2.Node.Edge
Subgraph = subgraph_pb2.Subgraph
ID_FEATURE_NAME = "#id"
_DIRECT_RUNNER = "DirectRunner"
_DATAFLOW_RUNNER = "DataflowRunner"


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


def create_adjacency_list(
    nodes: PCollection[Tuple[NodeSetName, NodeId, Example]],
    edges: PCollection[Tuple[EdgeSetName, NodeId, NodeId, Example]]
) -> PCollection[Tuple[NodeId, Node]]:
  """Creates Node objects, join in the given nodes with their outgoing edges.

  This function accepts collections of node and edges, as defined by tables with
  a special node ID and (source, target) IDs and aggregates all the features and
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


# Combine all subgraphs that share the same seed node.
class SubgraphCombiner(beam.CombineFn):
  """CombineFn implementation to combine subgraphs."""

  def create_accumulator(self) -> Tuple[Subgraph, Set[NodeId]]:
    """Returns an empty subgraph and an empty set of unique nodes."""
    return (Subgraph(), set({}))

  def add_input(self, accumulator: Tuple[Subgraph, Set[NodeId]],
                sg: Subgraph):
    accumulator_subgraph, unique_nodes = accumulator

    # At this point in the pipeline, all subgraphs will either have no features
    # or will replicate the seed_id and sample_id as tf.Example features
    # depending if `insert_sample_id` is true when the seed node set is created.
    if not accumulator_subgraph.HasField("features"):
      if sg.HasField("features"):
        accumulator_subgraph.features.CopyFrom(sg.features)

    # Only the initial subgraph generated by the seed op will have the
    # sample_id as a field on the subgraph so this field will only be set
    # once. If by chance there is a conflict, log a fatal error
    # (should never happen).
    if sg.HasField("sample_id"):
      if not accumulator_subgraph.HasField("sample_id"):
        accumulator_subgraph.sample_id = sg.sample_id
      elif accumulator_subgraph.sample_id != sg.sample_id:
        raise ValueError("sample_id mismatch: "
                         f"{accumulator_subgraph.sample_id} "
                         f"{sg.sample_id}")

    if sg.HasField("seed_node_id"):
      if not accumulator_subgraph.HasField("seed_node_id"):
        accumulator_subgraph.seed_node_id = sg.seed_node_id
      # This should never happen, each subgraph that is combined should have the
      # sample seed_node_id field if present.
      elif accumulator_subgraph.seed_node_id != sg.seed_node_id:
        raise ValueError("seed_node_id mismatch: "
                         f"{accumulator_subgraph.seed_node_id}"
                         f"{sg.seed_node_id}")

    for node in sg.nodes:
      if node.id not in unique_nodes:
        unique_nodes.add(node.id)
        accumulator_subgraph.nodes.add().CopyFrom(node)

    return (accumulator_subgraph, unique_nodes)

  def merge_accumulators(self, accumulators: Iterable[Tuple[Subgraph,
                                                            Set[NodeId]]]):
    if not accumulators:
      # Shouldn't happen but just want to make sure...
      logging.warning("Called merge_accumulator on empty accumulator iterable.")
    iterator = iter(accumulators)
    accumulator_subgraph, unique_nodes = next(iterator)

    # The boundary of the subgraph is no longer needed
    del accumulator_subgraph.boundary[:]

    for sg, _ in accumulators:
      for node in sg.nodes:
        if node.id not in unique_nodes:
          unique_nodes.add(node.id)
          accumulator_subgraph.nodes.add().CopyFrom(node)

    return (accumulator_subgraph, unique_nodes)

  def extract_output(self, accumulator):
    return accumulator[0]


class CombineSubgraphs(beam.PTransform):
  """A PTransform that combines subgraphs."""

  def expand(
      self, op_to_subgraphs: Dict[str, PCollection[Tuple[SampleId, Subgraph]]]
  ) -> PCollection[Tuple[SampleId, Subgraph]]:
    """Combine subgraphs keyed by SampleID.

    Args:
      op_to_subgraphs: A Dict[str, PCollection[Tuple[SampleId, Subgraph]]]
        mapping sampling operation names to PCollections of (SampleId, Subgraph)
        pairs.

    Returns:
      A PCollection[Tuple[SampleId, Subgraph]] mapping each unique SampleId to
        a single, aggregated subgraph.
    """
    return (op_to_subgraphs.values()
            | "FlattenSamplingOpsToSubgraph" >> beam.Flatten()
            |
            "CombineSubgraphsPerKey" >> beam.CombinePerKey(SubgraphCombiner()))


def sample_graph(
    nodes_map: PCollection[Tuple[NodeId, Node]], seeds: PCollection[NodeId],
    sampling_spec: sampling_spec_pb2.SamplingSpec
) -> PCollection[Tuple[SampleId, Subgraph]]:
  """Samples a graph and returns a collection of graph tensors.

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
      functools.partial(create_initial_set, True))
  subgraphs = ({
      "subgraph": node_subgraph_empty,
      "node": nodes_map
  }
               | "JoinInitial" >> beam.CoGroupByKey()
               | beam.FlatMap(join_initial))

  # Keep track of the subgraphs generated after each operation.
  # Note that the execution graph has already been validated previously.
  op_to_subgraph: Dict[str, PCollection[Tuple[SampleId, Subgraph]]] = {
      sampling_spec.seed_op.op_name: subgraphs
  }

  for sampling_op in sampling_spec.sampling_ops:
    input_subgraphs = [
        op_to_subgraph[input_op] for input_op in sampling_op.input_op_names
    ] | f"FlattenInputs{sampling_op.op_name}" >> beam.Flatten()
    subgraphs = sample_from_subgraphs(
        input_subgraphs, nodes_map,
        functools.partial(sample, sampling_op, get_weight_feature),
        f"Sample_{sampling_op.op_name}")

    op_to_subgraph[sampling_op.op_name] = subgraphs

  subgraphs = op_to_subgraph | "CombineSubgraphs" >> CombineSubgraphs()

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
  features = schema.context.features
  if "sample_id" not in schema.context.features:
    features["sample_id"].dtype = tf.string.as_datatype_enum
  if "seed_id" not in schema.context.features:
    features["seed_id"].dtype = tf.string.as_datatype_enum


def augment_schema_with_node_ids(schema: tfgnn.GraphSchema,
                                 node_id_feature: str):
  """Add node id feature to schema. Mutate `schema` in place."""
  for node_set in schema.node_sets.values():
    if node_id_feature not in node_set.features:
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


# TODO(tfgnn): Encapsulate in a beam.PTransform for better display in
# GCP UI (better for debugging and performance analysis).
def sample_from_subgraphs(sampling_frontier: PCollection[Tuple[SampleId,
                                                               Subgraph]],
                          adjacency_list: PCollection[Tuple[NodeId, Node]],
                          sampler_fn: Callable[[Subgraph], Iterable[NodeId]],
                          name: str) -> PCollection[Tuple[SampleId, Subgraph]]:
  """Generates a set of samples from subgraphs (as new subgraphs).

  Args:
    sampling_frontier: PCollection of SampleId-Subgraph pairs representing the
      sampling frontier (output) of a previous sampling operation.
    adjacency_list: PCollection of NodeId-Node pairs for all node ids.
    sampler_fn: A function that can generate NodeId samples from a subgraph.
    name: A unique prefix string for the name of the stage.

  Returns:
    A new set of node samples represented as a Subgraph object.
  """

  # Functor generating sampled (Node) IDs mapped to the original Subgraph ID.
  def sample_node_ids(
      keyed_subgraph: Tuple[SampleId, Subgraph]
  ) -> Iterable[Tuple[NodeId, SampleId]]:
    """Given keyed Subgraph generate a set of samples.

    Args:
      keyed_subgraph: A subgraph keyed by SampleId.

    Yields:
      A sequency of Tuple[NodeId, SampleId] mapping the new sample to the
        original sample ID.
    """
    sample_id, sg = keyed_subgraph
    for node_id in sampler_fn(sg):
      yield (node_id, sample_id)

  sampled_node_ids: PCollection[Tuple[NodeId, SampleId]] = (
      sampling_frontier | f"{name}_Enumerate" >> beam.FlatMap(sample_node_ids))

  def sampled_ids_to_nodes(
      join_result: Tuple[bytes, Dict[str,
                                     Any]]) -> Iterable[Tuple[SampleId, Node]]:
    """MapFn for join result emitting [SampleId, Node] pairs.

    Processes the join result of
      Join(PCol[NodeId, SampleId], PCol[NodeId, Node]) to map the full
      Node proto of the newly sampled NodeIds back to the SampleId it
      belongs to.

    Args:
      join_result: Tuple[NodeId, Dict[Str, Any]]

    Yields:
      Tuple[SampleId, Node]
    """
    (_, info) = join_result
    # List of subgraph IDs that contain the new sample
    sample_ids = info["sample_ids"]

    nodes = info["node_protos"]  # Node proto of the new sample
    if nodes:
      assert len(nodes) == 1
      node = nodes[0]
      # There may be more than one sample_id associated with a sampled node.
      for sample_id in sample_ids:
        yield sample_id, node

  sample_id_to_node: PCollection[Tuple[SampleId, Node]] = (
      {
          "sample_ids": sampled_node_ids,  # PCollection[NodeId, SampleId]
          "node_protos": adjacency_list  # PCollection[NodeId, Node]
      }
      | f"{name}_GroupRightByKey" >> beam.CoGroupByKey()
      | f"{name}_JoinRightNodes" >> beam.FlatMap(sampled_ids_to_nodes))

  # Generate new subgraphs from the sampled Nodes
  def sampled_nodes_to_subgraphs(
      sample_id: SampleId, nodes: Iterable[Node]) -> Tuple[SampleId, Subgraph]:
    """Generate new subgraphs from sampled nodes.

    Args:
      sample_id: The original SampleId from which the `nodes` were generated.
      nodes: List of sampled Nodes to combine into a Subgraph object.

    Returns:
    """
    sg = Subgraph()
    sg.nodes.extend(nodes)
    return sample_id, sg

  result: PCollection[Tuple[SampleId, Subgraph]] = (
      sample_id_to_node
      | f"{name}_GroupSampledNodes" >> beam.GroupByKey()
      | f"{name}_GenerateFrontierSubgraph" >>
      beam.MapTuple(sampled_nodes_to_subgraphs))

  return result


WeightFunc = Callable[[Edge], float]


def get_weight_feature(edge: Edge) -> float:
  """Return the weight feature or a default value, if not present."""
  feature = edge.features.feature.get("weight", None)
  if feature is None:
    return 1.0
  value = feature.float_list.value
  return value[0] if value else 1.0


def sample(sampling_op: sampling_spec_pb2.SamplingOp,
           weight_func: Optional[WeightFunc],
           frontier: Subgraph) -> Iterable[NodeId]:
  """Samples new nodes from a sampling boundary.

  Args:
    sampling_op: The specification for this sample.
    weight_func: A function that produces the weight. If 'None' is provided,
      weights are all 1.0.
    frontier: An instance of a subgraph representing a sampling frontier.

  Yields:
    Selected (sampled) node-id ids at the boundary of the subgraph.
    Those should be joined in the subgraph by further operations.
  """
  if weight_func is None:
    weight_func = lambda _: 1.0

  new_frontier = set()

  # TODO(tfgnn):  Pre-group edges by edgeset in order to speed this up.
  if sampling_op.strategy == sampling_spec_pb2.RANDOM_UNIFORM:
    for node in frontier.nodes:
      edges_to_sample = set()
      for edge in node.outgoing_edges:
        if edge.edge_set_name == sampling_op.edge_set_name:
          edges_to_sample.add((edge.neighbor_id, weight_func(edge)))

      for _ in range(sampling_op.sample_size):
        if edges_to_sample:
          e = random.choice(tuple(edges_to_sample))
          edges_to_sample.remove(e)
          new_frontier.add(e)
        else:
          break
  elif sampling_op.strategy == sampling_spec_pb2.TOP_K:
    for node in frontier.nodes:
      weights = collections.defaultdict(float)
      for edge in node.outgoing_edges:
        if edge.edge_set_name == sampling_op.edge_set_name:
          weights[edge.neighbor_id] += weight_func(edge)

      sorted_weights = sorted(
          weights.items(), key=lambda item: item[1], reverse=True)
      new_frontier.update(sorted_weights[:sampling_op.sample_size])
  else:
    raise NotImplementedError(
        "Sampling strategy not supported for: " +
        sampling_spec_pb2.SamplingStrategy.Name(sampling_op.strategy))

  for node_id, _ in new_frontier:
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
  if runner_name == _DIRECT_RUNNER:
    runner = beam.runners.DirectRunner()
  elif runner_name == _DATAFLOW_RUNNER:
    runner = beam.runners.DataflowRunner()
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
    if sampling_op.strategy != sampling_spec_pb2.SamplingStrategy.TOP_K and sampling_op.strategy != sampling_spec_pb2.SamplingStrategy.RANDOM_UNIFORM:
      raise ValueError(
          f"Unsupported sampling strategy {sampling_op.strategy} for operation "
          f"{sampling_op.op_name}, only TOP_K and RANDOM_UNIFORM are supported "
          "at this time.")
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


def run_sample_graph_pipeline(
    schema_filename: str,
    sampling_spec: sampling_spec_pb2.SamplingSpec,
    output_pattern: str,
    seeds_filename: Optional[str] = None,
    runner_name: Optional[str] = None,
    pipeline_options: Optional[PipelineOptions] = None,
    batching: bool = False):
  """Runs the pipeline on a graph, which may be homogeneous or heterogeneous.

  Args:
    schema_filename: Path specification to a GraphSchema proto message.
    sampling_spec: A SamplingSpec protobuf message.
    output_pattern: File specification for the output graph samples and schema.
    seeds_filename: An optional path specification to list of seed nodes.
    runner_name: An optional string specifying a beam pipeline runner. Only
      DirectRunner or DataflowRunner are currently supported.
    pipeline_options: Additional beam pipeline options that will be passed to
      the runner.
    batching: Write outputs in batches (should typically be False).
  """

  # Read the schema and validate it.
  filename = unigraph.find_schema_filename(schema_filename)
  logging.info("Reading schema...")
  schema = tfgnn.read_schema(filename)
  validate_schema(schema)
  logging.info("Schema read and validated.")

  # Validate configuration.
  _validate_sampling_spec(sampling_spec, schema)
  logging.info("Sampling specification validated. Pipeline commencing...")

  with beam.Pipeline(
      runner=create_beam_runner(runner_name), options=pipeline_options) as root:
    # Read the graph.
    graph_dict = unigraph.read_graph(schema, path.dirname(filename), root)

    # Read the seeds, or use the node ids as the seeds.
    if seeds_filename:
      seed_nodes = unigraph.read_node_set(root, seeds_filename, "seeds")
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

    # Add missing fields to schema
    augment_schema_with_sample_features(sampled_schema)
    augment_schema_with_node_ids(sampled_schema, ID_FEATURE_NAME)

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

    if batching:
      # Use IterableCoder if batching.
      _ = (
          graph_tensors
          | "Batching" >> beam.transforms.util.BatchElements(
              min_batch_size=10, max_batch_size=100)
          | "WriteGraphTensors" >> unigraph.WriteTable(
              output_pattern,
              coder=beam.coders.IterableCoder(beam.coders.ProtoCoder(Example))))
    else:
      # Write out the results in a file.
      _ = (
          graph_tensors
          | "WriteGraphTensors" >> unigraph.WriteTable(output_pattern))

    # Produce global count subgraph to tf.Example examples.
    _ = (
        subgraphs
        | beam.combiners.Count.Globally()
        | beam.Map(lambda size: logging.info("Produced %s subgraphs.", size)))

  # Produce output schema of the tensors.
  output_schema_filename = f"{path.dirname(output_pattern)}/schema.pbtxt"
  logging.info("Pipeline complete, writing output graph schema to: %s",
               output_schema_filename)

  tfgnn.write_schema(sampled_schema, output_schema_filename)

  logging.info("Sampling complete.")


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

  flags.DEFINE_bool("batching", False,
                    "Use batching when writing out TFRecords.")

  runner_choices = [_DIRECT_RUNNER, _DATAFLOW_RUNNER]
  flags.DEFINE_enum(
      "runner", None, runner_choices,
      "The underlying runner; if not specified, use the default runner.")

  flags.mark_flags_as_required(
      ["graph_schema", "sampling_spec", "output_samples"])

  flags.DEFINE_enum(
      "edge_aggregation_method", "node", EdgeAggregationMethod,
      "Given a subgraph sampling, specifies a method for retaining edges "
      "between nodes in the subgraph. `node` sampling includes all edges "
      "between the nodes in the sampled subgraph in the output.  `edge` "
      "aggregation only includes edges in the subgraph if the edge was "
      "traversed during sampling. NOTE: Currently, only `node` is supported."
  )


def app_main(argv):
  """Main sampler entrypoint.

  Args:
    argv: List of arguments passed by flags parser.
  """
  FLAGS = flags.FLAGS  # pylint: disable=invalid-name
  pipeline_args = argv[1:]
  logging.info("Additional pipeline args: %s", pipeline_args)
  pipeline_options = PipelineOptions(pipeline_args)

  # Make sure remote workers have access to variables/imports in the global
  # namespace.
  if FLAGS.runner == _DATAFLOW_RUNNER:
    pipeline_options.view_as(SetupOptions).save_main_session = True

  if FLAGS.edge_aggregation_method != EdgeAggregationMethod.NODE:
    raise ValueError("Only `node` subgraph generation is currently supported.")

  with tf.io.gfile.GFile(FLAGS.sampling_spec) as spec_file:
    spec = text_format.Parse(spec_file.read(), sampling_spec_pb2.SamplingSpec())

  logging.info("spec: %s", spec)
  logging.info("output_samples: %s", FLAGS.output_samples)

  run_sample_graph_pipeline(FLAGS.graph_schema, spec, FLAGS.output_samples,
                            FLAGS.input_seeds, FLAGS.runner, pipeline_options,
                            FLAGS.batching)


def main():
  define_flags()
  app.run(
      app_main, flags_parser=lambda argv: flags.FLAGS(argv, known_only=True))


if __name__ == "__main__":
  main()
