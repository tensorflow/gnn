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
"""Collection of graph sampling algorithms."""

import collections
import math
import random
from typing import Any, Callable, DefaultDict, Dict, List, Iterable, Iterator, Set, Tuple

from absl import logging
import apache_beam as beam
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.sampler import sampling_spec_pb2
from tensorflow_gnn.sampler import sampling_utils as utils
from tensorflow_gnn.sampler import subgraph_pb2

PCollection = beam.pvalue.PCollection
NodeId = bytes
SampleId = bytes
Example = tf.train.Example
Features = tf.train.Features
Node = subgraph_pb2.Node
Edge = subgraph_pb2.Node.Edge
SampledEdges = PCollection[Tuple[SampleId, Node]]
NodeFeatures = PCollection[Tuple[SampleId, Tuple[NodeId, Features]]]

WeightFunc = Callable[[Edge], float]

UniqueNodeIds = PCollection[Tuple[SampleId, List[NodeId]]]

inner_lookup_join = utils.balanced_inner_lookup_join


def get_weight_feature(edge: Edge, default_value: float = 1.0) -> float:
  """Return the weight feature or a default value (1.), if not present."""
  feature = edge.features.feature.get("weight", None)
  if feature is None:
    return default_value
  value = feature.float_list.value
  return value[0] if value else default_value


def create_sampling_weight_fn(sampling_op: sampling_spec_pb2.SamplingOp,
                              weight_fn: WeightFunc) -> WeightFunc:
  """Returns weight function that should be used for edge sampling.

  This function should be used to create weights for reservoir sampling. E.g.
  to sample K instances from a stream of edges according to the `sampling_op`
  one needs to extract top-k examples according to this funciton.

  For non-determinsitic sampling algorithms (like RANDOM_UNIFORM) this function
  could return different values for the same input.

  Args:
    sampling_op: The sampling operation.
    weight_fn: The function that reads the weight value of an edge.

  Returns:
    An edge weight function that should be used for reservoir edge sampling.

  Raises:
    ValueError: if sampling operation is not supported.
  """
  if sampling_op.strategy == sampling_spec_pb2.TOP_K:
    return weight_fn

  if sampling_op.strategy == sampling_spec_pb2.RANDOM_UNIFORM:
    return lambda _: random.uniform(0., 1.)

  if sampling_op.strategy == sampling_spec_pb2.RANDOM_WEIGHTED:

    def random_weighted_fn(edge: Edge) -> float:
      noise = random.uniform(0.0, 1.0)
      weight = weight_fn(edge)
      if weight <= 0.0 or noise <= 0.0:
        return float("-inf")
      return math.log(noise) / weight

    return random_weighted_fn

  raise ValueError(f"Unsupported samplign strategy f{sampling_op.strategy}.")


_DETERMINISTIC_SAMPLING_STRATEGIES = {
    sampling_spec_pb2.TOP_K: True,
    sampling_spec_pb2.RANDOM_UNIFORM: False,
    sampling_spec_pb2.RANDOM_WEIGHTED: False
}


def is_deterministic(sampling_op: sampling_spec_pb2.SamplingOp) -> bool:
  """Returns `True` if sampling operation produces resample_for_each_path results."""
  return _DETERMINISTIC_SAMPLING_STRATEGIES.get(sampling_op.strategy)


# Sampling frontier is a mapping from (sample id, node id) pair to the number
# of sampling paths that lead to that node id within the same subgraph.
Frontier = PCollection[Tuple[Tuple[SampleId, NodeId], int]]

_FROTIER_OUTPUT_TAG = "frontier"


class ResevoirEdgeSamplingFn(beam.DoFn):
  """Implements reservoir sampling without replacement.

  Sampling is done by accumulating top `sample_size` edges according to the
  weight function `weight_fn`.
  """

  def __init__(self, weight_fn: WeightFunc, sample_size: int,
               resample_for_each_path: bool):
    """Constructor.

    Args:
      weight_fn: A function that takes an edge as its input and returns weight
        to use for reservoir sampling as its output. Could be non-deterministic.
      sample_size: The upper bound on the number of sampled edges for each input
        node or (node, path) pair (see `resample_for_each_path`).
      resample_for_each_path: If `False`, the sampling is done once for each
        input node. If `True`, the sampling is repeated number of paths times.
    """
    self._weight_fn = weight_fn
    self._sample_size = sample_size
    self._resample_for_each_path = resample_for_each_path

  def process(self, element: Tuple[SampleId, Tuple[int, Node]]):
    """Samples edges for each input and computes new sampling frontier.

    Args:
      element: A tuple containing a sample id mapping to a tuple containing an
        integer count of the total number of sampling paths that lead to a given
        node.

    Yields:
      Tuple of sample id and sampled edged as the main output and new sampling
        frontier as `_FROTIER_OUTPUT_TAG` output.
    """
    sample_id, (num_samples, node) = element

    if len(node.outgoing_edges) <= self._sample_size:
      # Optimization: because sampling is done without replacement we could
      # return all edges if the sample size if greater than the total number of
      # outgoing edges.
      sampled_edges = node
      for edge in node.outgoing_edges:
        yield beam.pvalue.TaggedOutput(
            _FROTIER_OUTPUT_TAG, ((sample_id, edge.neighbor_id), num_samples))

    else:
      # Map from sampled edge indices (in `node.outgoing_edges`) to the total
      # number of times this edge was sampled for all sampling paths.
      edge_index_to_count = collections.defaultdict(int)
      if self._resample_for_each_path:
        num_resamples, count_step = num_samples, 1
      else:
        num_resamples, count_step = 1, num_samples
      for _ in range(num_resamples):
        weights: List[Tuple[float, int]] = []
        for edge_index, edge in enumerate(node.outgoing_edges):
          weights.append((self._weight_fn(edge), edge_index))
        weights.sort(reverse=True)
        for _, edge_index in weights[:self._sample_size]:
          edge_index_to_count[edge_index] += count_step

      sampled_edges = Node()
      sampled_edges.CopyFrom(node)
      sampled_edges.ClearField("outgoing_edges")
      sampled_edges.outgoing_edges.extend(
          [node.outgoing_edges[index] for index in edge_index_to_count.keys()])

      for edge_index, count in edge_index_to_count.items():
        edge = node.outgoing_edges[edge_index]
        yield beam.pvalue.TaggedOutput(_FROTIER_OUTPUT_TAG,
                                       ((sample_id, edge.neighbor_id), count))

    yield sample_id, sampled_edges


def sample_edges(
    sampling_spec: sampling_spec_pb2.SamplingSpec,
    seeds: Dict[tfgnn.NodeSetName, PCollection[Tuple[SampleId, NodeId]]],
    adj_lists: Dict[tfgnn.EdgeSetName, PCollection[Node]]
) -> Dict[tfgnn.EdgeSetName, SampledEdges]:
  """Samples edges from the graph according to the sampling specification.

  Sampling starts from the seed nodes provided in the `seeds` collection. Each
  edge sampling operation first aggregates nodes produced by all its input
  operations. Then it samples from nodes' outgoing edges independently for
  each node.

  Args:
    sampling_spec: The specification that defines how we will perform sampling.
    seeds: A mapping from node set name to (sample id, seed node id) pairs.
    adj_lists: A mapping from edge set names to adjacency lists. An adjacency
      list is a pair of a source node id and all outgoing edges for that source
      node (as Node message). Currently it is assumed that adjacency lists are
      "small" (much smaller than a single worker memory).

  Returns:
    A mapping from the edge set name to all sampled edges. Note that the
    function does not perform sampled edges deduplication so the same edge could
    appear multiple times for the same sample id.

  Raises:
    ValueError: If sampling specification is not correct.
  """

  def key_by_id(node: Node) -> Tuple[NodeId, Node]:
    return node.id, node

  adj_lists = {
      set_name:
      (nodes | f"SampleEdges/KeyByNodeId/{set_name}" >> beam.Map(key_by_id))
      for set_name, nodes in adj_lists.items()
  }
  op_to_frontier: Dict[str, Frontier] = {}
  for seed_op in [sampling_spec.seed_op]:
    node_set_name: tfgnn.NodeSetName = seed_op.node_set_name
    if seed_op.node_set_name not in seeds:
      raise ValueError(
          f"No seeds provided for '{seed_op.node_set_name}' from {seed_op}.")

    label = f"SampleEdges/CreateFrontier/{node_set_name}"
    op_to_frontier[seed_op.op_name] = (
        seeds[node_set_name] | label >> beam.Map(lambda item: (item, 1)))

  edge_set_to_sampled_edges: DefaultDict[
      tfgnn.EdgeSetName, List[SampledEdges]] = collections.defaultdict(list)

  def execute_op(sampling_op: sampling_spec_pb2.SamplingOp) -> None:
    if sampling_op.edge_set_name not in adj_lists:
      raise ValueError((
          f"Edge set '{sampling_op.edge_set_name}' from the"
          f" '{sampling_op.op_name}' sampling op "
          " is not defined in the graph schema."))

    stage_name = lambda suffix: f"SampleEdges/{sampling_op.op_name}/{suffix}"

    op_inputs = []
    for input_op_name in sampling_op.input_op_names:
      frontier_piece = op_to_frontier.get(input_op_name, None)
      if frontier_piece is None:
        raise ValueError(f"Could not match '{input_op_name}' input from"
                         f" the '{sampling_op.op_name}' sampling operation."
                         " Note that it is currently assumed that sampling"
                         " operations in the sampling spec are ordered in the"
                         " order of their possible execution.")
      op_inputs.append(frontier_piece)

    frontier: Frontier = (
        op_inputs
        | stage_name("FlattenInputs") >> beam.Flatten()
        | stage_name("GroupSamplingPaths") >> beam.CombinePerKey(sum))

    def extract_nodes(query: Tuple[SampleId, int],
                      value: Node) -> Tuple[SampleId, Tuple[int, Node]]:
      sample_id, num_paths = query
      return sample_id, (num_paths, value)

    def rekey_by_source_id(
        item: Tuple[SampleId, NodeId],
        num_paths: int) -> Tuple[NodeId, Tuple[SampleId, int]]:
      return item[1], (item[0], num_paths)

    nodes = (
        inner_lookup_join(
            stage_name("LookupNodes"),
            queries=(frontier
                     | stage_name("RekeyFrontierBySourceId") >>
                     beam.MapTuple(rekey_by_source_id)),
            lookup_table=adj_lists[sampling_op.edge_set_name])
        | stage_name("DropSourceIds") >> beam.Values()
        | stage_name("ExtractMatchedNodes") >> beam.MapTuple(extract_nodes))

    sampling_fn = ResevoirEdgeSamplingFn(
        create_sampling_weight_fn(sampling_op, get_weight_feature),
        sampling_op.sample_size,
        resample_for_each_path=not is_deterministic(sampling_op))
    sampled_edges, new_frontier = (
        nodes
        | stage_name("SampleEdges") >> beam.ParDo(sampling_fn).with_outputs(
            _FROTIER_OUTPUT_TAG, main="sampled_edges"))

    op_to_frontier[sampling_op.op_name] = new_frontier
    edge_set_to_sampled_edges[sampling_op.edge_set_name].append(sampled_edges)

  for sampling_op in sampling_spec.sampling_ops:
    execute_op(sampling_op)

  return {
      edge_set_name:
      (result_pieces | f"FlattenSampledEdges/{edge_set_name}" >> beam.Flatten())
      for edge_set_name, result_pieces in edge_set_to_sampled_edges.items()
  }


def create_adjacency_lists(
    graph_dict: Dict[str, Dict[str, PCollection]],
    sort_edges: bool = False
) -> Dict[tfgnn.EdgeSetName, PCollection[Node]]:
  """Creates adjacency lists for each edge set from the `graph_dict`.

  NOTE: it is assumed that the outgoing degrees of nodes are small, so that
  outgoing edges of any node can fit into a worker's memory. If it is not the
  case, the current recommended approach is to prune edges such that these
  constraints are met.

  Args:
    graph_dict: The map returned from a `unigraph` graph reading.
    sort_edges: If `True`, outgoing edges are sorted by `neighbor_id`.

  Returns:
    A mapping from edge set name to the collection of edges from that edge set
    grouped by their source node ids as Node messages.
  """

  def create_nodes(source_id: NodeId,
                   edges: Iterable[Tuple[NodeId, Example]]) -> Node:
    node = Node()
    node.id = source_id
    if sort_edges:
      edges = sorted(edges, key=lambda item: item[0])

    for edge_index, (target_id, features) in enumerate(edges):
      edge = node.outgoing_edges.add()
      edge.neighbor_id = target_id
      edge.edge_index = edge_index
      if features.feature:
        edge.features.CopyFrom(features)
    return node

  def key_by_source_id(
      source_id: NodeId, target_id: NodeId,
      edge_example: Example) -> Tuple[NodeId, Tuple[NodeId, Features]]:
    return source_id, (target_id, edge_example.features)

  def create_adjacency_list(
      edge_set_name: tfgnn.EdgeSetName,
      edge_set: PCollection[Tuple[NodeId, NodeId, Example]]
  ) -> PCollection[Tuple[NodeId, Node]]:
    stage_name = lambda prefix: f"CreateAdjLists/{prefix}/{edge_set_name}"

    return (edge_set
            | stage_name("KeyBySourceId") >> beam.MapTuple(key_by_source_id)
            | stage_name("GroupByKey") >> beam.GroupByKey()
            | stage_name("CreateNodes") >> beam.MapTuple(create_nodes))

  return {
      edge_set_name: create_adjacency_list(edge_set_name, edge_set)
      for edge_set_name, edge_set in graph_dict["edges"].items()
  }


def find_connecting_edges(
    schema: tfgnn.GraphSchema,
    incident_nodes: Dict[tfgnn.NodeSetName, UniqueNodeIds],
    adj_lists: Dict[tfgnn.EdgeSetName, PCollection[Node]]
) -> Dict[tfgnn.EdgeSetName, SampledEdges]:
  """Filters edges that connect given subset of incident nodes.

  Args:
    schema: The graph schema.
    incident_nodes: Allowed incident nodes grouped by node sets and sample ids.
    adj_lists: A mapping from edge set names to adjacency lists. An adjacency
      list is a pair of a source node id and all outgoing edges for that source
      node (as Node message).

  Returns:
    Filtered edges grouped by their edge sets and sample ids.
  """
  result = {}
  for edge_set_name, adj_list in adj_lists.items():
    source = schema.edge_sets[edge_set_name].source
    target = schema.edge_sets[edge_set_name].target
    source_ids = incident_nodes.get(source, None)
    target_ids = incident_nodes.get(target, None)
    if source_ids is None or target_ids is None:
      continue

    if source == target:
      source_to_targets = _create_homogeneous_edge_filter(
          edge_set_name, source_ids)
    else:
      source_to_targets = _create_heterogeneous_edge_filter(
          edge_set_name, source_ids, target_ids)

    result[edge_set_name] = _filter_edges_by_source_and_set_of_targets(
        edge_set_name, source_to_targets, adj_list)

  return result


def create_unique_node_ids(
    schema: tfgnn.GraphSchema,
    seeds: Dict[tfgnn.NodeSetName, Tuple[SampleId, NodeId]],
    edges: Dict[tfgnn.EdgeSetName, SampledEdges]
) -> Dict[tfgnn.NodeSetName, UniqueNodeIds]:
  """Aggregates unique node ids for each sampled subgraph.

  Args:
    schema: The graph schema.
    seeds: A mapping from node set name to (sample id, seed node id) pairs.
    edges: A mapping from edge set name to (sample id, node) pairs.

  Returns:
    A mapping from node set name to (sample id, list of unique node ids) pairs.
  """

  stage_name = lambda set_name, label: f"CreateUniqueNodeIds/{set_name}/{label}"

  node_set_to_ids = collections.defaultdict(list)

  def to_list(sample_id: SampleId,
              seed_node_id: NodeId) -> Tuple[SampleId, List[NodeId]]:
    return sample_id, [seed_node_id]

  for node_set_name, seed_ids in seeds.items():
    logging.info("[create_unique_node_ids][to_list] node_set_name: %s",
                 node_set_name)
    ids: UniqueNodeIds = (
        seed_ids
        | stage_name(node_set_name, "MoveSeedToList") >> beam.MapTuple(to_list))
    node_set_to_ids[node_set_name].append(ids)

  def filter_edge_sources(sample_id: SampleId,
                          node: Node) -> Tuple[SampleId, List[NodeId]]:
    assert node.id
    return sample_id, [node.id]

  def filter_edge_targets(sample_id: SampleId,
                          node: Node) -> Tuple[SampleId, List[NodeId]]:

    return sample_id, [edge.neighbor_id for edge in node.outgoing_edges]

  for edge_set_name, edge_set_edges in edges.items():
    logging.info("[create_unique_node_ids] edge_set_name: %s", edge_set_name)
    node_set_to_ids[schema.edge_sets[edge_set_name].source].append(
        edge_set_edges
        | stage_name(edge_set_name, "GetSourceIds") >> beam.MapTuple(
            filter_edge_sources))
    node_set_to_ids[schema.edge_sets[edge_set_name].target].append(
        edge_set_edges
        | stage_name(edge_set_name, "GetTargetIds") >> beam.MapTuple(
            filter_edge_targets))

  result = {}
  for node_set_name, ids in node_set_to_ids.items():
    result[node_set_name] = (
        ids
        | stage_name(node_set_name, "Flatten") >> beam.Flatten()
        | stage_name(node_set_name, "Dedup") >> beam.CombinePerKey(
            utils.unique_values_combiner))
  return result


def lookup_node_features(
    node_ids: Dict[tfgnn.NodeSetName, UniqueNodeIds],
    node_examples: Dict[tfgnn.NodeSetName, PCollection[Tuple[NodeId, Example]]]
) -> Dict[tfgnn.NodeSetName, NodeFeatures]:
  """Extracts node features using node ids.

  Args:
    node_ids: A mapping from node set name to (sample id, list of unique node
      ids) pairs.
    node_examples: A mapping from node set name to (node id, node example)
      pairs.

  Returns:
    A mapping from node set name to (node id, (sample id, node features)) for
    subset of nodes from node_ids that have examples.
  """

  def extract_features(
      node_id: NodeId, join_result: Tuple[SampleId, Example]
  ) -> Tuple[SampleId, Tuple[NodeId, Features]]:
    sample_id, example = join_result
    return (sample_id, (node_id, example.features))

  def unflatten_node_ids(
      sample_id: SampleId,
      node_ids: List[NodeId]) -> Iterator[Tuple[NodeId, SampleId]]:
    for node_id in node_ids:
      yield node_id, sample_id

  def lookup_features(
      node_set_name: tfgnn.NodeSetName, node_ids: UniqueNodeIds,
      examples: PCollection[Tuple[NodeId, Example]]) -> NodeFeatures:
    stage_prefix: str = f"LookupNodeFeatures/{node_set_name}"
    return (
        inner_lookup_join(
            stage_prefix,
            queries=(node_ids | f"{stage_prefix}/UnflattenIds" >>
                     beam.FlatMapTuple(unflatten_node_ids)),
            lookup_table=examples)
        | f"{stage_prefix}/ExtractFeatures" >> beam.MapTuple(extract_features))

  result = {}

  for node_set_name, node_set_node_ids in node_ids.items():
    result[node_set_name] = lookup_features(node_set_name, node_set_node_ids,
                                            node_examples[node_set_name])
  return result


def _filter_edges_by_source_and_set_of_targets(
    edge_set_name: tfgnn.EdgeSetName,
    incident_nodes: PCollection[Tuple[NodeId, Tuple[SampleId, List[NodeId]]]],
    adj_list: PCollection[Node]) -> PCollection[Tuple[NodeId, Node]]:
  """Filters edges that connect a source node to any of the target nodes."""

  def stage_name(suffix: str):
    return f"FilterEdgesBySourceAndSetOfTargets/{edge_set_name}/{suffix}"

  def filter_edges(edge_filter: Tuple[SampleId, List[NodeId]],
                   node: Node) -> Iterator[Tuple[NodeId, Node]]:
    sample_id, targets = edge_filter
    allowed_targets: Set[NodeId] = set(targets)
    filtered_edges = [
        edge for edge in node.outgoing_edges
        if edge.neighbor_id in allowed_targets
    ]
    if not filtered_edges:
      return

    fitlered_node = Node()
    fitlered_node.CopyFrom(node)
    fitlered_node.ClearField("outgoing_edges")
    fitlered_node.outgoing_edges.extend(filtered_edges)
    yield sample_id, fitlered_node

  def key_by_node_id(node: Node) -> Tuple[NodeId, Node]:
    return node.id, node

  return (inner_lookup_join(
      stage_name("JoinFilterWithEdges"),
      queries=incident_nodes,
      lookup_table=(adj_list
                    | stage_name("KeyBySourceId") >> beam.Map(key_by_node_id)))
          | stage_name("DropKey") >> beam.Values()
          | stage_name("FilterEdges") >> beam.FlatMapTuple(filter_edges))


def _create_homogeneous_edge_filter(
    edge_set_name: tfgnn.EdgeSetName, source_or_target_ids: UniqueNodeIds
) -> PCollection[Tuple[NodeId, Tuple[SampleId, List[NodeId]]]]:
  """Creates filter for edges that connect the same node set."""
  stage_name = lambda suffix: f"CreateEdgeFilter/XX/{edge_set_name}/{suffix}"

  def create_filter(
      sample_id: SampleId, ids: List[NodeId]
  ) -> Iterator[Tuple[NodeId, Tuple[SampleId, List[NodeId]]]]:
    for source_id in ids:
      yield source_id, (sample_id, ids)

  return (source_or_target_ids
          | stage_name("CreateFilter") >> beam.FlatMapTuple(create_filter))


def _create_heterogeneous_edge_filter(
    edge_set_name: tfgnn.EdgeSetName, source_ids: UniqueNodeIds,
    target_ids: UniqueNodeIds
) -> PCollection[Tuple[NodeId, Tuple[SampleId, List[NodeId]]]]:
  """Creates filter for edges that connect two distinct node sets."""
  stage_name = lambda suffix: f"CreateEdgeFilter/XY/{edge_set_name}/{suffix}"

  def create_filter(
      sample_id: SampleId, group: Dict[str, Any]
  ) -> Iterator[Tuple[NodeId, Tuple[SampleId, List[NodeId]]]]:
    source_ids: List[List[NodeId]] = list(group["source_ids"])
    target_ids: List[List[NodeId]] = list(group["target_ids"])
    if not source_ids or not target_ids:
      return
    assert len(source_ids) == 1, f"Not unique key {sample_id} for source nodes."
    assert len(target_ids) == 1, f"Not unique key {sample_id} for target nodes."
    for source_id in source_ids[0]:
      yield source_id, (sample_id, target_ids[0])

  return ({
      "source_ids": source_ids,
      "target_ids": target_ids
  } | stage_name("GroupBySampeIds") >> beam.CoGroupByKey()
          | stage_name("CreateFilter") >> beam.FlatMapTuple(create_filter))
