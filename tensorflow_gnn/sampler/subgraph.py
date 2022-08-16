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
"""Conversion of a Subgraph to tf.train.Example graph tensor.

This library contains code to convert an instance of a Subgraph proto, which
contains a sample of a graph, its graph topology, edges, and associated node and
edge features, to an encoded tf.train.Example proto that can be streamed to a
TensorFlow model.
"""
import collections

from typing import Any, Dict, List, Iterable, Mapping, Optional, Tuple, Union

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.sampler import subgraph_pb2

Example = tf.train.Example
Feature = tf.train.Feature
Features = tf.train.Features
Node = subgraph_pb2.Node
NodeId = bytes
SampleId = bytes
Subgraph = subgraph_pb2.Subgraph


def encode_subgraph_to_example(schema: tfgnn.GraphSchema,
                               subgraph: Subgraph) -> Example:
  """Convert a Subgraph to an encoded graph tensor."""

  # TODO(blais): Factor out the repeated bits from the static schema for reuse.

  # Copy context features. Nothing to aggregate here, simple copy.
  example = Example()
  for key, feature in schema.context.features.items():
    feature = subgraph.features.feature.get(key, None)
    if feature is not None:
      newkey = "context/{}".format(key)
      example.features.feature[newkey].CopyFrom(feature)

  # Prepare to store node and edge features.
  node_features_dicts: Dict[str, Dict[str, Feature]] = {}
  edge_features_dicts: Dict[str, Dict[str, Feature]] = {}
  for nset_name, nset_obj in schema.node_sets.items():
    node_features_dicts[nset_name] = _prepare_feature_dict(
        nset_name, nset_obj, "nodes", example)
  for eset_name, eset_obj in schema.edge_sets.items():
    edge_features_dicts[eset_name] = _prepare_feature_dict(
        eset_name, eset_obj, "edges", example)

  # Prepare to store edge indices.
  by_node_set_name: Dict[tfgnn.NodeSetName,
                         List[subgraph_pb2.Node]] = collections.defaultdict(
                             list)
  for node in subgraph.nodes:
    by_node_set_name[node.node_set_name].append(node)
  index_map: Dict[bytes, int] = {}
  for node_lists in by_node_set_name.values():
    index_map.update({node.id: i for i, node in enumerate(node_lists)})

  # Iterate over the nodes and edges.
  node_counter = collections.defaultdict(int)
  edge_counter = collections.defaultdict(int)
  for node in subgraph.nodes:
    node_counter[node.node_set_name] += 1
    node_features_dict = node_features_dicts[node.node_set_name]
    _copy_features(node.features, node_features_dict)

    # Iterate over outgoing edges.
    source_idx = index_map[node.id]
    for edge in node.outgoing_edges:
      # Append indices.
      target_idx = index_map.get(edge.neighbor_id, None)
      if target_idx is None:
        # Fail on edge references to a node that isn't in the graph.
        raise ValueError("Edge to node outside subgraph: '{}': {}".format(
            edge.neighbor_id, subgraph))

      source_name = "edges/{}.{}".format(edge.edge_set_name, tfgnn.SOURCE_NAME)
      target_name = "edges/{}.{}".format(edge.edge_set_name, tfgnn.TARGET_NAME)
      example.features.feature[source_name].int64_list.value.append(source_idx)
      example.features.feature[target_name].int64_list.value.append(target_idx)
      edge_counter[edge.edge_set_name] += 1

      # Store the edge features.
      edge_features_dict = edge_features_dicts[edge.edge_set_name]
      _copy_features(edge.features, edge_features_dict)

  # Produce size features.
  for node_set_name, num_nodes in node_counter.items():
    node_size_name = "nodes/{}.{}".format(node_set_name, tfgnn.SIZE_NAME)
    example.features.feature[node_size_name].int64_list.value.append(num_nodes)
  for edge_set_name, num_edges in edge_counter.items():
    edge_size_name = "edges/{}.{}".format(edge_set_name, tfgnn.SIZE_NAME)
    example.features.feature[edge_size_name].int64_list.value.append(num_edges)

  # Check the feature sizes (in aggregate).
  # TODO(blais): Support ragged features in this sampler eventually.
  for num_counter, features_dicts in [(node_counter, node_features_dicts),
                                      (edge_counter, edge_features_dicts)]:
    for set_name, features_dict in features_dicts.items():
      num = num_counter[set_name]
      for feature_name, out_feature in features_dict.items():
        out_length = get_feature_length(out_feature)
        if num > 0 and out_length % num != 0:
          raise ValueError(
              "Invalid number ({}) of features '{}' for set '{}' in subgraph '{}' for schema '{}'"
              .format(out_length, feature_name, set_name, subgraph, schema))

  strip_empty_features(example)
  return example


def encode_subgraph_pieces_to_example(
    schema: tfgnn.GraphSchema,
    seeds: Mapping[tfgnn.NodeSetName, Iterable[NodeId]],
    context: Features,
    node_sets: Mapping[tfgnn.NodeSetName, Mapping[NodeId, Features]],
    edge_sets: Mapping[tfgnn.EdgeSetName, Iterable[Node]]) -> Example:
  """Convert a subgraph pieces to an encoded graph tensor.

  For details on the serialization format, see the "Data Preparation and
  Sampling" guide.

  Extra features that are not present in the graph schema are ignored. It is
  required that all features defined by the graph schema are present in the
  input.

  Nodes from `seeds` are always added first. The output format allows multiple
  seed nodes from different node sets. Seed and not seed nodes are sorted by
  their node ids.

  Edges without matching nodes are filtered out. Edges are deduplicated and
  sorted using `(source node index, target node index, edge_index)` key.

  Args:
    schema: The graph schema.
    seeds: A mapping from a seed node set name to the list of seed node ids.
    context: Context features.
    node_sets: A mapping from a node set name to node set features.
    edge_sets: A mapping from an edge set name to edge set features.

  Returns:
    Graph tensor encoded as Tensorflow example.

  Raises:
    ValueError: if some features defined in the `schema` are not present.
  """
  result = Example()
  output_features = result.features.feature

  def add_features(feature_name_prefix: str, features: Features,
                   features_spec: Mapping[str, Any]) -> None:
    for fname in features_spec:
      fvalue = features.feature.get(fname, None)
      if fvalue is None:
        raise ValueError(f"Feature '{fname}' is missing from input {features}.")

      output_features[f"{feature_name_prefix}{fname}"].MergeFrom(fvalue)

  def add_size(feature_name_prefix: str, size: int) -> None:
    assert size > 0, "size=0 cannot not be added."
    fname = f"{feature_name_prefix}{tfgnn.SIZE_NAME}"
    output_features[fname].int64_list.value.append(size)

  add_features(f"{tfgnn.CONTEXT}/", context, schema.context.features)

  node_set_to_index = collections.defaultdict(dict)
  for node_set_name, node_set_spec in schema.node_sets.items():
    if _is_latent_node_set(schema, node_set_name):
      nodes = _create_empty_node_features(schema, node_set_name, edge_sets)
    else:
      nodes = list(node_sets.get(node_set_name, {}).items())

    if not nodes:
      continue

    nodes.sort(key=_create_nodes_sort_key(seeds.get(node_set_name, ())))

    feature_name_prefix = f"{tfgnn.NODES}/{node_set_name}."
    node_indices = node_set_to_index[node_set_name]
    for node_index, (node_id, features) in enumerate(nodes):
      add_features(feature_name_prefix, features, node_set_spec.features)
      node_indices[node_id] = node_index
    add_size(feature_name_prefix, len(nodes))

  for edge_set_name, edge_set_spec in schema.edge_sets.items():
    edges: Optional[Iterable[Node]] = edge_sets.get(edge_set_name, None)
    if edges is None:
      continue

    source_indexer = node_set_to_index[edge_set_spec.source]
    target_indexer = node_set_to_index[edge_set_spec.target]
    filtered_edges = []
    unique_edge_ids = set()
    for node in edges:
      source_index = source_indexer.get(node.id, None)
      if source_index is None:
        continue

      for edge in node.outgoing_edges:
        target_index = target_indexer.get(edge.neighbor_id, None)
        if target_index is None:
          continue

        unique_edge_id = (source_index, target_index, edge.edge_index)
        if unique_edge_id in unique_edge_ids:
          continue
        unique_edge_ids.add(unique_edge_id)

        filtered_edges.append((unique_edge_id, edge.features))
    if not filtered_edges:
      continue

    filtered_edges.sort(key=_create_edges_sort_key())
    feature_name_prefix = f"{tfgnn.EDGES}/{edge_set_name}."
    source_idx = output_features[f"{feature_name_prefix}{tfgnn.SOURCE_NAME}"]
    target_idx = output_features[f"{feature_name_prefix}{tfgnn.TARGET_NAME}"]
    for (source_index, target_index, _), features in filtered_edges:
      add_features(feature_name_prefix, features, edge_set_spec.features)
      source_idx.int64_list.value.append(source_index)
      target_idx.int64_list.value.append(target_index)
    add_size(feature_name_prefix, len(filtered_edges))

  return result


_FEATURE_FIELDS = frozenset(["float_list", "int64_list", "bytes_list"])


def strip_empty_features(example: Example):
  """Remove the empty features. Mutates in place."""
  remove_list = []
  for name, feature in example.features.feature.items():
    # pylint: disable=g-explicit-length-test
    if any(((feature.HasField(attrname) and
             len(getattr(feature, attrname).value) > 0)
            for attrname in _FEATURE_FIELDS)):
      continue
    remove_list.append(name)
  for remove in remove_list:
    del example.features.feature[remove]


def get_feature_values(
    feature: Feature) -> Optional[Union[List[float], List[int], List[bytes]]]:
  """Return the values of the feature, regardless of type."""
  for attr_name in "float_list", "bytes_list", "int64_list":
    if feature.HasField(attr_name):
      return getattr(feature, attr_name).value
  return None


def get_feature_length(feature: Feature) -> int:
  """Return the number of elements in the feature."""
  values = get_feature_values(feature)
  return 0 if values is None else len(values)


def _prepare_feature_dict(set_name: str, set_obj: Any, prefix: str,
                          example: Example) -> Dict[str, Feature]:
  """Prepare a dict of feature name to a Feature object."""
  features_dict = {}
  for feature_name in set_obj.features.keys():
    name = "{}/{}.{}".format(prefix, set_name, feature_name)
    features_dict[feature_name] = example.features.feature[name]
  return features_dict


def _copy_features(sg_features: tf.train.Feature,
                   ex_features_dict: Dict[str, tf.train.Feature]):
  """Copy features from sg_features to tf.train.Example."""
  for feature_name, ex_feature in ex_features_dict.items():
    sg_feature = sg_features.feature.get(feature_name, None)
    if sg_feature is None:
      # Feature is empty for that node. Fail for now, ragged tensors are not
      # supported by this conversion routine.
      raise ValueError("Feature '{}' is missing from input: {}".format(
          feature_name, sg_features))
    ex_feature.MergeFrom(sg_feature)


def _create_nodes_sort_key(seeds: Iterable[NodeId]):
  """Creates a sort key that puts seed nodes first and then sorts them by id."""

  seeds = set(seeds)

  def key(item: Tuple[NodeId, Any]) -> Tuple[bool, NodeId]:
    node_id: NodeId = item[0]
    return (node_id not in seeds, node_id)

  return key


def _create_edges_sort_key():
  """Create a sort key that puts seed nodes first and then sorts them by id."""

  def key(item: Tuple[Tuple[int, int, int], Any]) -> Tuple[int, int, int]:
    source_id, target_id, edge_index = item[0]
    return (source_id, target_id, edge_index)

  return key


def _is_latent_node_set(schema: tfgnn.GraphSchema,
                        node_set_name: tfgnn.NodeSetName) -> bool:
  """Return `True` if the node set has not features (s.c. latent)."""
  return not schema.node_sets[node_set_name].features


def _create_empty_node_features(
    schema: tfgnn.GraphSchema, node_set_name: tfgnn.NodeSetName,
    edges: Mapping[tfgnn.EdgeSetName, Iterable[Node]]
) -> List[Tuple[NodeId, Features]]:
  """Create empty features for all latent nodes referenced by edges."""
  unique_node_ids = set()
  for edge_set_name, edge_set in schema.edge_sets.items():
    edges = edges.get(edge_set_name, None)
    if not edges:
      continue
    if edge_set.source == node_set_name:
      for node in edges:
        unique_node_ids.add(node.id)
    if edge_set.target == node_set_name:
      for node in edges:
        unique_node_ids.update(
            [edge.neighbor_id for edge in node.outgoing_edges])
  dummy_example = Features()
  return [(node_id, dummy_example) for node_id in unique_node_ids]
