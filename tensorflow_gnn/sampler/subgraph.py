"""Conversion of Subgraph to tf.train.Example graph tensor.

This library contains code to convert an instance of a Subgraph proto, which
contains a sample of a graph, its graph topology, edges, and associated node and
edge features, to an encoded tf.train.Example proto that can be streamed to a
TensorFlow model.
"""
import collections

from typing import Any, Dict, List, Optional, Union

import tensorflow as tf
import tensorflow_gnn as gnn
from tensorflow_gnn.sampler import subgraph_pb2

Subgraph = subgraph_pb2.Subgraph
Example = tf.train.Example
Feature = tf.train.Feature


def encode_subgraph_to_example(schema: gnn.GraphSchema,
                               subgraph: Subgraph) -> Example:
  """Convert a subgraph to an encoded graph tensor."""

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
  by_node_set_name: Dict[
      gnn.NodeSetName, List[subgraph_pb2.Node]] = collections.defaultdict(list)
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

      source_name = "edges/{}.{}".format(edge.edge_set_name, gnn.SOURCE_NAME)
      target_name = "edges/{}.{}".format(edge.edge_set_name, gnn.TARGET_NAME)
      example.features.feature[source_name].int64_list.value.append(source_idx)
      example.features.feature[target_name].int64_list.value.append(target_idx)
      edge_counter[edge.edge_set_name] += 1

      # Store the edge features.
      edge_features_dict = edge_features_dicts[edge.edge_set_name]
      _copy_features(edge.features, edge_features_dict)

  # Produce size features.
  for node_set_name, num_nodes in node_counter.items():
    node_size_name = "nodes/{}.{}".format(node_set_name, gnn.SIZE_NAME)
    example.features.feature[node_size_name].int64_list.value.append(num_nodes)
  for edge_set_name, num_edges in edge_counter.items():
    edge_size_name = "edges/{}.{}".format(edge_set_name, gnn.SIZE_NAME)
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
  """Copy features from sg_features to example."""
  for feature_name, ex_feature in ex_features_dict.items():
    sg_feature = sg_features.feature.get(feature_name, None)
    if sg_feature is None:
      # Feature is empty for that node. Fail for now, ragged is not supported
      # by this conversion routine.
      raise ValueError("Feature '{}' is missing from input: {}".format(
          feature_name, sg_features))
    ex_feature.MergeFrom(sg_feature)
