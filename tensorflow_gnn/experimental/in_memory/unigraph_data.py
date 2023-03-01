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
"""Holds Unigraph data in-memory, for in-memory sampling and fullbatch learning.

Unigraphs (see data/unigraphs.py) are graphs that can be loaded using
`tfgnn.GraphSchema` proto. The proto node-sets and edge-sets should have the
`metadata` attribute populated, so-as to read the node features, edge features,
and adjacency lists, from various file sources, e.g., BigQuery, tfrecordio file,
CSV files. Please refer to unigraph.py, class `DictStreams`, for more.
"""

import collections
from typing import Dict, List, Mapping, MutableMapping, Optional, Tuple, Union


from absl import logging
import numpy as np
import pyarrow
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.data import unigraph
from tensorflow_gnn.experimental.in_memory import datasets
import tqdm

Example = tf.train.Example


class UnigraphData(datasets.InMemoryGraphData):
  """Implementation of in-memory dataset loader for unigraph format.

  For now, this class inherits from InMemoryGraphData to enable its use
  with in-memory graph sampling.

  TODO(haija): Redesign the GraphData classes to decouple the source data
  format (unigraph or other) from the prediction task.
  """

  def __init__(self,
               graph_schema: tfgnn.GraphSchema,
               keep_intermediate_examples=False,
               max_size: Optional[int] = None, use_tqdm=False):
    """Constructor to represent a unigraph in memory.

    Args:
      graph_schema: A tfgnn.GraphSchema protobuf message.
      keep_intermediate_examples: Used for testing. It keeps tf.Example protos
        saved in memory, in attributes `node_features` and `flat_edge_list`.
      max_size (int): If given, it will limit the number of records for all
        node sets and edge sets.
      use_tqdm: If set, all node set and edge set iterators will be wrapped with
        `tqdm.tqdm`.
    """
    super().__init__()
    self._graph_schema = graph_schema

    # Node Set Name -> Node ID ->  auto-incrementing int (`node_idx`)`.
    self.compression_maps: Dict[
        tfgnn.NodeSetName, Dict[bytes, int]] = collections.defaultdict(dict)
    # Node Set Name -> list of Node ID (from above) sorted per int (`node_idx`).
    self.rev_compression_maps: Dict[
        tfgnn.NodeSetName, List[bytes]] = collections.defaultdict(list)

    # Mapping from node set name to a mapping of node id to tf.train.Example
    # pairs.
    self.node_features: Dict[
        tfgnn.NodeSetName, Dict[bytes, tf.train.Example]] = {}

    # Mapping from an edge set name to a list of [src, target] pairs
    self.flat_edge_list: Dict[
        tfgnn.EdgeSetName, List[Tuple[str, str, tf.train.Example]]] = {}

    stream_dicts = unigraph.DictStreams.iter_graph_via_schema(graph_schema)

    # Node set name -> feature name -> feature tensor.
    # All features under a node set must have same `feature_tensor.shape[0]`.
    node_features_dict: Dict[
        tfgnn.NodeSetName, Dict[str, List[np.ndarray]]] = {}
    node_features_dict = collections.defaultdict(
        lambda: collections.defaultdict(list))

    for node_set_name, stream in stream_dicts[tfgnn.NODES].items():
      logging.info('Reading node set: %s', node_set_name)
      self.node_features[node_set_name] = {}
      node_schema = graph_schema.node_sets[node_set_name]
      feature_names = list(node_schema.features.keys())
      feature_lists = [node_features_dict[node_set_name][f]
                       for f in feature_names]
      if use_tqdm:
        stream = tqdm.tqdm(stream)
      #                                   `stream` is a once-iterator.
      for node_order, (node_id, example) in enumerate(stream):
        if max_size and node_order >= max_size:
          break
        if node_id in self.compression_maps[node_set_name]:
          raise ValueError('More than one node with ID %s' % node_id)
        self.compression_maps[node_set_name][node_id] = node_order
        self.rev_compression_maps[node_set_name].append(node_id)
        if keep_intermediate_examples:
          self.node_features[node_set_name][node_id] = example
        _append_features(feature_lists, feature_names, example)

    self._node_features_dict: Dict[tfgnn.NodeSetName, Dict[str, tf.Tensor]] = {}
    for node_set_name, features in node_features_dict.items():
      logging.info('Concatenating features for node set: %s', node_set_name)
      self._node_features_dict[node_set_name] = {}
      for feature_name, feature_list in features.items():
        if feature_list and isinstance(feature_list[0], pyarrow.Scalar):
          feature_list = [feat.as_py() for feat in feature_list]
        feature_schema = graph_schema.node_sets[node_set_name]
        feature_schema = feature_schema.features[feature_name]
        tf_dtype = tf.dtypes.as_dtype(feature_schema.dtype)
        np_dtype = tf_dtype.as_numpy_dtype
        dims = [d.size for d in feature_schema.shape.dim]
        dims.insert(0, len(feature_list))
        feature_np_array = np.array(feature_list, dtype=np_dtype).reshape(dims)
        self._node_features_dict[node_set_name][feature_name] = (
            tf.convert_to_tensor(feature_np_array))
    del node_features_dict  # Free-up memory.

    edge_lists: Dict[
        Tuple[tfgnn.NodeSetName, tfgnn.EdgeSetName, tfgnn.NodeSetName],
        List[np.ndarray]] = collections.defaultdict(list)
    new_nodes = collections.defaultdict(list)
    for edge_set_name, stream in stream_dicts[tfgnn.EDGES].items():
      if use_tqdm:
        stream = tqdm.tqdm(stream)
      edge_schema = graph_schema.edge_sets[edge_set_name]
      if unigraph.is_edge_reversed(edge_schema):
        continue
      logging.info('Reading edge set: %s', edge_set_name)
      source_node_set_name = edge_schema.source
      target_node_set_name = edge_schema.target
      self.flat_edge_list[edge_set_name] = []
      for edge_order, (src, target, example) in enumerate(stream):
        if max_size and edge_order >= max_size:
          break
        if keep_intermediate_examples:
          self.flat_edge_list[edge_set_name].append((src, target, example))
        edge_key = (source_node_set_name, edge_set_name, target_node_set_name)
        # Ignore edge features, for now.
        edge_endpoints = (
            self._compression_id(source_node_set_name, src,
                                 track_new=new_nodes[source_node_set_name]),
            self._compression_id(target_node_set_name, target,
                                 track_new=new_nodes[target_node_set_name]))
        edge_lists[edge_key].append(np.array(edge_endpoints))

    # If some edge accesses new node ID (that was not part of node streams),
    # then `_compression_id` assigns a new position, tracking newly-added node
    # IDs in `new_nodes`. We zero-pad feature tensors, such that, dim[0] of
    # every feature equals (MaxEndpointPosition+1).
    for node_set_name, new_node_list in new_nodes.items():
      num_new_nodes = len(new_node_list)
      if num_new_nodes == 0:
        continue
      node_set_features = self._node_features_dict.get(node_set_name, {})
      for feat_name in list(node_set_features.keys()):
        cur_feat = node_set_features[feat_name]
        zero_padding_shape = list(cur_feat.shape)
        zero_padding_shape[0] = num_new_nodes
        zero_padded_feat = tf.concat([
            cur_feat,
            tf.zeros(zero_padding_shape, dtype=cur_feat.dtype)
        ], axis=0)
        node_set_features[feat_name] = zero_padded_feat

    self._edge_lists: Dict[
        Tuple[tfgnn.NodeSetName, tfgnn.EdgeSetName, tfgnn.NodeSetName],
        tf.Tensor] = {}
    for edge_key, list_np_edge_list in edge_lists.items():
      self._edge_lists[edge_key] = tf.convert_to_tensor(
          np.stack(list_np_edge_list, -1))
    del edge_lists

  def _compression_id(self, node_set_name, node_id: bytes,
                      track_new: Optional[List[Tuple[int, bytes]]] = None):
    if node_id not in self.compression_maps[node_set_name]:
      assign_id = len(self.compression_maps[node_set_name])
      self.compression_maps[node_set_name][node_id] = assign_id
      self.rev_compression_maps[node_set_name].append(node_id)
      if track_new is not None:
        track_new.append((assign_id, node_id))

    return self.compression_maps[node_set_name][node_id]

  def get_adjacency_list(self) -> Dict[tfgnn.EdgeSetName, Dict[str, Example]]:
    """Returns weighted edges as an adjacency list of nested dictionaries.

    This function is useful for testing for fast access to edge features based
    on (source, target) IDs. Beware, this function will create an object that
    may increase memory usage.
    """
    adjacency_sets = collections.defaultdict(dict)
    for edge_set_name, flat_edge_list in self.flat_edge_list.items():
      adjacency_sets[edge_set_name] = collections.defaultdict(dict)
      for source, target, example in flat_edge_list:
        adjacency_sets[edge_set_name][source][target] = example

    return adjacency_sets

  def node_features_dicts(self) -> Mapping[
      tfgnn.NodeSetName, MutableMapping[tfgnn.FieldName, tf.Tensor]]:
    return self._node_features_dict

  def node_counts(self) -> Mapping[tfgnn.NodeSetName, int]:
    """Returns total number of graph nodes per node set."""
    return {node_set_name: len(ids)
            for node_set_name, ids in self.rev_compression_maps.items()}

  def edge_lists(self) -> Mapping[
      Tuple[tfgnn.NodeSetName, tfgnn.EdgeSetName, tfgnn.NodeSetName],
      tf.Tensor]:
    """Returns dict from "edge type tuple" to int array of shape (2, num_edges).

    "edge type tuple" string-tuple: (src_node_set, edge_set, target_node_set).
    """
    return self._edge_lists


def _append_features(feature_lists, feature_names, example):
  for feature_list, feature_name in zip(feature_lists, feature_names):
    if isinstance(example, tf.train.Example):
      feat_value = _get_feature_values(example.features.feature[feature_name])
    else:
      record_feat_name = feature_name
      if record_feat_name.startswith('#'):
        record_feat_name = record_feat_name[1:]
      feat_value = example[record_feat_name]
    feature_list.append(feat_value)


# Copied from tensorflow_gnn/graph/graph_tensor_random.py
def _get_feature_values(feature: tf.train.Feature) -> Union[List[str],
                                                            List[int],
                                                            List[float]]:
  """Return the values from a TF feature proto."""
  if feature.HasField('float_list'):
    return list(feature.float_list.value)
  elif feature.HasField('int64_list'):
    return list(feature.int64_list.value)
  elif feature.HasField('bytes_list'):
    return list(feature.bytes_list.value)
  return []

