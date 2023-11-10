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
"""A PyG Dataset adapter.

pyg_adapter functions as a tool that converts Pytorch Geometric Datasets
into graph tensors that are compatible with TensorFlow GNN.
"""

import collections
import dataclasses
from typing import List, Mapping, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from torch_geometric import data as tg_data


@dataclasses.dataclass
class GraphInfo:
  context_features: Mapping[str, List[np.ndarray]]
  node_features: Mapping[str, Mapping[str, List[np.ndarray]]]
  edge_features: Mapping[
      Union[Tuple[str, str, str], str], Mapping[str, List[np.ndarray]]
  ]
  graph_data: Mapping[
      Union[str, Tuple[str, str, str]], Mapping[str, List[np.ndarray]]
  ]


def build_graph_tensor_pyg(
    dataset: tg_data.InMemoryDataset,
) -> tfgnn.GraphTensor:
  """Returns a graph tensor.

  Args:
    dataset: PyG dataset to be converted into graph tensor.

  Returns:
    A graph tensor.

  Raises:
    ValueError: if input dataset is not a PyG InMemoryDataset.
  """
  if not isinstance(dataset, tg_data.InMemoryDataset):
    raise ValueError(
        f'dataset: {type(dataset)} is not a PyG InMemoryDataset.'
    )

  if isinstance(dataset[0], tg_data.data.Data):
    graph_info = _extract_features_homogenous(dataset)
  elif isinstance(dataset[0], tg_data.hetero_data.HeteroData):
    graph_info = _extract_features_heterogeneous(dataset)
  else:
    raise ValueError(f'dataset: {type(dataset)} is not currently supported.')

  graph = _build_graph_tensor(graph_info)
  return graph


def _extract_features_homogenous(dataset: tg_data.InMemoryDataset) -> GraphInfo:
  """Extracts graph structural data and features from homogenous dataset.

  Args:
    dataset: A homogenous PyG dataset.

  Returns:
    An instance of GraphInfo.

  Raises:
    ValueError: if input dataset is not homogenous.
  """

  if not isinstance(dataset[0], tg_data.data.Data):
    raise ValueError(f'dataset: {type(dataset)} is not a homogenous dataset.')

  node_features = {tfgnn.NODES: collections.defaultdict(list)}
  edge_features = {tfgnn.EDGES: collections.defaultdict(list)}
  context_features = collections.defaultdict(list)
  graph_data = {
      tfgnn.NODES: collections.defaultdict(list),
      tfgnn.EDGES: collections.defaultdict(list),
  }

  for graph in dataset:
    graph_map = graph.to_dict()
    list_of_fields = graph_map.keys() - {'num_nodes'}
    graph_data[tfgnn.NODES]['list_num_nodes'].append([graph.num_nodes])
    graph_data[tfgnn.EDGES]['list_num_edges'].append([graph.num_edges])

    for field in list_of_fields:
      if graph.is_node_attr(field):
        node_features[tfgnn.NODES][field].append(graph_map[field].numpy())

      elif graph.is_edge_attr(field):
        if field == 'edge_index':
          graph_data[tfgnn.EDGES]['source_edge'].append(
              graph_map[field].numpy()[0]
          )
          graph_data[tfgnn.EDGES]['target_edge'].append(
              graph_map[field].numpy()[1]
          )
        else:
          edge_features[tfgnn.EDGES][field].append(graph_map[field].numpy())

      elif len(graph_map[field]) == 1:
        context_features[field].append(graph_map[field].numpy())
      else:
        temp = graph_map[field].numpy()
        context_features[field].append(np.expand_dims(temp, axis=0))

  graph_info = GraphInfo(
      context_features, node_features, edge_features, graph_data
  )
  return graph_info


def _extract_features_heterogeneous(
    dataset: tg_data.InMemoryDataset,
) -> GraphInfo:
  """Extracts graph structural data and features from heterogeneous dataset.

  Args:
    dataset: A heterogeneous PyG dataset.

  Returns:
    An instance of GraphInfo.

  Raises:
    ValueError: if input dataset is not heterogeneous.
  """

  if not isinstance(dataset[0], tg_data.hetero_data.HeteroData):
    raise ValueError(
        f'dataset: {type(dataset)} is not a heterogeneous dataset.'
    )

  node_types = dataset[0].node_types
  edge_types = dataset[0].edge_types

  node_features = {}
  edge_features = {}
  context_features = collections.defaultdict(list)
  graph_data = collections.defaultdict(lambda: collections.defaultdict(list))

  for graph in dataset:
    graph_map = graph.to_dict()

    for node_type in node_types:
      if node_type not in node_features:
        node_features[node_type] = collections.defaultdict(list)
      node_data = graph_map.pop(node_type)

      if 'num_nodes' in node_data:
        graph_data[node_type]['list_num_nodes'].append(
            [node_data.pop('num_nodes')]
        )
      else:
        # appending size information only once for a given node_type.
        for node_feature in node_data.keys():
          graph_data[node_type]['list_num_nodes'].append(
              [len(node_data[node_feature])]
          )
          break
      for node_feature in node_data.keys():
        node_features[node_type][node_feature].append(
            node_data[node_feature].numpy()
        )

    for edge_type in edge_types:
      if edge_type not in edge_features:
        edge_features[edge_type] = collections.defaultdict(list)
      edge_data = graph_map.pop(edge_type)
      for edge_piece in edge_data.keys():
        if edge_piece == 'edge_index':
          graph_data[edge_type]['source_edge'].append(
              edge_data['edge_index'][0].numpy()
          )
          graph_data[edge_type]['target_edge'].append(
              edge_data['edge_index'][1].numpy()
          )
          graph_data[edge_type]['list_num_edges'].append(
              [len(edge_data['edge_index'][0].numpy())]
          )
        else:
          edge_features[edge_type][edge_piece].append(
              edge_data[edge_piece].numpy()
          )

    # The behavior of "HeteroData.to_dict" changes across PyG versions.
    # Before 2.4.0, the output `dict` has node_types, edge_types, and
    # whatever keys remain in '_global_store'.
    # In 2.4.0 and later, the '_global_store' is itself an element in the
    # returned dict.
    # This switch is necessary to make this code compatible across versions.
    if '_global_store' in graph_map:
      context_dict = graph_map['_global_store']
    else:
      context_dict = graph_map

    for context_feature in context_dict.keys():
      if len(context_dict[context_feature]) == 1:
        context_features[context_feature].append(
            context_dict[context_feature].numpy()
        )
      else:
        temp = context_dict[context_feature].numpy()
        context_features[context_feature].append(np.expand_dims(temp, axis=0))

  graph_info = GraphInfo(
      context_features, node_features, edge_features, graph_data
  )
  return graph_info


def _build_graph_tensor(graph_info: GraphInfo) -> tfgnn.GraphTensor:
  """Returns a graph tensor.

  Args:
    graph_info: a GraphInfo instance comprising node/context/edge structure and
      feature related data.

  Returns:
    A graph tensor
  """
  node_sets = _build_node_sets(graph_info)
  edge_sets = _build_edge_sets(graph_info)
  context_set = {}
  for _, context_feature in graph_info.context_features.items():
    if len(context_feature) == 1:
      context_set = {
          k: tf.constant(v[0]) for k, v in graph_info.context_features.items()
      }
    else:
      context_set = {
          k: tf.constant(v) for k, v in graph_info.context_features.items()
      }
    break
  graph_context_set = tfgnn.Context.from_fields(features=context_set)
  graph = tfgnn.GraphTensor.from_pieces(graph_context_set, node_sets, edge_sets)
  return graph


def _build_node_sets(graph_info: GraphInfo) -> Mapping[str, tfgnn.NodeSet]:
  """builds node sets.

  Args:
    graph_info: a GraphInfo instance comprising node/context/edge structure and
      feature related data.

  Returns:
    node sets
  """
  node_data = {}
  node_sets = {}
  for node_type in graph_info.node_features.keys():
    sizes = graph_info.graph_data[node_type]['list_num_nodes']
    if len(sizes) == 1:
      sizes = tf.constant(graph_info.graph_data[node_type]['list_num_nodes'][0])
      node_data[node_type] = {
          k: tf.constant(v[0])
          for k, v in graph_info.node_features[node_type].items()
      }
    else:
      sizes = tf.constant(graph_info.graph_data[node_type]['list_num_nodes'])
      node_data[node_type] = {
          k: tf.ragged.constant(v, ragged_rank=1)
          for k, v in graph_info.node_features[node_type].items()
      }
    node_sets[node_type] = tfgnn.NodeSet.from_fields(
        sizes=sizes, features=node_data[node_type]
    )
  return node_sets


def _build_edge_sets(graph_info: GraphInfo) -> Mapping[str, tfgnn.EdgeSet]:
  """builds edge sets.

  Args:
   graph_info: a GraphInfo isntance comprising node/context/edge structure and
     feature related data.

  Returns:
    edge sets
  """
  edge_data = {}
  edge_sets = {}
  for edge_type in graph_info.edge_features.keys():
    # PyG represents edge types in heterogeneous datasets as a tuple of strings
    # of length 3, with the following format:(source_node, relationship_name,
    # target_node). In converting into graph tensor, the tuple is concatenated
    # into a string.
    if isinstance(edge_type, str):
      edge_name = edge_type
      source = tfgnn.NODES
      target = tfgnn.NODES
    else:
      source, edge_name, target = edge_type
      # updating `edge_name` to `f'{source}-{edge_name}-{target}'` to ensure
      # `edge_name` is unique.
      edge_name = f'{source}-{edge_name}-{target}'
    sizes = graph_info.graph_data[edge_type]['list_num_edges']
    if len(sizes) == 1:
      sizes = tf.constant(graph_info.graph_data[edge_type]['list_num_edges'][0])
      edge_data[edge_name] = {
          k: tf.constant(v[0])
          for k, v in graph_info.edge_features[edge_type].items()
      }
      adjacency_source = tf.constant(
          graph_info.graph_data[edge_type]['source_edge'][0]
      )
      adjacency_target = tf.constant(
          graph_info.graph_data[edge_type]['target_edge'][0]
      )
    else:
      sizes = tf.constant(graph_info.graph_data[edge_type]['list_num_edges'])
      adjacency_source = tf.ragged.constant(
          graph_info.graph_data[edge_type]['source_edge'],
          ragged_rank=1,
      )
      adjacency_target = tf.ragged.constant(
          graph_info.graph_data[edge_type]['target_edge'],
          ragged_rank=1,
      )
      edge_data[edge_name] = {
          k: tf.ragged.constant(v, ragged_rank=1)
          for k, v in graph_info.edge_features[edge_type].items()
      }
    edge_sets[edge_name] = tfgnn.EdgeSet.from_fields(
        sizes=sizes,
        adjacency=tfgnn.Adjacency.from_indices(
            source=(
                source,
                adjacency_source,
            ),
            target=(
                target,
                adjacency_target,
            ),
        ),
        features=edge_data[edge_name],
    )
  return edge_sets
