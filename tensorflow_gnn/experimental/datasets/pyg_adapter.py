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
from typing import List, Mapping
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from torch_geometric import data as tg_data


def _extract_features_homogenous(
    dataset: tg_data.InMemoryDataset,
) -> Mapping[str, Mapping[str, List[np.ndarray]]]:
  """Returns a nested dictionary comprising buckets of edge, node, context, and misc graph structural data.

  Args:
    dataset: A homogenous PyG dataset.

  Returns:
    A nested dictionary comprising buckets of edge, node, context, and misc
    graph data.

  Raises:
    ValueError: if input dataset is not homogenous.
  """
  # ensuring input dataset is a homogenous graph dataset
  if not isinstance(dataset[0], tg_data.data.Data):
    raise ValueError(f'dataset: {type(dataset)} is not a homogenous dataset.')

  node_features = collections.defaultdict(list)
  edge_features = collections.defaultdict(list)
  context_features = collections.defaultdict(list)
  graph_data = collections.defaultdict(list)

  for graph in dataset:
    graph_map = graph.to_dict()
    list_of_fields = graph_map.keys() - {'num_nodes'}
    graph_data['list_num_nodes'].append([graph.num_nodes])
    graph_data['list_num_edges'].append([graph.num_edges])

    for field in list_of_fields:
      # extracting node related features
      if graph.is_node_attr(field):
        node_features[field].append(graph_map[field].numpy())

      # extracting edge related features
      elif graph.is_edge_attr(field):
        if field == 'edge_index':
          graph_data['source_edge'].append(graph_map[field].numpy()[0])
          graph_data['target_edge'].append(graph_map[field].numpy()[1])
        else:
          edge_features[field].append(graph_map[field].numpy())

      # extracting graph context related features.
      # additional check of len == 1 to ensure potential context related
      # features being added to the context set are of the right shape.
      elif len(graph_map[field]) == 1:
        context_features[field].append(graph_map[field].numpy())
      else:
        temp = graph_map[field].numpy()
        context_features[field].append(np.expand_dims(temp, axis=0))

  # building graph_info
  graph_info = {
      'node_features': node_features,
      'edge_features': edge_features,
      'context_features': context_features,
      'graph_data': graph_data,
  }
  return graph_info


def _build_graph_tensor_homogenous(
    *,
    graph_data: Mapping[str, List[np.ndarray]],
    node_features: Mapping[str, List[np.ndarray]],
    edge_features: Mapping[str, List[np.ndarray]],
    context_features: Mapping[str, List[np.ndarray]],
) -> tfgnn.GraphTensor:
  """Returns a graph tensor.

  Args:
    graph_data: a dictionary comprising node/edge size and edge index related
      data
    node_features: A dictionary where keys are labels of node related features
      and values are the associated node related features.
    edge_features: A dictionary where keys are labels of edge related features
      and values are the associated edge related features.
    context_features: A dictionary where keys are labels of graph context
      features and values are the associated graph context related data.

  Returns:
    A graph tensor.
  """
  # building node sets
  node_features = {
      k: tf.ragged.constant(v, ragged_rank=1) for k, v in node_features.items()
  }
  node_sets = {
      tfgnn.NODES: tfgnn.NodeSet.from_fields(
          sizes=tf.ragged.constant(graph_data['list_num_nodes'], ragged_rank=1),
          features=node_features,
      )
  }

  # building edge sets
  edge_features = {
      k: tf.ragged.constant(v, ragged_rank=1) for k, v in edge_features.items()
  }
  edge_sets = {
      tfgnn.EDGES: tfgnn.EdgeSet.from_fields(
          features=edge_features,
          sizes=tf.ragged.constant(graph_data['list_num_edges'], ragged_rank=1),
          adjacency=tfgnn.Adjacency.from_indices(
              source=(
                  tfgnn.NODES,
                  tf.ragged.constant(graph_data['source_edge'], ragged_rank=1),
              ),
              target=(
                  tfgnn.NODES,
                  tf.ragged.constant(graph_data['target_edge'], ragged_rank=1),
              ),
          ),
      )
  }

  # building context set
  context_features = {
      k: tf.ragged.constant(v, ragged_rank=1)
      for k, v in context_features.items()
  }
  graph_context_set = tfgnn.Context.from_fields(features=context_features)

  # assembling graph
  graph = tfgnn.GraphTensor.from_pieces(graph_context_set, node_sets, edge_sets)
  return graph


def build_graph_tensor_pyg(
    dataset: tg_data.InMemoryDataset,
) -> tfgnn.GraphTensor:
  """Returns a graph tensor.

  Args:
    dataset: PyG dataset to be converted into graph tensor.

  Returns:
    a graph tensor.

  Raises:
    ValueError: if input dataset is not a PyG in memory dataset.
  """
  if not isinstance(dataset, tg_data.InMemoryDataset):
    raise ValueError(
        f'dataset: {type(dataset)} is not a PyG in memory dataset.'
    )

  if isinstance(dataset[0], tg_data.data.Data):
    extracted_graph_info = _extract_features_homogenous(dataset)
    graph = _build_graph_tensor_homogenous(**extracted_graph_info)
    return graph
  else:
    raise ValueError(f'dataset: {type(dataset)} is not compatible.')
