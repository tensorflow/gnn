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

"""Tests for pyg_adapter."""

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow_gnn as tfgnn
from tensorflow_gnn.experimental.datasets import pyg_adapter
from torch_geometric import datasets as tg_datasets


class PygAdapterTest(parameterized.TestCase):

  def assertNumOfGraphs(self, num_graphs, graph_tensor):
    node_sets = graph_tensor.node_sets
    edge_sets = graph_tensor.edge_sets
    for node_type in node_sets.keys():
      self.assertEqual(
          node_sets[node_type].sizes.shape.as_list(),
          [num_graphs, 1],
      )
    for edge_type in edge_sets.keys():
      self.assertEqual(
          edge_sets[edge_type].sizes.shape.as_list(),
          [num_graphs, 1],
      )
      adjacency = edge_sets[edge_type].adjacency
      self.assertEqual(
          adjacency.target.shape.as_list(),
          [num_graphs, None],
      )
      self.assertEqual(
          adjacency.source.shape.as_list(),
          [num_graphs, None],
      )


class PygAdapterHomogenousTest(PygAdapterTest):

  def test_synthetic_node_classification(self):
    """Tests the conversion of a synthetic dataset into a graph tensor.

    This synthetic dataset is a homogenous multi-graph dataset comprising:
    -24 graphs
    -where a node has one feature 'x' with dimension 8.
    -an edge has one feature 'edge attr' with feature dimension 6.
    -and each node has a label feature 'y'.
    """
    dataset_synthetic = tg_datasets.FakeDataset(
        num_graphs=24,
        avg_num_nodes=19,
        avg_degree=8,
        num_channels=8,
        edge_dim=6,
        num_classes=10,
        task='node',
        is_undirected=True,
        transform=None,
        pre_transform=None,
    )
    graph_synthetic = pyg_adapter.build_graph_tensor_pyg(dataset_synthetic)
    self.assertNumOfGraphs(24, graph_synthetic)
    self.assertEqual(
        graph_synthetic.node_sets[tfgnn.NODES].features['x'].shape[2],
        8,
    )
    self.assertEqual(
        graph_synthetic.edge_sets[tfgnn.EDGES].features['edge_attr'].shape[2],
        6,
    )
    self.assertEmpty(graph_synthetic.context.features)

  def test_synthetic_graph_classification_without_node_attrs(self):
    """Tests the conversion of a synthetic dataset into a graph tensor.

    This synthetic dataset is a homogenous multi-graph dataset comprising:
    -3 graphs
    -where a node has no features
    -an edge has one feature 'edge attr' with feature dimension 6.
    -and each graph has a context feature 'y'.
    """
    dataset_synthetic = tg_datasets.FakeDataset(
        num_graphs=3,
        avg_num_nodes=19,
        avg_degree=8,
        num_channels=0,
        edge_dim=6,
        num_classes=10,
        task='auto',
        is_undirected=True,
        transform=None,
        pre_transform=None,
    )
    graph_synthetic = pyg_adapter.build_graph_tensor_pyg(dataset_synthetic)
    self.assertNumOfGraphs(3, graph_synthetic)
    self.assertEmpty(graph_synthetic.node_sets[tfgnn.NODES].features)
    self.assertEqual(
        graph_synthetic.edge_sets[tfgnn.EDGES].features['edge_attr'].shape[2],
        6,
    )
    self.assertEqual(
        graph_synthetic.context.features['y'].shape.as_list(), [3, 1]
    )

  def test_synthetic_graph_classification_with_edge_attrs(self):
    """Tests the conversion of a synthetic dataset into a graph tensor.

    This synthetic dataset is a homogenous multi-graph dataset comprising:
    -18 graphs
    -where a node has the feature 'x' with dimension 4
    -an edge has one feature 'edge attr' with feature dimension 4.
    -and each graph has a context feature 'y'.
    """
    dataset_synthetic = tg_datasets.FakeDataset(
        num_graphs=18,
        avg_num_nodes=19,
        avg_degree=8,
        num_channels=4,
        edge_dim=4,
        num_classes=10,
        task='auto',
        is_undirected=True,
        transform=None,
        pre_transform=None,
    )
    graph_synthetic = pyg_adapter.build_graph_tensor_pyg(dataset_synthetic)
    self.assertNumOfGraphs(18, graph_synthetic)
    self.assertEqual(
        graph_synthetic.node_sets[tfgnn.NODES].features['x'].shape[2], 4
    )
    self.assertEqual(
        graph_synthetic.edge_sets[tfgnn.EDGES].features['edge_attr'].shape[2],
        4,
    )
    self.assertEqual(
        graph_synthetic.context.features['y'].shape.as_list(), [18, 1]
    )

  def test_karateclub_dataset(self):
    """Tests the conversion of the karate club dataset into a graph tensor.

    The karate club dataset is a homogenous single graph comprising:
    -34 nodes with three features: 'x, 'y', and 'train_mask'
    -and 156 (undirected and unweighted) edges.
    """
    dataset_karate_club = tg_datasets.KarateClub()
    graph_karateclub = pyg_adapter.build_graph_tensor_pyg(dataset_karate_club)
    self.assertNumOfGraphs(1, graph_karateclub)
    node_features = graph_karateclub.node_sets[tfgnn.NODES].features
    for node_feature in node_features.keys():
      self.assertEqual(node_features[node_feature][0].shape[0], 34)
    self.assertEqual(
        node_features['x'][0].shape.as_list(),
        [34, 34],
    )
    self.assertEqual(
        node_features['y'][0].shape.as_list(),
        [34],
    )
    self.assertEqual(
        node_features['train_mask'][0].shape.as_list(),
        [34],
    )
    adjacency = graph_karateclub.edge_sets[tfgnn.EDGES].adjacency
    self.assertEqual(
        adjacency.source[0].shape[0],
        156,
    )
    self.assertEqual(
        adjacency.target[0].shape[0],
        156,
    )
    self.assertEmpty(graph_karateclub.edge_sets[tfgnn.EDGES].features)
    self.assertEmpty(graph_karateclub.context.features)


if __name__ == '__main__':
  absltest.main()
