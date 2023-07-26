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


class PygAdapterHomogenousTest(parameterized.TestCase):

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
    # checking for 24 graphs
    self.assertEqual(
        graph_synthetic.node_sets[tfgnn.NODES].features['x'].shape[0],
        24,
    )
    self.assertEqual(
        graph_synthetic.node_sets[tfgnn.NODES].features['y'].shape[0],
        24,
    )
    self.assertEqual(
        graph_synthetic.edge_sets[tfgnn.EDGES].adjacency.source.shape[0],
        24,
    )
    self.assertEqual(
        graph_synthetic.edge_sets[tfgnn.EDGES].adjacency.target.shape[0],
        24,
    )
    # checking that node feature 'x' has feature dimension 8.
    self.assertEqual(
        graph_synthetic.node_sets[tfgnn.NODES].features['x'].shape[2],
        8,
    )
    # checking that edge feature 'edge attr' has feature dimension 6.
    self.assertEqual(
        graph_synthetic.edge_sets[tfgnn.EDGES].features['edge_attr'].shape[2],
        6,
    )
    # checking for empty context features
    self.assertEmpty(graph_synthetic.context.features)

  def test_synthetic_without_node_attrs(self):
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
    # checking for 3 graphs
    self.assertEqual(
        graph_synthetic.edge_sets[tfgnn.EDGES].adjacency.source.shape[0],
        3,
    )
    self.assertEqual(
        graph_synthetic.edge_sets[tfgnn.EDGES].adjacency.target.shape[0],
        3,
    )
    # checking that nodeset 'tfgnn.NODES' has no features.
    self.assertEmpty(graph_synthetic.node_sets[tfgnn.NODES].features)
    # checking that edge feature 'edge attr' has feature dimension 6.
    self.assertEqual(
        graph_synthetic.edge_sets[tfgnn.EDGES].features['edge_attr'].shape[2],
        6,
    )
    # checking for graph context
    self.assertEqual(graph_synthetic.context.features['y'].shape[0], 3)

  def test_synthetic_with_edge_attrs(self):
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
    # checking for 18 graphs
    self.assertEqual(
        graph_synthetic.node_sets[tfgnn.NODES].features['x'].shape[0], 18
    )
    self.assertEqual(
        graph_synthetic.edge_sets[tfgnn.EDGES].adjacency.source.shape[0],
        18,
    )
    self.assertEqual(
        graph_synthetic.edge_sets[tfgnn.EDGES].adjacency.target.shape[0],
        18,
    )
    # checking that node feature 'x' has feature dimension .
    self.assertEqual(
        graph_synthetic.node_sets[tfgnn.NODES].features['x'].shape[2], 4
    )
    # checking that edge feature 'edge attr' has feature dimension 4.
    self.assertEqual(
        graph_synthetic.edge_sets[tfgnn.EDGES].features['edge_attr'].shape[2],
        4,
    )
    # checking for graph context
    self.assertEqual(graph_synthetic.context.features['y'].shape[0], 18)

  def test_karateclub_dataset(self):
    """Tests the conversion of the karate club dataset into a graph tensor.

    The karate club dataset is a homogenous single graph comprising:
    -34 nodes with three features: 'x, 'y', and 'train_mask'
    -and 156 (undirected and unweighted) edges.
    """
    dataset_karate_club = tg_datasets.KarateClub()
    graph_karateclub = pyg_adapter.build_graph_tensor_pyg(dataset_karate_club)
    # checking for single graph
    self.assertEqual(
        graph_karateclub.node_sets[tfgnn.NODES].features['x'].shape[0], 1
    )
    self.assertEqual(
        graph_karateclub.node_sets[tfgnn.NODES].features['y'].shape[0], 1
    )
    self.assertEqual(
        graph_karateclub.node_sets[tfgnn.NODES].features['train_mask'].shape[0],
        1,
    )
    self.assertEqual(graph_karateclub.edge_sets[tfgnn.EDGES].sizes.shape[0], 1)
    # checking for 34 nodes
    self.assertEqual(
        graph_karateclub.node_sets[tfgnn.NODES].features['x'][0].shape[0], 34
    )
    self.assertEqual(
        graph_karateclub.node_sets[tfgnn.NODES].features['y'][0].shape[0], 34
    )
    self.assertEqual(
        graph_karateclub.node_sets[tfgnn.NODES]
        .features['train_mask'][0]
        .shape[0],
        34,
    )
    # checking for the shape of 'x': 34 by 34
    self.assertEqual(
        graph_karateclub.node_sets[tfgnn.NODES]
        .features['x'][0]
        .shape.as_list(),
        [34, 34],
    )
    # checking for the shape of 'y': 34
    self.assertEqual(
        graph_karateclub.node_sets[tfgnn.NODES]
        .features['y'][0]
        .shape.as_list(),
        [34],
    )
    # checking for the shape of 'train_mask'
    self.assertEqual(
        graph_karateclub.node_sets[tfgnn.NODES]
        .features['train_mask'][0]
        .shape.as_list(),
        [34],
    )
    # checking for 156 edges
    self.assertEqual(
        graph_karateclub.edge_sets[tfgnn.EDGES].adjacency.source[0].shape[0],
        156,
    )
    self.assertEqual(
        graph_karateclub.edge_sets[tfgnn.EDGES].adjacency.target[0].shape[0],
        156,
    )
    # checking for unweighted/featureless edges
    self.assertEmpty(graph_karateclub.edge_sets[tfgnn.EDGES].features)
    # checking for empty context features
    self.assertEmpty(graph_karateclub.context.features)


if __name__ == '__main__':
  absltest.main()
