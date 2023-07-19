# Copyright 2023 The TensorFlow GNN Authors. All Rights Reserved.
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
"""Tests for networkx_data."""
from absl import logging
import tensorflow as tf
from tensorflow_gnn.experimental.in_memory import networkx_data

Example = tf.train.Example


class NetworkXDiGraphData(tf.test.TestCase):

  def test_karate_graph(self):
    try:
      import networkx as nx  # pylint: disable=g-import-not-at-top
    except ImportError:
      logging.warn('networkx not present, test skipped')
      return

    graph = nx.karate_club_graph()
    graph_data = networkx_data.NetworkXDiGraphData(graph)

    # check some invariants
    self.assertLen(graph_data.node_sets(), 1)
    self.assertLen(graph_data.edge_sets(), 2)
    self.assertIn('edges', graph_data.edge_sets())
    self.assertIn('rev_edges', graph_data.edge_sets())
    self.assertIn('nodes', graph_data.node_sets())

    self.assertEqual(graph_data.node_counts()['nodes'], graph.number_of_nodes())
    # number_of_edges() is calculating undirected (bidirectional edges) here.
    self.assertLen(
        graph_data.edge_sets()['edges'].adjacency.source,
        graph.number_of_edges() * 2,
    )

  def test_directed_graph(self):
    try:
      import networkx as nx  # pylint: disable=g-import-not-at-top
    except ImportError:
      logging.warn('networkx not present, test skipped')
      return

    graph = nx.gn_graph(10, seed=0)
    graph_data = networkx_data.NetworkXDiGraphData(graph)

    # check some invariants
    self.assertLen(graph_data.node_sets(), 1)
    self.assertLen(graph_data.edge_sets(), 2)
    self.assertIn('edges', graph_data.edge_sets())
    self.assertIn('rev_edges', graph_data.edge_sets())
    self.assertIn('nodes', graph_data.node_sets())

    self.assertEqual(graph_data.node_counts()['nodes'], graph.number_of_nodes())
    self.assertLen(
        graph_data.edge_sets()['edges'].adjacency.source,
        graph.number_of_edges(),
    )


if __name__ == '__main__':
  tf.test.main()
