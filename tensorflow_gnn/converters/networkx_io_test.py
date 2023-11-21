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
"""Tests for networkx converter."""

import tensorflow as tf
from tensorflow_gnn.converters import networkx_io
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt

as_tensor = tf.convert_to_tensor
as_ragged = tf.ragged.constant


class NetworkxTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    const.enable_graph_tensor_validation_at_runtime()

  def test_two_node_sets(self):
    tf_gnn_graph = gt.GraphTensor.from_pieces(
        node_sets={
            const.NODES: gt.NodeSet.from_fields(
                features={
                    '.#id': tf.constant([69429, 0]),
                },
                sizes=tf.constant([2]),
            ),
            'node_set2': gt.NodeSet.from_fields(
                features={
                    '.#id': tf.constant([1, 2, 3, 4]),
                },
                sizes=tf.constant([4]),
            ),
        },
        edge_sets={
            const.EDGES: gt.EdgeSet.from_fields(
                sizes=tf.constant([1]),
                adjacency=adj.Adjacency.from_indices(
                    source=(const.NODES, tf.constant([0])),
                    target=('node_set2', tf.constant([3])),
                ),
            ),
        },
    )

    nx_graph = networkx_io.to_networkx_graph(tf_gnn_graph)

    num_tf_nodes = int(tf_gnn_graph.node_sets['nodes'].sizes) + int(
        tf_gnn_graph.node_sets['node_set2'].sizes
    )
    num_tf_edges = int(tf_gnn_graph.edge_sets['edges'].sizes)

    self.assertEqual(num_tf_nodes, nx_graph.number_of_nodes())
    self.assertEqual(num_tf_edges, nx_graph.number_of_edges())

  def test_three_components(self):
    tf_gnn_graph = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(
            features={const.HIDDEN_STATE: tf.zeros([1, 5])},
            sizes=tf.constant([1]),
        ),
        node_sets={
            const.NODES: gt.NodeSet.from_fields(
                features={const.HIDDEN_STATE: tf.eye(7)}, sizes=tf.constant([7])
            )
        },
        edge_sets={
            const.EDGES: gt.EdgeSet.from_fields(
                features={const.HIDDEN_STATE: tf.ones([4, 3])},
                sizes=tf.constant([4]),
                adjacency=adj.Adjacency.from_indices(
                    source=(const.NODES, tf.constant([0, 3, 4, 5])),
                    target=(const.NODES, tf.constant([1, 2, 6, 4])),
                ),
            ),
        },
    )

    nx_graph = networkx_io.to_networkx_graph(tf_gnn_graph)

    num_tf_nodes = int(tf_gnn_graph.node_sets['nodes'].sizes)
    num_tf_edges = int(tf_gnn_graph.edge_sets['edges'].sizes)

    self.assertEqual(num_tf_nodes, nx_graph.number_of_nodes())
    self.assertEqual(num_tf_edges, nx_graph.number_of_edges())

    self.assertEqual(
        nx_graph.nodes()[0], {'#id': '0', '#type': 'nodes', '#local_id': 0}
    )
    self.assertEqual(list(nx_graph.edges(0)), [(0, 1)])


if __name__ == '__main__':
  tf.test.main()
