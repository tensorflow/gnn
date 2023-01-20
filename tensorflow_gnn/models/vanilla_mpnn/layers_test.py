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
"""Tests for VanillaMPNN."""
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import vanilla_mpnn


# The components of VanillaMPNNGraphUpdate have been tested elsewhere.
class VanillaMPNNTest(tf.test.TestCase, parameterized.TestCase):

  def testVanillaMPNNGraphUpdate(self):
    input_graph = _make_test_graph_abc()
    units = 2
    layer = vanilla_mpnn.VanillaMPNNGraphUpdate(
        units=units,
        message_dim=1,
        receiver_tag=tfgnn.TARGET,
        node_set_names=["b"],
        edge_feature="fab",
        kernel_initializer="ones")
    graph = layer(input_graph)
    # Nodes "a" and "c" are unchanged.
    self.assertAllEqual([[1.]], graph.node_sets["a"][tfgnn.HIDDEN_STATE])
    self.assertAllEqual([[8.]], graph.node_sets["c"][tfgnn.HIDDEN_STATE])
    # Node "b" receives message 1+2+16 = 19 and combines it with old state 2.
    self.assertAllEqual([[21.]*units], graph.node_sets["b"][tfgnn.HIDDEN_STATE])

  @parameterized.named_parameters(("WithoutLayerNorm", False),
                                  ("WithLayerNorm", True))
  def testVanillaMPNNGraphUpdateWithCustomKernelInitializer(
      self, use_layer_normalization):
    input_graph = _make_test_graph_abc()
    # To ensure that the updated node-state has non-identical entries
    kernel_initializer = tf.constant_initializer([[1., 1.],
                                                  [2., 0.],
                                                  [1., 1.]])
    units = 2
    layer = vanilla_mpnn.VanillaMPNNGraphUpdate(
        units=units,
        message_dim=2,
        receiver_tag=tfgnn.TARGET,
        node_set_names=["b"],
        edge_feature="fab",
        kernel_initializer=kernel_initializer,
        use_layer_normalization=use_layer_normalization)
    graph = layer(input_graph)

    # Nodes "a" and "c" are unchanged.
    self.assertAllEqual([[1.]], graph.node_sets["a"][tfgnn.HIDDEN_STATE])
    self.assertAllEqual([[8.]], graph.node_sets["c"][tfgnn.HIDDEN_STATE])
    # Node "b" receives message [b, a, fab] * Kw = [20., 18.]
    # Message is combined with "b" old state [b, 20., 18.] * Kw = [60, 20]
    # If use_layer_normalization flag is set, layer normalization is applied on
    # the updated node state
    if use_layer_normalization:
      want = [[1., -1.]]
    else:
      want = [[60., 20.]]
    self.assertAllClose(want, graph.node_sets["b"][tfgnn.HIDDEN_STATE])


def _make_test_graph_abc():
  return tfgnn.GraphTensor.from_pieces(
      node_sets={
          "a": tfgnn.NodeSet.from_fields(
              sizes=tf.constant([1]),
              features={tfgnn.HIDDEN_STATE: tf.constant([[1.]])}),
          "b": tfgnn.NodeSet.from_fields(
              sizes=tf.constant([1]),
              features={tfgnn.HIDDEN_STATE: tf.constant([[2.]])}),
          "c": tfgnn.NodeSet.from_fields(
              sizes=tf.constant([1]),
              features={tfgnn.HIDDEN_STATE: tf.constant([[8.]])})},
      edge_sets={
          "a->b": tfgnn.EdgeSet.from_fields(
              sizes=tf.constant([1]),
              adjacency=tfgnn.Adjacency.from_indices(
                  ("a", tf.constant([0])),
                  ("b", tf.constant([0]))),
              features={"fab": tf.constant([[16.]])}),
          "c->c": tfgnn.EdgeSet.from_fields(
              sizes=tf.constant([1]),
              adjacency=tfgnn.Adjacency.from_indices(
                  ("c", tf.constant([0])),
                  ("c", tf.constant([0]))))})


if __name__ == "__main__":
  tf.test.main()
