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
"""Tests for graph_update Keras layers."""

import collections
import os

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.keras import builders
from tensorflow_gnn.keras.layers import convolutions
from tensorflow_gnn.keras.layers import graph_update as graph_update_lib
from tensorflow_gnn.keras.layers import next_state as next_state_lib

IdentityLayer = tf.keras.layers.Layer


class ConvGNNBuilderTest(tf.test.TestCase, parameterized.TestCase):

  def testHomogeneousCase(self):
    input_graph = _make_test_graph_with_singleton_node_sets(
        [("node", [1.])], [("node", "node", [100.])])
    gnn_builder = builders.ConvGNNBuilder(
        lambda _: convolutions.SimpleConv(IdentityLayer()),
        lambda _: next_state_lib.NextStateFromConcat(IdentityLayer()))
    graph = gnn_builder.Convolve()(input_graph)
    self.assertAllEqual([[1., 1., 1.]],
                        graph.node_sets["node"][const.HIDDEN_STATE])
    self.assertAllEqual([[100.]],
                        graph.edge_sets["node->node"][const.HIDDEN_STATE])

  def testReceivingRequired(self):
    input_graph = _make_test_graph_with_singleton_node_sets(
        [("node", [1.]), ("isolnode", [1.])], [("node", "node", [100.])])
    gnn_builder = builders.ConvGNNBuilder(
        lambda _: convolutions.SimpleConv(IdentityLayer()),
        lambda _: next_state_lib.NextStateFromConcat(IdentityLayer()))
    _ = gnn_builder.Convolve(["node"])(input_graph)
    with self.assertRaisesRegex(ValueError,
                                r"not .* from any edge set.*isolnode"):
      _ = gnn_builder.Convolve(["node", "isolnode"])(input_graph)

  def testCallbacks(self):
    conv_result = None
    def convolutions_factory(edge_set_name, receiver_tag):
      self.assertEqual(edge_set_name, "node->node")
      nonlocal conv_result
      conv_result = convolutions.SimpleConv(IdentityLayer(),
                                            receiver_tag=receiver_tag)
      return conv_result

    next_state_result = None
    def nodes_next_state_factory(node_set_name):
      self.assertEqual(node_set_name, "node")
      nonlocal next_state_result
      next_state_result = next_state_lib.NextStateFromConcat(IdentityLayer())
      return next_state_result

    node_set_update_result = None
    def node_set_update_factory(node_set_name, edge_set_inputs, next_state):
      self.assertEqual(node_set_name, "node")
      self.assertCountEqual(edge_set_inputs.keys(), ["node->node"])
      self.assertIs(edge_set_inputs["node->node"], conv_result)
      self.assertIs(next_state, next_state_result)
      nonlocal node_set_update_result
      node_set_update_result = graph_update_lib.NodeSetUpdate(
          edge_set_inputs, next_state)
      return node_set_update_result

    graph_update_result = None
    def graph_update_factory(deferred_init_callback, name):
      # The effects of deferred_init_callback() are tested later on.
      self.assertEqual(name, "my_update")
      nonlocal graph_update_result
      graph_update_result = graph_update_lib.GraphUpdate(
          deferred_init_callback=deferred_init_callback, name=name)
      return graph_update_result

    input_graph = _make_test_graph_with_singleton_node_sets(
        [("node", [1.])], [("node", "node", [100.])])
    gnn_builder = builders.ConvGNNBuilder(
        convolutions_factory,
        nodes_next_state_factory,
        node_set_update_factory=node_set_update_factory,
        graph_update_factory=graph_update_factory,
        receiver_tag=const.TARGET)
    graph_update = gnn_builder.Convolve(name="my_update")
    graph = graph_update(input_graph)
    self.assertIs(graph_update, graph_update_result)
    self.assertIs(graph_update._node_set_updates["node"],
                  node_set_update_result)
    self.assertAllEqual([[1., 1., 1.]],
                        graph.node_sets["node"][const.HIDDEN_STATE])
    self.assertAllEqual([[100.]],
                        graph.edge_sets["node->node"][const.HIDDEN_STATE])

  @parameterized.named_parameters(
      ("DuplicateNodeSets", False, ["node", "node", "node"]),
      ("ReadoutIgnored", True, None))
  def testCallCounts(self, add_readout, node_sets):
    call_counts = collections.defaultdict(lambda: 0)
    def convolutions_factory(edge_set_name, receiver_tag):
      call_counts[edge_set_name] += 1
      return convolutions.SimpleConv(IdentityLayer(), receiver_tag=receiver_tag)
    def nodes_next_state_factory(node_set_name):
      call_counts[node_set_name] += 1
      return next_state_lib.NextStateFromConcat(IdentityLayer())
    input_graph = _make_test_graph_with_singleton_node_sets(
        [("node", [1.])], [("node", "node", [100.])], add_readout=add_readout)
    gnn_builder = builders.ConvGNNBuilder(convolutions_factory,
                                          nodes_next_state_factory,
                                          receiver_tag=const.TARGET)
    graph = gnn_builder.Convolve(node_sets)(input_graph)
    self.assertDictEqual({"node": 1, "node->node": 1}, call_counts)
    self.assertAllEqual([[1., 1., 1.]],
                        graph.node_sets["node"][const.HIDDEN_STATE])
    self.assertAllEqual([[100.]],
                        graph.edge_sets["node->node"][const.HIDDEN_STATE])

  @parameterized.named_parameters(
      ("Default", dict(), [1.0], [2.0, 2.0, 2.0]),  # Behaves like Target.
      ("Target", dict(receiver_tag=const.TARGET), [1.0], [2.0, 2.0, 2.0]),
      ("Source", dict(receiver_tag=const.SOURCE), [2.0, 2.0, 2.0], [1.0]),
  )
  def testReceiverTag(self, receiver_tag_kwarg, expected_a, expected_b):
    def make_doubling_layer():
      return tf.keras.layers.Dense(
          3, use_bias=False,
          kernel_initializer=tf.keras.initializers.Identity(gain=2.0))
    input_graph = _make_test_graph_with_singleton_node_sets(
        [("a", [1.]), ("b", [1.])], [("a", "b", [100.])])
    if receiver_tag_kwarg:
      # pylint: disable=g-long-lambda
      gnn_builder = builders.ConvGNNBuilder(
          lambda _, *, receiver_tag: convolutions.SimpleConv(
              IdentityLayer(), receiver_tag=receiver_tag),
          lambda _: next_state_lib.NextStateFromConcat(make_doubling_layer()),
          **receiver_tag_kwarg)
    else:
      gnn_builder = builders.ConvGNNBuilder(
          lambda _: convolutions.SimpleConv(IdentityLayer()),
          lambda _: next_state_lib.NextStateFromConcat(make_doubling_layer()))
    graph = gnn_builder.Convolve()(input_graph)
    self.assertAllEqual([expected_a],
                        graph.node_sets["a"][const.HIDDEN_STATE])
    self.assertAllEqual([expected_b],
                        graph.node_sets["b"][const.HIDDEN_STATE])
    self.assertAllEqual([[100.]],  # Unchanged.
                        graph.edge_sets["a->b"][const.HIDDEN_STATE])

  def testModelSaving(self):

    def sum_sources_conv(_):
      return convolutions.SimpleConv(
          message_fn=tf.keras.layers.Dense(
              1,
              use_bias=False,
              kernel_initializer=tf.keras.initializers.Ones()),
          sender_edge_feature=const.HIDDEN_STATE,
          receiver_feature=None,
          reduce_type="sum")

    def add_edges_state(_):
      return next_state_lib.NextStateFromConcat(
          tf.keras.layers.Dense(
              1,
              use_bias=False,
              kernel_initializer=tf.keras.initializers.Ones()))

    gnn_builder = builders.ConvGNNBuilder(sum_sources_conv, add_edges_state)
    gnn_layer = tf.keras.models.Sequential([
        gnn_builder.Convolve({"a"}),
        gnn_builder.Convolve({"b"}),
    ])
    input_graph = _make_test_graph_with_singleton_node_sets([("a", [1.]),
                                                             ("b", [2.])],
                                                            [("a", "b", [100.]),
                                                             ("b", "a", [10.])])

    inputs = tf.keras.layers.Input(type_spec=input_graph.spec)
    outputs = gnn_layer(inputs)
    model = tf.keras.Model(inputs, outputs)

    export_dir = os.path.join(self.get_temp_dir(), "stdlayer-tf")
    tf.saved_model.save(model, export_dir)
    restored_model = tf.saved_model.load(export_dir)
    graph = restored_model(input_graph)

    def node_state(node_set_name):
      return graph.node_sets[node_set_name][const.HIDDEN_STATE]

    self.assertAllEqual([[2. + 1. + 10.]], node_state("a"))
    self.assertAllEqual([[13. + 2. + 100.]], node_state("b"))

  def testParallelUpdates(self):
    input_graph = _make_test_graph_with_singleton_node_sets(
        [("a", [1.]), ("b", [2.]), ("c", [4.])], [("a", "c", [100.]),
                                                  ("b", "c", [100.]),
                                                  ("c", "a", [100.]),
                                                  ("b", "a", [100.])])
    conv_sum_sources = convolutions.SimpleConv(
        message_fn=tf.keras.layers.Dense(
            1, use_bias=False, kernel_initializer=tf.keras.initializers.Ones()),
        receiver_feature=None,
        reduce_type="sum")
    conv_sum_endpoints = convolutions.SimpleConv(
        message_fn=tf.keras.layers.Dense(
            1, use_bias=False, kernel_initializer=tf.keras.initializers.Ones()),
        # receiver_feature=const.HIDDEN_STATE,  # The default.
        reduce_type="sum")
    state_add_edges = next_state_lib.NextStateFromConcat(
        tf.keras.layers.Dense(
            1, use_bias=False, kernel_initializer=tf.keras.initializers.Ones()))
    double_state_add_edges = next_state_lib.ResidualNextState(
        tf.keras.layers.Dense(
            1, use_bias=False, kernel_initializer=tf.keras.initializers.Ones()))

    def convolutions_factory(edge_set: const.EdgeSetName):
      return conv_sum_endpoints if edge_set == "b->c" else conv_sum_sources

    def next_state_factory(node_set: const.NodeSetName):
      return state_add_edges if node_set == "c" else double_state_add_edges

    gnn_builder = builders.ConvGNNBuilder(convolutions_factory,
                                          next_state_factory)
    model = tf.keras.models.Sequential([
        gnn_builder.Convolve({"a", "c"}),
        gnn_builder.Convolve({"c"}),
    ])
    graph = model(input_graph)

    def node_state(node_set_name):
      return graph.node_sets[node_set_name][const.HIDDEN_STATE]

    def edge_state(edge_set_name):
      return graph.edge_sets[edge_set_name][const.HIDDEN_STATE]

    # Node sets are updated in parallel.
    # 1st convolution:
    #   a has 1, gets 1 by skipconn, 2 from b->a and 4 from c->a, totalling 8.
    # 2nd convolution:
    #   a has 8 and stays unchanged.
    self.assertAllEqual([[8.]], node_state("a"))
    # 1st, 2nd convolutions: b has 2 and stays unchanged.
    self.assertAllEqual([[2.]], node_state("b"))
    # 1st convolution:
    #   c has 4, gets 2+4 from b->c and 1 from a->c, totalling 11.
    # 2nd convolution:
    #   c has 11, gets 2+11 from b->c and 8 from a->c, totalling 32.
    self.assertAllEqual([[32.]], node_state("c"))
    # Edge sets are unchanged.
    self.assertAllEqual([[100.]], edge_state("a->c"))
    self.assertAllEqual([[100.]], edge_state("b->c"))
    self.assertAllEqual([[100.]], edge_state("c->a"))
    self.assertAllEqual([[100.]], edge_state("b->a"))

  def testAuxNodeSetRequested(self):
    def convolutions_factory(edge_set_name, receiver_tag):
      del edge_set_name  # Unused.
      return convolutions.SimpleConv(IdentityLayer(), receiver_tag=receiver_tag)
    def nodes_next_state_factory(node_set_name):
      del node_set_name  # Unused.
      return next_state_lib.NextStateFromConcat(IdentityLayer())
    input_graph = _make_test_graph_with_singleton_node_sets(
        [("node", [1.]), ("_extra", [9.])], [("node", "node", [100.])])
    gnn_builder = builders.ConvGNNBuilder(convolutions_factory,
                                          nodes_next_state_factory,
                                          receiver_tag=const.TARGET)
    _ = gnn_builder.Convolve(["node"])(input_graph)
    with self.assertRaises(ValueError):
      _ = gnn_builder.Convolve(["node", "_extra"])(input_graph)

  def testAuxNodeSetDiscoveredFromEdgeSet(self):
    def convolutions_factory(edge_set_name, receiver_tag):
      del edge_set_name  # Unused.
      return convolutions.SimpleConv(IdentityLayer(), receiver_tag=receiver_tag)
    def nodes_next_state_factory(node_set_name):
      del node_set_name  # Unused.
      return next_state_lib.NextStateFromConcat(IdentityLayer())
    input_graph = _make_test_graph_with_singleton_node_sets(
        [("node", [1.]), ("_extra", [9.])],
        [("node", "_extra", [100.])])
    self.assertCountEqual(
        ["node->_extra"],  # This edge set name looks non-auxiliary.
        input_graph.edge_sets)
    gnn_builder = builders.ConvGNNBuilder(convolutions_factory,
                                          nodes_next_state_factory,
                                          receiver_tag=const.TARGET)
    with self.assertRaisesRegex(
        ValueError,
        r"Node set '_extra' is auxiliary but the incident "
        r"edge set 'node->_extra' \(at tag 1\) is not\."):
      _ = gnn_builder.Convolve()(input_graph)


def _make_test_graph_with_singleton_node_sets(nodes, edges, add_readout=False):
  """Returns graph with singleton node sets and edge sets of given values."""
  # pylint: disable=g-complex-comprehension
  node_sets = {
      name: gt.NodeSet.from_fields(
          sizes=tf.constant([1]),
          features={const.HIDDEN_STATE: tf.constant([value])})
      for name, value in nodes
  }
  edge_sets = {
      f"{src}->{dst}": gt.EdgeSet.from_fields(
          sizes=tf.constant([1]),
          adjacency=adj.Adjacency.from_indices((src, tf.constant([0])),
                                               (dst, tf.constant([0]))),
          features={const.HIDDEN_STATE: tf.constant([value])})
      for src, dst, value in edges
  }

  if add_readout:
    source_name = next(iter(node_sets.keys()))
    node_sets["_readout"] = gt.NodeSet.from_fields(sizes=tf.constant([1]))
    edge_sets["_readout/seed"] = gt.EdgeSet.from_fields(
        sizes=tf.constant([1]),
        adjacency=adj.Adjacency.from_indices((source_name, tf.constant([0])),
                                             ("_readout", tf.constant([0]))))

  return gt.GraphTensor.from_pieces(node_sets=node_sets, edge_sets=edge_sets)


if __name__ == "__main__":
  tf.test.main()
