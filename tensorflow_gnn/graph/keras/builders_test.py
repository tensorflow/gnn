"""Tests for graph_update Keras layers."""

import os
import tensorflow as tf

from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph.keras import builders
from tensorflow_gnn.graph.keras.layers import convolutions
from tensorflow_gnn.graph.keras.layers import next_state as next_state_lib

IdentityLayer = tf.keras.layers.Layer


class ConvGNNBuilderTest(tf.test.TestCase):  # , parameterized.TestCase):

  def testHomogeneousCase(self):
    input_graph = _make_test_graph_with_singleton_node_sets(
        [("node", [1.])], [("node", "node", [100.])])
    gnn_builder = builders.ConvGNNBuilder(
        lambda _: convolutions.SimpleConvolution(IdentityLayer()),
        lambda _: next_state_lib.NextStateFromConcat(IdentityLayer()))
    graph = gnn_builder.Convolve()(input_graph)
    self.assertAllEqual([[1., 1., 1.]],
                        graph.node_sets["node"][const.DEFAULT_STATE_NAME])
    self.assertAllEqual([[100.]],
                        graph.edge_sets["node->node"][const.DEFAULT_STATE_NAME])

  def testModelSaving(self):

    def sum_sources_conv(_):
      return convolutions.SimpleConvolution(
          node_input_tags=[const.SOURCE],
          edge_input_feature=const.DEFAULT_STATE_NAME,
          message_fn=tf.keras.layers.Dense(
              1,
              use_bias=False,
              kernel_initializer=tf.keras.initializers.Ones()),
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
      return graph.node_sets[node_set_name][const.DEFAULT_STATE_NAME]

    self.assertAllEqual([[2. + 1. + 10.]], node_state("a"))
    self.assertAllEqual([[13. + 2. + 100.]], node_state("b"))

  def testParallelUpdates(self):
    input_graph = _make_test_graph_with_singleton_node_sets(
        [("a", [1.]), ("b", [2.]), ("c", [4.])], [("a", "c", [100.]),
                                                  ("b", "c", [100.]),
                                                  ("c", "a", [100.]),
                                                  ("b", "a", [100.])])
    conv_sum_sources = convolutions.SimpleConvolution(
        node_input_tags=[const.SOURCE],
        message_fn=tf.keras.layers.Dense(
            1, use_bias=False, kernel_initializer=tf.keras.initializers.Ones()),
        reduce_type="sum")
    conv_sum_endpoints = convolutions.SimpleConvolution(
        # node_input_tags=[const.SOURCE, const.TARGET],  # The default.
        message_fn=tf.keras.layers.Dense(
            1, use_bias=False, kernel_initializer=tf.keras.initializers.Ones()),
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
      return graph.node_sets[node_set_name][const.DEFAULT_STATE_NAME]

    def edge_state(edge_set_name):
      return graph.edge_sets[edge_set_name][const.DEFAULT_STATE_NAME]

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


def _make_test_graph_with_singleton_node_sets(nodes, edges):
  """Returns graph with singleton node sets and edge sets of given values."""
  # pylint: disable=g-complex-comprehension
  return gt.GraphTensor.from_pieces(
      node_sets={
          name: gt.NodeSet.from_fields(
              sizes=tf.constant([1]),
              features={const.DEFAULT_STATE_NAME: tf.constant([value])})
          for name, value in nodes
      },
      edge_sets={
          f"{src}->{dst}": gt.EdgeSet.from_fields(
              sizes=tf.constant([1]),
              adjacency=adj.Adjacency.from_indices((src, tf.constant([0])),
                                                   (dst, tf.constant([0]))),
              features={const.DEFAULT_STATE_NAME: tf.constant([value])})
          for src, dst, value in edges
      })


if __name__ == "__main__":
  tf.test.main()
