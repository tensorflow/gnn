"""Tests for graph_update Keras layers."""

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph.keras.layers import convolutions
from tensorflow_gnn.graph.keras.layers import graph_ops
from tensorflow_gnn.graph.keras.layers import graph_update
from tensorflow_gnn.graph.keras.layers import next_state as next_state_lib


class GraphUpdateTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("InstantInit", False),
      ("DeferredInit", True))
  def testEndToEndWithConvolutions(self, use_deferred_init):
    input_graph = _make_test_graph_with_singleton_node_sets(
        [("a", [1.]), ("b", [2.]), ("c", [4.])],
        [("a", "c", [100.]), ("b", "c", [100.]),
         ("c", "a", [100.]), ("b", "a", [100.])])

    def get_kwargs(graph_tensor_spec):
      self.assertEqual(graph_tensor_spec, input_graph.spec)
      conv_sum_sources = convolutions.SimpleConvolution(
          node_input_tags=[const.SOURCE],
          message_fn=tf.keras.layers.Dense(
              1, use_bias=False, kernel_initializer="ones"),
          reduce_type="sum")
      conv_sum_endpoints = convolutions.SimpleConvolution(
          # node_input_tags=[const.SOURCE, const.TARGET],  # The default.
          message_fn=tf.keras.layers.Dense(
              1, use_bias=False, kernel_initializer="ones"),
          reduce_type="sum")
      state_add_edges = next_state_lib.NextStateFromConcat(
          tf.keras.layers.Dense(
              1, use_bias=False, kernel_initializer="ones"))
      double_state_add_edges = next_state_lib.ResidualNextState(
          tf.keras.layers.Dense(
              1, use_bias=False, kernel_initializer="ones"))
      node_sets = {
          "c": graph_update.NodeSetUpdate(
              {"b->c": conv_sum_endpoints,
               "a->c": conv_sum_sources},
              state_add_edges),
          "a": graph_update.NodeSetUpdate(
              {"c->a": conv_sum_sources,
               "b->a": conv_sum_sources},
              double_state_add_edges)}
      return dict(node_sets=node_sets)

    if use_deferred_init:
      update = graph_update.GraphUpdate(deferred_init_callback=get_kwargs)
    else:
      update = graph_update.GraphUpdate(**get_kwargs(input_graph.spec))
    graph = update(input_graph)

    def node_state(node_set_name):
      return graph.node_sets[node_set_name][const.DEFAULT_STATE_NAME]
    def edge_state(edge_set_name):
      return graph.edge_sets[edge_set_name][const.DEFAULT_STATE_NAME]

    # Node sets are updated in parallel.
    # a has 1, gets 1 by skipconn, 2 from b->a and 4 from c->a, totalling 8.
    self.assertAllEqual([[8.]], node_state("a"))
    # b has 2 and stays unchanged.
    self.assertAllEqual([[2.]], node_state("b"))
    # c has 4, gets 2+4 from b->c and 1 from a->c, totalling 11.
    self.assertAllEqual([[11.]], node_state("c"))
    # Edge sets are unchanged.
    self.assertAllEqual([[100.]], edge_state("a->c"))
    self.assertAllEqual([[100.]], edge_state("b->c"))
    self.assertAllEqual([[100.]], edge_state("c->a"))
    self.assertAllEqual([[100.]], edge_state("b->a"))

  @parameterized.named_parameters(
      ("InstantInit", False),
      ("DeferredInit", True))
  def testEndToEndWithEdgeStates(self, use_deferred_init):
    input_graph = _make_test_graph_with_singleton_node_sets(
        [("a", [1.]), ("b", [2.]), ("c", [4.])],
        [("a", "c", [100.]), ("b", "c", [100.]),
         ("c", "a", [100.]), ("b", "a", [100.])])

    def get_kwargs(graph_tensor_spec):
      self.assertEqual(graph_tensor_spec, input_graph.spec)
      edge_sum_sources = graph_update.EdgeSetUpdate(
          next_state_lib.NextStateFromConcat(
              tf.keras.layers.Dense(
                  1, use_bias=False, kernel_initializer="ones")),
          node_input_tags=[const.SOURCE],
          edge_input_feature=())
      edge_sum_endpoints = graph_update.EdgeSetUpdate(
          next_state_lib.NextStateFromConcat(
              tf.keras.layers.Dense(
                  1, use_bias=False,
                  kernel_initializer="ones")),
          # node_input_tags=[const.SOURCE, const.TARGET] is the default.
          edge_input_feature=())
      state_add_edges = next_state_lib.NextStateFromConcat(
          tf.keras.layers.Dense(
              1, use_bias=False, kernel_initializer="ones"))
      double_state_add_edges = next_state_lib.ResidualNextState(
          tf.keras.layers.Dense(
              1, use_bias=False, kernel_initializer="ones"))
      ctx_add_node_sets = next_state_lib.NextStateFromConcat(
          tf.keras.layers.Dense(
              1, use_bias=False, kernel_initializer="ones"))
      edge_sets = {
          "b->c": edge_sum_endpoints,
          "a->c": edge_sum_sources,
          "c->a": edge_sum_sources,
          "b->a": edge_sum_sources}
      node_sets = {
          "c": graph_update.NodeSetUpdate(
              {"b->c": graph_ops.Pool(const.TARGET, "sum"),
               "a->c": graph_ops.Pool(const.TARGET, "sum")},
              state_add_edges),
          "a": graph_update.NodeSetUpdate(
              {"c->a": graph_ops.Pool(const.TARGET, "sum"),
               "b->a": graph_ops.Pool(const.TARGET, "sum")},
              double_state_add_edges)}
      context = graph_update.ContextUpdate(
          {node_set_name: graph_ops.Pool(const.CONTEXT, "sum")
           for node_set_name in ["a", "b", "c"]},
          ctx_add_node_sets, context_input_feature=())
      return dict(edge_sets=edge_sets, node_sets=node_sets, context=context)

    if use_deferred_init:
      update = graph_update.GraphUpdate(deferred_init_callback=get_kwargs)
    else:
      update = graph_update.GraphUpdate(**get_kwargs(input_graph.spec))
    graph = update(input_graph)

    def edge_state(edge_set_name):
      return graph.edge_sets[edge_set_name][const.DEFAULT_STATE_NAME]
    def node_state(node_set_name):
      return graph.node_sets[node_set_name][const.DEFAULT_STATE_NAME]

    # Edge sets are updated first.
    self.assertAllEqual([[1.]], edge_state("a->c"))
    self.assertAllEqual([[6.]], edge_state("b->c"))
    self.assertAllEqual([[4.]], edge_state("c->a"))
    self.assertAllEqual([[2.]], edge_state("b->a"))
    # Node sets are updated in parallel after edge sets.
    # a has 1, gets 1 by skipconn, 2 from b->a and 4 from c->a, totalling 8.
    self.assertAllEqual([[8.]], node_state("a"))
    # b has 2 and stays unchanged.
    self.assertAllEqual([[2.]], node_state("b"))
    # c has 4, gets 2+4 from b->c and 1 from a->c, totalling 11.
    self.assertAllEqual([[11.]], node_state("c"))


def _make_test_graph_with_singleton_node_sets(nodes, edges):
  """Returns graph with singleton node sets and edge sets of given values."""
  # pylint: disable=g-complex-comprehension
  return gt.GraphTensor.from_pieces(
      node_sets={
          name: gt.NodeSet.from_fields(
              sizes=tf.constant([1]),
              features={const.DEFAULT_STATE_NAME: tf.constant([value])})
          for name, value in nodes},
      edge_sets={
          f"{src}->{dst}": gt.EdgeSet.from_fields(
              sizes=tf.constant([1]),
              adjacency=adj.Adjacency.from_indices(
                  (src, tf.constant([0])),
                  (dst, tf.constant([0]))),
              features={const.DEFAULT_STATE_NAME: tf.constant([value])})
          for src, dst, value in edges})


if __name__ == "__main__":
  tf.test.main()
