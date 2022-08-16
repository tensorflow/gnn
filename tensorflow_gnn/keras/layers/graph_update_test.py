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

import enum
import os

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.keras.layers import convolutions
from tensorflow_gnn.keras.layers import graph_ops
from tensorflow_gnn.keras.layers import graph_update
from tensorflow_gnn.keras.layers import next_state as next_state_lib


class ReloadModel(int, enum.Enum):
  """Controls how to reload a model for further testing after saving."""
  SKIP = 0
  SAVED_MODEL = 1
  KERAS = 2


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
      conv_sum_sources = convolutions.SimpleConv(
          message_fn=tf.keras.layers.Dense(
              1, use_bias=False, kernel_initializer="ones"),
          receiver_feature=None,
          reduce_type="sum")
      conv_sum_endpoints = convolutions.SimpleConv(
          message_fn=tf.keras.layers.Dense(
              1, use_bias=False, kernel_initializer="ones"),
          # receiver_feature=const.HIDDEN_STATE,  # The default.
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
      return graph.node_sets[node_set_name][const.HIDDEN_STATE]
    def edge_state(edge_set_name):
      return graph.edge_sets[edge_set_name][const.HIDDEN_STATE]

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
      ("InstantInit", False, ReloadModel.SKIP),
      ("InstantInitRestored", False, ReloadModel.SAVED_MODEL),
      ("InstantInitRestoredKeras", False, ReloadModel.KERAS),
      ("DeferredInit", True, ReloadModel.SKIP),
      ("DeferredInitRestored", True, ReloadModel.SAVED_MODEL),
      ("DeferredInitRestoredKeras", True, ReloadModel.KERAS))
  def testEndToEndInModelWithEdgeStates(self, use_deferred_init, reload_model):
    input_graph = _make_test_graph_with_singleton_node_sets(
        [("a", [1.]), ("b", [2.]), ("c", [4.])],
        [("a", "c", [100.]), ("b", "c", [100.]),
         ("c", "a", [100.]), ("b", "a", [100.])],
        context=[8.])

    def get_kwargs(graph_tensor_spec):
      self.assertEqual(graph_tensor_spec, input_graph.spec)
      edge_sum_sources = graph_update.EdgeSetUpdate(
          next_state_lib.NextStateFromConcat(
              tf.keras.layers.Dense(
                  1, use_bias=False, kernel_initializer="ones")),
          node_input_tags=[const.SOURCE],
          edge_input_feature=())
      edge_sum_sources_context = graph_update.EdgeSetUpdate(
          next_state_lib.NextStateFromConcat(
              tf.keras.layers.Dense(
                  1, use_bias=False, kernel_initializer="ones")),
          node_input_tags=[const.SOURCE],
          edge_input_feature=(),
          context_input_feature=const.HIDDEN_STATE)
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
          "b->a": edge_sum_sources_context}
      node_sets = {
          "c": graph_update.NodeSetUpdate(
              {"b->c": graph_ops.Pool(const.TARGET, "sum"),
               "a->c": graph_ops.Pool(const.TARGET, "sum")},
              state_add_edges,
              context_input_feature=const.HIDDEN_STATE),
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

    # Build a Model around the Layer, possibly saved and restored.
    inputs = tf.keras.layers.Input(type_spec=input_graph.spec)
    outputs = update(inputs)
    model = tf.keras.Model(inputs, outputs)
    _ = model(input_graph)  # Trigger building.
    if reload_model:
      export_dir = os.path.join(self.get_temp_dir(), "edge-update-model")
      model.save(export_dir, include_optimizer=False)
      if reload_model == ReloadModel.KERAS:
        model = tf.keras.models.load_model(export_dir)
        # Check that from_config() worked, no fallback to a function trace, see
        # https://www.tensorflow.org/guide/keras/save_and_serialize#how_savedmodel_handles_custom_objects
        self.assertIsInstance(model.get_layer(index=1),
                              graph_update.GraphUpdate)
      else:
        model = tf.saved_model.load(export_dir)

    graph = model(input_graph)

    def edge_state(edge_set_name):
      return graph.edge_sets[edge_set_name][const.HIDDEN_STATE]
    def node_state(node_set_name):
      return graph.node_sets[node_set_name][const.HIDDEN_STATE]

    # Edge sets are updated first.
    self.assertAllEqual([[1.]], edge_state("a->c"))
    self.assertAllEqual([[6.]], edge_state("b->c"))
    self.assertAllEqual([[4.]], edge_state("c->a"))
    self.assertAllEqual([[10.]], edge_state("b->a"))
    # Node sets are updated in parallel after edge sets.
    # a has 1, gets 1 by skipconn, 10 from b->a and 4 from c->a, totalling 16.
    self.assertAllEqual([[16.]], node_state("a"))
    # b has 2 and stays unchanged.
    self.assertAllEqual([[2.]], node_state("b"))
    # c has 4, gets 2+4 from b->c, 1 from a->c and 8 from context, totalling 19.
    self.assertAllEqual([[19.]], node_state("c"))
    # Context is updated last, gets overwritten with sum of nodes.
    self.assertAllEqual([[16. + 2.+ 19.]],
                        graph.context[const.HIDDEN_STATE])


def _make_test_graph_with_singleton_node_sets(nodes, edges, context=None):
  """Returns graph with singleton node sets and edge sets of given values."""
  # pylint: disable=g-complex-comprehension
  return gt.GraphTensor.from_pieces(
      context=gt.Context.from_fields(
          features=None if context is None else {
              const.HIDDEN_STATE: tf.constant([context])}),
      node_sets={
          name: gt.NodeSet.from_fields(
              sizes=tf.constant([1]),
              features={const.HIDDEN_STATE: tf.constant([value])})
          for name, value in nodes},
      edge_sets={
          f"{src}->{dst}": gt.EdgeSet.from_fields(
              sizes=tf.constant([1]),
              adjacency=adj.Adjacency.from_indices(
                  (src, tf.constant([0])),
                  (dst, tf.constant([0]))),
              features={const.HIDDEN_STATE: tf.constant([value])})
          for src, dst, value in edges})


if __name__ == "__main__":
  tf.test.main()
