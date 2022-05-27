"""Tests for gnn_template.modeling."""

import functools

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models.gnn_template import modeling


class ModelingTemplateTest(tf.test.TestCase, parameterized.TestCase):

  def testVanillaMPNNModel(self):
    input_graph = tfgnn.GraphTensor.from_pieces(
        node_sets={"nodes": tfgnn.NodeSet.from_fields(sizes=[2])})
    def init_states_fn(graph):
      return graph.replace_features(node_sets={
          "nodes": {"state": tf.constant([[1.], [2.]])}})
    def pass_messages_fn(graph):
      return graph.replace_features(node_sets={
          "nodes": {"state": 2. * graph.node_sets["nodes"]["state"]}})
    model = modeling.vanilla_mpnn_model(
        input_graph.spec,
        init_states_fn=init_states_fn, pass_messages_fn=pass_messages_fn)
    graph = model(input_graph)
    self.assertAllEqual(
        graph.node_sets["nodes"]["state"], [[2.], [4.]])

  def testInitStates(self):
    input_graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            "dense": tfgnn.NodeSet.from_fields(
                sizes=[2],
                features={tfgnn.HIDDEN_STATE: tf.constant([[1.], [2.]])}),
            "ragged": tfgnn.NodeSet.from_fields(
                sizes=[2],
                features={tfgnn.HIDDEN_STATE: tf.ragged.constant(
                    [[1], [2, 3]])}),
            "other": tfgnn.NodeSet.from_fields(
                sizes=[2],
                features={tfgnn.HIDDEN_STATE: tf.constant([[3.], [1.]])}),
        })
    init_states_fn = functools.partial(
        modeling.init_states_by_embed_and_transform,
        node_transformations={"dense": 2},
        node_embeddings={"ragged": (4, 2)},
        l2_regularization=0.01)
    model = modeling.vanilla_mpnn_model(
        input_graph.spec,
        init_states_fn=init_states_fn, pass_messages_fn=lambda x: x)
    _ = model(input_graph)  # Build weights.
    weights = {v.name: v for v in model.trainable_weights}
    weights["embedding/embeddings:0"].assign(
        [[0., 1.], [10., 11.], [20., 21.,], [30., 31.]])
    weights["dense/kernel:0"].assign([[1., 2.]])
    weights["dense/bias:0"].assign([1., 1.])

    graph = model(input_graph)
    self.assertAllEqual(
        graph.node_sets["dense"][tfgnn.HIDDEN_STATE],
        [[2., 3.,], [3., 5.]])
    self.assertAllEqual(
        graph.node_sets["ragged"][tfgnn.HIDDEN_STATE],
        [[10., 11.], [(20.+30.)/2., (21.+31.)/2.]])
    self.assertAllEqual(
        graph.node_sets["other"][tfgnn.HIDDEN_STATE],
        [[3.,], [1.]])
    self.assertAllClose(
        tf.add_n(model.losses),
        # l2_regularization factor applied to dense layer weights.
        0.01 * sum(x**2 for x in [1., 2., 1., 1.]))

  @parameterized.named_parameters(
      ("RawGnnOps", "raw_gnn_ops", False),
      ("RawGnnOpsReverse", "raw_gnn_ops", True),
      ("EdgeNodeUpdates", "edge_node_updates", False),
      ("NodeUpdates", "node_updates", False),
      ("GnnBuilder", "gnn_builder", False),
      ("GnnBuilderReverse", "gnn_builder", True))
  def testPassSimpleMessages(self, modeling_flavor, reverse):
    adjacency_args = [("a", tf.constant([0, 1])),
                      ("b", tf.constant([0, 0]))]
    if reverse:
      adjacency_args.reverse()
      receiver_tag = tfgnn.SOURCE
    else:
      receiver_tag = tfgnn.TARGET
    # Input graph: a0 -> b0 <- a1 (or reversed).
    input_graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            "a": tfgnn.NodeSet.from_fields(
                sizes=[2],
                features={tfgnn.HIDDEN_STATE: tf.constant([[1.], [2.]])}),
            "b": tfgnn.NodeSet.from_fields(
                sizes=[1],
                features={tfgnn.HIDDEN_STATE: tf.constant([[1., 2.]])})},
        edge_sets={
            "edges": tfgnn.EdgeSet.from_fields(
                sizes=[2],
                adjacency=tfgnn.Adjacency.from_indices(*adjacency_args))})
    pass_messages_fn = functools.partial(
        modeling.pass_simple_messages,
        num_message_passing=2,
        receiver_tag=receiver_tag,
        message_dim=1,
        h_next_dim=3,
        l2_regularization=0.01,
        modeling_flavor=modeling_flavor)
    model = modeling.vanilla_mpnn_model(
        input_graph.spec,
        init_states_fn=lambda x: x, pass_messages_fn=pass_messages_fn)
    _ = model(input_graph)  # Build weights.
    self.assertLen(model.trainable_weights, 8)
    for v in model.trainable_weights:
      if v.shape.rank == 1:  # Bias.
        v.assign(tf.zeros_like(v))
      elif v.shape == [3, 1]:  # 1st round, messages to "b":
        v.assign([[10.], [-1.], [1.]])  # sum(10a - 1 + 2 for a in [1,2]) = 32.
      elif v.shape == [3, 3]:  # 1st round, new state for "b":
        v.assign([[1., 0., 0.],  # [1, 2, 32/8] = [1, 2, 4].
                  [0., 1., 0.],
                  [0., 0., 0.125]])
      elif v.shape == [4, 1]:  # 2nd round, messages to "b".
        v.assign([[5.], [1.], [0.], [1.]])  # sum(5a+1+4 for a in [1,2]) = 25.
      elif v.shape == [4, 3]:  # 2nd round, new state for "b":
        v.assign([[0., 1., 0.],  # [4, 1+2, -4+25] = [4, 3, 21].
                  [0., 1., 0.],
                  [1., 0., -1.],
                  [0., 0., 1.]])
      else:
        self.fail(f"Unexpected weight '{v.name}' of shape {v.shape}")

    graph = model(input_graph)
    self.assertAllEqual(
        graph.node_sets["a"][tfgnn.HIDDEN_STATE],
        [[1.], [2.]])  # Unchanged.
    self.assertAllEqual(
        graph.node_sets["b"][tfgnn.HIDDEN_STATE],
        [[4., 3., 21.]])
    self.assertAllClose(
        tf.add_n(model.losses),
        # l2_regularization factor applied to all weights assigned above.
        0.01 * (10.**2 + 11 * 1.**2 + 5.**2 + 0.125**2))

  @parameterized.named_parameters(
      ("RawGnnOps", "raw_gnn_ops"),
      ("EdgeNodeUpdates", "edge_node_updates"),
      ("NodeUpdates", "node_updates"),
      ("GnnBuilder", "gnn_builder"))
  def testDropoutInPassSimpleMessages(self, modeling_flavor):
    dim = 20
    dropout_rate = 0.4
    # Input graph a0 -> b0.
    input_graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            "a": tfgnn.NodeSet.from_fields(
                sizes=[1],
                features={tfgnn.HIDDEN_STATE: tf.constant([[1.]*dim])}),
            "b": tfgnn.NodeSet.from_fields(
                sizes=[1],
                features={tfgnn.HIDDEN_STATE: tf.constant([[]])})},
        edge_sets={
            "edges": tfgnn.EdgeSet.from_fields(
                sizes=[1],
                adjacency=tfgnn.Adjacency.from_indices(
                    ("a", tf.constant([0])),
                    ("b", tf.constant([0]))))})
    # Set up message passing as the identity function on the node state of "a",
    # but with dropout applied both for the message and the state update.
    pass_messages_fn = functools.partial(
        modeling.pass_simple_messages,
        num_message_passing=1,
        receiver_tag=tfgnn.TARGET,
        message_dim=dim,
        h_next_dim=dim,
        dropout_rate=dropout_rate,
        modeling_flavor=modeling_flavor)
    model = modeling.vanilla_mpnn_model(
        input_graph.spec,
        init_states_fn=lambda x: x, pass_messages_fn=pass_messages_fn)
    _ = model(input_graph)  # Build weights.
    self.assertLen(model.trainable_weights, 4)
    for v in model.trainable_weights:
      if v.shape == [dim]:  # Biases.
        v.assign(np.zeros(dim))
      elif v.shape == [dim, dim]:  # Kernels.
        v.assign(np.eye(dim))
      else:
        self.fail(f"Unexpected weight '{v.name}' of shape {v.shape}")

    # Baseline: no dropout.
    graph = model(input_graph)
    node_state = graph.node_sets["b"][tfgnn.HIDDEN_STATE]
    self.assertAllEqual(node_state, [[1.]*(dim)])

    # Test dropout. It's random, but tf.test.TestCase fixes the random seed, so
    # we can accept a small failure probability without making this test flaky.
    graph = model(input_graph, training=True)
    node_state_with_dropout = graph.node_sets["b"][tfgnn.HIDDEN_STATE]
    self.assertAllEqual(node_state_with_dropout.shape, [1, dim])
    # Check for non-dropped out units, scaled up to preserve the expected value.
    # We'll see this for any one unit with probability (1-0.4)**2 = 0.36, so
    # the probability to find none among 20 iid units is 0.64**20 < 1.4e-4.
    self.assertAllClose(tf.reduce_max(node_state_with_dropout),
                        1. * 1./(1.-dropout_rate)**2)
    # Check for dropped out units.
    # We'll see this for any one unit with probability 1-(1-0.4)**2 = 0.64, so
    # the probability to find none among 20 iid units is 0.36**20 < 1.4e-9.
    self.assertEqual(tf.reduce_min(node_state_with_dropout), 0.)


if __name__ == "__main__":
  tf.test.main()
