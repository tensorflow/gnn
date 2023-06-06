# pyformat: mode=yapf
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
"""Tests for Multi-Head Attention."""

import enum
import math
import os

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import multi_head_attention


class ReloadModel(int, enum.Enum):
  """Controls how to reload a model for further testing after saving."""
  SKIP = 0
  SAVED_MODEL = 1
  KERAS = 2


class MultiHeadAttentionTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(("", False), ("TransformAfter", True))
  def testBasic(self, transform_values_after_pooling):
    """Tests that a single-headed MHA is correct given predefined weights."""
    # NOTE: Many following tests use minor variations of the explicit
    # construction of weights and results introduced here.

    # Construct a graph with three nodes 0, 1, 2, and six edges:
    # a cycle 0->1->2->0 (let's call it clockwise)
    # and the reverse cycle 0->2->1->0 (counterclockwise).
    gt_input = _get_test_bidi_cycle_graph(
        tf.constant(
            # Node states have dimension 4.
            # The first three dimensions one-hot encode the node_id i.
            # The fourth dimension holds a distinct payload value i+1.
            [
                [1., 0., 0., 1.],
                [0., 1., 0., 2.],
                [0., 0., 1., 3.],
            ]))

    conv = multi_head_attention.MultiHeadAttentionConv(
        num_heads=1,
        per_head_channels=3,
        receiver_tag=tfgnn.TARGET,
        activation="relu",  # Let's keep it simple.
        transform_values_after_pooling=transform_values_after_pooling,
    )

    _ = conv(gt_input, edge_set_name="edges")  # Build weights.
    weights = {v.name: v for v in conv.trainable_weights}
    self.assertLen(weights, 6)

    weights["multi_head_attention_conv/query/kernel:0"].assign(
        # The space of attention computation of the single head has dimension 3.
        # The three dimensions are used to represent the transformed query
        # in one-hot manner and hand-construct attention scores
        # (see the running example below) that favor the counterclockwise
        # incoming edge over the other. Recall that weight matrices are
        # multiplied from the right onto batched inputs (in rows).
        #
        # For example, the query vector of node 0 is [0, 1, 0], and ...
        [
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 0., 0.],
        ])
    weights["multi_head_attention_conv/query/bias:0"].assign([0., 0., 0.])

    log20 = tf.math.log(20.).numpy()
    log2 = tf.math.log(2.).numpy()
    # Using an inverse scaling factor to cancel out the score scaling.
    inverse_scaling_factor = tf.math.sqrt(3.)
    weights["multi_head_attention_conv/key_node/kernel:0"].assign(
        # ... the key vectors of node 1 and 2, resp., are \sqrt(3) *
        # [log(2), log(20), 0.] and [0., log(2), log(20)]. Therefore, the node 0
        # query vector [0., 1., 0.] dot-product on the key vectors of node 1
        # and 2 will give \sqrt(3) * log(20) (favored) and \sqrt(3) * log(2)
        # (not favored), and the \sqrt(3) is canceled out after scaling.
        inverse_scaling_factor * [
            [log20, 0., log2],
            [log2, log20, 0.],
            [0., log2, log20],
            [0., 0., 0.],
        ])
    weights["multi_head_attention_conv/key_node/bias:0"].assign([0., 0., 0.])

    # The attention coefficients are computed by the dot-product of transformed
    # query and key. In our specific case, for example, the attention from
    # node 0 to node 1 has a pre-softmax score log(20) after score scaling.
    # Likewise, the attention from node 0 to node 2 has a pre-softmax score
    # log(2). The two scores become 10/11 and 1/11 after softmax.

    if not transform_values_after_pooling:
      weights["multi_head_attention_conv/value_node/kernel:0"].assign(
          # ... the value vectors of node 1 and 2, resp., are [-1, 0, 2.2]
          # and [-1, -1, 3.3], and only positive value (the fourth dimension)
          # will be kept after the final ReLU activation.
          [
              [0., -1., 0.],
              [-1., 0., 0.],
              [-1., -1., 0.],
              [0., 0., 1.1],
          ])
      weights["multi_head_attention_conv/value_node/bias:0"].assign(
          [0., 0., 0.])
    else:
      # Same weights, but as Einsum kernel "hvc".
      weights["multi_head_attention_conv/value_pooled/kernel:0"].assign([[
          [0., -1., 0.],
          [-1., 0., 0.],
          [-1., -1., 0.],
          [0., 0., 1.1],
      ]])
      weights["multi_head_attention_conv/value_pooled/bias:0"].assign(
          [[0., 0., 0.]])

    got = conv(gt_input, edge_set_name="edges")

    # ... The softmax-weighed key vectors on the incoming edges of node 0
    # are  10/11 * [-1, 0, 2.2]  +  1/11 * [-1, -1, 3.3].
    # The final ReLU takes out the non-positive components and leaves 2 + 0.3
    # in the last component of the first row in the resulting node states.
    want = tf.constant([
        [0., 0., 2.3],  # Node 0.
        [0., 0., 3.1],  # Node 1.
        [0., 0., 1.2],  # Node 2.
    ])
    self.assertAllEqual(got.shape, (3, 3))
    self.assertAllClose(got, want, atol=.0001)

    # For node states with more than one feature dimension, MultiHeadAttention
    # works in parallel on the vectors from the innermost dimension, so we can
    # repeat the previous computation and an alternative with different values
    # in the last component and reversed orientation:
    gt_input_2 = _get_test_bidi_cycle_graph(
        tf.constant([
            [[1., 0., 0., 1.], [0., 0., 1., 3.]],
            [[0., 1., 0., 2.], [0., 1., 0., 6.]],
            [[0., 0., 1., 3.], [1., 0., 0., 9.]],
        ]))
    got_2 = conv(gt_input_2, edge_set_name="edges")
    want_2 = tf.constant([
        [[0., 0., 2.3], [0., 0., 9.6]],
        [[0., 0., 3.1], [0., 0., 3.9]],
        [[0., 0., 1.2], [0., 0., 6.3]],
    ])
    self.assertAllEqual(got_2.shape, (3, 2, 3))
    self.assertAllClose(got_2, want_2, atol=.0001)

  def testAttentionActivation(self):
    """Tests that a single-headed MHA correctly applies attention activations."""

    # The same test graph as in the testBasic above.
    gt_input = _get_test_bidi_cycle_graph(
        tf.constant([
            [1., 0., 0., 1.],
            [0., 1., 0., 2.],
            [0., 0., 1., 3.],
        ]))

    def get_conv(attention_activation=None):
      """Constructs a MultiHeadAttentionConv with the given attention_activation."""

      conv = multi_head_attention.MultiHeadAttentionConv(
          num_heads=1,
          per_head_channels=3,
          receiver_tag=tfgnn.TARGET,
          attention_activation=attention_activation,
          activation=None,
          score_scaling="none",
      )

      _ = conv(gt_input, edge_set_name="edges")  # Build weights.
      weights = {v.name: v for v in conv.trainable_weights}
      self.assertLen(weights, 6)

      weights["multi_head_attention_conv/query/kernel:0"].assign(
          # The node states times the query kernel should be:
          #
          # [[0., 1., 0.],
          #  [0., 0., -1.],
          #  [1., 0., 0.]]
          #
          # i.e. the second query vector has negative values, which, after
          # activation with the `relu` function, should be all zeros.
          [
              [0., 1., 0.],
              [0., 0., -1.],
              [1., 0., 0.],
              [0., 0., 0.],
          ])
      weights["multi_head_attention_conv/query/bias:0"].assign([0., 0., 0.])

      weights["multi_head_attention_conv/key_node/kernel:0"].assign(
          # The key_node kernel is chosen such that the the product with the
          # node states is:
          #
          # [[-1., 0., 0.],
          #  [0., 1., 0.],
          #  [0., 0., 1.]]
          #
          # i.e. the third key vector has negative values, which, after
          # activation with the `relu` function, should be all zeros.
          [
              [-1., 0., 0.],
              [0., 1., 0.],
              [0., 0., 1.],
              [0., 0., 0.],
          ])
      weights["multi_head_attention_conv/key_node/bias:0"].assign([0., 0., 0.])

      # The attention scores are computed as the product of the transformed
      # queries and keys (with a zero diagonal since there are no self edges and
      # hence no self-attention) and should be:
      #
      # [[0., 1., 0.],
      #  [0., 0., a],
      #  [a, 0., 0.]]
      #
      # where the value `a` should be `-1` if no attention activation is used,
      # and `0` when the attention activation is set to `relu`.
      #
      # Attention weights are computed by applying softmax to each row except
      # the diagonal element. Recall that
      #    softmax([1, 0])= [e, 1] / (e + 1);
      #    softmax([0, 0])= [1, 1] / 2, for a == 0;
      #    softmax([0, -1]) = softmax([1, 0]) = [e, 1] / (e + 1), for a == - 1;
      # which explains the expected values below.

      weights["multi_head_attention_conv/value_node/kernel:0"].assign(
          # Identity matrix such that the transformed node states are `eye(3)`.
          [
              [1., 0., 0.],
              [0., 1., 0.],
              [0., 0., 1.],
              [0., 0., 0.],
          ])
      weights["multi_head_attention_conv/value_node/bias:0"].assign(
          [0., 0., 0.])

      return conv

    with self.subTest("without_attention_activation"):
      conv = get_conv(attention_activation=None)
      got = conv(gt_input, edge_set_name="edges")

      # Since the transformed values are just the identity matrix, we recover
      # the attention weights for each query.
      e = tf.math.exp(1.).numpy()
      want = tf.constant([
          [0., e, 1.],
          [e, 0., 1.],
          [1., e, 0.],
      ]) / tf.constant(
          e + 1., dtype=tf.float32)
      self.assertAllEqual(got.shape, (3, 3))
      self.assertAllClose(got, want, atol=.0001)

    with self.subTest("with_attention_activation"):
      conv = get_conv(attention_activation="relu")
      got = conv(gt_input, edge_set_name="edges")

      # Since the transformed values are just the identity matrix, we recover
      # the attention weights for each query.
      want = tf.constant([
          [0., e, 1.],
          [1., 0., 1.],
          [1., 1., 0.],
      ])
      want = want / tf.reduce_sum(want, axis=-1, keepdims=True)
      self.assertAllEqual(got.shape, (3, 3))
      self.assertAllClose(got, want, atol=.0001)

  def testScoreScalingTypes(self):
    """Tests that the different types of score scaling are applied correctly."""

    # The same test graph as in the testBasic above.
    gt_input = _get_test_bidi_cycle_graph(
        tf.constant([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
        ]))

    def get_conv(score_scaling=None):
      """Constructs a MultiHeadAttentionConv with the given score_scaling."""

      conv = multi_head_attention.MultiHeadAttentionConv(
          num_heads=1,
          per_head_channels=3,
          receiver_tag=tfgnn.TARGET,
          activation=None,
          score_scaling=score_scaling,
      )

      _ = conv(gt_input, edge_set_name="edges")  # Build weights.
      weights = {v.name: v for v in conv.trainable_weights}
      if score_scaling == "trainable_sigmoid":
        # Additional trainable weight for the score scaling.
        self.assertLen(weights, 7)
      else:
        self.assertLen(weights, 6)

      weights["multi_head_attention_conv/query/kernel:0"].assign(
          # The node states times the query kernel should be:
          #
          # [[0., 1., 0.],
          #  [0., 0., -1.],
          #  [1., 0., 0.]]
          #
          # i.e. the second query vector has negative values, which, after
          # activation with the `relu` function, should be all zeros.
          [
              [0.0, 1.0, 0.0],
              [0.0, 0.0, -1.0],
              [1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0],
          ])
      weights["multi_head_attention_conv/query/bias:0"].assign([0.0, 0.0, 0.0])

      weights["multi_head_attention_conv/key_node/kernel:0"].assign(
          # The key_node kernel is chosen such that the the product with the
          # node states is:
          #
          # [[-1., 0., 0.],
          #  [0., 1., 0.],
          #  [0., 0., 1.]]
          #
          # i.e. the third key vector has negative values, which, after
          # activation with the `relu` function, should be all zeros.
          [
              [-1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0],
              [0.0, 0.0, 0.0],
          ])
      weights["multi_head_attention_conv/key_node/bias:0"].assign(
          [0.0, 0.0, 0.0])

      # The attention scores are computed as the product of the transformed
      # queries and keys (with a zero diagonal since there are no self edges and
      # hence no self-attention), scaled by a factor s.
      #
      #  s * [[0., 1., 0.],              [[ 0,  s,  0],
      #       [0., 0., -1],      ==       [ 0,  0, -s],
      #       [-1, 0., 0.]]               [-s,  0,  0]]
      #
      # Attention weights are computed by applying softmax to each row except
      # the diagonal element. Recall that
      #    softmax([s,  0])                   = [exp(s), 1] / (exp(s) + 1), and
      #    softmax([0, -s]) = softmax([s, 0]) = [exp(s), 1] / (exp(s) + 1),
      # which explains the expected values below, with w = exp(s).

      weights["multi_head_attention_conv/value_node/kernel:0"].assign(
          # Identity matrix such that the transformed node states are `eye(3)`.
          [
              [1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0],
              [0.0, 0.0, 0.0],
          ])
      weights["multi_head_attention_conv/value_node/bias:0"].assign(
          [0.0, 0.0, 0.0])

      return conv

    named_scalings = {
        "none": 1.0,
        "rsqrt_dim": 1.0 / math.sqrt(3.0),
        "trainable_sigmoid": tf.keras.activations.sigmoid(-5.0),
    }

    for scaling_name, scaling_factor in named_scalings.items():
      with self.subTest(f"with_{scaling_name}"):
        conv = get_conv(score_scaling=scaling_name)
        got = conv(gt_input, edge_set_name="edges")

        # Since the transformed values are just the identity matrix, we recover
        # the attention weights for each query.
        w = tf.math.exp(scaling_factor).numpy()
        want = tf.constant([
            [0.0, w, 1.0],
            [w, 0.0, 1.0],
            [1.0, w, 0.0],
        ]) / tf.constant(
            w + 1.0, dtype=tf.float32)
        self.assertAllEqual(got.shape, (3, 3))
        self.assertAllClose(got, want, atol=0.0001)

  def testNoTransformKeys(self):
    """Tests that the no key transformation variant of MHA is correct."""

    # The same test graph as in the testBasic above.
    gt_input = _get_test_bidi_cycle_graph(
        tf.constant([
            [1., 0., 0., 1.],
            [0., 1., 0., 2.],
            [0., 0., 1., 3.],
        ]))

    conv = multi_head_attention.MultiHeadAttentionConv(
        num_heads=1,
        per_head_channels=3,
        receiver_tag=tfgnn.TARGET,
        activation="relu",  # Let's keep it simple.
        transform_keys=False,  # Turn off the key transformation.
    )

    _ = conv(gt_input, edge_set_name="edges")  # Build weights.
    weights = {v.name: v for v in conv.trainable_weights}
    # No key transformation weights.
    self.assertLen(weights, 4)

    log20 = tf.math.log(20.).numpy()
    log2 = tf.math.log(2.).numpy()
    # Using an inverse scaling factor to cancel out the score scaling.
    inverse_scaling_factor = tf.math.sqrt(4.)

    weights["multi_head_attention_conv/query/kernel:0"].assign(
        # The space of attention computation of the single head has dimension 4,
        # since we do NOT transform the keys and thus need to transform queries
        # to match the keys width.
        # The first three dimensions are used to represent the transformed query
        # in one-hot manner.
        #
        # For example, the query vector of node 0 is
        # inverse_scaling_factor * [0, log(20), log(2), 0], and ...
        inverse_scaling_factor * [
            [0., log20, log2, 0.],
            [log2, 0., log20, 0.],
            [log20, log2, 0., 0.],
            [0., 0., 0., 0.],
        ])
    weights["multi_head_attention_conv/query/bias:0"].assign([0., 0., 0., 0.])

    # The attention coefficients are computed by the dot-product of transformed
    # query and key. We manually assign the weights to get the same attention
    # score in the testBasic above. For example, node 0 favors node 1 (10/11)
    # and does not favor node 2 (1/11).

    weights["multi_head_attention_conv/value_node/kernel:0"].assign(
        # ... the value vectors of node 1 and 2, resp., are [-1, 0, 2.2]
        # and [-1, -1, 3.3], and only positive value (the fourth dimension)
        # will be kept after the final ReLU activation.
        [
            [0., -1., 0.],
            [-1., 0., 0.],
            [-1., -1., 0.],
            [0., 0., 1.1],
        ])
    weights["multi_head_attention_conv/value_node/bias:0"].assign([0., 0., 0.])

    got = conv(gt_input, edge_set_name="edges")

    # ... The softmax-weighed key vectors on the incoming edges of node 0
    # are  10/11 * [-1, 0, 2.2]  +  1/11 * [-1, -1, 3.3].
    # The final ReLU takes out the non-positive components and leaves 2 + 0.3
    # in the last component of the first row in the resulting node states.
    want = tf.constant([
        [0., 0., 2.3],  # Node 0.
        [0., 0., 3.1],  # Node 1.
        [0., 0., 1.2],  # Node 2.
    ])
    self.assertAllEqual(got.shape, (3, 3))
    self.assertAllClose(got, want, atol=.0001)

    # For node states with more than one feature dimension, MultiHeadAttention
    # works in parallel on the vectors from the innermost dimension, so we can
    # repeat the previous computation and an alternative with different values
    # in the last component and reversed orientation:
    gt_input_2 = _get_test_bidi_cycle_graph(
        tf.constant([
            [[1., 0., 0., 1.], [0., 0., 1., 3.]],
            [[0., 1., 0., 2.], [0., 1., 0., 6.]],
            [[0., 0., 1., 3.], [1., 0., 0., 9.]],
        ]))
    got_2 = conv(gt_input_2, edge_set_name="edges")
    want_2 = tf.constant([
        [[0., 0., 2.3], [0., 0., 9.6]],
        [[0., 0., 3.1], [0., 0., 3.9]],
        [[0., 0., 1.2], [0., 0., 6.3]],
    ])
    self.assertAllEqual(got_2.shape, (3, 2, 3))
    self.assertAllClose(got_2, want_2, atol=.0001)

  @parameterized.named_parameters(("", False), ("TransformAfter", True))
  def testMultihead(self, transform_values_after_pooling):
    """Extends testBasic with multiple attention heads."""
    # The same test graph as in the testBasic above.
    gt_input = _get_test_bidi_cycle_graph(
        tf.constant([
            [1., 0., 0., 1.],
            [0., 1., 0., 2.],
            [0., 0., 1., 3.],
        ]))

    conv = multi_head_attention.MultiHeadAttentionConv(
        num_heads=2,
        per_head_channels=3,
        receiver_tag=tfgnn.TARGET,
        activation="relu",
        use_bias=False,  # Don't create /bias variables.
        score_scaling="none",  # Disable score scaling.
        transform_values_after_pooling=transform_values_after_pooling,
    )

    _ = conv(gt_input, edge_set_name="edges")  # Build weights.
    weights = {v.name: v for v in conv.trainable_weights}
    self.assertLen(weights, 3)

    weights["multi_head_attention_conv/query/kernel:0"].assign(
        # Attention head 0 uses the first three dimensions, which are used
        # in the same way as for the testBasic test above.
        # Attention head 1 uses the last three dimensions, in which we
        # now favor the clockwise incoming edges.
        [
            [0., 1., 0., 0., 0., 1.],
            [0., 0., 1., 1., 0., 0.],
            [1., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 0.],
        ])

    log20 = tf.math.log(20.).numpy()
    log2 = tf.math.log(2.).numpy()
    # No need for an inverse scaling factor as score scaling is disabled.
    weights["multi_head_attention_conv/key_node/kernel:0"].assign(
        # To favor the clockwise incoming edges, we use the last three
        # dimensions, and assign 100 and 0 to corresponding neighbors,
        # which gives weights 1 to clockwise incoming edges and weights 0
        # to counterclockwise incoming edges.
        [
            [log20, 0., log2, 100., 0., 0.],
            [log2, log20, 0., 0., 100., 0.],
            [0., log2, log20, 0., 0., 100.],
            [0., 0., 0., 0., 0., 0.],
        ])

    if not transform_values_after_pooling:
      # No matter where the -1s are, they got eliminated by ReLU.
      weights["multi_head_attention_conv/value_node/kernel:0"].assign([
          [0., -1., 0., 0., -1., 0.],
          [-1., 0., 0., -1., 0., 0.],
          [-1., -1., 0., -1., -1., 0.],
          [0., 0., 1.1, 0., 0., 1.],
      ])
    else:
      # Same weights, but as Einsum kernel with axes "hvc".
      weights["multi_head_attention_conv/value_pooled/kernel:0"].assign([[
          [0., -1., 0.],
          [-1., 0., 0.],
          [-1., -1., 0.],
          [0., 0., 1.1],
      ], [
          [0., -1., 0.],
          [-1., 0., 0.],
          [-1., -1., 0.],
          [0., 0., 1.],
      ]])

    got = conv(gt_input, edge_set_name="edges")

    # Attention head 0 generates the first four output dimensions as in the
    # testBasic above, with weights 10/11 and 1/11,
    # Attention head 1 uses weights 0 and 1 (note the reversed preference).
    want = tf.constant([
        [0., 0., 2.3, 0., 0., 3.0],
        [0., 0., 3.1, 0., 0., 1.0],
        [0., 0., 1.2, 0., 0., 2.0],
    ])
    self.assertAllEqual(got.shape, (3, 6))
    self.assertAllClose(got, want, atol=.0001)

  @parameterized.named_parameters(
      ("", ReloadModel.SKIP, False), ("TransformAfter", ReloadModel.SKIP, True),
      ("Restored", ReloadModel.SAVED_MODEL, False),
      ("RestoredTransformAfter", ReloadModel.SAVED_MODEL, True),
      ("RestoredKeras", ReloadModel.KERAS, False),
      ("RestoredKerasTransformAfter", ReloadModel.KERAS, True))
  def testFullModel(self, reload_model, transform_values_after_pooling):
    """Tests MultiHeadAttentionHomGraphUpdate in a Model with edge input."""
    # The same example as in the testBasic above, but with extra inputs
    # from edges.
    gt_input = _get_test_bidi_cycle_graph(
        # Node i has value i+1 in the last component, which will be mapped
        # into the fourth component of the value.
        tf.constant([
            [1., 0., 0., 1.],  # Node 0.
            [0., 1., 0., 2.],  # Node 1.
            [0., 0., 1., 3.],  # Node 2.
        ]),
        # Edges out of node i have value 2*(i+1) for the clockwise edges favored
        # by attention and 3*(i+1) for the counterclockwise edges not favored.
        tf.constant([
            [3.],  # Edge from node 0 (clockwise, not favored).
            [6.],  # Edge from node 1 (clockwise, not favored).
            [9.],  # Edge from node 2 (clockwise, not favored).
            [2.],  # Edge from node 0 (counterclockwise, favored).
            [6.],  # Edge from node 2 (counterclockwise, favored).
            [4.],  # Edge from node 1 (counterclockwise, favored).
        ]))

    layer = multi_head_attention.MultiHeadAttentionHomGraphUpdate(
        num_heads=1,
        per_head_channels=4,
        receiver_tag=tfgnn.TARGET,
        sender_edge_feature=tfgnn.HIDDEN_STATE,  # Activate edge input.
        attention_activation="relu",
        kernel_initializer="zeros",
        kernel_regularizer=tf.keras.regularizers.L2(0.05),  # Add a regularizer.
        transform_values_after_pooling=transform_values_after_pooling,
    )

    _ = layer(gt_input)  # Build weights.
    weights = {v.name: v for v in layer.trainable_weights}
    if not transform_values_after_pooling:
      self.assertLen(weights, 8)
    else:
      self.assertLen(weights, 7)

    # Check the initial weights.
    self.assertAllClose(
        weights["multi_head_attention/node_set_update/" +
                "multi_head_attention_conv/query/kernel:0"],
        tf.zeros((4, 4)),
        atol=.0001)
    self.assertAllClose(
        weights["multi_head_attention/node_set_update/" +
                "multi_head_attention_conv/key_edge/kernel:0"],
        tf.zeros((1, 4)),
        atol=.0001)
    self.assertAllClose(
        weights["multi_head_attention/node_set_update/" +
                "multi_head_attention_conv/key_node/kernel:0"],
        tf.zeros((4, 4)),
        atol=.0001)
    if not transform_values_after_pooling:
      self.assertAllClose(
          weights["multi_head_attention/node_set_update/" +
                  "multi_head_attention_conv/value_edge/kernel:0"],
          tf.zeros((1, 4)),
          atol=.0001)
      self.assertAllClose(
          weights["multi_head_attention/node_set_update/" +
                  "multi_head_attention_conv/value_node/kernel:0"],
          tf.zeros((4, 4)),
          atol=.0001)
    else:
      self.assertAllClose(
          weights["multi_head_attention/node_set_update/" +
                  "multi_head_attention_conv/value_pooled/kernel:0"],
          tf.zeros((1, 5, 4)),
          atol=.0001)

    log20 = tf.math.log(20.).numpy()
    log2 = tf.math.log(2.).numpy()
    # Using an inverse scaling factor to cancel out the score scaling.
    inverse_scaling_factor = tf.math.sqrt(4.)

    weights["multi_head_attention/node_set_update/" +
            "multi_head_attention_conv/query/kernel:0"].assign([
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [1., 0., 0., 0.],
                [0., 0., 0., 0.],
            ])
    weights[
        "multi_head_attention/node_set_update/" +
        "multi_head_attention_conv/key_edge/kernel:0"].assign(
            # Edge values are projected to the fourth dimension. However, it
            # will have no effects on the dot-product since the fourth
            # dimension of the transformed query is filled with zeros.
            [[0., 0., 0., 1.]])
    weights[
        "multi_head_attention/node_set_update/" +
        "multi_head_attention_conv/key_node/kernel:0"].assign(
            # Similar transformation as in testBasic, except adding the
            # fourth dimension to with all zeros (placeholder for edge
            # values).
            inverse_scaling_factor * [
                [log20, 0., log2, 0.],
                [log2, log20, 0., 0.],
                [0., log2, log20, 0.],
                [0., 0., 0., 0.],
            ])
    # Edge values and node payloads are put into final components of the value
    # space, with the same adjustment for softmax-weighting by 1/11 or 10/11.
    if not transform_values_after_pooling:
      weights["multi_head_attention/node_set_update/" +
              "multi_head_attention_conv/value_node/kernel:0"].assign([
                  [0., 0., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 1.1, 0.],
              ])
      weights["multi_head_attention/node_set_update/" +
              "multi_head_attention_conv/value_edge/kernel:0"].assign(
                  [[0., 0., 0., 1.1]])
    else:
      weights["multi_head_attention/node_set_update/" +
              "multi_head_attention_conv/value_pooled/kernel:0"].assign([[
                  [0., 0., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 1.1, 0.],
                  [0., 0., 0., 1.1],
              ]])

    # Assign zeros to all the bias terms.
    weights["multi_head_attention/node_set_update/" +
            "multi_head_attention_conv/query/bias:0"].assign([0., 0., 0., 0.])
    if not transform_values_after_pooling:
      weights["multi_head_attention/node_set_update/" +
              "multi_head_attention_conv/key_node/bias:0"].assign(
                  [0., 0., 0., 0.])
      weights["multi_head_attention/node_set_update/" +
              "multi_head_attention_conv/value_node/bias:0"].assign(
                  [0., 0., 0., 0.])
    else:
      weights["multi_head_attention/node_set_update/" +
              "multi_head_attention_conv/value_pooled/bias:0"].assign(
                  [[0., 0., 0., 0.]])

    # Build a Model around the Layer, possibly saved and restored.
    inputs = tf.keras.layers.Input(type_spec=gt_input.spec)
    outputs = layer(inputs)
    model = tf.keras.Model(inputs, outputs)
    if reload_model:
      export_dir = os.path.join(self.get_temp_dir(), "edge-input-model")
      model.save(export_dir, include_optimizer=False)
      if reload_model == ReloadModel.KERAS:
        model = tf.keras.models.load_model(export_dir)
        # Check that from_config() worked, no fallback to a function trace, see
        # https://www.tensorflow.org/guide/keras/save_and_serialize#how_savedmodel_handles_custom_objects
        self.assertIsInstance(
            model.get_layer(index=1), tfgnn.keras.layers.GraphUpdate)
      else:
        model = tf.saved_model.load(export_dir)

    got_gt = model(gt_input)
    got = got_gt.node_sets["nodes"][tfgnn.HIDDEN_STATE]

    # The fourth column with values x.y from nodes is analogous to the
    # testBasic test above, with the contribution x from the favored
    # input before the decimal dot and the other contribution y after.
    # The fifth column with values (2x).(3y) is from edges, with the
    # multipliers 2 and 3 used above in setting up the edge features.
    want = tf.constant([
        [0., 0., 2.3, 4.9],
        [0., 0., 3.1, 6.3],
        [0., 0., 1.2, 2.6],
    ])
    self.assertAllEqual(got.shape, (3, 4))
    self.assertAllClose(got, want, atol=.0001)

    # Check the L2 regularization.
    want = 0.05 * (4 * 1.0**2 + 3 * (inverse_scaling_factor * log20)**2 + 3 *
                   (inverse_scaling_factor * log2)**2 + 2 * 1.1**2)
    if reload_model != ReloadModel.SAVED_MODEL:
      self.assertAllClose(tf.add_n(model.losses), want)

  def testConvolutionReceivers(self):
    """Tests convolution to context and equivalence to a special node set."""

    # An example graph with two components:
    # The first component has the example from testBasic split into a "sources"
    # node set with the first two nodes and a "targets" node set with the third
    # node and a context feature equal to the target node.
    # The second component repeats that construction, but with a variation in
    # the value of the target node / context.
    gt_input = tfgnn.GraphTensor.from_pieces(
        node_sets={
            "sources":
                tfgnn.NodeSet.from_fields(
                    sizes=[2, 2],
                    features={
                        tfgnn.HIDDEN_STATE:
                            tf.constant([
                                [1., 0., 0., 1.],
                                [0., 1., 0., 2.],
                            ] * 2)  # Repeated for both components.
                    }),
            "targets":
                tfgnn.NodeSet.from_fields(
                    sizes=[1, 1],
                    features={
                        tfgnn.HIDDEN_STATE:
                            tf.constant([
                                [0., 0., 1., 3.],
                                [0., 0., 1., 4.],
                            ])
                    }),
        },
        context=tfgnn.Context.from_fields(
            # Same as "targets".
            features={
                tfgnn.HIDDEN_STATE:
                    tf.constant([
                        [0., 0., 1., 3.],
                        [0., 0., 1., 4.],
                    ])
            }),
        edge_sets={
            "edges":
                tfgnn.EdgeSet.from_fields(
                    sizes=[2, 2],
                    # Same as membership of "sources" in components.
                    adjacency=tfgnn.Adjacency.from_indices(
                        ("sources", tf.constant([0, 1, 2, 3])),
                        ("targets", tf.constant([0, 0, 1, 1]))))
        })

    conv = multi_head_attention.MultiHeadAttentionConv(
        num_heads=1, per_head_channels=3, activation="relu", use_bias=False)

    # Build weights, then initialize as in testBasic.
    _ = conv(gt_input, node_set_name="sources", receiver_tag=tfgnn.CONTEXT)
    weights = {v.name: v for v in conv.trainable_weights}
    self.assertLen(weights, 3)

    log20 = tf.math.log(20.).numpy()
    log2 = tf.math.log(2.).numpy()
    # Using an inverse scaling factor to cancel out the score scaling.
    inverse_scaling_factor = tf.math.sqrt(3.)

    weights["multi_head_attention_conv/query/kernel:0"].assign([
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 0., 0.],
        [0., 0., 0.],
    ])
    weights["multi_head_attention_conv/key_node/kernel:0"].assign(
        inverse_scaling_factor * [
            [log20, 0., log2],
            [log2, log20, 0.],
            [0., log2, log20],
            [0., 0., 0.],
        ])
    weights["multi_head_attention_conv/value_node/kernel:0"].assign([
        [0., -1., 0.],
        [-1., 0., 0.],
        [-1., -1., 0.],
        [0., 0., 1.1],
    ])

    # The convolution object can be called interchangeably for convolving
    # "sources" to context, or along "edges" to "targets" with the same
    # features as context.
    got_context = conv(
        gt_input, node_set_name="sources", receiver_tag=tfgnn.CONTEXT)
    got_targets = conv(
        gt_input, edge_set_name="edges", receiver_tag=tfgnn.TARGET)
    want = tf.constant([
        [0., 0., 1.2],  # As in testBasic for node 2.
        [0., 0., 1.2],
    ])
    self.assertAllClose(got_context, want, atol=.0001)
    self.assertAllClose(got_targets, want, atol=.0001)

    # The same object can even be used for convolving over the edge set in
    # reverse direction, that is, to "sources". The result is boring, though:
    # Every "source" gets the same value from the sole "target" of the component
    # (so softmax reduces to a no-op), which is scaled with 1.1 by the
    # bottom-right element of value/kernel.
    got_sources = conv(
        gt_input, edge_set_name="edges", receiver_tag=tfgnn.SOURCE)
    want_sources = tf.constant([
        [0., 0., 3.3],
        [0., 0., 3.3],
        [0., 0., 4.4],
        [0., 0., 4.4],
    ])
    self.assertAllClose(got_sources, want_sources, atol=.0001)

  def testEdgePoolReceivers(self):
    """Tests MultiHeadAttentionEdgePool for pooling to nodes and to context."""
    # This example is similar to testConvolutionReceivers, except that
    # the sender features are now on the edges, not the source nodes.
    gt_input = tfgnn.GraphTensor.from_pieces(
        edge_sets={
            "edges":
                tfgnn.EdgeSet.from_fields(
                    sizes=[2, 2],
                    adjacency=tfgnn.Adjacency.from_indices(
                        ("sources", tf.constant([0, 1, 2, 3])),
                        ("targets", tf.constant([0, 0, 1, 1]))),
                    features={
                        tfgnn.HIDDEN_STATE:
                            tf.constant([
                                [1., 0., 0., 1.],
                                [0., 1., 0., 2.],
                            ] * 2)  # Repeated for both components.
                    }),
        },
        node_sets={
            "sources":
                tfgnn.NodeSet.from_fields(sizes=[2, 2]),  # No features.
            "targets":
                tfgnn.NodeSet.from_fields(
                    sizes=[1, 1],
                    features={
                        tfgnn.HIDDEN_STATE:
                            tf.constant([
                                [0., 0., 1., 3.],
                                [0., 0., 1., 4.],
                            ])
                    }),
        },
        context=tfgnn.Context.from_fields(
            # Same as "targets".
            features={
                tfgnn.HIDDEN_STATE:
                    tf.constant([
                        [0., 0., 1., 3.],
                        [0., 0., 1., 4.],
                    ])
            }))

    layer = multi_head_attention.MultiHeadAttentionEdgePool(
        num_heads=1,
        per_head_channels=3,
        attention_activation="relu",
        use_bias=False)

    # Build weights, then initialize analogous to testConvolutionReceivers,
    # with "value_edge" instead of "value_node".
    _ = layer(gt_input, edge_set_name="edges", receiver_tag=tfgnn.CONTEXT)
    weights = {v.name: v for v in layer.trainable_weights}
    self.assertLen(weights, 3)

    log20 = tf.math.log(20.).numpy()
    log2 = tf.math.log(2.).numpy()
    # Using an inverse scaling factor to cancel out the score scaling.
    inverse_scaling_factor = tf.math.sqrt(3.)

    weights["multi_head_attention_edge_pool/query/kernel:0"].assign([
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 0., 0.],
        [0., 0., 0.],
    ])
    weights["multi_head_attention_edge_pool/key_edge/kernel:0"].assign(
        inverse_scaling_factor * [
            [log20, 0., log2],
            [log2, log20, 0.],
            [0., log2, log20],
            [0., 0., 0.],
        ])
    weights["multi_head_attention_edge_pool/value_edge/kernel:0"].assign([
        [0., -1., 0.],
        [-1., 0., 0.],
        [-1., -1., 0.],
        [0., 0., 1.1],
    ])

    # The EdgePool object can be called interchangeably for attention-pooling
    # the "edges" to context or to each component's unique node in "targets".
    got_context = layer(
        gt_input, edge_set_name="edges", receiver_tag=tfgnn.CONTEXT)
    got_targets = layer(
        gt_input, edge_set_name="edges", receiver_tag=tfgnn.TARGET)
    want = tf.constant([[0., 0., 1.2], [0., 0., 1.2]])
    self.assertAllClose(got_context, want, atol=.0001)
    self.assertAllClose(got_targets, want, atol=.0001)

  @parameterized.named_parameters(("", ReloadModel.SKIP),
                                  ("Restored", ReloadModel.SAVED_MODEL),
                                  ("RestoredKeras", ReloadModel.KERAS))
  def testEdgeDropout(self, reload_model):
    """Tests dropout, esp. the switch between training and inference modes."""
    # Avoid flakiness.
    tf.random.set_seed(42)

    # This test graph has many source nodes feeding into one target node.
    # The node features are a one-hot encoding of node ids.
    num_nodes = 32
    target_node_id = 7
    gt_input = tfgnn.GraphTensor.from_pieces(
        node_sets={
            "nodes":
                tfgnn.NodeSet.from_fields(
                    sizes=[num_nodes],
                    features={tfgnn.HIDDEN_STATE: tf.eye(num_nodes)}),
        },
        edge_sets={
            "edges":
                tfgnn.EdgeSet.from_fields(
                    sizes=[num_nodes],
                    adjacency=tfgnn.Adjacency.from_indices(
                        ("nodes", tf.constant(list(range(num_nodes)))),
                        ("nodes", tf.constant([target_node_id] * num_nodes))))
        })

    # On purpose, this test is not for MultiHeadAttentionConv directly,
    # but for its common usage in a GraphUpdate, to make sure the
    # training/inference mode is propagated correctly.
    layer = multi_head_attention.MultiHeadAttentionHomGraphUpdate(
        num_heads=1,
        per_head_channels=num_nodes,
        receiver_tag=tfgnn.TARGET,
        edge_dropout=1. / 3.,  # Note here.
        activation="linear",
        attention_activation="linear",
        use_bias=False)

    _ = layer(gt_input)  # Build weights.
    weights = {v.name: v for v in layer.trainable_weights}
    self.assertLen(weights, 3)
    # Set up equal attention to all inputs.
    weights["multi_head_attention/node_set_update/" +
            "multi_head_attention_conv/query/kernel:0"].assign(
                [[0.] * num_nodes] * num_nodes)
    # Values are one-hot node ids, scaled up to undo the softmax normalization.
    weights["multi_head_attention/node_set_update/" +
            "multi_head_attention_conv/key_node/kernel:0"].assign(
                num_nodes * tf.eye(num_nodes))
    weights["multi_head_attention/node_set_update/" +
            "multi_head_attention_conv/value_node/kernel:0"].assign(
                num_nodes * tf.eye(num_nodes))

    # Build a Model around the Layer, possibly saved and restored.
    inputs = tf.keras.layers.Input(type_spec=gt_input.spec)
    outputs = layer(inputs)
    model = tf.keras.Model(inputs, outputs)
    if reload_model:
      export_dir = os.path.join(self.get_temp_dir(), "dropout-model")
      model.save(export_dir, include_optimizer=False)
      if reload_model == ReloadModel.KERAS:
        model = tf.keras.models.load_model(export_dir)
        # Check that from_config() worked, no fallback to a function trace, see
        # https://www.tensorflow.org/guide/keras/save_and_serialize#how_savedmodel_handles_custom_objects
        self.assertIsInstance(
            model.get_layer(index=1), tfgnn.keras.layers.GraphUpdate)
      else:
        model = tf.saved_model.load(export_dir)

    # The output is a one-hot encoding of the nodes that have been attended to.
    # For inference without dropout, it's an all-ones vector, so min = max = 1.
    # For training with dropout rate 1/3, there are some zeros (for dropped-out
    # edges) and some entries with value 3/2 (for kept edges, such that the
    # expected value remains at (1-1/3) * 3/2 == 1), so min = 0 and max = 1.5.
    def min_max(**kwargs):
      got_gt = model(gt_input, **kwargs)
      got = got_gt.node_sets["nodes"][tfgnn.HIDDEN_STATE][target_node_id]
      return [tf.reduce_min(got), tf.reduce_max(got)]

    self.assertAllEqual(min_max(), [1., 1.])  # Inference is the default.
    self.assertAllEqual(min_max(training=False), [1., 1.])
    self.assertAllClose(min_max(training=True), [0., 1.5])

  def testInputsDropout(self):
    """Tests dropout, esp. the switch between training and inference modes."""
    # Avoid flakiness.
    tf.random.set_seed(42)

    # This test graph has many source nodes feeding into one target node.
    # The node features are all ones, the edge features are one-hot encodings of
    # the source node IDs.
    num_nodes = 32
    target_node_id = 7
    gt_input = tfgnn.GraphTensor.from_pieces(
        node_sets={
            "nodes":
                tfgnn.NodeSet.from_fields(
                    sizes=[num_nodes],
                    features={
                        tfgnn.HIDDEN_STATE:
                            tf.ones(
                                shape=(num_nodes, num_nodes), dtype=tf.float32)
                    }),
        },
        edge_sets={
            "edges":
                tfgnn.EdgeSet.from_fields(
                    sizes=[num_nodes],
                    adjacency=tfgnn.Adjacency.from_indices(
                        ("nodes", tf.constant(list(range(num_nodes)))),
                        ("nodes", tf.constant([target_node_id] * num_nodes))),
                    features={tfgnn.HIDDEN_STATE: tf.eye(num_nodes)})
        })

    # On purpose, this test is not for MultiHeadAttentionConv directly,
    # but for its common usage in a GraphUpdate, to make sure the
    # training/inference mode is propagated correctly.
    inputs_dropout_rate = 0.25
    layer = multi_head_attention.MultiHeadAttentionHomGraphUpdate(
        num_heads=1,
        per_head_channels=num_nodes,
        receiver_tag=tfgnn.TARGET,
        sender_edge_feature=tfgnn.HIDDEN_STATE,
        inputs_dropout=inputs_dropout_rate,  # Note here.
        activation="linear",
        attention_activation="linear",
        use_bias=False)

    _ = layer(gt_input)  # Build weights.
    weights = {v.name: v for v in layer.trainable_weights}
    self.assertLen(weights, 5)
    # Do not transform the queries.
    weights["multi_head_attention/node_set_update/" +
            "multi_head_attention_conv/query/kernel:0"].assign(
                tf.eye(num_nodes))
    # Filter out the key edge features, keep only the node features.
    weights["multi_head_attention/node_set_update/" +
            "multi_head_attention_conv/key_node/kernel:0"].assign(
                tf.eye(num_nodes))
    weights["multi_head_attention/node_set_update/" +
            "multi_head_attention_conv/key_edge/kernel:0"].assign(
                tf.zeros(shape=(num_nodes, num_nodes), dtype=tf.float32))
    # Filter out the value node features, keep only the one-hot edge features.
    weights["multi_head_attention/node_set_update/" +
            "multi_head_attention_conv/value_node/kernel:0"].assign(
                tf.zeros(shape=(num_nodes, num_nodes), dtype=tf.float32))
    weights["multi_head_attention/node_set_update/" +
            "multi_head_attention_conv/value_edge/kernel:0"].assign(
                tf.eye(num_nodes))

    # Build a Model around the Layer, possibly saved and restored.
    inputs = tf.keras.layers.Input(type_spec=gt_input.spec)
    outputs = layer(inputs)
    model = tf.keras.Model(inputs, outputs)

    # Without dropout, i.e. during inference, The transformed queries and keys
    # should be all ones, so the softmaxe'd scores should be all equal. Since
    # the transformed values are one-hots, their scaled sum should be a vector
    # with the normalized scores.
    self.assertAllEqual(
        model(gt_input).node_sets["nodes"][tfgnn.HIDDEN_STATE][target_node_id],
        [1 / num_nodes] * num_nodes)

    # With dropout, i.e. during training, the scores will be binomially
    # distributed with n=32 and p=0.5625 (probability of neither the query nor
    # the key value to have been dropped out). Additionally, ~25% of the
    # resulting values will be zero if the one-hot encoded value was dropped
    # out.
    outputs = model(
        gt_input,
        training=True).node_sets["nodes"][tfgnn.HIDDEN_STATE][target_node_id]
    outputs = tf.boolean_mask(outputs, mask=outputs > 0)
    num_zero_outputs = num_nodes - tf.size(outputs)

    # Check that some values were dropped out completely.
    self.assertGreater(num_zero_outputs, 0)

    # Check that the stdv of the remaining values is not zero, i.e. the
    # remaining scores are not all identical.
    self.assertGreater(tf.math.reduce_std(outputs), 0.0)


def _get_test_bidi_cycle_graph(node_state, edge_state=None):
  return tfgnn.GraphTensor.from_pieces(
      node_sets={
          "nodes":
              tfgnn.NodeSet.from_fields(
                  sizes=[3], features={tfgnn.HIDDEN_STATE: node_state}),
      },
      edge_sets={
          "edges":
              tfgnn.EdgeSet.from_fields(
                  sizes=[6],
                  adjacency=tfgnn.Adjacency.from_indices(
                      ("nodes", tf.constant([0, 1, 2, 0, 2, 1])),
                      ("nodes", tf.constant([1, 2, 0, 2, 1, 0]))),
                  features=(None if edge_state is None else {
                      tfgnn.HIDDEN_STATE: edge_state
                  })),
      })


# The components of GATv2MPNNGraphUpdate have been tested elsewhere.
class MultiHeadAttentionMPNNGraphUpdateTest(tf.test.TestCase,
                                            parameterized.TestCase):

  def testBasic(self):
    input_graph = _make_test_graph_abc()
    message_dim = 6
    layer = multi_head_attention.MultiHeadAttentionMPNNGraphUpdate(
        units=1,
        message_dim=message_dim,
        num_heads=2,
        receiver_tag=tfgnn.TARGET,
        node_set_names=["b"],
        edge_feature="fab",
        kernel_initializer="ones")
    graph = layer(input_graph)
    # Nodes "a" and "c" are unchanged.
    self.assertAllEqual([[1.]], graph.node_sets["a"][tfgnn.HIDDEN_STATE])
    self.assertAllEqual([[8.]], graph.node_sets["c"][tfgnn.HIDDEN_STATE])
    # Node "b" receives message of dimension 6 from sender node and edge
    # in which all elements are 1+16=17. The hidden state is updated from
    # 2 to 2 + 6*17.
    self.assertAllEqual([[2. + message_dim * 17]],
                        graph.node_sets["b"][tfgnn.HIDDEN_STATE])

  def testMessageUnitsNotDivisible(self):
    with self.assertRaisesRegex(ValueError,
                                r"must be divisible by num_heads, got 5 and 2"):
      _ = multi_head_attention.MultiHeadAttentionMPNNGraphUpdate(
          message_dim=5, num_heads=2, units=12, receiver_tag=tfgnn.TARGET)


class MultiHeadAttentionMPNNTFLiteTest(tf.test.TestCase,
                                       parameterized.TestCase):

  @parameterized.named_parameters(
      ("Simplest", "none", False, False),
      ("TransformedKeys", "rsqrt_dim", True, False),
      ("AllOps", "trainable_sigmoid", True, True))
  def testBasic(
      self,
      score_scaling,
      transform_keys,
      transform_values_after_pooling,
  ):
    test_graph_1_dict = {
        # We care that the TFLite interpreter gives the same output as the
        # model, which was tested separately (although not for the randomly
        # initialized weights that we keep here).
        "source":
            tf.constant([0, 1, 2, 0, 2, 1]),
        "target":
            tf.constant([1, 2, 0, 2, 1, 0]),
        "node_features":
            tf.constant([
                [1., 0., 0., 1.],
                [0., 1., 0., 2.],
                [0., 0., 1., 3.],
            ]),
        "edge_features":
            tf.constant([
                [3.],
                [6.],
                [9.],
                [2.],
                [6.],
                [4.]]),
    }
    # TODO(b/276291104): Remove when TF 2.11+ is required by all of TFGNN
    if tf.__version__.startswith("2.10."):
      self.skipTest("GNN models are unsupported in TFLite until TF 2.11 but "
                    f"got TF {tf.__version__}")
    layer = multi_head_attention.MultiHeadAttentionHomGraphUpdate(
        num_heads=2,
        per_head_channels=2,
        receiver_tag=tfgnn.TARGET,
        sender_edge_feature=tfgnn.HIDDEN_STATE,  # Activate edge input.
        attention_activation="relu",
        kernel_initializer="zeros",
        kernel_regularizer=tf.keras.regularizers.L2(0.05),  # Add a regularizer.
        score_scaling=score_scaling,
        transform_keys=transform_keys,
        transform_values_after_pooling=transform_values_after_pooling,
    )

    inputs = {
        "node_features": tf.keras.Input([4], None, "node_features", tf.float32),
        "source": tf.keras.Input([], None, "source", tf.int32),
        "target": tf.keras.Input([], None, "target", tf.int32),
        "edge_features": tf.keras.Input([1], None, "edge_features", tf.float32),
    }
    graph_in = _MakeGraphTensor()(inputs)
    graph_out = layer(graph_in)
    outputs = tf.keras.layers.Layer(name="final_node_states")(
        graph_out.node_sets["nodes"][tfgnn.HIDDEN_STATE])
    model = tf.keras.Model(inputs, outputs)

    # The other unit tests should verify that this is correct
    expected = model(test_graph_1_dict).numpy()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    model_content = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=model_content)
    signature_runner = interpreter.get_signature_runner("serving_default")
    obtained = signature_runner(**test_graph_1_dict)["final_node_states"]
    self.assertAllClose(expected, obtained)


# TODO(b/274779989): Replace this layer with a more standard representation
# of GraphTensor as a dict of plain Tensors.
class _MakeGraphTensor(tf.keras.layers.Layer):
  """Makes a homogeneous GraphTensor of rank 0 with a single component."""

  def call(self, inputs):
    node_sizes = tf.shape(inputs["node_features"])[0]
    edge_sizes = tf.shape(inputs["edge_features"])[0]
    return tfgnn.GraphTensor.from_pieces(
        node_sets={
            "nodes":
                tfgnn.NodeSet.from_fields(
                    sizes=tf.expand_dims(node_sizes, axis=0),
                    features={tfgnn.HIDDEN_STATE: inputs["node_features"]})
        },
        edge_sets={
            "edges":
                tfgnn.EdgeSet.from_fields(
                    sizes=tf.expand_dims(edge_sizes, axis=0),
                    adjacency=tfgnn.Adjacency.from_indices(
                        ("nodes", inputs["source"]),
                        ("nodes", inputs["target"])),
                    features={tfgnn.HIDDEN_STATE: inputs["edge_features"]})
        })


def _make_test_graph_abc():
  return tfgnn.GraphTensor.from_pieces(
      node_sets={
          "a":
              tfgnn.NodeSet.from_fields(
                  sizes=tf.constant([1]),
                  features={tfgnn.HIDDEN_STATE: tf.constant([[1.]])}),
          "b":
              tfgnn.NodeSet.from_fields(
                  sizes=tf.constant([1]),
                  features={tfgnn.HIDDEN_STATE: tf.constant([[2.]])}),
          "c":
              tfgnn.NodeSet.from_fields(
                  sizes=tf.constant([1]),
                  features={tfgnn.HIDDEN_STATE: tf.constant([[8.]])})
      },
      edge_sets={
          "a->b":
              tfgnn.EdgeSet.from_fields(
                  sizes=tf.constant([1]),
                  adjacency=tfgnn.Adjacency.from_indices(
                      ("a", tf.constant([0])), ("b", tf.constant([0]))),
                  features={"fab": tf.constant([[16.]])}),
          "c->c":
              tfgnn.EdgeSet.from_fields(
                  sizes=tf.constant([1]),
                  adjacency=tfgnn.Adjacency.from_indices(
                      ("c", tf.constant([0])), ("c", tf.constant([0]))))
      })


if __name__ == "__main__":
  tf.test.main()
