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
"""Tests for GATv2."""

import enum
import os

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import gat_v2


class ReloadModel(int, enum.Enum):
  """Controls how to reload a model for further testing after saving."""

  SKIP = 0
  SAVED_MODEL = 1
  KERAS = 2


class GATv2Test(tf.test.TestCase, parameterized.TestCase):

  def testBasic(self):
    """Tests that a single-headed GAT is correct given predefined weights."""
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
            [[1.0, 0.0, 0.0, 1.0],
             [0.0, 1.0, 0.0, 2.0],
             [0.0, 0.0, 1.0, 3.0]]
        )
    )

    conv = gat_v2.GATv2Conv(
        num_heads=1,
        per_head_channels=4,
        receiver_tag=tfgnn.TARGET,
        attention_activation="relu",  # Let's keep it simple.
    )

    _ = conv(gt_input, edge_set_name="edges")  # Build weights.
    weights = {v.name: v for v in conv.trainable_weights}
    self.assertLen(weights, 5)
    weights["gat_v2_conv/query/kernel:0"].assign(
        # The space of attention computation of the single head has dimension 4.
        # The last dimension is used only in the key to carry the node's value,
        # multiplied by 11/10.
        # The first three dimensions are used to hand-construct attention scores
        # (see the running example below) that favor the counterclockwise
        # incoming edge over the other. Recall that weight matrices are
        # multiplied from the right onto batched inputs (in rows).
        #
        # For example, the query vector of node 0 is [0, 1, 0, 0], and ...
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    weights["gat_v2_conv/value_node/kernel:0"].assign(
        # ... the key vectors of node 1 and 2, resp., are [-1, 0, -1, 2.2]
        # and [-1, -1, 0, 3.3]. Therefore, ...
        [
            [0.0, -1.0, -1.0, 0.0],
            [-1.0, 0.0, -1.0, 0.0],
            [-1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.1],
        ]
    )
    log10 = tf.math.log(10.0).numpy()
    weights["gat_v2_conv/attn_logits/kernel:0"].assign(
        # ... attention from node 0 to node 1 has a sum of key and query vector
        # [-1, 1, -1, 2.2], which gets turned by ReLU and the attention weights
        # below into a pre-softmax score of log(10). Likewise,
        # attention from node 0 to node 2 has a vector sum [-1, 0, 0, 3.3]
        # and pre-softmax score of 0. Altogether, this means: ...
        [[log10], [log10], [log10], [0.0]]
    )
    weights["gat_v2_conv/query/bias:0"].assign([0.0, 0.0, 0.0, 0.0])
    weights["gat_v2_conv/value_node/bias:0"].assign([0.0, 0.0, 0.0, 0.0])

    got = conv(gt_input, edge_set_name="edges")

    # ... The softmax-weighed key vectors on the incoming edges of node 0
    # are  10/11 * [-1, 0, -1, 2.2]  +  1/11 * [-1, -1, 0, 3.3].
    # The final ReLU takes out the non-positive components and leaves 2 + 0.3
    # in the last component of the first row in the resulting node states.
    want = tf.constant([
        [0.0, 0.0, 0.0, 2.3],  # Node 0.
        [0.0, 0.0, 0.0, 3.1],  # Node 1.
        [0.0, 0.0, 0.0, 1.2],
    ])  # Node 2.
    self.assertAllEqual(got.shape, (3, 4))
    self.assertAllClose(got, want, atol=0.0001)

    # For node states with more than one feature dimension, GATv2 works in
    # parallel on the vectors from the innermost dimension, so we can repeat the
    # previous computation and an alternative with different values in the last
    # component and reversed orientation:
    gt_input_2 = _get_test_bidi_cycle_graph(
        tf.constant([
            [[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 3.0]],
            [[0.0, 1.0, 0.0, 2.0], [0.0, 1.0, 0.0, 6.0]],
            [[0.0, 0.0, 1.0, 3.0], [1.0, 0.0, 0.0, 9.0]],
        ])
    )
    got_2 = conv(gt_input_2, edge_set_name="edges")
    want_2 = tf.constant([
        [[0.0, 0.0, 0.0, 2.3], [0.0, 0.0, 0.0, 9.6]],
        [[0.0, 0.0, 0.0, 3.1], [0.0, 0.0, 0.0, 3.9]],
        [[0.0, 0.0, 0.0, 1.2], [0.0, 0.0, 0.0, 6.3]],
    ])
    self.assertAllEqual(got_2.shape, (3, 2, 4))
    self.assertAllClose(got_2, want_2, atol=0.0001)

  @parameterized.named_parameters(
      ("ConcatMerge", "concat"), ("MeanMerge", "mean")
  )
  def testMultihead(self, merge_type):
    """Extends testBasic with multiple attention heads."""
    # The same test graph as in the testBasic above.
    gt_input = _get_test_bidi_cycle_graph(
        tf.constant(
            [[1.0, 0.0, 0.0, 1.0],
             [0.0, 1.0, 0.0, -2.0],
             [0.0, 0.0, 1.0, 3.0]]
        )
    )

    conv = gat_v2.GATv2Conv(
        num_heads=2,
        per_head_channels=4,
        receiver_tag=tfgnn.TARGET,
        attention_activation=tf.keras.layers.LeakyReLU(alpha=0.0),
        use_bias=False,  # Don't create /bias variables.
        heads_merge_type=merge_type,
    )

    _ = conv(gt_input, edge_set_name="edges")  # Build weights.
    weights = {v.name: v for v in conv.trainable_weights}
    self.assertLen(weights, 3)

    weights["gat_v2_conv/query/kernel:0"].assign(
        # Attention head 0 uses the first four dimensions, which are used
        # in the same way as for the testBasic test above.
        # Attention head 1 uses the last four dimensions, in which we
        # now favor the clockwise incoming edges and omit the scaling by 11/10.
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    weights["gat_v2_conv/value_node/kernel:0"].assign([
        [0.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0],
        [-1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0],
        [-1.0, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.0],
    ])
    log10 = tf.math.log(10.0).numpy()
    weights["gat_v2_conv/attn_logits/kernel:0"].assign(
        # Attention head 0 works out to softmax weights 10/11 and 1/11 as above.
        # Attention head 1 creates very large pre-softmax scores that
        # work out to weights 1 and 0 within floating-point precision.
        [[log10, 100.0], [log10, 100.0], [log10, 100.0], [0.0, 0.0]]
    )

    got = conv(gt_input, edge_set_name="edges")

    # Attention head 0 generates the first four output dimensions as in the
    # testBasic above, with weights 10/11 and 1/11,
    # Attention head 1 uses weights 0 and 1 (note the reversed preference).
    want_logits = tf.constant([
        [0.0, 0.0, 0.0, -2 + 0.3, 0.0, 0.0, 0.0, 3.0],
        [0.0, 0.0, 0.0, 3 + 0.1, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1 - 0.2, 0.0, 0.0, 0.0, -2.0],
    ])

    if merge_type == "mean":
      # Expect mean of heads, followed by activation.
      want = tf.nn.relu((want_logits[:, :4] + want_logits[:, 4:]) / 2)
    elif merge_type == "concat":
      want = tf.nn.relu(want_logits)

    self.assertAllClose(got, want, atol=0.0001)

  @parameterized.named_parameters(
      ("", ReloadModel.SKIP),
      ("Restored", ReloadModel.SAVED_MODEL),
      ("RestoredKeras", ReloadModel.KERAS),
  )
  def testFullModel(self, reload_model):
    """Tests GATv2HomGraphUpdate in Model (incl. saving) with edge input."""
    # The same example as in the testBasic above, but with extra inputs
    # from edges.
    gt_input = _get_test_bidi_cycle_graph(
        # Node i has value i+1 in the last component, which will be mapped
        # into the fourth component of the value.
        tf.constant([
            [1.0, 0.0, 0.0, 1.0],  # Node 0.
            [0.0, 1.0, 0.0, 2.0],  # Node 1.
            [0.0, 0.0, 1.0, 3.0],  # Node 2.
        ]),
        # Edges out of node i have value 2*(i+1) for the clockwise edges favored
        # by attention and 3*(i+1) for the counterclockwise edges not favored.
        tf.constant([
            [3.0],  # Edge from node 0 (clockwise, not favored).
            [6.0],  # Edge from node 1 (clockwise, not favored).
            [9.0],  # Edge from node 2 (clockwise, not favored).
            [2.0],  # Edge from node 0 (counterclockwise, favored).
            [6.0],  # Edge from node 2 (counterclockwise, favored).
            [4.0],  # Edge from node 1 (counterclockwise, favored).
        ]),
    )

    l2reg = 1e-2
    layer = gat_v2.GATv2HomGraphUpdate(
        num_heads=1,
        per_head_channels=5,
        receiver_tag=tfgnn.TARGET,
        sender_edge_feature=tfgnn.HIDDEN_STATE,  # Activate edge input.
        kernel_regularizer=tf.keras.regularizers.l2(l2reg),
        # Test with a non-None initializer for b/238163789.
        kernel_initializer=tf.keras.initializers.GlorotNormal(),
        attention_activation="relu",
    )

    _ = layer(gt_input)  # Build weights.
    weights = {v.name: v for v in layer.trainable_weights}
    self.assertLen(weights, 6)
    weights["gat_v2/node_set_update/gat_v2_conv/value_edge/kernel:0"].assign(
        # Edge values are put into a new final component of the value space,
        # with the same adjustment for softmax-weighting by 1/11 or 10/11.
        [[0.0, 0.0, 0.0, 0.0, 1.1]]
    )
    weights["gat_v2/node_set_update/gat_v2_conv/query/kernel:0"].assign([
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    weights["gat_v2/node_set_update/gat_v2_conv/value_node/kernel:0"].assign([
        [0.0, -1.0, -1.0, 0.0, 0.0],
        [-1.0, 0.0, -1.0, 0.0, 0.0],
        [-1.0, -1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.1, 0.0],
    ])
    log10 = tf.math.log(10.0).numpy()
    weights["gat_v2/node_set_update/gat_v2_conv/attn_logits/kernel:0"].assign(
        [[log10], [log10], [log10], [0.0], [0.0]]
    )
    weights["gat_v2/node_set_update/gat_v2_conv/query/bias:0"].assign(
        [0.0, 0.0, 0.0, 0.0, 0.0]
    )
    # NOTE: There is value_node/bias but no redundant value_edge/bias.
    weights["gat_v2/node_set_update/gat_v2_conv/value_node/bias:0"].assign(
        [0.0, 0.0, 0.0, 0.0, 0.0]
    )

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
            model.get_layer(index=1), tfgnn.keras.layers.GraphUpdate
        )
      else:
        model = tf.saved_model.load(export_dir)  # Gives _UserObject

    got_gt = model(gt_input)
    got = got_gt.node_sets["nodes"][tfgnn.HIDDEN_STATE]

    # Verify kernel regularization is attached on model kernel variables.
    if not reload_model or reload_model == ReloadModel.KERAS:
      # Model.losses only works on Keras models. tf.saved_model.load(), however,
      # does not return a Keras model. See:
      # https://www.tensorflow.org/api_docs/python/tf/saved_model/load
      kernel_variables = [
          v for v in model.trainable_variables if "/kernel:0" in v.name
      ]
      self.assertLen(kernel_variables, 4)  # 4 kernel variables per `weights[]`.
      self.assertLen(model.losses, 4)  # One loss term per kernel variable.
      expected_model_losses = [
          tf.reduce_sum(v**2) * l2reg for v in kernel_variables
      ]
      self.assertAllClose(model.losses, expected_model_losses)

    # The fourth column with values x.y from nodes is analogous to the
    # testBasic test above, with the contribution x from the favored
    # input before the decimal dot and the other contribution y after.
    # The fifth column with values (2x).(3y) is from edges, with the
    # multipliers 2 and 3 used above in setting up the edge features.
    want = tf.constant([
        [0.0, 0.0, 0.0, 2.3, 4.9],
        [0.0, 0.0, 0.0, 3.1, 6.3],
        [0.0, 0.0, 0.0, 1.2, 2.6],
    ])
    self.assertAllEqual(got.shape, (3, 5))
    self.assertAllClose(got, want, atol=0.0001)

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
            "sources": tfgnn.NodeSet.from_fields(
                sizes=[2, 2],
                features={
                    tfgnn.HIDDEN_STATE: tf.constant(
                        [[1.0, 0.0, 0.0, 1.0],  # Repeated for both components.
                         [0.0, 1.0, 0.0, 2.0]] * 2
                    )
                },
            ),
            "targets": tfgnn.NodeSet.from_fields(
                sizes=[1, 1],
                features={
                    tfgnn.HIDDEN_STATE: tf.constant(
                        [[0.0, 0.0, 1.0, 3.0], [1.0, 0.0, 0.0, 4.0]]
                    )
                },
            ),
        },
        context=tfgnn.Context.from_fields(
            # Same as "targets".
            features={
                tfgnn.HIDDEN_STATE: tf.constant(
                    [[0.0, 0.0, 1.0, 3.0], [1.0, 0.0, 0.0, 4.0]]
                )
            }
        ),
        edge_sets={
            "edges": tfgnn.EdgeSet.from_fields(
                sizes=[2, 2],
                # Same as membership of "sources" in components.
                adjacency=tfgnn.Adjacency.from_indices(
                    ("sources", tf.constant([0, 1, 2, 3])),
                    ("targets", tf.constant([0, 0, 1, 1])),
                ),
            )
        },
    )

    conv = gat_v2.GATv2Conv(
        num_heads=1,
        per_head_channels=4,
        attention_activation="relu",
        use_bias=False,
    )

    # Build weights, then initialize as in testBasic.
    _ = conv(gt_input, node_set_name="sources", receiver_tag=tfgnn.CONTEXT)
    weights = {v.name: v for v in conv.trainable_weights}
    self.assertLen(weights, 3)
    weights["gat_v2_conv/query/kernel:0"].assign([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    weights["gat_v2_conv/value_node/kernel:0"].assign([
        [0.0, -1.0, -1.0, 0.0],
        [-1.0, 0.0, -1.0, 0.0],
        [-1.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.1],
    ])
    log10 = tf.math.log(10.0).numpy()
    weights["gat_v2_conv/attn_logits/kernel:0"].assign(
        [[log10], [log10], [log10], [0.0]]
    )

    # The convolution object can be called interchangeably for convolving
    # "sources" to context, or along "edges" to "targets" with the same
    # features as context.
    got_context = conv(
        gt_input, node_set_name="sources", receiver_tag=tfgnn.CONTEXT
    )
    got_targets = conv(
        gt_input, edge_set_name="edges", receiver_tag=tfgnn.TARGET
    )
    want = tf.constant([
        [0.0, 0.0, 0.0, 1.2],  # As in testBasic for node 2.
        [0.0, 0.0, 0.0, 2.1],  # Opposite preference.
    ])
    self.assertAllClose(got_context, want, atol=0.0001)
    self.assertAllClose(got_targets, want, atol=0.0001)

    # The same object can even be used for convolving over the edge set in
    # reverse direction, that is, to "sources". The result is boring, though:
    # Every "source" gets the same value from the sole "target" of the component
    # (so softmax reduces to a no-op), which is scaled with 1.1 by the
    # bottom-right element of value_node/kernel.
    got_sources = conv(
        gt_input, edge_set_name="edges", receiver_tag=tfgnn.SOURCE
    )
    want_sources = tf.constant([
        [0.0, 0.0, 0.0, 3.3],
        [0.0, 0.0, 0.0, 3.3],
        [0.0, 0.0, 0.0, 4.4],
        [0.0, 0.0, 0.0, 4.4],
    ])
    self.assertAllClose(got_sources, want_sources, atol=0.0001)

  def testEdgePoolReceivers(self):
    """Tests GATv2EdgePool for pooling to nodes and to context."""
    # This example is similar to testConvolutionReceivers, except that
    # the sender features are now on the edges, not the source nodes.
    gt_input = tfgnn.GraphTensor.from_pieces(
        edge_sets={
            "edges": tfgnn.EdgeSet.from_fields(
                sizes=[2, 2],
                adjacency=tfgnn.Adjacency.from_indices(
                    ("sources", tf.constant([0, 1, 2, 3])),
                    ("targets", tf.constant([0, 0, 1, 1])),
                ),
                features={
                    tfgnn.HIDDEN_STATE: tf.constant(
                        [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0]] * 2
                    )
                },
            ),  # Repeated for both components.
        },
        node_sets={
            "sources": tfgnn.NodeSet.from_fields(sizes=[2, 2]),  # No features.
            "targets": tfgnn.NodeSet.from_fields(
                sizes=[1, 1],
                features={
                    tfgnn.HIDDEN_STATE: tf.constant(
                        [[0.0, 0.0, 1.0, 3.0], [1.0, 0.0, 0.0, 4.0]]
                    )
                },
            ),
        },
        context=tfgnn.Context.from_fields(
            # Same as "targets".
            features={
                tfgnn.HIDDEN_STATE: tf.constant(
                    [[0.0, 0.0, 1.0, 3.0], [1.0, 0.0, 0.0, 4.0]]
                )
            }
        ),
    )

    layer = gat_v2.GATv2EdgePool(
        num_heads=1, per_head_channels=4, attention_activation="relu"
    )

    # Build weights, then initialize analogous to testConvolutionReceivers,
    # with "value_edge" instead of "value_node".
    _ = layer(gt_input, edge_set_name="edges", receiver_tag=tfgnn.CONTEXT)
    weights = {v.name: v for v in layer.trainable_weights}
    self.assertLen(weights, 5)
    weights["gat_v2_edge_pool/query/kernel:0"].assign([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    weights["gat_v2_edge_pool/value_edge/kernel:0"].assign([
        [0.0, -1.0, -1.0, 0.0],
        [-1.0, 0.0, -1.0, 0.0],
        [-1.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.1],
    ])
    log10 = tf.math.log(10.0).numpy()
    weights["gat_v2_edge_pool/attn_logits/kernel:0"].assign(
        [[log10], [log10], [log10], [0.0]]
    )
    weights["gat_v2_edge_pool/query/bias:0"].assign([0.0, 0.0, 0.0, 0.0])
    # NOTE: There is a value_edge/bias but not value_node/bias.
    weights["gat_v2_edge_pool/value_edge/bias:0"].assign([0.0, 0.0, 0.0, 0.0])

    # The EdgePool object can be called interchangeably for attention-pooling
    # the "edges" to context or to each component's unique node in "targets".
    got_context = layer(
        gt_input, edge_set_name="edges", receiver_tag=tfgnn.CONTEXT
    )
    got_targets = layer(
        gt_input, edge_set_name="edges", receiver_tag=tfgnn.TARGET
    )
    want = tf.constant([[0.0, 0.0, 0.0, 1.2], [0.0, 0.0, 0.0, 2.1]])
    self.assertAllClose(got_context, want, atol=0.0001)
    self.assertAllClose(got_targets, want, atol=0.0001)

  @parameterized.named_parameters(
      ("", ReloadModel.SKIP),
      ("Restored", ReloadModel.SAVED_MODEL),
      ("RestoredKeras", ReloadModel.KERAS),
  )
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
            "nodes": tfgnn.NodeSet.from_fields(
                sizes=[num_nodes],
                features={tfgnn.HIDDEN_STATE: tf.eye(num_nodes)},
            ),
        },
        edge_sets={
            "edges": tfgnn.EdgeSet.from_fields(
                sizes=[num_nodes],
                adjacency=tfgnn.Adjacency.from_indices(
                    ("nodes", tf.constant(list(range(num_nodes)))),
                    ("nodes", tf.constant([target_node_id] * num_nodes)),
                ),
            )
        },
    )

    # On purpose, this test is not for GATv2Conv directly, but for its
    # common usage in a GraphUpdate, to make sure the training/inference mode
    # is propagated correctly.
    layer = gat_v2.GATv2HomGraphUpdate(
        num_heads=1,
        per_head_channels=num_nodes,
        receiver_tag=tfgnn.TARGET,
        edge_dropout=1.0 / 3.0,  # Note here.
        activation="linear",
        attention_activation="linear",
        use_bias=False,
    )

    _ = layer(gt_input)  # Build weights.
    weights = {v.name: v for v in layer.trainable_weights}
    self.assertLen(weights, 3)
    # Set up equal attention to all inputs.
    weights["gat_v2/node_set_update/gat_v2_conv/query/kernel:0"].assign(
        [[0.0] * num_nodes] * num_nodes
    )
    weights["gat_v2/node_set_update/gat_v2_conv/attn_logits/kernel:0"].assign(
        [[0.0]] * num_nodes
    )
    # Values are one-hot node ids, scaled up to undo the softmax normalization.
    weights["gat_v2/node_set_update/gat_v2_conv/value_node/kernel:0"].assign(
        num_nodes * tf.eye(num_nodes)
    )

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
            model.get_layer(index=1), tfgnn.keras.layers.GraphUpdate
        )
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

    self.assertAllEqual(min_max(), [1.0, 1.0])  # Inference is the default.
    self.assertAllEqual(min_max(training=False), [1.0, 1.0])
    self.assertAllClose(min_max(training=True), [0.0, 1.5])


def _get_test_bidi_cycle_graph(node_state, edge_state=None):
  return tfgnn.GraphTensor.from_pieces(
      node_sets={
          "nodes": tfgnn.NodeSet.from_fields(
              sizes=[3], features={tfgnn.HIDDEN_STATE: node_state}
          ),
      },
      edge_sets={
          "edges": tfgnn.EdgeSet.from_fields(
              sizes=[6],
              adjacency=tfgnn.Adjacency.from_indices(
                  ("nodes", tf.constant([0, 1, 2, 0, 2, 1])),
                  ("nodes", tf.constant([1, 2, 0, 2, 1, 0])),
              ),
              features=(
                  None
                  if edge_state is None
                  else {tfgnn.HIDDEN_STATE: edge_state}
              ),
          ),
      },
  )


# The components of GATv2MPNNGraphUpdate have been tested elsewhere.
class GATv2MPNNGraphUpdateTest(tf.test.TestCase, parameterized.TestCase):

  def testBasic(self):
    input_graph = _make_test_graph_abc()
    message_dim = 6
    layer = gat_v2.GATv2MPNNGraphUpdate(
        units=1,
        message_dim=message_dim,
        num_heads=2,
        receiver_tag=tfgnn.TARGET,
        node_set_names=["b"],
        edge_feature="fab",
        kernel_initializer="ones",
    )
    graph = layer(input_graph)
    # Nodes "a" and "c" are unchanged.
    self.assertAllEqual([[1.0]], graph.node_sets["a"][tfgnn.HIDDEN_STATE])
    self.assertAllEqual([[8.0]], graph.node_sets["c"][tfgnn.HIDDEN_STATE])
    # Node "b" receives message of dimension 6 from sender node and edge
    # in which all elements are 1+16=17. The hidden state is updated from
    # 2 to 2 + 6*17.
    self.assertAllEqual(
        [[2.0 + message_dim * 17]], graph.node_sets["b"][tfgnn.HIDDEN_STATE]
    )

  def testMessageUnitsNotDivisible(self):
    with self.assertRaisesRegex(
        ValueError, r"must be divisible by num_heads, got 5 and 2"
    ):
      _ = gat_v2.GATv2MPNNGraphUpdate(
          message_dim=5, num_heads=2, units=12, receiver_tag=tfgnn.SOURCE
      )


class GATv2TFLiteTest(tf.test.TestCase, parameterized.TestCase):

  def testBasic(self):
    test_graph_1_dict = {
        # We care that the TFLite interpreter gives the same output as the
        # model, which was tested separately (although not for the randomly
        # initialized weights that we keep here).
        "source": tf.constant([0, 1, 2, 0, 2, 1]),
        "target": tf.constant([1, 2, 0, 2, 1, 0]),
        "node_features": tf.constant([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
        ]),
        "edge_features": tf.constant([
            [3.0],
            [6.0],
            [9.0],
            [2.0],
            [6.0],
            [4.0]]),
    }

    layer = gat_v2.GATv2MPNNGraphUpdate(
        units=4,
        message_dim=4,
        num_heads=2,
        receiver_tag=tfgnn.TARGET,
        node_set_names=["nodes"],
        kernel_initializer="ones",
        edge_feature=tfgnn.HIDDEN_STATE,
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
        graph_out.node_sets["nodes"][tfgnn.HIDDEN_STATE]
    )
    model = tf.keras.Model(inputs, outputs)

    # The other unit tests should verify that this is correct
    expected = model(test_graph_1_dict).numpy()

    # TODO(b/276291104): Remove when TF 2.11+ is required by all of TFGNN
    if tf.__version__.startswith("2.10."):
      self.skipTest("GNN models are unsupported in TFLite until TF 2.11 but "
                    f"got TF {tf.__version__}")
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
            "nodes": tfgnn.NodeSet.from_fields(
                sizes=tf.expand_dims(node_sizes, axis=0),
                features={tfgnn.HIDDEN_STATE: inputs["node_features"]},
            )
        },
        edge_sets={
            "edges": tfgnn.EdgeSet.from_fields(
                sizes=tf.expand_dims(edge_sizes, axis=0),
                adjacency=tfgnn.Adjacency.from_indices(
                    ("nodes", inputs["source"]), ("nodes", inputs["target"])
                ),
                features={tfgnn.HIDDEN_STATE: inputs["edge_features"]},
            )
        },
    )


def _make_test_graph_abc():
  return tfgnn.GraphTensor.from_pieces(
      node_sets={
          "a": tfgnn.NodeSet.from_fields(
              sizes=tf.constant([1]),
              features={tfgnn.HIDDEN_STATE: tf.constant([[1.0]])},
          ),
          "b": tfgnn.NodeSet.from_fields(
              sizes=tf.constant([1]),
              features={tfgnn.HIDDEN_STATE: tf.constant([[2.0]])},
          ),
          "c": tfgnn.NodeSet.from_fields(
              sizes=tf.constant([1]),
              features={tfgnn.HIDDEN_STATE: tf.constant([[8.0]])},
          ),
      },
      edge_sets={
          "a->b": tfgnn.EdgeSet.from_fields(
              sizes=tf.constant([1]),
              adjacency=tfgnn.Adjacency.from_indices(
                  ("a", tf.constant([0])), ("b", tf.constant([0]))
              ),
              features={"fab": tf.constant([[16.0]])},
          ),
          "c->c": tfgnn.EdgeSet.from_fields(
              sizes=tf.constant([1]),
              adjacency=tfgnn.Adjacency.from_indices(
                  ("c", tf.constant([0])), ("c", tf.constant([0]))
              ),
          ),
      },
  )


if __name__ == "__main__":
  tf.test.main()
