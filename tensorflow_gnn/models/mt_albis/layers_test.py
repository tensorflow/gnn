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
"""Tests for Model Template "Albis"."""

import enum
import json
import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models.mt_albis import layers


class ReloadModel(int, enum.Enum):
  """Controls how to reload a model for further testing after saving."""
  SKIP = 0
  SAVED_MODEL = 1
  KERAS = 2


class MtAlbisNextNodeStateTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("Basic", "concat", "none", "dense", "linear", False, ReloadModel.SKIP),
      ("BasicRestored", "concat", "none", "dense", "linear", False,
       ReloadModel.SAVED_MODEL),
      ("BasicRestoredKeras", "concat", "none", "dense", "linear", False,
       ReloadModel.KERAS),
      ("EdgeSum", "sum", "none", "dense", "linear", False, ReloadModel.SKIP),
      ("LayerNorm", "concat", "layer", "dense", "linear", False,
       ReloadModel.SKIP),
      ("Residual", "concat", "none", "residual", "linear", False,
       ReloadModel.SKIP),
      ("Relu", "concat", "none", "dense", "relu", False, ReloadModel.SKIP),
      ("Context", "concat", "none", "dense", "linear", True, ReloadModel.SKIP),
      ("AllOptions", "sum", "layer", "residual", "relu", True,
       ReloadModel.SKIP),
      ("AllOptionsRestored", "sum", "layer", "residual", "relu", True,
       ReloadModel.SAVED_MODEL),
      ("AllOptionsRestoredKeras", "sum", "layer", "residual", "relu", True,
       ReloadModel.KERAS),
  )
  def test(self, edge_set_combine_type, normalization_type, next_state_type,
           activation, use_context, reload_model):
    """Tests computation of MtAlbisNextNodeState."""
    input_graph = _make_test_graph_abuv()

    # Build a test model that contains a MtAlbisNextNodeState in its natural
    # place: an EdgeSetUpdate layer.
    pool = tfgnn.keras.layers.Pool(tfgnn.SOURCE, "sum")
    layer = tfgnn.keras.layers.NodeSetUpdate(
        {"u": pool, "v": pool},
        layers.MtAlbisNextNodeState(
            4,
            next_state_type=next_state_type,
            dropout_rate=0.0,
            normalization_type=normalization_type,
            edge_set_combine_type=edge_set_combine_type,
            activation=activation,
        ),
        context_input_feature="cf" if use_context else None)
    inputs = tf.keras.layers.Input(type_spec=input_graph.spec)
    outputs = layer(inputs, node_set_name="a")
    model = tf.keras.models.Model(inputs, outputs)
    _ = model(input_graph)  # Trigger the actual building.

    # Initialize weights for predictable results.
    weights = {v.name: v for v in model.trainable_weights}
    self.assertLen(weights,
                   2 if normalization_type == "none" else 4)
    weights["node_set_update/mt_albis_next_node_state/dense/kernel:0"].assign(
        # 4 rows for the old state [3, 1, 1, 1].
        [[2., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]] +
        # 4 or 2 rows for the concat or summed edge inputs [1, 0], [1, 1].
        ([[0., 4., 0., 0.],
          [0., 3., 0., 0.]] if edge_set_combine_type == "concat" else []) +
        [[0., 2., 0., 0.],
         [0., 1., 0., 0.]] +
        # 0 or 1 rows for the context input [2].
        ([[.25, 0., 0., 0.]] if use_context else []))
    weights["node_set_update/mt_albis_next_node_state/dense/bias:0"].assign(
        # Put extreme values in the two unused final positions
        # to achieve predictable signs after normalization (if any).
        [0., 0., -10., 10.])
    if normalization_type == "layer":
      # Let LayerNorm do only the normalization part.
      weights[
          "node_set_update/mt_albis_next_node_state/layer_normalization/beta:0"
      ].assign([0., 0., 0., 0.])
      weights[
          "node_set_update/mt_albis_next_node_state/layer_normalization/gamma:0"
      ].assign([1., 1., 1., 1.])

    # Optionally test with a round-trip through serialization.
    model = self._maybe_reload_model(model, reload_model, "basic-next-state")

    expected = np.array([[6. if not use_context else 6.5,
                          7. if edge_set_combine_type == "concat" else 5.,
                          -10.,  # Negative; even after normalization.
                          10.]])
    if normalization_type == "layer":
      expected = _normalize(expected).numpy()
    if next_state_type == "residual":
      expected += input_graph.node_sets["a"][tfgnn.HIDDEN_STATE].numpy()
    if activation == "relu":
      expected[0, 2] = 0.  # Zero out the negative element.
    self.assertAllClose(expected, model(input_graph), rtol=1e-5)

  @parameterized.named_parameters(
      ("", ReloadModel.SKIP),
      ("Restored", ReloadModel.SAVED_MODEL),
      ("RestoredKeras", ReloadModel.KERAS),
  )
  def testDropout(self, reload_model):
    """Tests dropout, esp. the switch between training and inference modes."""
    # Avoid flakiness.
    tf.random.set_seed(42)

    input_graph = _make_test_graph_abuv()

    # Build a test model that contains a MtAlbisNextNodeState in its natural
    # place: an EdgeSetUpdate layer.
    pool = tfgnn.keras.layers.Pool(tfgnn.SOURCE, "sum")
    layer = tfgnn.keras.layers.NodeSetUpdate(
        {"u": pool, "v": pool},
        layers.MtAlbisNextNodeState(
            32,
            next_state_type="dense",
            dropout_rate=1./3,
            normalization_type="none"))
    inputs = tf.keras.layers.Input(type_spec=input_graph.spec)
    outputs = layer(inputs, node_set_name="a")
    model = tf.keras.models.Model(inputs, outputs)
    _ = model(input_graph)  # Trigger the actual building.

    # Initialize weights for predictable results.
    weights = {v.name: v for v in model.trainable_weights}
    self.assertLen(weights, 2)
    weights["node_set_update/mt_albis_next_node_state/dense/kernel:0"].assign(
        tf.zeros([8, 32]))
    weights["node_set_update/mt_albis_next_node_state/dense/bias:0"].assign(
        tf.ones([32]))

    model = self._maybe_reload_model(model, reload_model, "dropout-next-state")

    def min_max(x):
      return [tf.reduce_min(x), tf.reduce_max(x)]

    # In inference mode (the default), dropout does nothing to the all-ones
    # input, so min = max = 1.
    self.assertAllEqual([1., 1.], min_max(model(input_graph)))
    self.assertAllEqual([1., 1.], min_max(model(input_graph, training=False)))
    # In training mode, each input is zeroed out independently at random with
    # a probability of 1/3, and the remaining entries are scaled up to 3/2, such
    # that the expected value remains at (1-1/3) * 3/2 == 1. The odds of seeing
    # no zeroes in any of the 32 positions is 2/3^32 < 2e-15, and this depends
    # only on the seed (not the particular run of this test). The odds of seeing
    # only zeros is even smaller. Hence with very high probability we will see
    # min = 0 and max = 3/2.
    self.assertAllClose([0., 1.5], min_max(model(input_graph, training=True)))

  # TODO(b/265776928): Maybe refactor to share.
  def _maybe_reload_model(self, model: tf.keras.Model,
                          reload_model: ReloadModel, subdir_name: str):
    if reload_model == ReloadModel.SKIP:
      return model
    export_dir = os.path.join(self.get_temp_dir(), subdir_name)
    model.save(export_dir, include_optimizer=False,
               save_traces=(reload_model == ReloadModel.SAVED_MODEL)
               )
    if reload_model == ReloadModel.SAVED_MODEL:
      return tf.saved_model.load(export_dir)
    elif reload_model == ReloadModel.KERAS:
      return tf.keras.models.load_model(export_dir)


def _normalize(x):
  mean = tf.math.reduce_mean(x, axis=-1, keepdims=True)
  stddev = tf.math.reduce_std(x, axis=-1, keepdims=True)
  return tf.math.divide(tf.math.subtract(x, mean), stddev)


class MtAlbisGraphUpdateTest(tf.test.TestCase, parameterized.TestCase):

  def testConvType(self):
    """Tests the selection of convolution type."""
    input_graph = _make_test_graph_abuv()
    layer = layers.MtAlbisGraphUpdate(
        units=16,
        message_dim=8,
        receiver_tag=tfgnn.SOURCE,
        attention_type="multi_head",
        attention_edge_set_names=["v"],  # ... but not "u".
    )
    inputs = tf.keras.layers.Input(type_spec=input_graph.spec)
    outputs = layer(inputs)
    model = tf.keras.models.Model(inputs, outputs)
    _ = model(input_graph)
    config = json.loads(model.to_json())
    node_set_config = config["config"]["layers"][1]["config"]["node_sets/a"]

    # Check for `class_name` in `node_set_config` is serialization format
    # dependent. This conditional allows for compatibility with multiple
    # versions of OSS users.
    if node_set_config["config"]["edge_set_inputs/u"].get(
        "registered_name", None
    ):
      self.assertEqual(
          node_set_config["config"]["edge_set_inputs/u"]["class_name"],
          "SimpleConv",
      )
      self.assertEqual(
          node_set_config["config"]["edge_set_inputs/v"]["class_name"],
          "MultiHeadAttentionConv",
      )
    else:
      self.assertEqual(
          node_set_config["config"]["edge_set_inputs/u"]["class_name"],
          "GNN>SimpleConv",
      )
      self.assertEqual(
          node_set_config["config"]["edge_set_inputs/v"]["class_name"],
          "GNN>models>multi_head_attention>MultiHeadAttentionConv",
      )


class MtAlbisTFLiteTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("SumDenseNoNorm", "sum", "dense", "none"),
      ("ConcatResidualLayerNorm", "concat", "residual", "layer"),
  )
  def test(self,
           edge_set_combine_type,
           next_state_type,
           normalization_type):
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
    units = 4
    layer = layers.MtAlbisGraphUpdate(
        units=units,
        message_dim=units,
        receiver_tag=tfgnn.SOURCE,
        # The Conv classes chosen by other attention_type values are presumed
        # to have their own TFLite integration tests.
        attention_type="none",
        normalization_type=normalization_type,
        next_state_type=next_state_type,
        edge_set_combine_type=edge_set_combine_type,
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


def _make_test_graph_abuv():
  return tfgnn.GraphTensor.from_pieces(
      node_sets={
          "a": tfgnn.NodeSet.from_fields(
              sizes=tf.constant([1]),
              features={tfgnn.HIDDEN_STATE: tf.constant([[3., 1., 1., 1.]])}),
          "b": tfgnn.NodeSet.from_fields(
              sizes=tf.constant([1]),
              features={tfgnn.HIDDEN_STATE: tf.constant([[1., 1.]])}),
      },
      edge_sets={
          "u": tfgnn.EdgeSet.from_fields(
              sizes=tf.constant([1]),
              features={tfgnn.HIDDEN_STATE: tf.constant([[1., 0.]])},
              adjacency=tfgnn.Adjacency.from_indices(
                  ("a", tf.constant([0])),
                  ("b", tf.constant([0])))),
          "v": tfgnn.EdgeSet.from_fields(
              sizes=tf.constant([1]),
              features={tfgnn.HIDDEN_STATE: tf.constant([[1., 1.]])},
              adjacency=tfgnn.Adjacency.from_indices(
                  ("a", tf.constant([0])),
                  ("b", tf.constant([0])))),
      },
      context=tfgnn.Context.from_fields(
          sizes=tf.constant([1]),
          features={"cf": tf.constant([[2.]])}))


if __name__ == "__main__":
  tf.test.main()
