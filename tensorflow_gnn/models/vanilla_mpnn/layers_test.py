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


class VanillaMPNNTFLiteTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(("WithoutLayerNorm", False),
                                  ("WithLayerNorm", True))
  def testBasic(self, use_layer_normalization):
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
    layer = vanilla_mpnn.VanillaMPNNGraphUpdate(
        units=units,
        message_dim=2,
        receiver_tag=tfgnn.TARGET,
        node_set_names=["nodes"],
        edge_feature=tfgnn.HIDDEN_STATE,
        use_layer_normalization=use_layer_normalization)

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
