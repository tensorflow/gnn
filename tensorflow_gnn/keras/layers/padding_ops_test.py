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
"""Tests for padding_ops Keras layers."""

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import preprocessing_common
from tensorflow_gnn.keras import keras_tensors  # For registration. pylint: disable=unused-import
from tensorflow_gnn.keras.layers import padding_ops
from tensorflow_gnn.utils import tf_test_utils as tftu
# pylint: disable=g-direct-tensorflow-import
from ai_edge_litert import interpreter as tfl_interpreter
# pylint: enable=g-direct-tensorflow-import


class PadToTotalSizesTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    const.enable_graph_tensor_validation_at_runtime()

  def _make_test_graph(self):
    return gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(
            features={"label": tf.constant([42])}),
        node_sets={"nodes": gt.NodeSet.from_fields(
            sizes=tf.constant([1]),
            features={"feature": tf.constant([[1., 2.]])})},
        edge_sets={"edges": gt.EdgeSet.from_fields(
            sizes=tf.constant([1]),
            adjacency=adj.Adjacency.from_indices(("nodes", tf.constant([0])),
                                                 ("nodes", tf.constant([0]))),
            features={"weight": tf.constant([1.0])})})

  @parameterized.named_parameters(
      ("", tftu.ModelReloading.SKIP, False),
      ("ViaFeature", tftu.ModelReloading.SKIP, True),
      ("Restored", tftu.ModelReloading.SAVED_MODEL, False),
      ("RestoredKeras", tftu.ModelReloading.KERAS, False),
      ("RestoredKerasViaFeature", tftu.ModelReloading.KERAS, True),
  )
  def test(self, model_reloading, via_feature):
    input_graph = self._make_test_graph()
    sc = preprocessing_common.SizeConstraints(
        total_num_components=2,
        total_num_nodes={"nodes": 3},
        total_num_edges={"edges": tf.constant(4)})  # Test conversion to int.

    inputs = tf.keras.layers.Input(type_spec=input_graph.spec)
    if via_feature:
      pad = padding_ops.PadToTotalSizes(
          sc, mask_output_feature_name="mask", return_mask=False)
      padded_graph = pad(inputs)
      padding_mask = padded_graph.context["mask"]
    else:
      pad = padding_ops.PadToTotalSizes(sc)
      padded_graph, padding_mask = pad(inputs)
      self.assertNotIn("mask", padded_graph.context.features)
    model = tf.keras.Model(inputs, (padded_graph, padding_mask))
    model = tftu.maybe_reload_model(self, model, model_reloading,
                                    "padding-model")

    graph, mask = model(input_graph)
    self.assertAllEqual([True, False], mask)
    self.assertAllEqual(2, graph.num_components)
    self.assertAllEqual([42, 0], graph.context["label"])
    nodes = graph.node_sets["nodes"]
    self.assertAllEqual([1, 2], nodes.sizes)
    self.assertAllEqual([[1., 2.], [0., 0.], [0., 0.]], nodes["feature"])
    edges = graph.edge_sets["edges"]
    self.assertAllEqual([1, 3], edges.sizes)
    self.assertAllEqual([1., 0., 0., 0.], edges["weight"])


class PadToTotalSizesTFLiteTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    const.enable_graph_tensor_validation_at_runtime()

  def testBasic(self):
    test_graph_1_dict = {
        # We care that the TFLite interpreter gives the same output as the
        # model, which was tested separately (although not for the randomly
        # initialized weights that we keep here).
        "source": tf.constant([0]),
        "target": tf.constant([0]),
        "node_features": tf.constant([[1., 2.]]),
        "edge_weights": tf.constant([1.0]),
        "context_label": tf.constant([42]),
    }

    sc = preprocessing_common.SizeConstraints(
        total_num_components=2,
        total_num_nodes={"nodes": 3},
        total_num_edges={"edges": tf.constant(4)})  # Test conversion to int.
    pad = padding_ops.PadToTotalSizes(sc)

    inputs = {
        "node_features": tf.keras.Input([2], None, "node_features", tf.float32),
        "source": tf.keras.Input([], None, "source", tf.int32),
        "target": tf.keras.Input([], None, "target", tf.int32),
        "edge_weights": tf.keras.Input([], None, "edge_weights", tf.float32),
        "context_label": tf.keras.Input([], None, "context_label", tf.int32),
    }
    graph_in = _MakeGraphTensor()(inputs)
    graph_out, _ = pad(graph_in)
    outputs = tf.keras.layers.Layer(name="final_node_states")(
        graph_out.node_sets["nodes"]["feature"]
    )
    model = tf.keras.Model(inputs, outputs)

    # The other unit tests should verify that this is correct
    expected = model(test_graph_1_dict).numpy()

    # TODO(b/275338236): Un-skip when padding ops are supported.
    self.skipTest("Padding ops are unsupported in TFLite.")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    model_content = converter.convert()
    interpreter = tfl_interpreter.Interpreter(model_content=model_content)
    signature_runner = interpreter.get_signature_runner("serving_default")
    obtained = signature_runner(**test_graph_1_dict)["final_node_states"]
    self.assertAllClose(expected, obtained)


# TODO(b/274779989): Replace this layer with a more standard representation
# of GraphTensor as a dict of plain Tensors.
class _MakeGraphTensor(tf.keras.layers.Layer):
  """Makes a homogeneous GraphTensor of rank 0 with a single component."""

  def call(self, inputs):
    node_sizes = tf.shape(inputs["node_features"])[0]
    edge_sizes = tf.shape(inputs["edge_weights"])[0]
    return gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(
            features={"label": inputs["context_label"]}),
        node_sets={
            "nodes": gt.NodeSet.from_fields(
                sizes=tf.expand_dims(node_sizes, axis=0),
                features={"feature": inputs["node_features"]},
            )
        },
        edge_sets={
            "edges": gt.EdgeSet.from_fields(
                sizes=tf.expand_dims(edge_sizes, axis=0),
                adjacency=adj.Adjacency.from_indices(
                    ("nodes", inputs["source"]), ("nodes", inputs["target"])
                ),
                features={"weight": inputs["edge_weights"]},
            )
        },
    )

if __name__ == "__main__":
  tf.test.main()
