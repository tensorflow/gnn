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
"""Tests for convolutions."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.keras.layers import convolutions
from tensorflow_gnn.utils import tf_test_utils as tftu


class SimpleConvTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    const.enable_graph_tensor_validation_at_runtime()

  @parameterized.named_parameters(
      ("Forward", False, False, tftu.ModelReloading.SKIP),
      ("ForwardWithEdgeFeatureRestoredKeras",
       True, False, tftu.ModelReloading.KERAS),
      ("BackwardRestoredKeras",
       False, True, tftu.ModelReloading.KERAS),
      ("BackwardWithEdgeFeatureRestored",
       True, True, tftu.ModelReloading.SAVED_MODEL))
  def testSourcesAndReceiver(self, include_edges, reverse, model_reloading):
    values = dict(edges=tf.constant([[1.], [2.]]),
                  nodes=tf.constant([[4.], [8.], [16.]]))
    input_graph = _make_test_graph_01into2(values)
    message_fn = tf.keras.layers.Dense(1, use_bias=False,
                                       kernel_initializer="ones")
    input_kwargs = {}
    if include_edges:
      input_kwargs["sender_edge_feature"] = const.HIDDEN_STATE
    # Use the SOURCE node feature, irrespective of direction
    # (just to test kwargs and their defaults, not for any modeling reason).
    if reverse:
      input_kwargs["sender_node_feature"] = None
    else:
      input_kwargs["receiver_feature"] = None

    conv = convolutions.SimpleConv(
        message_fn, combine_type="sum", **input_kwargs,
        receiver_tag=const.SOURCE if reverse else const.TARGET)

    # Build a Model around the Layer, possibly saved and restored.
    inputs = tf.keras.layers.Input(type_spec=input_graph.spec)
    outputs = conv(inputs, edge_set_name="edges")
    model = tf.keras.Model(inputs, outputs)
    _ = model(input_graph)  # Trigger building.
    model = tftu.maybe_reload_model(self, model, model_reloading,
                                    "simple-convolution")

    # combine_type="sum" uses the same kernel size for any number of inputs.
    self.assertEqual(tf.TensorShape([1, 1]),
                     conv._message_fn.kernel.shape)

    actual = model(input_graph)
    if reverse:
      expected = tf.constant([
          [4. + include_edges*1.],
          [8. + include_edges*2.],
          [0.]])  # No outgoing edge.
    else:
      expected = tf.constant([
          [0.], [0.],  # No incoming edges,
          [(4. + 8.) + include_edges*(1. + 2.)]])
    self.assertAllEqual(expected, actual)

  @parameterized.named_parameters(("Concat", "concat"), ("Sum", "sum"))
  def testCombineType(self, combine_type):
    values = dict(nodes=tf.constant([[1.], [2.], [4.]]))
    input_graph = _make_test_graph_01into2(values)

    if combine_type == "sum":
      kernel = np.array([[1.]])
      sender_scale = 1.
    elif combine_type == "concat":
      kernel = np.array([[2.], [1.]])
      sender_scale = 2.
    else:
      self.fail(f"missing a case for combine_type='{combine_type}'")
    message_fn = tf.keras.layers.Dense(
        1, use_bias=False,
        kernel_initializer=tf.keras.initializers.Constant(kernel))

    if combine_type == "concat":
      combine_type_kwarg = dict()  # Expected as the default.
    else:
      combine_type_kwarg = dict(combine_type=combine_type)
    conv = convolutions.SimpleConv(
        message_fn, receiver_tag=const.SOURCE, **combine_type_kwarg)

    actual = conv(input_graph, edge_set_name="edges")
    expected = tf.constant([
        [1. + sender_scale*4.],
        [2. + sender_scale*4.],
        [0.]])  # No edges.
    self.assertAllEqual(expected, actual)

  def testTFLite(self):
    self.skipTest(
        "SimpleConv TFLite functionality is tested in models/mt_albis")


def _make_test_graph_01into2(values):
  """Returns GraphTensor for [v0] --e0--> [v2] <-e1-- [v1] with values."""
  def maybe_features(key):
    features = {const.HIDDEN_STATE: values[key]} if key in values else {}
    return dict(features=features)
  graph = gt.GraphTensor.from_pieces(
      context=gt.Context.from_fields(**maybe_features("context")),
      node_sets={"nodes": gt.NodeSet.from_fields(
          sizes=tf.constant([3]), **maybe_features("nodes"))},
      edge_sets={"edges": gt.EdgeSet.from_fields(
          sizes=tf.constant([2]),
          adjacency=adj.Adjacency.from_indices(("nodes", tf.constant([0, 1])),
                                               ("nodes", tf.constant([2, 2]))),
          **maybe_features("edges"))})
  return graph


if __name__ == "__main__":
  tf.test.main()
