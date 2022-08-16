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
"""Tests for AnyToAnyConvolutionBase."""

import enum
import os
import re

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import normalization_ops
from tensorflow_gnn.keras.layers import convolution_base
from tensorflow_gnn.keras.layers import graph_update


# From AnyToAnyConvolutionBase.__init__.__doc__, except symbol `tfgnn.`.
@tf.keras.utils.register_keras_serializable(package="GNNtesting")
class ExampleConvolution(convolution_base.AnyToAnyConvolutionBase):

  def __init__(self, units, **kwargs):
    super().__init__(**kwargs)
    self._message_fn = tf.keras.layers.Dense(units, "relu")

  def get_config(self):
    return dict(units=self._message_fn.units, **super().get_config())

  def convolve(
      self, *,
      sender_node_input, sender_edge_input, receiver_input,
      broadcast_from_sender_node, broadcast_from_receiver, pool_to_receiver,
      training):
    inputs = []
    if sender_node_input is not None:
      inputs.append(broadcast_from_sender_node(sender_node_input))
    if sender_edge_input is not None:
      inputs.append(sender_edge_input)
    if receiver_input is not None:
      inputs.append(broadcast_from_receiver(receiver_input))
    messages = self._message_fn(tf.concat(inputs, axis=-1))
    return pool_to_receiver(messages, reduce_type="sum")


# From AnyToAnyConvolutionBase.__init__.__doc__, except symbol `tfgnn.`.
def ExampleEdgePool(*args, sender_feature=const.HIDDEN_STATE, **kwargs):  # To be called like a class initializer.  pylint: disable=invalid-name
  return ExampleConvolution(*args, sender_node_feature=None,
                            sender_edge_feature=sender_feature, **kwargs)


class SoftmaxBySumConvolution(convolution_base.AnyToAnyConvolutionBase):

  def __init__(self, **kwargs):
    super().__init__(sender_edge_feature=None, receiver_feature=None,
                     extra_receiver_ops={"softmax": normalization_ops.softmax},
                     **kwargs)

  def convolve(
      self, *,
      sender_node_input, sender_edge_input, receiver_input,
      broadcast_from_sender_node, broadcast_from_receiver, pool_to_receiver,
      extra_receiver_ops, training):
    assert self.takes_sender_node_input
    assert sender_node_input is not None
    assert not self.takes_sender_edge_input
    assert sender_edge_input is None
    assert not self.takes_receiver_input
    assert receiver_input is None

    inputs = broadcast_from_sender_node(sender_node_input)
    scores = tf.reduce_sum(inputs, axis=-1)
    multipliers = extra_receiver_ops["softmax"](scores)
    messages = tf.multiply(
        inputs,  # [num_items, feature_depth]
        tf.expand_dims(multipliers, axis=-1))  # [num_items, 1]
    return pool_to_receiver(messages, reduce_type="sum")


class ReloadModel(int, enum.Enum):
  """Controls how to reload a model for further testing after saving."""
  SKIP = 0
  SAVED_MODEL = 1
  KERAS = 2


class AnyToAnyConvolutionBaseTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("", False, False, False),
      ("FromConfig", False, False, True),
      ("WithEdgeInput", True, False, False),
      ("WithReceiverInput", False, True, False),
      ("WithEdgeAndReceiverInput", True, True, False),
      ("WithEdgeAndReceiverInputFromConfig", True, True, True))
  def testExampleNodeToNode(self, use_sender_edge_input, use_receiver_input,
                            from_config):
    values = dict(nodes=tf.constant([[1.], [2.], [4.]]),
                  edges=tf.constant([[8.], [16.]]))
    example_graph = _make_test_graph_01into2(values)

    conv_kwargs = {}
    if use_sender_edge_input:
      conv_kwargs["sender_edge_feature"] = const.HIDDEN_STATE
    if not use_receiver_input:
      conv_kwargs["receiver_feature"] = None
    conv = ExampleConvolution(1, **conv_kwargs)
    if from_config:
      conv = ExampleConvolution.from_config(conv.get_config())

    # Build weights and set them to known values.
    _ = conv(example_graph, edge_set_name="edges", receiver_tag=const.TARGET)
    weights = {_drop_prefix_re("example_convolution/dense.*/", v.name): v
               for v in conv.trainable_weights}
    self.assertLen(weights, 2)
    num_inputs = 1 + use_sender_edge_input + use_receiver_input
    weights["kernel:0"].assign([[1.]] * num_inputs)
    weights["bias:0"].assign([0.])

    actual = conv(example_graph, edge_set_name="edges",
                  receiver_tag=const.TARGET)
    expected = np.array([
        [0.],  # TARGET of no edges.
        [0.],  # TARGET of no edges.
        [(1. + use_sender_edge_input*8. + use_receiver_input*4.) +
         (2. + use_sender_edge_input*16. + use_receiver_input*4.)]])
    self.assertAllEqual(expected, actual)

    actual = conv(example_graph, edge_set_name="edges",
                  receiver_tag=const.SOURCE)
    expected = np.array([
        [4. + use_sender_edge_input*8. + use_receiver_input*1.],
        [4. + use_sender_edge_input*16. + use_receiver_input*2.],
        [0.]])  # SOURCE of no edges.
    self.assertAllEqual(expected, actual)

  @parameterized.named_parameters(
      ("", False, False),
      ("FromConfig", False, True),
      ("WithReceiverInput", True, False),
      ("WithReceiverInputFromConfig", True, True))
  def testExampleNodeToContext(self, use_receiver_input, from_config):
    values = dict(context=tf.constant([[1.]]),
                  nodes=tf.constant([[2.], [4.], [8.]]))
    example_graph = _make_test_graph_01into2(values)

    conv_kwargs = {}
    if not use_receiver_input:
      conv_kwargs["receiver_feature"] = None
    conv = ExampleConvolution(1, receiver_tag=const.CONTEXT, **conv_kwargs)
    if from_config:
      conv = ExampleConvolution.from_config(conv.get_config())

    # Build weights and set them to known values.
    _ = conv(example_graph, node_set_name="nodes")
    weights = {_drop_prefix_re("example_convolution/dense.*/", v.name): v
               for v in conv.trainable_weights}
    self.assertLen(weights, 2)
    num_inputs = 1 + use_receiver_input
    weights["kernel:0"].assign([[1.]] * num_inputs)
    weights["bias:0"].assign([0.])

    actual = conv(example_graph, node_set_name="nodes")
    expected = np.array([
        [(2. + use_receiver_input*1.) +
         (4. + use_receiver_input*1.) +
         (8. + use_receiver_input*1.)]])
    self.assertAllEqual(expected, actual)

  @parameterized.named_parameters(
      ("", False, False),
      ("FromConfig", False, True),
      ("WithReceiverInput", True, False),
      ("WithReceiverInputFromConfig", True, True))
  def testExampleEdgePool(self, use_receiver_input, from_config):
    values = dict(context=tf.constant([[1.]]),
                  nodes=tf.constant([[2.], [4.], [8.]]),
                  edges=tf.constant([[16.], [32.]]))
    example_graph = _make_test_graph_01into2(values)

    conv_kwargs = {}
    if not use_receiver_input:
      conv_kwargs["receiver_feature"] = None
    pool = ExampleEdgePool(1, **conv_kwargs)
    if from_config:
      pool = ExampleConvolution.from_config(pool.get_config())

    # Build weights and set them to known values.
    _ = pool(example_graph, edge_set_name="edges", receiver_tag=const.CONTEXT)
    weights = {_drop_prefix_re("example_convolution/dense.*/", v.name): v
               for v in pool.trainable_weights}
    self.assertLen(weights, 2)
    num_inputs = 1 + use_receiver_input
    weights["kernel:0"].assign([[1.]] * num_inputs)
    weights["bias:0"].assign([0.])

    actual = pool(example_graph, edge_set_name="edges",
                  receiver_tag=const.CONTEXT)
    expected = np.array([
        [(16. + use_receiver_input*1.) +
         (32. + use_receiver_input*1.)]])
    self.assertAllEqual(expected, actual)

    actual = pool(example_graph, edge_set_name="edges",
                  receiver_tag=const.TARGET)
    expected = np.array([
        [0.],  # TARGET of no edges.
        [0.],  # TARGET of no edges.
        [(16. + use_receiver_input*8.) +
         (32. + use_receiver_input*8.)]])
    self.assertAllEqual(expected, actual)

    actual = pool(example_graph, edge_set_name="edges",
                  receiver_tag=const.SOURCE)
    expected = np.array([
        [16. + use_receiver_input*2.],
        [32. + use_receiver_input*4.],
        [0.]])  # SOURCE of no edges.
    self.assertAllEqual(expected, actual)

  @parameterized.named_parameters(
      ("", ReloadModel.SKIP),
      ("Restored", ReloadModel.SAVED_MODEL),
      ("RestoredKeras", ReloadModel.KERAS))
  def testExampleEndToEnd(self, reload_model):
    values = dict(nodes=tf.constant([[1.], [2.], [4.]]),
                  context=tf.constant([[1.]]))
    example_graph = _make_test_graph_01into2(values)

    # Test two instances of ExampleConvolution as part of a GraphUpdate.
    update = graph_update.GraphUpdate(
        node_sets={
            "nodes": graph_update.NodeSetUpdate(
                {"edges": ExampleConvolution(1, receiver_tag=const.SOURCE)},
                NextStateFromSingleInput())},
        context=graph_update.ContextUpdate(
            {"nodes": ExampleConvolution(1, receiver_tag=const.CONTEXT)},
            NextStateFromSingleInput()))

    # Build weights and set them to known values.
    _ = update(example_graph)
    weights = {_drop_prefix_re("graph_update/", v.name): v
               for v in update.trainable_weights}
    self.assertLen(weights, 4)
    weights["node_set_update/example_convolution/dense/kernel:0"].assign(
        [[1.], [1.]])
    weights["node_set_update/example_convolution/dense/bias:0"].assign([0.5])
    weights["context_update/example_convolution_1/dense_1/kernel:0"].assign(
        [[1.], [100.]])
    weights["context_update/example_convolution_1/dense_1/bias:0"].assign([0.])

    # Build a Model around the GraphUpdate, possibly saved and restored.
    inputs = tf.keras.layers.Input(type_spec=example_graph.spec)
    outputs = update(inputs)
    model = tf.keras.Model(inputs, outputs)
    if reload_model:
      export_dir = os.path.join(self.get_temp_dir(), "example-end2end-model")
      model.save(export_dir, include_optimizer=False)
      if reload_model == ReloadModel.KERAS:
        model = tf.keras.models.load_model(export_dir)
        # Check that from_config() worked, no fallback to a function trace, see
        # https://www.tensorflow.org/guide/keras/save_and_serialize#how_savedmodel_handles_custom_objects
        self.assertIsInstance(model.get_layer(index=1),
                              graph_update.GraphUpdate)
      else:
        model = tf.saved_model.load(export_dir)

    # Check expected values.
    example_output = model(example_graph)
    self.assertAllEqual(
        [[1. + 4. + 0.5], [2. + 4. + 0.5], [0.]],
        example_output.node_sets["nodes"][const.HIDDEN_STATE])
    self.assertAllEqual(
        [[(5.5 + 100*1) + (6.5 + 100*1) + (0. + 100*1)]],
        example_output.context[const.HIDDEN_STATE])

  def testCustomReceiverOp(self):
    log2 = np.log(2)
    log3 = np.log(3)
    log5 = np.log(5)
    values = dict(nodes=tf.constant([
        [log2, 0., 0.],
        [0., log3, 0.],
        [0., 0., log5]]))
    example_graph = _make_test_graph_01into2(values)

    # Scales each input by the softmax of its sum.
    conv = SoftmaxBySumConvolution()

    actual = conv(example_graph, node_set_name="nodes",
                  receiver_tag=const.CONTEXT)
    expected = np.array([[0.2*log2, 0.3*log3, 0.5*log5]])
    self.assertAllClose(expected, actual)

    actual = conv(example_graph, edge_set_name="edges",
                  receiver_tag=const.TARGET)
    expected = np.array([
        [0., 0., 0.],  # TARGET of no edges.
        [0., 0., 0.],  # TARGET of no edges.
        [0.4*log2, 0.6*log3, 0.]])
    self.assertAllClose(expected, actual)

  # Like testExampleNodeToNode(use_sender_edge_input=False,
  #                            use_receiver_input=False, ...)
  # but with receiver tag set in different places.
  @parameterized.named_parameters(
      ("TargetInit", const.TARGET, None, [[0.], [0.], [3.]]),
      ("TargetCall", None, const.TARGET, [[0.], [0.], [3.]]),
      ("TargetBoth", const.TARGET, const.TARGET, [[0.], [0.], [3.]]),
      ("SourceInit", const.SOURCE, None, [[4.], [4.], [0.]]),
      ("Contradictory", const.SOURCE, const.TARGET, None,
       "ExampleConvolution.* contradictory value receiver_tag=1"),
      ("Missing", None, None, None,
       "ExampleConvolution requires.* receiver_tag"))
  def testGetReceiverTag(self, init_receiver_tag, call_receiver_tag, expected,
                         raises_regex=None):
    values = dict(nodes=tf.constant([[1.], [2.], [4.]]))
    example_graph = _make_test_graph_01into2(values)
    init_kwarg = (dict() if init_receiver_tag is None else
                  dict(receiver_tag=init_receiver_tag))
    conv = ExampleConvolution(1, receiver_feature=None, **init_kwarg)
    call_kwarg = (dict() if call_receiver_tag is None else
                  dict(receiver_tag=call_receiver_tag))
    def call():
      return conv(example_graph, edge_set_name="edges", **call_kwarg)

    if raises_regex:
      self.assertRaisesRegex(ValueError, raises_regex, call)
    else:
      # Call to build weights and set them to known values.
      _ = call()
      weights = {_drop_prefix_re("example_convolution/dense.*/", v.name): v
                 for v in conv.trainable_weights}
      self.assertLen(weights, 2)
      weights["kernel:0"].assign([[1.]])
      weights["bias:0"].assign([0.])
      # Call again for predictable output.
      self.assertAllEqual(expected, call())


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


def _drop_prefix_re(prefix_re, string):
  new_string, subs_made = re.subn("^" + prefix_re, "", string, count=1)
  assert subs_made == 1, f"Missing prefix '{prefix_re}' in '{string}'"
  return new_string


@tf.keras.utils.register_keras_serializable(package="GNNtesting")
class NextStateFromSingleInput(tf.keras.layers.Layer):

  def call(self, inputs):
    unused_old_state, main_input, third_input = inputs
    assert not third_input, f"expected nothing, got {third_input}"
    single_input, = main_input.values()  # Unpack.
    return single_input


if __name__ == "__main__":
  tf.test.main()
