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
"""Tests for the NextState layers."""

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.keras.layers import next_state as next_state_lib
from tensorflow_gnn.utils import tf_test_utils as tftu


class NextStateFromConcatTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    const.enable_graph_tensor_validation_at_runtime()

  def testBasic(self):
    init_double = tf.keras.initializers.Identity(gain=2.0)
    next_state = next_state_lib.NextStateFromConcat(
        tf.keras.layers.Dense(5, use_bias=False,
                              kernel_initializer=init_double))
    actual = next_state((tf.constant([[1.]]),
                         {const.SOURCE: {"h0": tf.constant([[2.]]),
                                         "h1": tf.constant([[3.]])},
                          const.TARGET: tf.constant([[4.]])},
                         tf.constant([[5.]])))
    self.assertAllEqual([[2., 4., 6., 8., 10.]], actual)

  @parameterized.named_parameters(
      ("", tftu.ModelReloading.SKIP),
      ("Restored", tftu.ModelReloading.SAVED_MODEL),
      ("RestoredKeras", tftu.ModelReloading.KERAS))
  def testModel(self, model_reloading):
    test_input = tf.constant([[1.]])
    model_input = tf.keras.Input([1], dtype=tf.float32)
    init_double = tf.keras.initializers.Identity(gain=2.0)
    next_state = next_state_lib.NextStateFromConcat(
        tf.keras.layers.Dense(1, use_bias=False,
                              kernel_initializer=init_double))
    model_output = next_state((model_input, {}, {}))
    model = tf.keras.Model(model_input, model_output)
    _ = model(test_input)  # Trigger model building.
    model = tftu.maybe_reload_model(self, model, model_reloading,
                                    "next-state-from-concat")
    actual = model(test_input)
    self.assertAllEqual([[2.]], actual)

  def testTFLite(self):
    self.skipTest(
        "NextStateFromConcat TFLite functionality is tested in models/mt_albis")


class ResidualNextStateTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    const.enable_graph_tensor_validation_at_runtime()

  @parameterized.named_parameters(
      ("SoleFeature",),
      ("DefaultFeature", const.HIDDEN_STATE),
      ("CustomFeature", "my_feature", "my_feature"))
  def test(self, input_dict_key=None, skip_connection_feature_name=None):
    first_input = tf.constant([[64.]])
    second_input = {"foo": tf.constant([[8.]]),
                    "bar": tf.constant([[16.]])}
    third_input = tf.constant([[32.]])
    if input_dict_key:
      # Repurpose the third input as a non-state self-input.
      first_input = {input_dict_key: first_input, "other": third_input}
      third_input = []
    num_input_tensors = 4

    init_div8 = tf.keras.initializers.Constant([[0.125]] * num_input_tensors)
    kwargs = {}
    if skip_connection_feature_name is not None:
      kwargs["skip_connection_feature_name"] = skip_connection_feature_name
    next_state = next_state_lib.ResidualNextState(
        tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=init_div8),
        **kwargs)

    actual = next_state((first_input, second_input, third_input))
    self.assertAllEqual([[64. + 15.]], actual)

  @parameterized.named_parameters(
      ("", tftu.ModelReloading.SKIP),
      ("Restored", tftu.ModelReloading.SAVED_MODEL),
      ("RestoredKeras", tftu.ModelReloading.KERAS))
  def testModel(self, model_reloading):
    test_input = [tf.constant([[1., 1.]]), tf.constant([[2., 2.]])]
    model_input = [tf.keras.Input([2], dtype=tf.float32) for _ in range(2)]

    init_plus_minus = tf.keras.initializers.Constant([[+1., -1.]] * 4)
    next_state = next_state_lib.ResidualNextState(
        tf.keras.layers.Dense(2, use_bias=False,
                              kernel_initializer=init_plus_minus),
        activation="relu", skip_connection_feature_name="my_feature")
    model_output = next_state((
        {"my_feature": model_input[0], "other_feature": model_input[1]},
        {}, {}
    ))
    model = tf.keras.Model(model_input, model_output)
    _ = model(test_input)  # Trigger model building.
    model = tftu.maybe_reload_model(self, model, model_reloading,
                                    "residual-next-state")
    actual = model(test_input)
    expected = [[
        1. + (1. + 1. + 2. + 2.),
        0.,  #  = relu(1. - (...))
    ]]
    self.assertAllClose(expected, actual)

  def testEmptyState(self):
    first_input = {const.HIDDEN_STATE: tf.constant([[]], tf.float32),
                   "other": tf.constant([[2.]])}
    second_input = {"foo": tf.constant([[4.]]),
                    "bar": tf.constant([[8.]])}
    third_input = {}

    init_div2 = tf.keras.initializers.Constant(0.5)
    next_state = next_state_lib.ResidualNextState(
        tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=init_div2))

    actual = next_state((first_input, second_input, third_input))
    self.assertAllEqual([[1. + 2. + 4.]], actual)

  def testMissingDictionaryThrowsException(self):
    first_input = tf.constant([[32.0]])
    init_div2 = tf.keras.initializers.Constant(0.5)
    next_state = next_state_lib.ResidualNextState(
        tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=init_div2),
        skip_connection_feature_name="my_feature",
    )

    with self.assertRaises(KeyError):
      _ = next_state((first_input))

  def testTFLite(self):
    test_input_dict = {
        "first_input": tf.constant([[64.0]]),
        "second_input_0": tf.constant([[8.0]]),
        "second_input_1": tf.constant([[16.0]]),
        "third_input": tf.constant([[32.0]]),
    }
    inputs = {
        "first_input": tf.keras.Input([1], None, "first_input", tf.float32),
        "second_input_0": tf.keras.Input(
            [1], None, "second_input_0", tf.float32
        ),
        "second_input_1": tf.keras.Input(
            [1], None, "second_input_1", tf.float32
        ),
        "third_input": tf.keras.Input([1], None, "third_input", tf.float32),
    }
    layer = next_state_lib.ResidualNextState(
        tf.keras.layers.Dense(1, use_bias=False), name="residual_next_state")
    outputs = layer(
        (inputs["first_input"],
         {"second_input_0": inputs["second_input_0"],
          "second_input_1": inputs["second_input_1"]},
         inputs["third_input"],))
    model = tf.keras.Model(inputs, outputs)

    # The other unit tests should verify that this is correct
    expected = model(test_input_dict).numpy()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    model_content = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=model_content)
    signature_runner = interpreter.get_signature_runner("serving_default")
    obtained = signature_runner(**test_input_dict)["residual_next_state"]
    self.assertAllClose(expected, obtained)


class SingleInputNextStateTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    const.enable_graph_tensor_validation_at_runtime()

  def testSingleInput(self):
    next_state = next_state_lib.SingleInputNextState()
    input_triple = (tf.constant([[1.]]),
                    {"edge": tf.constant([[2.]])},
                    {})
    self.assertAllEqual([[2.]], next_state(input_triple))

  def testErrorWithMultipleInputs(self):
    next_state = next_state_lib.SingleInputNextState()
    input_triple = (tf.constant([[1.]]),
                    {const.SOURCE: tf.constant([[2.]]),
                     const.TARGET: tf.constant([[4.]])},
                    {})
    self.assertRaisesRegex(ValueError,
                           ("This layer should take only a single input"),
                           lambda: next_state(input_triple))

  def testThreePassedIn(self):
    next_state = next_state_lib.SingleInputNextState()
    input_triple = (tf.constant([[1.]]),
                    {"edge": tf.constant([[2.]])},
                    tf.constant([[1.]]))
    self.assertRaisesRegex(ValueError,
                           ("GraphPieceUpdate should only pass 2 inputs "
                            "to SingleInputNextState"),
                           lambda: next_state(input_triple))

  @parameterized.named_parameters(
      ("", tftu.ModelReloading.SKIP),
      ("Restored", tftu.ModelReloading.SAVED_MODEL),
      ("RestoredKeras", tftu.ModelReloading.KERAS))
  def testModel(self, model_reloading):
    test_input = [tf.constant([[1.]]), tf.constant([[2.]])]
    model_input = [tf.keras.Input([1], dtype=tf.float32) for _ in range(2)]
    input_triple = (model_input[0],
                    {"edge": model_input[1]},
                    {})
    next_state = next_state_lib.SingleInputNextState()
    model_output = next_state(input_triple)
    model = tf.keras.Model(model_input, model_output)
    _ = model(test_input)  # Trigger model building.
    model = tftu.maybe_reload_model(self, model, model_reloading,
                                    "single-input-next-state")
    actual = model(test_input)
    self.assertAllEqual([[2.]], actual)


if __name__ == "__main__":
  tf.test.main()
