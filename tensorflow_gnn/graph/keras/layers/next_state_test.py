"""Tests for the NextState layers."""

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph.keras.layers import next_state as next_state_lib


class NextStateFromConcatTest(tf.test.TestCase):

  def test(self):
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


class ResidualNextStateTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("SoleFeature",),
      ("DefaultFeature", const.DEFAULT_STATE_NAME),
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


if __name__ == "__main__":
  tf.test.main()
