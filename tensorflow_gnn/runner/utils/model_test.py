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
"""Tests for model."""
from typing import Sequence, Union

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.runner.utils import model as model_utils


def model(shape: Sequence[int],
          *,
          tag_value: int,
          num_outputs: int = 1) -> tf.keras.Model:
  """Builds a model with input shape, tag value and number of outputs."""
  inputs = tf.keras.layers.Input(shape=shape)
  if num_outputs > 1:
    outputs = tuple(inputs + tag_value for _ in range(num_outputs))
  elif num_outputs == 1:
    outputs = inputs + tag_value
  else:
    raise ValueError(f"`num_outputs` must be >= 1 (got {num_outputs})")
  return tf.keras.Model(inputs, outputs)


class ModelTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="SingleOutputs",
          m1=model((4,), tag_value=332),
          m2=model((4,), tag_value=8191),
          first_output_only=True,
          inputs=tf.range(0, 4),
          expected_outputs=tf.range(332 + 8191, 332 + 8191 + 4),
      ),
      dict(
          testcase_name="MultipleM1Output",
          m1=model((4,), tag_value=332, num_outputs=2),
          m2=model((4,), tag_value=8191),
          first_output_only=True,
          inputs=tf.range(0, 4),
          expected_outputs=tf.range(332 + 8191, 332 + 8191 + 4),
      ),
      dict(
          testcase_name="MultipleM2Output",
          m1=model((4,), tag_value=332),
          m2=model((4,), tag_value=8191, num_outputs=2),
          first_output_only=True,
          inputs=tf.range(0, 4),
          expected_outputs=tf.range(332 + 8191, 332 + 8191 + 4),
      ),
      dict(
          testcase_name="MultipleOutputs",
          m1=model((4,), tag_value=332, num_outputs=2),
          m2=model((4,), tag_value=8191, num_outputs=2),
          first_output_only=True,
          inputs=tf.range(0, 4),
          expected_outputs=tf.range(332 + 8191, 332 + 8191 + 4),
      ),
      dict(
          testcase_name="NestOutputSingleOutputs",
          m1=model((4,), tag_value=332),
          m2=model((4,), tag_value=8191),
          first_output_only=False,
          inputs=tf.range(0, 4),
          expected_outputs=tf.range(332 + 8191, 332 + 8191 + 4),
      ),
      dict(
          testcase_name="NestOutputMultipleM1Output",
          m1=model((4,), tag_value=332, num_outputs=2),
          m2=model((4,), tag_value=8191),
          first_output_only=False,
          inputs=tf.range(0, 4),
          expected_outputs=[
              tf.range(332 + 8191, 332 + 8191 + 4),
              tf.range(332, 332 + 4)
          ],
      ),
      dict(
          testcase_name="NestOutputMultipleM2Output",
          m1=model((4,), tag_value=332),
          m2=model((4,), tag_value=8191, num_outputs=2),
          first_output_only=False,
          inputs=tf.range(0, 4),
          expected_outputs=[
              tf.range(332 + 8191, 332 + 8191 + 4),
              tf.range(332 + 8191, 332 + 8191 + 4)
          ],
      ),
      dict(
          testcase_name="NestOutputMultipleOutputs",
          m1=model((4,), tag_value=332, num_outputs=2),
          m2=model((4,), tag_value=8191, num_outputs=2),
          first_output_only=False,
          inputs=tf.range(0, 4),
          expected_outputs=[
              tf.range(332 + 8191, 332 + 8191 + 4),
              tf.range(332, 332 + 4),
              tf.range(332 + 8191, 332 + 8191 + 4)
          ],
      ),
  ])
  def test_chain_first_output(
      self,
      m1: tf.keras.Model,
      m2: tf.keras.Model,
      first_output_only: bool,
      inputs: tf.Tensor,
      expected_outputs: Union[tf.Tensor, Sequence[tf.Tensor]]):
    actual = model_utils.chain_first_output(m1, m2, first_output_only)
    self.assertAllClose(actual(inputs), expected_outputs)

  def test_chain_first_output_fails(self):
    inputs = tf.keras.Input(shape=(4,))
    m1 = tf.keras.Model(inputs, {"outputs": inputs + 8191})
    m2 = tf.keras.Model({"outputs": inputs}, {"outputs": inputs ** 2})
    with self.assertRaisesRegex(
        ValueError,
        "Only Sequence nested structures are supported"):
      _ = model_utils.chain_first_output(m1, m2)

if __name__ == "__main__":
  tf.test.main()
