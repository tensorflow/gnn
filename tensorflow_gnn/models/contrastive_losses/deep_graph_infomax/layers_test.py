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
"""Tests for layers."""
from typing import Any
from absl.testing import parameterized

import tensorflow as tf
from tensorflow_gnn.models.contrastive_losses.deep_graph_infomax import layers


class DeepGraphInfomaxLogitsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="NonSequenceInput",
          inputs=tf.constant(range(8)),
          expected_error=r"Expected `Sequence` \(got .*\)",
      ),
      dict(
          testcase_name="InvalidSequenceInput",
          inputs=[tf.constant(range(8))] * 4,
          expected_error=r"Expected `Sequence` of length 2 \(got .*\)",
      ),
      dict(
          testcase_name="IncompatabileShapeInput",
          inputs=[tf.constant(range(8)), tf.constant(range(4))],
          expected_error=r"Shapes \(8,\) and \(4,\) are incompatible",
      ),
      dict(
          testcase_name="UndefinedInnerDimension",
          inputs=[tf.keras.Input((2, None)), tf.keras.Input((2, None))],
          expected_error=r"Expected a defined inner dimension \(got .*\)",
      ),
  ])
  def test_value_error(
      self,
      inputs: Any,
      expected_error: str):
    """Verifies invalid input raises `ValueError`."""
    with self.assertRaisesRegex(ValueError, expected_error):
      _ = layers.DeepGraphInfomaxLogits()(inputs)

  def test_deep_graph_infomax(self):
    """Verifies output logits shape."""
    x_clean = tf.random.uniform((1, 4))
    x_corrupted = tf.random.uniform((1, 4))
    logits = layers.DeepGraphInfomaxLogits()((x_clean, x_corrupted))

    # Clean and corrupted logits (i.e., shape (1, 2)).
    self.assertEqual(logits.shape, (1, 2))


if __name__ == "__main__":
  tf.test.main()
