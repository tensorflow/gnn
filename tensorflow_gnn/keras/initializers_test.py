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
"""Tests for initializers.py."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_gnn.keras import initializers


class CloneInitializerTest(tf.test.TestCase, parameterized.TestCase):

  def testProblem(self):
    """Verifies that there even is a problem to fix (b/268648226)."""
    initializer = tf.keras.initializers.RandomUniform(-2., 5.)
    self.assertAllClose(initializer([10]), initializer([10]))

  @parameterized.named_parameters(
      ("None", None),
      ("Str", "ones"),
      ("Dict", {"class_name": "Ones", "config": {}}),
  )
  def testUnchangedValue(self, input_value):
    output_value = initializers.clone_initializer(input_value)
    self.assertIs(input_value, output_value)

  def testClonedObject(self):
    input_value = tf.keras.initializers.RandomUniform(-2., 5.)
    output_value = initializers.clone_initializer(input_value)
    # The output is a separate object...
    self.assertIsNot(input_value, output_value)
    # ...but with the same non-default config.
    self.assertEqual(input_value.minval, output_value.minval)
    self.assertEqual(input_value.maxval, output_value.maxval)
    # Verify that the results are actually different.
    self.assertNotAllClose(input_value([10]), output_value([10]))


if __name__ == "__main__":
  tf.test.main()
