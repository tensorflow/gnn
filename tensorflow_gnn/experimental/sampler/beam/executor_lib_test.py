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
"""Tests for executor_lib."""

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.coders import typecoders
from apache_beam.typehints import trivial_inference
import numpy as np
import tensorflow as tf
from tensorflow_gnn.experimental.sampler.beam import executor_lib

PCollection = beam.PCollection


class NDArrayCoderTest(tf.test.TestCase, parameterized.TestCase):

  def test_registration(self):
    value = np.array([1, 2, 3])
    value_type = trivial_inference.instance_to_type(value)
    value_coder = typecoders.registry.get_coder(value_type)
    self.assertIsInstance(value_coder, executor_lib.NDArrayCoder)

  @parameterized.parameters(
      (np.array([], np.int32),),
      (np.array([], np.float32),),
      (np.array([[]], np.int64),),
      (np.array([[], []], np.float32),),
      (np.array([1, 2, 3]),),
      (np.array([[1], [2], [3]]),),
      (np.array(['1', '2', '3']),),
      (np.array([['a', 'b'], ['c', 'd']]),),
      (np.array([b'a', b'bbb', b'cccc', b'ddddd']),),
      (np.array([1.0, 2.0, 3.0]),),
      (np.array([[1.0, 2.0], [3.0, 4.0]]),),
      (np.array([[[True], [False]], [[False], [True]]]),),
  )
  def test_encoding_and_decoding(self, value):
    coder = executor_lib.NDArrayCoder()
    encoded = coder.encode(value)
    decoded = coder.decode(encoded)
    self.assertAllEqual(value, decoded)


if __name__ == '__main__':
  tf.test.main()
