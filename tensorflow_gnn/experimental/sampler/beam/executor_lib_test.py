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
import os

from absl.testing import parameterized

import apache_beam as beam
from apache_beam.coders import typecoders
from apache_beam.testing import util
from apache_beam.typehints import trivial_inference

import numpy as np
import tensorflow as tf

from tensorflow_gnn.experimental import sampler
from tensorflow_gnn.experimental.sampler.beam import executor_lib

from google.protobuf import text_format

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


class Identity(sampler.CompositeLayer):

  def symbolic_call(self, inputs):
    return inputs


class ExecutorTestBase(tf.test.TestCase, parameterized.TestCase):

  def sampling_results_equal(self, expected, actual) -> bool:
    e_sample_id, e_data = expected
    a_sample_id, a_data = actual
    if e_sample_id != a_sample_id:
      return False
    e_data = text_format.Parse(e_data, tf.train.Example())
    try:
      self.assertProtoEquals(e_data, a_data)
    except AssertionError:
      return False
    return True


class TFModelStageTest(ExecutorTestBase):

  def test_primitive(self):
    i = tf.keras.Input(
        type_spec=tf.RaggedTensorSpec(
            [None, None], dtype=tf.string, ragged_rank=1
        ),
        name='input',
    )
    o1 = tf.strings.join(['x', i], separator='-')
    o2 = tf.strings.join(['y', i], separator='-')
    o = tf.concat([o1, o2], axis=-1)
    model = tf.keras.Model(inputs=i, outputs=o)
    program, artifacts = sampler.create_program(model)
    temp_dir = self.create_tempdir().full_path
    for name, model in artifacts.models.items():
      sampler.save_model(model, os.path.join(temp_dir, name))

    values = {
        b's1': [[np.array([b'a'], np.object_), np.array([1], np.int64)]],
        b's2': [[
            np.array([b'a', b'b'], np.object_),
            np.array([2], np.int64),
        ]],
    }
    with beam.Pipeline() as root:
      values = root | beam.Create(values)
      result = executor_lib.execute(
          program, {'input': values}, artifacts_path=temp_dir
      )
      util.assert_that(
          result,
          util.equal_to(
              [
                  (
                      b's1',
                      """features {
                        feature {
                          key: "__output__"
                          value {
                            bytes_list { value: ["x-a", "y-a"] }
                          }
                        }
                      }""",
                  ),
                  (
                      b's2',
                      """features {
                        feature {
                          key: "__output__"
                          value {
                            bytes_list { value: ["x-a", "x-b", "y-a", "y-b"] }
                          }
                        }
                      }""",
                  ),
              ],
              self.sampling_results_equal,
          ),
      )

  def test_multiple_inputs(self):
    i1 = tf.keras.Input(
        type_spec=tf.RaggedTensorSpec(
            [None, None], dtype=tf.string, ragged_rank=1
        ),
        name='i1',
    )
    i2 = tf.keras.Input(
        type_spec=tf.RaggedTensorSpec(
            [None, None], dtype=tf.string, ragged_rank=1
        ),
        name='i2',
    )
    o1 = tf.strings.join(['x', i1], separator='-')
    o2 = tf.strings.join(['y', i2], separator='-')
    o = tf.concat([o1, o2], axis=-1)
    model = tf.keras.Model(inputs=[i1, i2], outputs=o)
    program, artifacts = sampler.create_program(model)
    temp_dir = self.create_tempdir().full_path
    for name, model in artifacts.models.items():
      sampler.save_model(model, os.path.join(temp_dir, name))

    i1_values = {
        b's': [[
            np.array([b'a', b'b'], np.object_),
            np.array([2], np.int64),
        ]],
    }
    i2_values = {
        b's': [[
            np.array([b'1', b'2', b'3'], np.object_),
            np.array([3], np.int64),
        ]],
    }

    with beam.Pipeline() as root:
      i1_values = root | 'i1' >> beam.Create(i1_values)
      i2_values = root | 'i2' >> beam.Create(i2_values)
      result = executor_lib.execute(
          program, {'i1': i1_values, 'i2': i2_values}, artifacts_path=temp_dir
      )
      util.assert_that(
          result,
          util.equal_to(
              [
                  (
                      b's',
                      """features {
                        feature {
                          key: "__output__"
                          value {
                            bytes_list {
                              value: ["x-a", "x-b", "y-1", "y-2", "y-3"]
                            }
                          }
                        }
                      }""",
                  ),
              ],
              self.sampling_results_equal,
          ),
      )

  def test_any_composite(self):
    i = tf.keras.Input([2], name='input')
    o = i
    o = tf.keras.layers.Lambda(tf.sparse.from_dense)(o)
    o = tf.keras.layers.Lambda(tf.math.negative)(o)
    # Add identity composite layer which splits eval data on five pieces:
    # <input> => <to negative sparse> => identity -> <to dense> => output.
    o = Identity()(o)

    def fn(t):
      return tf.sparse.to_dense(t)

    o = tf.keras.layers.Lambda(fn)(o)
    model = tf.keras.Model(inputs=i, outputs=o)
    program, artifacts = sampler.create_program(model)
    self.assertLen(program.eval_dag.stages, 5)
    temp_dir = self.create_tempdir().full_path
    for name, model in artifacts.models.items():
      sampler.save_model(model, os.path.join(temp_dir, name))

    values = {
        b's1': [[np.array([[1.0, 2.0]], np.float32)]],
        b's2': [[np.array([[3.0, 4.0]], np.float32)]],
        b's3': [[np.array([[5.0, 6.0]], np.float32)]],
    }
    with beam.Pipeline() as root:
      values = root | beam.Create(values)
      result = executor_lib.execute(
          program, {'input': values}, artifacts_path=temp_dir
      )
      feat_template = """features {
                        feature {
                          key: "__output__"
                          value {
                            float_list {
                                value: %s
                            }
                          }
                        }
                      }"""

      util.assert_that(
          result,
          util.equal_to(
              [
                  (b's1', feat_template % '[-1., -2.]'),
                  (b's2', feat_template % '[-3., -4.]'),
                  (b's3', feat_template % '[-5., -6.]'),
              ],
              self.sampling_results_equal,
          ),
      )


if __name__ == '__main__':
  tf.test.main()
