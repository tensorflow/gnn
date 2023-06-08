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
"""Tests for accessors."""
import os

import apache_beam as beam
from apache_beam.testing import util

import numpy as np
import tensorflow as tf

from tensorflow_gnn.experimental import sampler
from tensorflow_gnn.experimental.sampler.beam import accessors  # pylint: disable=unused-import
from tensorflow_gnn.experimental.sampler.beam import executor_lib

from google.protobuf import text_format

PCollection = beam.PCollection


rt = tf.ragged.constant


class ExecutorTestBase(tf.test.TestCase):

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


class KeyToBytesAccessorTest(ExecutorTestBase):

  def test_string_keys(self):
    table = {
        b'a': text_format.Merge(
            r"""
              features {
                feature { key: "s" value {int64_list {value: [1]} } }
              }""",
            tf.train.Example(),
        ).SerializeToString(),
        b'b': text_format.Merge(
            r"""
              features {
                feature { key: "s" value {int64_list {value: [2]} } }
              }""",
            tf.train.Example(),
        ).SerializeToString(),
    }
    layer = sampler.KeyToTfExampleAccessor(
        sampler.InMemStringKeyToBytesAccessor(
            keys_to_values=table, name='table'
        ),
        features_spec={
            's': tf.TensorSpec([], tf.int64),
        },
        default_values={'s': -1},
    )
    keys = tf.keras.Input(
        type_spec=tf.RaggedTensorSpec(
            [None, None], dtype=tf.string, ragged_rank=1
        ),
        name='keys',
    )
    output = layer(keys)
    model = tf.keras.Model(inputs=keys, outputs=output)
    program, artifacts = sampler.create_program(model)
    temp_dir = self.create_tempdir().full_path
    for name, model in artifacts.models.items():
      sampler.save_model(model, os.path.join(temp_dir, name))

    keys = {
        b's1': [[
            np.array([b'a', b'x'], np.object_),
            np.array([2], np.int64),
        ]],
        b's2': [[
            np.array([b'a', b'b', b'a'], np.object_),
            np.array([2], np.int64),
        ]],
        b's3': [[np.array([b'x'], np.object_), np.array([1], np.int64)]],
        b's4': [[
            np.array([b'x', b'y'], np.object_),
            np.array([2], np.int64),
        ]],
        b's5': [[
            np.array([], np.object_),
            np.array([0], np.int64),
        ]],
    }
    with beam.Pipeline() as root:
      keys = root | 'keys' >> beam.Create(keys)

      table = root | 'table' >> beam.Create(table)
      result = executor_lib.execute(
          program,
          {'keys': keys},
          feeds={'table': table},
          artifacts_path=temp_dir,
      )
      util.assert_that(
          result,
          util.equal_to(
              [
                  (
                      b's1',
                      """features {
                        feature {
                          key: "s"
                          value { int64_list { value: [1, -1] } }
                        }
                      }""",
                  ),
                  (
                      b's2',
                      """features {
                        feature {
                          key: "s"
                          value { int64_list { value: [1, 2, 1] } }
                        }
                      }""",
                  ),
                  (
                      b's3',
                      """features {
                        feature {
                          key: "s"
                          value { int64_list { value: [-1] } }
                        }
                      }""",
                  ),
                  (
                      b's4',
                      """features {
                        feature {
                          key: "s"
                          value { int64_list { value: [-1, -1] } }
                        }
                      }""",
                  ),
                  (
                      b's5',
                      """features {
                        feature {
                          key: "s"
                          value { int64_list { value: [] } }
                        }
                      }""",
                  ),
              ],
              self.sampling_results_equal,
          ),
      )

  def test_integer_keys(self):
    table = {
        10: text_format.Merge(
            r"""
              features {
                feature { key: "f" value {float_list {value: [1.0, 2.0]} } }
              }
            """,
            tf.train.Example(),
        ).SerializeToString(),
        20: text_format.Merge(
            r"""
              features {
                feature { key: "f" value {float_list {value: [3.0, 4.0]} } }
              }
            """,
            tf.train.Example(),
        ).SerializeToString(),
    }
    layer = sampler.KeyToTfExampleAccessor(
        sampler.InMemIntegerKeyToBytesAccessor(
            keys_to_values=table, name='table'
        ),
        features_spec={
            'f': tf.TensorSpec([2], tf.float32),
        },
    )
    keys = tf.keras.Input(
        type_spec=tf.RaggedTensorSpec(
            [None, None],
            dtype=tf.int32,
            ragged_rank=1,
            row_splits_dtype=tf.int32,
        ),
        name='keys',
    )
    output = layer(keys)
    model = tf.keras.Model(inputs=keys, outputs=output)
    program, artifacts = sampler.create_program(model)
    temp_dir = self.create_tempdir().full_path
    for name, model in artifacts.models.items():
      sampler.save_model(model, os.path.join(temp_dir, name))

    keys = {
        b's1': [[
            np.array([10, -1], np.int32),
            np.array([2], np.int32),
        ]],
        b's2': [[
            np.array([-1, 20, 10], np.int32),
            np.array([3], np.int32),
        ]],
        b's3': [[
            np.array([], np.int32),
            np.array([0], np.int32),
        ]],
        b's4': [[
            np.array([-1], np.int32),
            np.array([1], np.int32),
        ]],
    }
    with beam.Pipeline() as root:
      keys = root | 'keys' >> beam.Create(keys)

      table = root | 'table' >> beam.Create(table)
      result = executor_lib.execute(
          program,
          {'keys': keys},
          feeds={'table': table},
          artifacts_path=temp_dir,
      )
      util.assert_that(
          result,
          util.equal_to(
              [
                  (
                      b's1',
                      """features {
                          feature {
                            key: "f"
                            value {
                              float_list {
                                value: [1.0, 2.0, 0.0, 0.0]
                              }
                            }
                          }
                        }""",
                  ),
                  (
                      b's2',
                      """features {
                          feature {
                            key: "f"
                            value {
                              float_list {
                                value: [0.0, 0.0, 3.0, 4.0, 1.0, 2.0]
                              }
                            }
                          }
                        }""",
                  ),
                  (
                      b's3',
                      """features {
                          feature {
                            key: "f"
                            value { float_list { value: [] } }
                          }
                        }""",
                  ),
                  (
                      b's4',
                      """features {
                          feature {
                            key: "f"
                            value { float_list { value: [0.0, 0.0] } }
                          }
                        }""",
                  ),
              ],
              self.sampling_results_equal,
          ),
      )


if __name__ == '__main__':
  tf.test.main()
