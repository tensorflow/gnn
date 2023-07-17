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
"""Tests for edge_samplers."""
import os

from typing import Tuple

import apache_beam as beam
from apache_beam.testing import util

import numpy as np
import tensorflow as tf

from tensorflow_gnn.experimental import sampler
from tensorflow_gnn.experimental.sampler.beam import accessors  # pylint: disable=unused-import
from tensorflow_gnn.experimental.sampler.beam import edge_samplers  # pylint: disable=unused-import
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


class EdgeSamplersTest(ExecutorTestBase):

  def test_sampling(self):
    edges = {1: 2, 2: 3}
    feats = {
        1: text_format.Merge(
            r"""
                features {
                  feature {
                      key: "feat"
                      value { float_list {value: [1., 2., 3.]} }
                  }
                }""",
            tf.train.Example(),
        ).SerializeToString(),
        2: text_format.Merge(
            r"""
                features {
                  feature {
                      key: "feat"
                      value { float_list {value: [2., 3., 4.]} }
                  }
                }""",
            tf.train.Example(),
        ).SerializeToString(),
        3: text_format.Merge(
            r"""
                features {
                  feature {
                      key: "feat"
                      value { float_list {value: [3., 4., 5.]} }
                  }
                }""",
            tf.train.Example(),
        ).SerializeToString(),
    }
    edges_layer = sampler.UniformEdgesSampler(
        sampler.KeyToTfExampleAccessor(
            sampler.InMemIntegerKeyToBytesAccessor(
                keys_to_values={0: b''}, name='edges'
            ),
            features_spec={
                'neighbors': tf.TensorSpec([None], tf.int64),
            },
        ),
        sample_size=3,
        edge_target_feature_name='neighbors',
        name='edges_sampler',
    )
    feats_layer = sampler.KeyToTfExampleAccessor(
        sampler.InMemIntegerKeyToBytesAccessor(
            keys_to_values=feats, name='feats'
        ),
        features_spec={
            'feat': tf.TensorSpec([3], tf.float32),
        },
    )
    seeds = tf.keras.Input(
        type_spec=tf.RaggedTensorSpec(
            [None, None], dtype=tf.int64, ragged_rank=1
        ),
        name='seeds',
    )
    hop_1 = edges_layer(seeds)
    hop_2 = edges_layer(hop_1['#target'])
    all_ids = tf.concat([seeds, hop_1['#target'], hop_2['#target']], axis=-1)
    unique_ids = sampler.ragged_unique(all_ids)

    graph = sampler.build_graph_tensor(
        node_sets={'nodes': {'#id': unique_ids, **feats_layer(unique_ids)}},
        edge_sets={'nodes,edges,nodes': [hop_1, hop_2]},
    )
    model = tf.keras.Model(inputs=seeds, outputs=graph)
    program, artifacts = sampler.create_program(model)
    temp_dir = self.create_tempdir().full_path
    for name, model in artifacts.models.items():
      sampler.save_model(model, os.path.join(temp_dir, name))

    seeds = {
        b's1': [[np.array([1]), np.array([1], np.int64)]],
        b's2': [[np.array([2]), np.array([1], np.int64)]],
    }
    with beam.Pipeline() as root:
      seeds = root | 'seeds' >> beam.Create(seeds)
      edges = root | 'edges' >> beam.Create(edges)
      feats = root | 'feats' >> beam.Create(feats)
      result = executor_lib.execute(
          program,
          {'seeds': seeds},
          feeds={'edges_sampler': edges, 'feats': feats},
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
                          key: "edges/edges.#size"
                          value { int64_list { value: 2 } }
                        }
                        feature {
                          key: "edges/edges.#source"
                          value { int64_list { value: [0, 1] } }
                        }
                        feature {
                          key: "edges/edges.#target"
                          value { int64_list { value: [1, 2] } }
                        }
                        feature {
                          key: "nodes/nodes.#size"
                          value { int64_list { value: 3 } }
                        }
                        feature {
                          key: "nodes/nodes.#id"
                          value { int64_list { value: [1, 2, 3] } }
                        }
                        feature {
                          key: "nodes/nodes.feat"
                          value {
                            float_list {
                              value: [1., 2., 3., 2., 3., 4., 3., 4., 5. ]
                            }
                          }
                        }
                      }
                      """,
                  ),
                  (
                      b's2',
                      """features {
                        feature {
                          key: "edges/edges.#size"
                          value { int64_list { value: 1 } }
                        }
                        feature {
                          key: "edges/edges.#source"
                          value { int64_list { value: [0] } }
                        }
                        feature {
                          key: "edges/edges.#target"
                          value { int64_list { value: [1] } }
                        }
                        feature {
                          key: "nodes/nodes.#size"
                          value { int64_list { value: 2 } }
                        }
                        feature {
                          key: "nodes/nodes.#id"
                          value { int64_list { value: [2, 3] } }
                        }
                        feature {
                          key: "nodes/nodes.feat"
                          value {
                            float_list {
                              value: [2., 3., 4., 3., 4., 5. ]
                            }
                          }
                        }
                      }
                      """,
                  ),
              ],
              self.sampling_results_equal,
          ),
      )

  def test_sampling_stats(self):
    edges = [(b'a', b'b'), (b'a', b'c'), (b'a', b'd')]
    edges_layer = sampler.UniformEdgesSampler(
        sampler.KeyToTfExampleAccessor(
            sampler.InMemStringKeyToBytesAccessor(
                keys_to_values={b'?': b''}, name='edges'
            ),
            features_spec={
                '#target': tf.TensorSpec([None], tf.string),
            },
        ),
        sample_size=2,
        name='edges_sampler',
    )
    seeds = tf.keras.Input(
        type_spec=tf.RaggedTensorSpec(
            [None, None], dtype=tf.string, ragged_rank=1
        ),
        name='seeds',
    )
    model = tf.keras.Model(inputs=seeds, outputs=edges_layer(seeds))
    program, artifacts = sampler.create_program(model)
    temp_dir = self.create_tempdir().full_path
    for name, model in artifacts.models.items():
      sampler.save_model(model, os.path.join(temp_dir, name))

    seeds = {
        b's1': [[
            np.array([b'a'] * 10_000, np.object_),
            np.array([10_000], np.int64),
        ]],
    }

    def _asert_stats(inputs: Tuple[bytes, tf.train.Example]):
      key, values = inputs
      self.assertEqual(key, b's1')
      values = values.features.feature
      self.assertAllEqual(values['#source'].bytes_list.value, [b'a'] * 20_000)
      targets = np.array(values['#target'].bytes_list.value, np.object_)
      self.assertLen(targets, 20_000)
      self.assertTrue(np.all(targets[0::2] != targets[1::2]))
      values, counts = np.unique(targets, return_counts=True)
      self.assertAllEqual(values, [b'b', b'c', b'd'])
      self.assertTrue(np.all(counts > [1_000, 1_000, 1_000]))

    with beam.Pipeline() as root:
      seeds = root | 'seeds' >> beam.Create(seeds)
      edges = root | 'edges' >> beam.Create(edges)
      _ = executor_lib.execute(
          program,
          {'seeds': seeds},
          feeds={'edges_sampler': edges},
          artifacts_path=temp_dir,
      ) | beam.Map(_asert_stats)

  def test_missing_values(self):
    edges = [(1, 2), (1, 3), (2, 3)]
    edges_layer = sampler.UniformEdgesSampler(
        sampler.KeyToTfExampleAccessor(
            sampler.InMemStringKeyToBytesAccessor(
                keys_to_values={b'?': b''}, name='edges'
            ),
            features_spec={
                '#target': tf.TensorSpec([None], tf.int64),
            },
        ),
        sample_size=2,
        name='edges_sampler',
    )
    seeds = tf.keras.Input(
        type_spec=tf.RaggedTensorSpec(
            [None, None], dtype=tf.int64, ragged_rank=1
        ),
        name='seeds',
    )
    model = tf.keras.Model(inputs=seeds, outputs=edges_layer(seeds))
    program, artifacts = sampler.create_program(model)

    temp_dir = self.create_tempdir().full_path
    for name, model in artifacts.models.items():
      sampler.save_model(model, os.path.join(temp_dir, name))

    seeds = {
        b's1': [[
            np.array([1, -1, -1, 2, -1], np.int64),
            np.array([5], np.int64),
        ]],
        b's2': [[
            np.array([-1], np.int64),
            np.array([1], np.int64),
        ]],
        b's3': [[
            np.array([], np.int64),
            np.array([0], np.int64),
        ]],
    }

    with beam.Pipeline() as root:
      seeds = root | 'seeds' >> beam.Create(seeds)
      edges = root | 'edges' >> beam.Create(edges)
      result = executor_lib.execute(
          program,
          {'seeds': seeds},
          feeds={'edges_sampler': edges},
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
                            key: "#source"
                            value { int64_list { value: [1, 1, 2] } }
                          }
                          feature {
                            key: "#target"
                            value { int64_list { value: [2, 3, 3] } }
                          }
                        }
                        """,
                  ),
                  (
                      b's2',
                      """features {
                          feature {
                            key: "#source"
                            value { int64_list { value: [] } }
                          }
                          feature {
                            key: "#target"
                            value { int64_list { value: [] } }
                          }
                        }
                        """,
                  ),
                  (
                      b's3',
                      """features {
                          feature {
                            key: "#source"
                            value { int64_list { value: [] } }
                          }
                          feature {
                            key: "#target"
                            value { int64_list { value: [] } }
                          }
                        }
                        """,
                  ),
              ],
              self.sampling_results_equal,
          ),
      )


if __name__ == '__main__':
  tf.test.main()
