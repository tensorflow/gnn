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
"""Tests for executor_lib using sampling model from subgraph_pipeline."""
import os

from absl.testing import parameterized

import apache_beam as beam
from apache_beam.testing import util

import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn.experimental import sampler
from tensorflow_gnn.experimental.sampler.beam import executor_lib

# pylint: disable=g-bad-import-order
# The inputs below are required to register specializations for stages.
from tensorflow_gnn.experimental.sampler.beam import accessors  # pylint: disable=unused-import
from tensorflow_gnn.experimental.sampler.beam import edge_samplers  # pylint: disable=unused-import
# pylint: enable=g-bad-import-order

PCollection = beam.PCollection


class SubgraphPipelineTest(tf.test.TestCase, parameterized.TestCase):

  def _get_sampling_model(self, ids_dtype) -> tf.keras.Model:
    graph_schema = tfgnn.GraphSchema()
    graph_schema.node_sets['a'].description = 'node set a'
    graph_schema.node_sets['a'].description = 'node set b'
    graph_schema.edge_sets['a->b'].source = 'a'
    graph_schema.edge_sets['a->b'].target = 'b'
    graph_schema.edge_sets['a->a'].features['weights'].dtype = 1

    sampling_spec = tfgnn.sampler.SamplingSpec()
    sampling_spec.seed_op.op_name = 'seed'
    sampling_spec.seed_op.node_set_name = 'a'
    sampling_spec.sampling_ops.add(
        op_name='hop1', edge_set_name='a->b', sample_size=100
    ).input_op_names.append('seed')

    def edge_sampler_factory(sampling_op):
      self.assertEqual(sampling_op.edge_set_name, 'a->b')
      if ids_dtype == tf.string:
        accessor = sampler.InMemStringKeyToBytesAccessor(
            keys_to_values={b'a': b''}
        )
      else:
        accessor = sampler.InMemIntegerKeyToBytesAccessor(
            keys_to_values={0: b''}
        )
      return sampler.UniformEdgesSampler(
          sample_size=sampling_op.sample_size,
          outgoing_edges_accessor=sampler.KeyToTfExampleAccessor(
              accessor,
              features_spec={
                  tfgnn.TARGET_NAME: tf.TensorSpec([None], ids_dtype),
                  'weights': tf.TensorSpec([None], tf.float32),
              },
          ),
      )

    return sampler.create_sampling_model_from_spec(
        graph_schema,
        sampling_spec,
        edge_sampler_factory,
        seed_node_dtype=ids_dtype,
    )

  @parameterized.parameters([tf.string, tf.int32, tf.int64])
  def test(self, ids_dtype: tf.DType):
    if ids_dtype == tf.string:
      source, target = (b'node:a', b'node:b')
    else:
      source, target = (100, 200)

    program, artifacts = sampler.create_program(
        self._get_sampling_model(ids_dtype)
    )
    temp_dir = self.create_tempdir().full_path
    for name, model in artifacts.models.items():
      sampler.save_model(model, os.path.join(temp_dir, name))

    seeds = {
        b'sample.1': [[
            np.array([source], dtype=ids_dtype.as_numpy_dtype),
            np.array([1], dtype=np.int64),
        ]],
    }
    edges = [
        _as_tf_example({
            tfgnn.SOURCE_NAME: [source],
            tfgnn.TARGET_NAME: [target],
            'weights': [0.5],
        })
    ]

    with beam.Pipeline() as root:
      result = executor_lib.execute(
          program,
          {'Input': root | 'seeds' >> beam.Create(seeds)},
          artifacts_path=temp_dir,
          feeds={'uniform_edges_sampler': root | 'a->b' >> beam.Create(edges)},
      )
      util.assert_that(
          result,
          util.equal_to(
              [
                  (
                      b'sample.1',
                      _as_tf_example({
                          'nodes/a.#size': [1],
                          'nodes/a.#id': [source],
                          'nodes/b.#size': [1],
                          'nodes/b.#id': [target],
                          'edges/a->b.#size': [1],
                          'edges/a->b.#source': [0],
                          'edges/a->b.#target': [0],
                          'edges/a->b.weights': [0.5],
                      }),
                  ),
              ],
          ),
      )


def _as_tf_example(values) -> tf.train.Example:
  example = tf.train.Example()
  features = example.features.feature
  for name, value in values.items():
    assert isinstance(value, list) and value
    if isinstance(value[0], (str, bytes)):
      features[name].bytes_list.value.extend(value)
    elif isinstance(value[0], int):
      features[name].int64_list.value.extend(value)
    elif isinstance(value[0], float):
      features[name].float_list.value.extend(value)
    else:
      raise ValueError(f'Unsupported value type {value[0]}')
  return example


if __name__ == '__main__':
  tf.test.main()
