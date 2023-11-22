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
from typing import List
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.experimental.sampler import core
from tensorflow_gnn.experimental.sampler import eval_dag as lib
from tensorflow_gnn.experimental.sampler import interfaces
from tensorflow_gnn.experimental.sampler import proto as pb


class EdgesSamplerTest(tf.test.TestCase, parameterized.TestCase):

  def _get_test_data(
      self, *, edge_target_feature_name: str, extra_feature_names: List[str]
  ):
    table = core.InMemIntegerKeyToBytesAccessor(
        keys_to_values={
            0: b'',
        },
        name='edges',
    )
    return core.KeyToTfExampleAccessor(
        table,
        features_spec={
            edge_target_feature_name: tf.TensorSpec([None], tf.string),
            **{
                k: tf.TensorSpec([None], tf.float32)
                for k in extra_feature_names
            },
        },
    )

  @parameterized.parameters([
      dict(sample_size=1, impl='UniformEdgesSampler'),
      dict(sample_size=8, impl='UniformEdgesSampler'),
      dict(
          sample_size=8,
          edge_target_feature_name='outgoing_edges',
          impl='UniformEdgesSampler',
      ),
      dict(sample_size=1, impl='UniformEdgesSampler'),
      dict(sample_size=1, impl='InMemUniformEdgesSampler'),
      dict(sample_size=8, impl='InMemUniformEdgesSampler'),
  ])
  def testUniformEdgesSampler(
      self,
      sample_size: int,
      impl: str,
      *,
      edge_target_feature_name: str = tfgnn.TARGET_NAME,
  ):
    def check_config(config):
      self.assertIsInstance(config, pb.EdgeSamplingConfig)
      self.assertEqual(config.edge_set_name, 'edges')
      self.assertEqual(config.sample_size, sample_size)
      self.assertEqual(
          config.edge_target_feature_name, edge_target_feature_name
      )
      self.assertTrue(config.HasField('edge_feature_names'))
      self.assertSetEqual(
          set(config.edge_feature_names.feature_names),
          {'weight', 'score', edge_target_feature_name, tfgnn.SOURCE_NAME},
      )

    if impl == 'UniformEdgesSampler':
      layer = core.UniformEdgesSampler(
          self._get_test_data(
              edge_target_feature_name=edge_target_feature_name,
              extra_feature_names=['weight', 'score'],
          ),
          sample_size=sample_size,
          edge_target_feature_name=edge_target_feature_name,
          name='uniform_edges_sampler',
      )
    elif impl == 'InMemUniformEdgesSampler':
      layer = core.InMemUniformEdgesSampler(
          num_source_nodes=3,
          source=tf.constant([2, 0, 0], tf.int64),
          target=tf.constant([0, 1, 2], tf.int64),
          edge_features={'weight': [0.1, 0.2, 0.3], 'score': [1.0, 2.0, 3.0]},
          sample_size=sample_size,
          name='uniform_edges_sampler',
          edge_set_name='edges',
      )
    else:
      raise NotImplementedError(impl)

    self.assertIsInstance(layer, interfaces.UniformEdgesSampler)

    i = tf.keras.Input(
        type_spec=tf.RaggedTensorSpec(
            [None, None], dtype=tf.int32, ragged_rank=1
        ),
        name='input',
    )
    model = tf.keras.Model(inputs=i, outputs=layer(i))
    program, _ = lib.create_program(model)
    self.assertIn('uniform_edges_sampler', program.layers)
    config = pb.EdgeSamplingConfig()
    self.assertTrue(program.layers['uniform_edges_sampler'].HasField('config'))
    program.layers['uniform_edges_sampler'].config.Unpack(config)
    check_config(config)

  @parameterized.parameters([
      dict(sample_size=1),
      dict(sample_size=8),
      dict(
          edge_target_feature_name='outgoing_edges',
      ),
      dict(
          weight_feature_name='scores',
      ),
      dict(
          weight_feature_name='scores',
          edge_target_feature_name='outgoing_edges',
      ),
  ])
  def testTopKEdgesSampler(
      self,
      *,
      sample_size: int = 1,
      edge_target_feature_name: str = tfgnn.TARGET_NAME,
      weight_feature_name: str = 'weight',
  ):
    def check_config(config):
      self.assertIsInstance(config, pb.EdgeSamplingConfig)
      self.assertEqual(config.edge_set_name, 'edges')
      self.assertEqual(config.sample_size, sample_size)
      self.assertEqual(
          config.edge_target_feature_name, edge_target_feature_name
      )
      self.assertEqual(config.weight_feature_name, weight_feature_name)
      self.assertTrue(config.HasField('edge_feature_names'))
      self.assertSetEqual(
          set(config.edge_feature_names.feature_names),
          {weight_feature_name, edge_target_feature_name, tfgnn.SOURCE_NAME},
      )

    layer = core.TopKEdgesSampler(
        self._get_test_data(
            edge_target_feature_name=edge_target_feature_name,
            extra_feature_names=[weight_feature_name],
        ),
        sample_size=sample_size,
        edge_target_feature_name=edge_target_feature_name,
        weight_feature_name=weight_feature_name,
        name='topk_edges_sampler',
    )

    self.assertIsInstance(layer, interfaces.TopKEdgesSampler)
    i = tf.keras.Input(
        type_spec=tf.RaggedTensorSpec(
            [None, None], dtype=tf.int32, ragged_rank=1
        ),
        name='input',
    )
    model = tf.keras.Model(inputs=i, outputs=layer(i))
    program, _ = lib.create_program(model)
    self.assertIn('topk_edges_sampler', program.layers)
    config = pb.EdgeSamplingConfig()
    self.assertTrue(program.layers['topk_edges_sampler'].HasField('config'))
    program.layers['topk_edges_sampler'].config.Unpack(config)
    check_config(config)


if __name__ == '__main__':
  tf.test.main()
