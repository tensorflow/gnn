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
import tempfile
from typing import Optional

from absl.testing import parameterized

import google.protobuf.text_format as pbtext
import tensorflow as tf

from tensorflow_gnn.experimental.sampler import core

rt = tf.ragged.constant


def save_and_load(model: tf.keras.Model) -> tf.keras.Model:
  with tempfile.TemporaryDirectory() as tmpdir:
    tf.keras.models.save_model(model, tmpdir)
    return tf.keras.models.load_model(tmpdir)


class Adder(core.CompositeLayer):

  def __init__(self, value, **argw):
    super().__init__(**argw)
    self._value = value

  def get_config(self):
    return dict(value=self._value, **super().get_config())

  def symbolic_call(self, inputs):
    value = tf.reshape(self._value, [-1])
    return tf.keras.layers.add([inputs, value], name='add_value')


class Add10(core.CompositeLayer):

  def symbolic_call(self, inputs):
    return Adder(10)(inputs)


class Dense32(core.CompositeLayer):

  def symbolic_call(self, inputs):
    return tf.keras.layers.Dense(32)(inputs)


class ResetIfTrainingImpl(tf.keras.layers.Layer):

  def call(self, inputs, training=None):
    return tf.zeros_like(inputs) if training else inputs


class ResetIfTraining(core.CompositeLayer):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._impl = ResetIfTrainingImpl()

  def symbolic_call(self, inputs):
    return self._impl(inputs)


class CompositeLayerTest(tf.test.TestCase):

  def testLvl1(self):
    def check(adder):
      self.assertAllEqual(adder(tf.convert_to_tensor([10])), [11])
      adder_model = adder.layers[1].wrapped_model
      self.assertAllEqual(adder_model([tf.convert_to_tensor([100])]), [101])
      self.assertEqual(adder_model.layers[1].name, 'add_value')

    i = tf.keras.Input([], dtype=tf.int32)
    o = Adder(1)(i)
    adder = tf.keras.Model(i, o)
    check(adder)

    with tf.keras.utils.custom_object_scope({'Adder': Adder}):
      adder_1 = save_and_load(adder)
      check(adder_1)
      adder_2 = save_and_load(adder_1)
      check(adder_2)

  def testLvl2(self):
    i = tf.keras.Input([], dtype=tf.int32)
    o = Add10()(i)
    adder = tf.keras.Model(i, o)
    self._check_lvl2(adder)

    with tf.keras.utils.custom_object_scope({'Add10': Add10, 'Adder': Adder}):
      adder_1 = save_and_load(adder)
      self._check_lvl2(adder_1)
      adder_2 = save_and_load(adder_1)
      self._check_lvl2(adder_2)

  def testLazyInitAfterLoad1(self):
    i = tf.keras.Input([], dtype=tf.int32)
    o = Add10()(i)
    adder = tf.keras.Model(i, o)
    self._check_lvl2(adder)

    with tf.keras.utils.custom_object_scope({'Add10': Add10, 'Adder': Adder}):
      adder_1 = save_and_load(adder)
      self.assertIsInstance(adder_1.layers[1].wrapped_model.layers[1], Adder)
      adder_2 = save_and_load(adder_1)
      self.assertIsInstance(adder_1.layers[1].wrapped_model.layers[1], Adder)
      self._check_lvl2(adder_2)

  def testLazyInitAfterLoad2(self):
    i = tf.keras.Input([], dtype=tf.int32)
    o = Add10()(i)
    adder = tf.keras.Model(i, o)
    self._check_lvl2(adder)

    with tf.keras.utils.custom_object_scope({'Add10': Add10, 'Adder': Adder}):
      adder_1 = save_and_load(adder)
      adder_2 = save_and_load(adder_1)
      adder_3 = save_and_load(adder_2)
      self._check_lvl2(adder_3)

  def testMultipleModels(self):
    def check(adder):
      self.assertAllEqual(adder(tf.convert_to_tensor([1])), [1111])

    i = tf.keras.Input([], dtype=tf.int32)
    o = Add10()(i)
    o = Adder(100)(o)
    o = Adder(1000)(o)
    adder = tf.keras.Model(i, o)
    check(adder)

    with tf.keras.utils.custom_object_scope({'Add10': Add10, 'Adder': Adder}):
      adder_1 = save_and_load(adder)
      check(adder_1)
      adder_2 = save_and_load(adder_1)
      check(adder_2)

  def testIsTrainable(self):
    i = tf.keras.Input([32], dtype=tf.float32)
    o = Dense32()(i)
    model = tf.keras.Model(i, o)
    self.assertEqual(model.trainable_weights[0].shape.as_list(), [32, 32])

    with tf.keras.utils.custom_object_scope({'Dense32': Dense32}):
      model_1 = save_and_load(model)
      self.assertEqual(model_1.trainable_weights[0].shape.as_list(), [32, 32])

  def testTrainingFlag(self):
    def check(model):
      self.assertAllEqual(model(tf.ones([3]), training=True), tf.zeros([3]))
      self.assertAllEqual(model(tf.ones([3]), training=False), tf.ones([3]))
      self.assertAllEqual(model(tf.ones([3])), tf.ones([3]))
      rng = tf.range(1, 4, dtype=tf.float32)
      self.assertAllEqual(model(rng, training=True), tf.zeros([3]))
      self.assertAllEqual(model(rng, training=False), rng)

    i = tf.keras.Input([3])
    o = ResetIfTraining()(i)
    model = tf.keras.Model(i, o)
    check(model)

    with tf.keras.utils.custom_object_scope({
        'ResetIfTraining': ResetIfTraining,
        'ResetIfTrainingImpl': ResetIfTrainingImpl,
    }):
      model_1 = save_and_load(model)
      check(model_1)
      model_2 = save_and_load(model_1)
      check(model_2)

  def testRaisesOnArgumentTypeChange(self):
    adder = Add10()
    adder(tf.convert_to_tensor([1], tf.int32))
    adder(tf.convert_to_tensor([2], tf.int32))
    with self.assertRaisesRegex(
        ValueError, 'called with different argument types'
    ):
      adder(tf.convert_to_tensor([2], tf.int64))

  def _check_lvl2(self, adder):
    self.assertAllEqual(adder(tf.convert_to_tensor([100])), [110])
    lvl1_model = adder.layers[1].wrapped_model
    lvl2_model = lvl1_model.layers[1].wrapped_model
    self.assertAllEqual(lvl2_model([tf.convert_to_tensor([100])]), [110])
    self.assertEqual(lvl2_model.layers[1].name, 'add_value')


class TfExamplesParserTest(tf.test.TestCase):

  def testWithoutDefault(self):
    serialized = [
        pbtext.Merge(
            r"""
            features {
              feature {key: "s" value {bytes_list {value: ['1']} } }
              feature {key: "v" value {int64_list {value: [1, 2]} } }
              feature {key: "r" value {float_list {value: [1.]} } }
            }""",
            tf.train.Example(),
        ).SerializeToString(),
        pbtext.Merge(
            r"""
            features {
              feature {key: "s" value {bytes_list {value: ['2']} } }
              feature {key: "v" value {int64_list {value: [3, 4]} } }
              feature {key: "r" value {float_list {value: [1., 2.]} } }
            }
            """,
            tf.train.Example(),
        ).SerializeToString(),
        pbtext.Merge(
            r"""
            features {
              feature {key: "s" value {bytes_list {value: ['3']} } }
              feature {key: "v" value {int64_list {value: [5, 6]} } }
            }
            """,
            tf.train.Example(),
        ).SerializeToString(),
    ]
    layer = core.TfExamplesParser({
        's': tf.TensorSpec([], tf.string),
        'v': tf.TensorSpec([2], tf.int64),
        'r': tf.TensorSpec([None], tf.float32),
    })

    ragged = tf.RaggedTensor.from_row_lengths(serialized, [2, 1])
    result = layer(ragged)
    self.assertAllEqual(result['s'], rt([['1', '2'], ['3']]))
    self.assertAllEqual(
        result['v'], rt([[[1, 2], [3, 4]], [[5, 6]]], ragged_rank=1)
    )
    self.assertAllEqual(
        result['r'], rt([[[1.0], [1.0, 2.0]], [[]]], ragged_rank=2)
    )

    dense1 = tf.reshape(serialized, [-1])
    result = layer(dense1)
    self.assertAllEqual(result['s'], ['1', '2', '3'])
    self.assertAllEqual(result['v'], [[1, 2], [3, 4], [5, 6]])
    self.assertAllEqual(result['r'], rt([[1.0], [1.0, 2.0], []]))

    dense2 = tf.reshape(serialized, [-1, 1])
    result = layer(dense2)
    self.assertAllEqual(result['s'], [['1'], ['2'], ['3']])
    self.assertAllEqual(result['v'], [[[1, 2]], [[3, 4]], [[5, 6]]])
    self.assertAllEqual(
        result['r'], rt([[[1.0]], [[1.0, 2.0]], [[]]], ragged_rank=2)
    )

  def testWithDefault(self):
    serialized = [
        pbtext.Merge(
            r"""
            features {
              feature {key: "s" value {bytes_list {value: ['1']} } }
              feature {key: "v" value {int64_list {value: [1, 2]} } }
            }""",
            tf.train.Example(),
        ).SerializeToString(),
        pbtext.Merge(
            r"""
            features {
              feature {key: "v" value {int64_list {value: [3, 4]} } }
            }
            """,
            tf.train.Example(),
        ).SerializeToString(),
        pbtext.Merge(
            r"""
            features {
              feature {key: "s" value {bytes_list {value: ['3']} } }
            }
            """,
            tf.train.Example(),
        ).SerializeToString(),
    ]
    serialized = tf.RaggedTensor.from_row_lengths(serialized, [1, 0, 2])
    layer = core.TfExamplesParser(
        {
            's': tf.TensorSpec([], tf.string),
            'v': tf.TensorSpec([2], tf.int64),
        },
        default_values={'s': 'x', 'v': [-1, -1]},
    )
    result = layer(serialized)
    self.assertAllEqual(result['s'], rt([['1'], [], ['x', '3']]))
    self.assertAllEqual(
        result['v'], rt([[[1, 2]], [], [[3, 4], [-1, -1]]], ragged_rank=1)
    )


class LookupLayersTest(tf.test.TestCase):

  def testStrKeyLookup(self):
    layer = core.InMemStringKeyToBytesAccessor(
        keys_to_values={
            'a': b'v.a',
            'b': b'v.b',
        }
    )
    result = layer(rt([['a', 'b'], ['x']]))
    self.assertAllEqual(result, rt([['v.a', 'v.b'], ['']]))

  def testStrKeyLookupWithDefault(self):
    layer = core.InMemStringKeyToBytesAccessor(
        keys_to_values={
            'a': b'v.a',
            'b': b'v.b',
        },
        default_value='v.?',
    )
    result = layer(rt([['a', 'x'], ['x']]))
    self.assertAllEqual(result, rt([['v.a', 'v.?'], ['v.?']]))

  def testStrKeyLookupWithNoDefault(self):
    layer = core.InMemStringKeyToBytesAccessor(
        keys_to_values={
            'a': b'v.a',
            'b': b'v.b',
        },
        default_value=None,
    )
    result = layer(rt([['a'], ['b', 'a']]))
    self.assertAllEqual(result, rt([['v.a'], ['v.b', 'v.a']]))
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError, 'inputs should be in vocabulary'
    ):
      layer(rt([['x']]))

  def testReourceName(self):
    layer = core.InMemStringKeyToBytesAccessor(
        name='lookup_table',
        keys_to_values={
            'x': b'?',
        },
    )
    self.assertEqual(layer.name, 'lookup_table')
    self.assertEqual(layer.resource_name, 'lookup_table')


class KeyToTfExampleTest(tf.test.TestCase):

  def testEager(self):
    table = core.InMemStringKeyToBytesAccessor(
        keys_to_values={
            'a': pbtext.Merge(
                r"""
                features {
                  feature {key: "s" value {int64_list {value: [1]} } }
                }""",
                tf.train.Example(),
            ).SerializeToString(),
            'b': pbtext.Merge(
                r"""
                features {
                  feature {key: "s" value {int64_list {value: [2]} } }
                }""",
                tf.train.Example(),
            ).SerializeToString(),
        }
    )
    layer = core.KeyToTfExampleAccessor(
        table,
        features_spec={
            's': tf.TensorSpec([], tf.int64),
        },
    )

    actual = layer(rt([['a', 'b'], ['c']]))['s']
    expected = rt([[1, 2], [0]])
    self.assertAllEqual(actual, expected)

  def testSymbolic(self):
    table = core.InMemIntegerKeyToBytesAccessor(
        keys_to_values={
            1: pbtext.Merge(
                r"""
                features {
                  feature {key: "f" value {float_list {value: [1.0, 2.0]} } }
                }""",
                tf.train.Example(),
            ).SerializeToString(),
        }
    )
    layer = core.KeyToTfExampleAccessor(
        table,
        features_spec={
            'f': tf.TensorSpec([2]),
        },
        default_values={'f': [-1, -1]},
    )

    def check_results(model):
      result = model(rt([[1, 0], [0]]))
      expected = rt([[[1.0, 2.0], [-1.0, -1.0]], [[-1.0, -1.0]]], ragged_rank=1)
      self.assertAllEqual(result['f'], expected)

    i = tf.keras.Input(type_spec=tf.RaggedTensorSpec([None, None], tf.int64))
    o = layer(i)
    model = tf.keras.Model(inputs=i, outputs=o)
    check_results(model)

    restored_model = save_and_load(model)
    check_results(restored_model)
    self.assertIsInstance(restored_model.layers[1], core.KeyToTfExampleAccessor)
    restored_accessor = restored_model.layers[1].wrapped_model.layers[1]
    self.assertIsInstance(
        restored_accessor, core.InMemIntegerKeyToBytesAccessor
    )

  def testAccessorSharing(self):
    table = core.InMemIntegerKeyToBytesAccessor(
        keys_to_values={
            1: pbtext.Merge(
                r"""
                features {
                  feature {key: "f" value {float_list {value: [1.0, 2.0]} } }
                }""",
                tf.train.Example(),
            ).SerializeToString(),
        }
    )
    layer1 = core.KeyToTfExampleAccessor(
        table,
        features_spec={
            'f': tf.TensorSpec([2]),
        },
        default_values={'f': [-1, -1]},
    )
    layer2 = core.KeyToTfExampleAccessor(
        table,
        features_spec={
            'f': tf.TensorSpec([2]),
        },
        default_values={'f': [-2, -2]},
    )

    def check_results(model):
      result = model(rt([[1, 0], [0]]))
      expected = rt(
          [[[11.0, 22.0], [-21.0, -21.0]], [[-21.0, -21.0]]], ragged_rank=1
      )
      self.assertAllEqual(result, expected)

    i = tf.keras.Input(type_spec=tf.RaggedTensorSpec([None, None], tf.int64))
    o = layer1(i)['f'] + 10.0 * layer2(i)['f']
    model = tf.keras.Model(inputs=i, outputs=o)

    restored_model = save_and_load(model)
    check_results(restored_model)
    self.assertIsInstance(restored_model.layers[1], core.KeyToTfExampleAccessor)
    restored_accessor = restored_model.layers[1].wrapped_model.layers[1]
    self.assertIsInstance(
        restored_accessor, core.InMemIntegerKeyToBytesAccessor
    )

  def testMultipleLayers(self):
    layer1 = core.KeyToTfExampleAccessor(
        core.InMemStringKeyToBytesAccessor(
            keys_to_values={
                'a': pbtext.Merge(
                    r"""
                    features {
                      feature {key: "f" value {float_list {value: 1.0} } }
                    }""",
                    tf.train.Example(),
                ).SerializeToString(),
            }
        ),
        features_spec={
            'f': tf.TensorSpec([], tf.float32),
        },
    )
    layer2 = core.KeyToTfExampleAccessor(
        core.InMemStringKeyToBytesAccessor(
            keys_to_values={
                'a': pbtext.Merge(
                    r"""
                    features {
                      feature {key: "i" value {int64_list {value: 10} } }
                    }""",
                    tf.train.Example(),
                ).SerializeToString(),
            }
        ),
        features_spec={
            'i': tf.TensorSpec([], tf.int64),
        },
    )

    def check_results(model):
      result = model(rt([['a']]))
      expected = rt([[10.0]], ragged_rank=1)
      self.assertAllEqual(result, expected)

    i = tf.keras.Input(type_spec=tf.RaggedTensorSpec([None, None], tf.string))
    o = layer1(i)['f'] * tf.cast(layer2(i)['i'], tf.float32)
    model = tf.keras.Model(inputs=i, outputs=o)
    check_results(model)

    restored_model = save_and_load(model)
    check_results(restored_model)
    self.assertIsInstance(restored_model.layers[1], core.KeyToTfExampleAccessor)
    restored_layer1 = restored_model.layers[1].wrapped_model.layers[1]
    self.assertIsInstance(restored_layer1, core.InMemStringKeyToBytesAccessor)
    restored_layer2 = restored_model.layers[2].wrapped_model.layers[1]
    self.assertIsInstance(restored_layer2, core.InMemStringKeyToBytesAccessor)

  def testReourceName(self):
    layer = core.KeyToTfExampleAccessor(
        core.InMemStringKeyToBytesAccessor(
            keys_to_values={
                '?': b'',
            },
            name='node_features',
        ),
        features_spec={
            'f': tf.TensorSpec([], tf.float32),
        },
    )
    self.assertEqual(layer.resource_name, 'node_features')


class UniformEdgesSamplerTest(tf.test.TestCase, parameterized.TestCase):

  def _get_test_data(
      self, edge_target_feature_name: str, table_name: Optional[str] = None
  ):
    table = core.InMemStringKeyToBytesAccessor(
        keys_to_values={
            'a': pbtext.Merge(
                r"""
                features {
                  feature {
                      key: "%s"
                      value { bytes_list {value: ['b', 'c']} }
                  }
                  feature {
                      key: "weights"
                      value { float_list {value: [2., 3.]} }
                  }
                }""" % edge_target_feature_name,
                tf.train.Example(),
            ).SerializeToString(),
            'b': pbtext.Merge(
                r"""
                features {
                  feature {
                      key: "%s"
                      value { bytes_list {value: ['a']} }
                  }
                  feature {
                      key: "weights"
                      value { float_list {value: [1.]} }
                  }
                }""" % edge_target_feature_name,
                tf.train.Example(),
            ).SerializeToString(),
        },
        name=table_name,
    )
    return core.KeyToTfExampleAccessor(
        table,
        features_spec={
            edge_target_feature_name: tf.TensorSpec([None], tf.string),
            'weights': tf.TensorSpec([None], tf.float32),
        },
    )

  @parameterized.product(table_name=['nodes', 'edges'], sample_size=[1, 8])
  def testAttributes(self, table_name: str, sample_size: int):
    layer = core.UniformEdgesSampler(
        self._get_test_data('neighbors', table_name=table_name),
        sample_size=sample_size,
        edge_target_feature_name='neighbors',
    )
    self.assertEqual(layer.resource_name, table_name)
    self.assertEqual(layer.sample_size, sample_size)

  @parameterized.parameters(list(range(10)))
  def testSampling1(self, seed):
    layer = core.UniformEdgesSampler(
        self._get_test_data('neighbors'),
        sample_size=1,
        edge_target_feature_name='neighbors',
        seed=seed,
    )
    result = layer(rt([['b']]))
    self.assertSetEqual(set(result.keys()), {'#source', '#target', 'weights'})
    self.assertAllEqual(result['weights'], rt([[1.0]]))
    self.assertAllEqual(result['#source'], rt([['b']]))
    self.assertAllEqual(result['#target'], rt([['a']]))

    result = layer(rt([['a'], ['xxx'], []]))
    self.assertSetEqual(set(result.keys()), {'#source', '#target', 'weights'})
    for example_id in [1, 2]:
      self.assertAllEqual(
          result['weights'][example_id, :], tf.convert_to_tensor([], tf.float32)
      )
      self.assertAllEqual(
          result['#source'][example_id, :],
          tf.convert_to_tensor([], tf.string),
      )
      self.assertAllEqual(
          result['#target'][example_id, :],
          tf.convert_to_tensor([], tf.string),
      )

    target_ids = list(result['#target'][0, :].numpy())
    self.assertLen(target_ids, 1)
    self.assertIn(target_ids[0], [b'b', b'c'])
    self.assertAllEqual(result['#source'][0, :], ['a'])
    self.assertAllEqual(
        result['weights'][0, :], [2.0 if target_ids[0] == b'b' else 3.0]
    )

  @parameterized.parameters([2, 3, 10])
  def testSampling2(self, sample_size):
    layer = core.UniformEdgesSampler(
        self._get_test_data('#target'),
        sample_size=sample_size,
        seed=2,
    )
    result = layer(rt([['a', 'b'], ['a'], ['b']]))
    self.assertSetEqual(
        set(result['#target'][0, :].numpy()), {b'a', b'b', b'c'}
    )
    self.assertSetEqual(set(result['#source'][0, :].numpy()), {b'a', b'b'})
    self.assertSetEqual(set(result['weights'][0, :].numpy()), {1.0, 2.0, 3.0})
    self.assertSetEqual(set(result['#target'][1, :].numpy()), {b'b', b'c'})
    self.assertSetEqual(set(result['#source'][1, :].numpy()), {b'a'})
    self.assertSetEqual(set(result['weights'][1, :].numpy()), {2.0, 3.0})
    self.assertSetEqual(set(result['#target'][2, :].numpy()), {b'a'})
    self.assertSetEqual(set(result['#source'][2, :].numpy()), {b'b'})
    self.assertSetEqual(set(result['weights'][2, :].numpy()), {1.0})

  @parameterized.parameters([2, 3, 10])
  def testFeaturesOrder(self, sample_size):
    layer = core.UniformEdgesSampler(
        self._get_test_data('neighbors'),
        sample_size=sample_size,
        edge_target_feature_name='neighbors',
        seed=42,
    )
    result = layer(rt([['a']]))
    target_ids = list(result['#target'][0, :].numpy())
    self.assertAllEqual(
        result['weights'][0, :],
        [2.0, 3.0] if target_ids[0] == b'b' else [3.0, 2.0],
    )

  def testRandomness(self):
    layer = core.UniformEdgesSampler(
        self._get_test_data('#target'),
        sample_size=1,
        seed=42,
    )
    result = layer(rt([['a'] * 1000]))
    self.assertSetEqual(
        set(result['#target'].flat_values.numpy()), {b'b', b'c'}
    )

  def testSerialization(self):
    keys = tf.keras.Input(
        type_spec=tf.RaggedTensorSpec(
            [None, None], dtype=tf.string, ragged_rank=1
        )
    )

    layer = core.UniformEdgesSampler(
        self._get_test_data('#target'),
        sample_size=1,
        seed=1,
    )

    result = layer(keys)

    def check_results(model):
      self.assertIsInstance(
          model.layers[1],
          core.UniformEdgesSampler,
      )
      self.assertIsInstance(
          model.layers[1].wrapped_model.layers[1],
          core.KeyToTfExampleAccessor,
      )

      result = model(rt([['a'] * 1000]))
      self.assertSetEqual(
          set(result['#target'].flat_values.numpy()), {b'b', b'c'}
      )

    model = tf.keras.Model(keys, result)
    check_results(model)
    restored_model = save_and_load(model)
    check_results(restored_model)


class GraphTensorBuilderTest(tf.test.TestCase):

  def testContext(self):
    graph = core.build_graph_tensor(context={'label': [['G'], ['B']]})
    self.assertAllEqual(graph.shape, tf.TensorShape([2]))
    self.assertAllEqual(graph.context.shape, tf.TensorShape([2]))
    self.assertAllEqual(graph.context.sizes, [[1], [1]])
    self.assertAllEqual(graph.context['label'], [['G'], ['B']])

  def testOneNodeSet(self):
    graph = core.build_graph_tensor(
        context={'label': [['G'], ['B'], ['G']]},
        node_sets={'node': {'#id': rt([[1, 2, 3], [1, 2], []])}},
    )
    self.assertAllEqual(graph.shape, tf.TensorShape([3]))
    self.assertAllEqual(graph.node_sets['node'].sizes, [[3], [2], [0]])
    self.assertAllEqual(
        graph.node_sets['node']['#id'], rt([[1, 2, 3], [1, 2], []])
    )

  def testTwoNodeSets(self):
    graph = core.build_graph_tensor(
        node_sets={
            'A': {
                '#id': rt([[1, 2, 3], [1, 2]]),
                'f.s': rt([[1.0, 2.0, 3.0], [1.0, 2.0]]),
            },
            'B': {
                '#id': rt([['a', 'b'], ['a']]),
                'f.s': rt([['1', '2'], ['1']]),
                'f.v': rt([[[1, 2], [2, 3]], [[1, 2]]], ragged_rank=1),
            },
        }
    )
    self.assertAllEqual(graph.node_sets['A'].sizes, [[3], [2]])
    self.assertAllEqual(
        graph.node_sets['A']['f.s'], rt([[1.0, 2.0, 3.0], [1.0, 2.0]])
    )

    self.assertAllEqual(graph.node_sets['B'].sizes, [[2], [1]])
    self.assertAllEqual(graph.node_sets['B']['f.s'], rt([['1', '2'], ['1']]))
    self.assertAllEqual(
        graph.node_sets['B']['f.v'],
        rt([[[1, 2], [2, 3]], [[1, 2]]], ragged_rank=1),
    )

  def testHomogeneous(self):
    graph = core.build_graph_tensor(
        node_sets={
            'A': [
                {
                    '#id': rt([['a'], ['c']]),
                    'f.s': rt([[0], [2]]),
                },
                {
                    '#id': rt([['b'], []]),
                    'f.s': rt([[1], []]),
                },
            ]
        },
        edge_sets={
            'A,A->A,A': [
                {
                    '#source': rt([['a'], ['c']]),
                    '#target': rt([['b'], ['c']]),
                    'f.s': rt([[0], [2]]),
                },
                {
                    '#source': rt([['b'], ['c']]),
                    '#target': rt([['a'], ['c']]),
                    'f.s': rt([[1], [2]]),
                },
            ],
        },
    )
    self.assertAllEqual(graph.node_sets['A'].sizes, [[2], [1]])
    self.assertAllEqual(graph.node_sets['A']['#id'], rt([['a', 'b'], ['c']]))
    self.assertAllEqual(graph.node_sets['A']['f.s'], rt([[0, 1], [2]]))

    self.assertAllEqual(graph.edge_sets['A->A'].sizes, [[2], [2]])
    self.assertAllEqual(
        graph.edge_sets['A->A'].adjacency.source, rt([[0, 1], [0, 0]])
    )
    self.assertAllEqual(
        graph.edge_sets['A->A'].adjacency.target, rt([[1, 0], [0, 0]])
    )
    self.assertAllEqual(graph.edge_sets['A->A']['f.s'], rt([[0, 1], [2, 2]]))

  def testLatentNodeSets(self):
    graph = core.build_graph_tensor(
        edge_sets={
            'A,A->A,A': {
                '#source': rt([['a', 'a', 'a'], ['c', 'c']]),
                '#target': rt([['a', 'b', 'a'], ['c', 'c']]),
                'f.s': rt([[1, 2, 3], [1, 2]]),
            },
        }
    )
    self.assertAllEqual(graph.node_sets['A'].sizes, [[2], [1]])
    self.assertAllEqual(graph.node_sets['A']['#id'], rt([['a', 'b'], ['c']]))

    self.assertAllEqual(graph.edge_sets['A->A'].sizes, [[3], [2]])
    self.assertAllEqual(
        graph.edge_sets['A->A'].adjacency.source, rt([[0, 0, 0], [0, 0]])
    )
    self.assertAllEqual(
        graph.edge_sets['A->A'].adjacency.target, rt([[0, 1, 0], [0, 0]])
    )

  def testHeterogeneous1(self):
    graph = core.build_graph_tensor(
        node_sets={
            'A': {
                '#id': rt([[1, 2, 3], [1, 2], [3]]),
            },
        },
        edge_sets={
            'A,A->B,B': {
                '#source': rt([[1, 2, 2, 1], [2, 2, 2], [3]]),
                '#target': rt([[1, 2, 3, 4], [5, 6, 7], [1]]),
            }
        },
    )
    self.assertAllEqual(graph.shape, tf.TensorShape([3]))
    self.assertAllEqual(graph.node_sets['A'].sizes, [[3], [2], [1]])
    self.assertAllEqual(
        graph.node_sets['A']['#id'], rt([[1, 2, 3], [1, 2], [3]])
    )
    self.assertAllEqual(graph.node_sets['B'].sizes, [[4], [3], [1]])
    self.assertAllEqual(
        graph.node_sets['B']['#id'], rt([[1, 2, 3, 4], [5, 6, 7], [1]])
    )

    self.assertAllEqual(graph.edge_sets['A->B'].sizes, [[4], [3], [1]])
    self.assertAllEqual(
        graph.edge_sets['A->B'].adjacency.source,
        rt([[0, 1, 1, 0], [1, 1, 1], [0]]),
    )
    self.assertAllEqual(
        graph.edge_sets['A->B'].adjacency.target,
        rt([[0, 1, 2, 3], [0, 1, 2], [0]]),
    )

  def testHeterogeneous2(self):
    graph = core.build_graph_tensor(
        node_sets={
            'A': {
                '#id': rt([[1, 2, 3], [1, 2], [3]]),
            },
            'B': {
                '#id': rt([[4, 3, 2, 1], [7, 6, 5], [1]]),
            },
        },
        edge_sets={
            'A,A->B,B': {
                '#source': rt([[1, 2, 2, 1], [2, 2, 2], [3]]),
                '#target': rt([[1, 2, 3, 4], [5, 6, 7], [1]]),
            },
            'B,B->A,A': {
                '#source': rt([[1], [5], [1]]),
                '#target': rt([[1], [2], [3]]),
            },
        },
    )
    self.assertAllEqual(graph.shape, tf.TensorShape([3]))
    self.assertAllEqual(graph.node_sets['A'].sizes, [[3], [2], [1]])
    self.assertAllEqual(
        graph.node_sets['A']['#id'], rt([[1, 2, 3], [1, 2], [3]])
    )
    self.assertAllEqual(graph.node_sets['B'].sizes, [[4], [3], [1]])
    self.assertAllEqual(
        graph.node_sets['B']['#id'], rt([[4, 3, 2, 1], [7, 6, 5], [1]])
    )

    self.assertAllEqual(graph.edge_sets['A->B'].sizes, [[4], [3], [1]])
    self.assertAllEqual(
        graph.edge_sets['A->B'].adjacency.source,
        rt([[0, 1, 1, 0], [1, 1, 1], [0]]),
    )
    self.assertAllEqual(
        graph.edge_sets['A->B'].adjacency.target,
        rt([[3, 2, 1, 0], [2, 1, 0], [0]]),
    )

    self.assertAllEqual(graph.edge_sets['B->A'].sizes, [[1], [1], [1]])
    self.assertAllEqual(
        graph.edge_sets['B->A'].adjacency.source, rt([[3], [2], [0]])
    )
    self.assertAllEqual(
        graph.edge_sets['B->A'].adjacency.target, rt([[0], [1], [0]])
    )


if __name__ == '__main__':
  tf.test.main()
