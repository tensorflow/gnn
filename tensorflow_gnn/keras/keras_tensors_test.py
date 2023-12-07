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
"""Tests for KerasTensor specializations for GraphTensor pieces."""

import functools
import inspect
import os
from typing import Mapping

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_gnn import runner
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import broadcast_ops
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import graph_tensor_ops
from tensorflow_gnn.graph import pool_ops
from tensorflow_gnn.keras import keras_tensors as kt
from tensorflow_gnn.utils import tf_test_utils as tftu

# Enables tests for graph pieces that are members of test classes.
const.enable_graph_tensor_validation_at_runtime()

as_tensor = tf.convert_to_tensor
as_ragged = tf.ragged.constant

_TEST_GRAPH_TENSOR = gt.GraphTensor.from_pieces(
    gt.Context.from_fields(features={'f': as_tensor([1])}),
    node_sets={
        'a': gt.NodeSet.from_fields(
            features={'f': as_tensor([1, 2]), 'r': as_ragged([[1], [2, 3]])},
            sizes=as_tensor([2]),
        ),
        'b': gt.NodeSet.from_fields(
            features={'f': as_tensor([1, 2, 3])},
            sizes=as_tensor([3]),
        ),
    },
    edge_sets={
        'a->b': gt.EdgeSet.from_fields(
            features={'f': as_tensor([1, 2, 3])},
            sizes=as_tensor([3]),
            adjacency=adj.Adjacency.from_indices(
                ('a', as_tensor([0, 1, 1])), ('b', as_tensor([0, 0, 1]))
            ),
        ),
        'b->b': gt.EdgeSet.from_fields(
            features={'f': as_tensor([1])},
            sizes=as_tensor([1]),
            adjacency=adj.Adjacency.from_indices(
                ('b', as_tensor([0])), ('b', as_tensor([1]))
            ),
        ),
    },
)


def _min_pool_v2_feature_value(graph_tensor):
  a_b = graph_tensor.edge_sets['a->b']['f'] * 10
  b_b = graph_tensor.edge_sets['b->b']['f'] * 100
  return pool_ops.pool_v2(
      graph_tensor,
      to_tag=const.TARGET,
      edge_set_name=['a->b', 'b->b'],
      reduce_type='min_no_inf',
      feature_value=[a_b, b_b],
  )


def _broadcast_v2_feature_value(graph_tensor):
  feature_value = graph_tensor.node_sets['b']['f'] * 100
  return broadcast_ops.broadcast_v2(
      graph_tensor,
      from_tag=const.SOURCE,
      edge_set_name='b->b',
      feature_value=feature_value,
  )


def _shuffle_nodes(graph_tensor):
  result = graph_tensor_ops.shuffle_nodes(graph_tensor, node_sets=('a', 'b'))
  set_a = result.node_sets['a']
  return tf.math.reduce_sum(set_a['r']) + tf.math.reduce_sum(set_a['r'])


_OPS_TEST_CASES = [
    (
        'sum_pool',
        'pool_edges_to_node',
        functools.partial(
            pool_ops.pool_edges_to_node,
            edge_set_name='a->b',
            node_tag=const.TARGET,
            reduce_type='sum',
            feature_name='f',
        ),
    ),
    (
        'max_pool_v2',
        'pool',
        functools.partial(
            pool_ops.pool_v2,
            to_tag=const.SOURCE,
            edge_set_name='b->b',
            reduce_type='max_no_inf',
            feature_name='f',
        ),
    ),
    (
        'min_pool_v2_feature_value',
        'pool',
        _min_pool_v2_feature_value,
    ),
    (
        'broadcast_v2_feature_value',
        'broadcast',
        _broadcast_v2_feature_value,
    ),
    (
        'broadcast_dense',
        'broadcast_node_to_edges',
        functools.partial(
            broadcast_ops.broadcast_node_to_edges,
            edge_set_name='a->b',
            node_tag=const.SOURCE,
            feature_name='f',
        ),
    ),
    (
        'broadcast_ragged',
        'broadcast_node_to_edges',
        functools.partial(
            broadcast_ops.broadcast_node_to_edges,
            edge_set_name='a->b',
            node_tag=const.SOURCE,
            feature_name='r',
        ),
    ),
    (
        'shuffle_nodes',
        'shuffle_nodes',
        _shuffle_nodes,
    ),
    (
        'gather_first_node',
        'gather_first_node',
        functools.partial(
            graph_tensor_ops.gather_first_node,
            node_set_name='a',
            feature_name='r',
        ),
    ),
]


class _TestBase(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    const.enable_graph_tensor_validation_at_runtime()

  def assertFieldsEqual(self, actual: const.Fields, expected: const.Fields):
    self.assertIsInstance(actual, Mapping)
    self.assertAllEqual(actual.keys(), expected.keys())
    for key in actual.keys():
      self.assertAllEqual(actual[key], expected[key], msg=f'feature={key}')


class _SaveAndLoadTestBase(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    const.enable_graph_tensor_validation_at_runtime()

  def _save_and_load_inference_model(
      self, model: tf.keras.Model, use_legacy_model_save: bool
  ) -> tf.keras.Model:
    if tf.__version__.startswith('2.12.') and not use_legacy_model_save:
      self.skipTest('Model.export() does not work for TF < 2.13')
    path = os.path.join(self.get_temp_dir(), 'model')
    runner.export_model(model, path,
                        use_legacy_model_save=use_legacy_model_save)
    return tf.saved_model.load(path)


class ClassMethodsTest(_TestBase):

  @parameterized.named_parameters([
      ('scalar', as_tensor([1])),
      ('vector', as_tensor([[1], [2], [3]])),
      ('ragged', as_ragged([[1, 2], [3]])),
  ])
  def testContext(self, test_feature):
    test_feature_spec = tf.type_spec_from_value(test_feature)
    feature = tf.keras.Input(type_spec=test_feature_spec)
    context = gt.Context.from_fields(features={'feat': feature})
    self.assertIsInstance(context, kt.ContextKerasTensor)
    self.assertEqual(context.features['feat'].type_spec, test_feature_spec)
    model = tf.keras.Model(feature, context)

    self.assertAllEqual(model(test_feature)['feat'], test_feature)

    # test symbolic call
    context = model(feature)
    self.assertIsInstance(context, kt.ContextKerasTensor)
    self.assertEqual(context.features['feat'].type_spec, test_feature_spec)

  @parameterized.product(
      indices_dtype=[tf.int32, tf.int64],
      shape=[tf.TensorShape([]), tf.TensorShape([2]), [], [2]],
  )
  def testWithNonTensorArguments(
      self, indices_dtype: tf.DType, shape: tf.TensorShape
  ):
    feature = tf.keras.Input(type_spec=tf.TensorSpec([2, 2]))
    context = gt.Context.from_fields(
        features={'feat': feature}, indices_dtype=indices_dtype, shape=shape
    )
    self.assertIsInstance(context, kt.ContextKerasTensor)
    self.assertEqual(context.shape, tf.TensorShape(shape))
    self.assertEqual(context.indices_dtype, indices_dtype)

  @parameterized.named_parameters([
      ('scalar', as_tensor([3]), as_tensor([1, 2, 3])),
      ('ragged', as_tensor([3]), as_ragged([[1], [], [1, 2, 3]])),
  ])
  def testNodeSet(self, test_sizes, test_feature):
    test_sizes_spec = tf.type_spec_from_value(test_sizes)
    test_feature_spec = tf.type_spec_from_value(test_feature)
    sizes = tf.keras.Input(type_spec=test_sizes_spec)
    feature = tf.keras.Input(type_spec=test_feature_spec)

    node_set = gt.NodeSet.from_fields(features={'feat': feature}, sizes=sizes)
    self.assertIsInstance(node_set, kt.NodeSetKerasTensor)
    self.assertEqual(node_set.features['feat'].type_spec, feature.type_spec)
    self.assertEqual(node_set.sizes.type_spec, sizes.type_spec)

    model = tf.keras.Model([sizes, feature], node_set)
    result = model([test_sizes, test_feature])
    self.assertAllEqual(result.sizes, test_sizes)
    self.assertAllEqual(result['feat'], test_feature)

    result = model([test_sizes, test_feature])
    self.assertAllEqual(result.sizes, test_sizes)
    self.assertAllEqual(result['feat'], test_feature)

    # test symbolic call
    node_set = model([sizes, feature])
    self.assertIsInstance(node_set, kt.NodeSetKerasTensor)
    self.assertEqual(node_set.sizes.type_spec, sizes.type_spec)
    self.assertEqual(node_set.features['feat'].type_spec, feature.type_spec)

  @parameterized.named_parameters([
      ('list', [2]),
      ('np.array', np.array([3])),
      ('tf.constant', tf.constant([1, 2])),
  ])
  def testConstantInput(self, sizes):
    feature = tf.keras.Input(type_spec=tf.TensorSpec([None]))
    node_set = gt.NodeSet.from_fields(features={'feat': feature}, sizes=sizes)
    self.assertIsInstance(node_set, kt.NodeSetKerasTensor)
    self.assertEqual(node_set.features['feat'].type_spec, feature.type_spec)
    self.assertEqual(node_set.sizes.type_spec, tf.type_spec_from_value(sizes))

  def testAdjacency(self):
    index = tf.keras.Input(type_spec=tf.TensorSpec([None], dtype=tf.int64))
    self.assertIsInstance(
        adj.HyperAdjacency.from_indices({
            0: ('a', index),
            1: ('b', index),
            2: ('c', index),
        }),
        kt.HyperAdjacencyKerasTensor,
    )
    self.assertIsInstance(
        adj.Adjacency.from_indices(
            ('a', index),
            ('b', index),
        ),
        kt.AdjacencyKerasTensor,
    )

  def testEdgeSet(self):
    source = tf.keras.Input(type_spec=tf.TensorSpec([None], dtype=tf.int32))
    target = tf.keras.Input(type_spec=tf.TensorSpec([None], dtype=tf.int32))

    adjacency = adj.Adjacency.from_indices(('a', source), ('b', target))
    self.assertIsInstance(adjacency, kt.AdjacencyKerasTensor)
    edge_set = gt.EdgeSet.from_fields(
        sizes=tf.expand_dims(tf.shape(source)[0], axis=-1),
        adjacency=adjacency,
    )
    self.assertIsInstance(edge_set, kt.EdgeSetKerasTensor)
    self.assertEqual(edge_set.indices_dtype, tf.int32)
    model = tf.keras.Model([source, target], edge_set)

    result = model([as_tensor([1]), as_tensor([1])])
    self.assertAllEqual(result.sizes, [1])

    result = model([as_tensor([0, 1, 2]), as_tensor([0, 0, 0])])
    self.assertAllEqual(result.sizes, [3])

  def testGraphTensor(self):
    source = tf.keras.Input(type_spec=tf.TensorSpec([None], dtype=tf.int64))
    target = tf.keras.Input(type_spec=tf.TensorSpec([None], dtype=tf.int64))
    num_nodes = tf.keras.Input(type_spec=tf.TensorSpec([], dtype=tf.int64))

    edge_set = gt.EdgeSet.from_fields(
        sizes=[tf.shape(source)[0]],
        adjacency=adj.Adjacency.from_indices(
            ('node', source), ('node', target)
        ),
    )

    node_set = gt.NodeSet.from_fields(
        sizes=[num_nodes],
    )
    graph = gt.GraphTensor.from_pieces(
        node_sets={'node': node_set}, edge_sets={'edge': edge_set}
    )

    self.assertIsInstance(graph, kt.GraphKerasTensor)

    model = tf.keras.Model([source, target, num_nodes], graph)
    graph = model([as_tensor([0, 1]), as_tensor([0, 0]), as_tensor(2)])
    self.assertAllEqual(graph.node_sets['node'].sizes, [2])
    adjacency = graph.edge_sets['edge'].adjacency
    self.assertAllEqual(adjacency.source, [0, 1])
    self.assertAllEqual(adjacency.target, [0, 0])


class ClassMethodsSavingTest(_SaveAndLoadTestBase):

  def _get_test_model(self):
    source = tf.keras.Input(type_spec=tf.TensorSpec(
        [None], dtype=tf.int64, name='source'))
    target = tf.keras.Input(type_spec=tf.TensorSpec(
        [None], dtype=tf.int64, name='target'))
    num_nodes = tf.keras.Input(type_spec=tf.TensorSpec(
        [], dtype=tf.int64, name='num_nodes'))

    edge_set = gt.EdgeSet.from_fields(
        sizes=[tf.shape(source)[0]],
        adjacency=adj.Adjacency.from_indices(
            ('node', source), ('node', target)
        ),
    )

    node_set = gt.NodeSet.from_fields(
        sizes=[num_nodes],
    )
    graph = gt.GraphTensor.from_pieces(
        node_sets={'node': node_set}, edge_sets={'edge': edge_set}
    )

    self.assertIsInstance(graph, kt.GraphKerasTensor)

    return tf.keras.Model([source, target, num_nodes], graph)

  @parameterized.named_parameters(
      ('Baseline', tftu.ModelReloading.SKIP),
      ('SavedModel', tftu.ModelReloading.SAVED_MODEL),
      ('Keras', tftu.ModelReloading.KERAS),
      ('KerasV3', tftu.ModelReloading.KERAS_V3))
  def testKerasModelSaving(self, model_reloading):
    model = self._get_test_model()
    model = tftu.maybe_reload_model(self, model, model_reloading,
                                    'class-methods-model')
    graph = model([as_tensor([0, 1], tf.int64), as_tensor([0, 0], tf.int64),
                   as_tensor(2, tf.int64)])
    self.assertAllEqual(graph.node_sets['node'].sizes, [2])
    adjacency = graph.edge_sets['edge'].adjacency
    self.assertAllEqual(adjacency.source, [0, 1])
    self.assertAllEqual(adjacency.target, [0, 0])

  @parameterized.parameters([True, False])
  def testSavingForInference(self, use_legacy_model_save: bool):
    model = self._get_test_model()
    total_sizes = tf.reduce_sum(
        model.output.node_sets['node'].sizes
    ) + tf.reduce_sum(model.output.edge_sets['edge'].sizes)
    outputs = tf.keras.layers.Layer(name='total_sizes')(total_sizes)
    inference_model = tf.keras.Model(inputs=model.inputs, outputs=outputs)
    restored_model = self._save_and_load_inference_model(
        inference_model, use_legacy_model_save
    )
    signature = restored_model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    actual_total_sizes = signature(
        source=as_tensor([0, 1], tf.int64),
        target=as_tensor([0, 0], tf.int64),
        num_nodes=as_tensor(2, tf.int64),
    )['total_sizes']
    self.assertAllEqual(actual_total_sizes, 2 + 2)


class FunctionalModelTest(_TestBase):
  features = {
      'v': as_tensor([[0], [1]]),
      'r': as_ragged([[1, 2], []]),
  }

  @parameterized.parameters([
      gt.Context.from_fields(features=features),
      gt.NodeSet.from_fields(
          features=features,
          sizes=as_tensor([2]),
      ),
      gt.EdgeSet.from_fields(
          features=features,
          sizes=as_tensor([2]),
          adjacency=adj.Adjacency.from_indices(
              ('node', as_tensor([0, 1])), ('node', as_tensor([0, 0]))
          ),
      ),
  ])
  def testFeatures(self, example):
    piece_input = tf.keras.Input(type_spec=example.spec)
    f = piece_input.get_features_dict()
    f['s'] = tf.squeeze(piece_input['v'], axis=1)
    f['s'] += tf.reduce_sum(piece_input['r'], axis=1)
    piece_output = piece_input.replace_features(f)
    self.assertIsInstance(piece_output, piece_input.__class__)
    model = tf.keras.Model(piece_input, piece_output)
    result = model(example)
    self.assertFieldsEqual(
        result.features,
        {
            's': as_tensor([0 + 1 + 2, 1]),
            'v': as_tensor([[0], [1]]),
            'r': as_ragged([[1, 2], []]),
        },
    )

  @parameterized.parameters([
      gt.Context.from_fields(
          sizes=as_tensor([1, 1]),
      ),
      gt.NodeSet.from_fields(
          features={},
          sizes=as_tensor([1, 1]),
      ),
      gt.EdgeSet.from_fields(
          features={},
          sizes=as_tensor([1, 1]),
          adjacency=adj.Adjacency.from_indices(
              ('node', as_tensor([0, 1])), ('node', as_tensor([0, 0]))
          ),
      ),
  ])
  def testSizes(self, example):
    piece_input = tf.keras.Input(type_spec=example.spec)
    sizes = tf.keras.Model(piece_input, piece_input.sizes)(example)
    self.assertAllEqual(sizes, [1, 1])

    total_size = tf.keras.Model(piece_input, piece_input.total_size)(example)
    self.assertAllEqual(total_size, 2)

  @parameterized.parameters([
      gt.Context.from_fields(
          sizes=as_tensor([[1], [2]]),
      ),
      gt.NodeSet.from_fields(
          features={},
          sizes=as_tensor([[2], [1]]),
      ),
      gt.EdgeSet.from_fields(
          features={},
          sizes=as_tensor([[1], [1]]),
          adjacency=adj.Adjacency.from_indices(
              ('node', as_tensor([[0], [1]])), ('node', as_tensor([[0], [0]]))
          ),
      ),
      gt.GraphTensor.from_pieces(
          node_sets={
              'node': gt.NodeSet.from_fields(
                  features={},
                  sizes=as_tensor([[2], [1]]),
              )
          }
      ),
  ])
  def testNumComponents(self, example):
    piece_input = tf.keras.Input(type_spec=example.spec)
    num_components_model = tf.keras.Model(
        piece_input, piece_input.num_components
    )
    self.assertAllEqual(num_components_model(example), [1, 1])

    total_num_components_model = tf.keras.Model(
        piece_input, piece_input.total_num_components
    )
    self.assertAllEqual(total_num_components_model(example), 2)

  def testAdjacency(self):
    adjacency = adj.Adjacency.from_indices(
        ('node.a', as_tensor([0, 1])), ('node.b', as_tensor([0, 0]))
    )
    adjacency_input = tf.keras.Input(type_spec=adjacency.spec)
    self.assertEqual(adjacency.source_name, 'node.a')
    self.assertEqual(adjacency.target_name, 'node.b')
    self.assertAllEqual(
        tf.keras.Model(adjacency_input, adjacency_input.source)(adjacency),
        [0, 1],
    )
    self.assertAllEqual(
        tf.keras.Model(adjacency_input, adjacency_input.target)(adjacency),
        [0, 0],
    )
    self.assertIsInstance(adjacency_input.get_indices_dict(), dict)

  def testHyperAdjacency(self):
    adjacency = adj.HyperAdjacency.from_indices(
        {0: ('node', as_tensor([0, 1]))}
    )
    adjacency_input = tf.keras.Input(type_spec=adjacency.spec)
    self.assertEqual(adjacency.node_set_name(0), 'node')
    self.assertAllEqual(
        tf.keras.Model(adjacency_input, adjacency_input[0])(adjacency), [0, 1]
    )
    self.assertIsInstance(adjacency_input.get_indices_dict(), dict)

  def testGraphTensorOps(self):
    example = gt.GraphTensor.from_pieces(
        gt.Context.from_fields(features={'f': as_tensor([1])}),
        node_sets={
            'node': gt.NodeSet.from_fields(
                features={'f': as_tensor([1, 2])},
                sizes=as_tensor([2]),
            )
        },
        edge_sets={
            'edge': gt.EdgeSet.from_fields(
                features={'f': as_tensor([1, 2, 3])},
                sizes=as_tensor([3]),
                adjacency=adj.Adjacency.from_indices(
                    ('node', as_tensor([0, 1, 1])),
                    ('node', as_tensor([0, 0, 1])),
                ),
            )
        },
    )

    graph_input = tf.keras.Input(type_spec=example.spec)
    self.assertIsInstance(graph_input, kt.GraphKerasTensor)
    graph_output = graph_input.replace_features(
        context={
            'f': pool_ops.pool_edges_to_context(
                graph_input, 'edge', feature_name='f'
            ) + pool_ops.pool_nodes_to_context(
                graph_input, 'node', feature_name='f'
            )
        },
        node_sets={
            'node': {
                'f': broadcast_ops.broadcast_context_to_nodes(
                    graph_input, 'node', feature_name='f'
                ) + pool_ops.pool_edges_to_node(
                    graph_input, 'edge', const.TARGET, feature_name='f'
                )
            }
        },
        edge_sets={
            'edge': {
                'f': broadcast_ops.broadcast_context_to_edges(
                    graph_input, 'edge', feature_name='f'
                ) + broadcast_ops.broadcast_node_to_edges(
                    graph_input, 'edge', const.SOURCE, feature_name='f'
                )
            }
        },
    )
    self.assertIsInstance(graph_output, kt.GraphKerasTensor)
    features_output = {
        'context.f': graph_output.context['f'],
        'node.f': graph_output.node_sets['node']['f'],
        'edge.f': graph_output.edge_sets['edge']['f'],
    }
    result = tf.keras.Model(graph_input, features_output)(example)

    self.assertFieldsEqual(
        result,
        {
            'context.f': [(1 + 2 + 1) + (2 + 3)],
            'node.f': [1 + (1 + 2), 1 + (3)],
            'edge.f': [1 + 1, 1 + 2, 1 + 2],
        },
    )

  def testRemoveFeatures(self):
    example = gt.GraphTensor.from_pieces(
        gt.Context.from_fields(
            features={'f': as_tensor([1]), 'c': as_tensor([10])}
        ),
        node_sets={
            'node': gt.NodeSet.from_fields(
                features={'f': as_tensor([2, 3]), 'n': as_tensor([11, 12])},
                sizes=as_tensor([2]),
            )
        },
        edge_sets={
            'edge': gt.EdgeSet.from_fields(
                features={'f': as_tensor([4]), 'e': as_tensor([13])},
                sizes=as_tensor([1]),
                adjacency=adj.Adjacency.from_indices(
                    ('node', as_tensor([0])), ('node', as_tensor([1]))
                ),
            )
        },
    )

    graph_input = tf.keras.Input(type_spec=example.spec)
    self.assertIsInstance(graph_input, kt.GraphKerasTensor)
    graph_output = graph_input.remove_features(
        context=['c'], node_sets={'node': ['n']}, edge_sets={'edge': ['e']}
    )
    self.assertIsInstance(graph_output, kt.GraphKerasTensor)
    result = tf.keras.Model(graph_input, graph_output)(example)

    self.assertFieldsEqual(result.context.features, {'f': [1]})
    self.assertFieldsEqual(result.node_sets['node'].features, {'f': [2, 3]})
    self.assertFieldsEqual(result.edge_sets['edge'].features, {'f': [4]})

  @parameterized.named_parameters(
      ('Baseline', tftu.ModelReloading.SKIP, False),
      ('SavedModel', tftu.ModelReloading.SAVED_MODEL, False),
      ('SavedModelStatic', tftu.ModelReloading.SAVED_MODEL, True),
      ('Keras', tftu.ModelReloading.KERAS, False),
      ('KerasStatic', tftu.ModelReloading.KERAS, True))
  def testModelSaving(self, model_reloading, static_shapes):
    # A GraphTensorSpec for a homogeneous graph with an indeterminate number
    # of components flattened into a scalar graph (suitable for model
    # computations). Each node has a state of shape [2] and each edge has a
    # weight of shape [1].
    # Test the model (original and restored) on a graph with one component:
    #
    #           /--  0.5 -->>
    #    [10, 0]             [12, 0]
    #           <<-- -0.5 --/
    def create_graph_tensor():
      return gt.GraphTensor.from_pieces(
          edge_sets={
              'edge': gt.EdgeSet.from_fields(
                  features={
                      'edge_weight': as_tensor([[0.5], [-0.5]], tf.float32)
                  },
                  sizes=as_tensor([2]),
                  adjacency=adj.HyperAdjacency.from_indices(
                      indices={
                          const.SOURCE: ('node', as_tensor([0, 1])),
                          const.TARGET: ('node', as_tensor([1, 0])),
                      }
                  ),
              )
          },
          node_sets={
              'node': gt.NodeSet.from_fields(
                  features={
                      'state': as_tensor([[10, 0.0], [12.0, 0.0]], tf.float32)
                  },
                  sizes=as_tensor([2]),
              )
          },
      )

    def get_input_spec():
      spec = create_graph_tensor().spec
      if static_shapes:
        self.assertAllEqual(
            spec.edge_sets_spec['edge']['edge_weight'],
            tf.TensorSpec(tf.TensorShape([2, 1]), tf.float32))
      else:
        spec = spec.relax(num_nodes=True, num_edges=True)
        self.assertAllEqual(
            spec.edge_sets_spec['edge']['edge_weight'],
            tf.TensorSpec(tf.TensorShape([None, 1]), tf.float32))
      return spec

    # A Keras Model that inputs and outputs such a GraphTensor.
    fnn = tf.keras.layers.Dense(
        units=2,
        name='swap_node_state_coordinates',
        use_bias=False,
        kernel_initializer=tf.keras.initializers.Constant(
            [[0.0, 1.0], [1.0, 0.0]]
        ),
    )

    inputs = tf.keras.layers.Input(type_spec=get_input_spec())
    graph = inputs

    weight = graph.edge_sets['edge']['edge_weight']
    node_state = graph.node_sets['node']['state']
    source_value = broadcast_ops.broadcast_node_to_edges(
        graph, 'edge', const.SOURCE, feature_name='state'
    )
    message = tf.multiply(weight, source_value)
    pooled_message = pool_ops.pool_edges_to_node(
        graph, 'edge', const.TARGET, feature_value=message
    )
    node_updates = fnn(pooled_message)
    node_state += node_updates
    outputs = graph.replace_features(node_sets={'node': {'state': node_state}})
    model = tf.keras.Model(inputs, outputs)
    # Save and restore the model.
    model = tftu.maybe_reload_model(self, model, model_reloading, 'graph-model')

    graph_1 = create_graph_tensor()
    self.assertAllClose([[10.0, -6.0], [12.0, 5.0]],
                        model(graph_1).node_sets['node']['state'])


@kt.delegate_keras_tensors
def plus_one(t):
  return t + 1.0


@kt.delegate_keras_tensors(name='as_dict')
def as_dict_v2(t, key: str):
  return {key: t}


@kt.disallow_keras_tensors
def size(t: tf.Tensor) -> tf.Tensor:
  return tf.size(t)


@kt.disallow_keras_tensors(alternative='tf.shape(t)[0]')
def size_v2(t: tf.Tensor) -> tf.Tensor:
  return tf.size(t)


class WrappedOpsTest(_TestBase):

  def setUp(self):
    super().setUp()
    tf.keras.backend.clear_session()

  def testSimple(self):
    t = i = tf.keras.Input([])
    t = plus_one(t)
    t = plus_one(t)
    t = plus_one(t)
    model = tf.keras.Model(i, t)

    self.assertEqual(model.layers[1].name, 'plus_one')
    self.assertIsInstance(model.layers[1], kt.TFGNNOpLambda)
    self.assertEqual(model.layers[2].name, 'plus_one_1')
    self.assertIsInstance(model.layers[2], kt.TFGNNOpLambda)
    self.assertEqual(model.layers[3].name, 'plus_one_2')
    self.assertIsInstance(model.layers[3], kt.TFGNNOpLambda)

    self.assertAllClose(model(tf.convert_to_tensor([1.0])), [4.0])

  def testLayerNamesOverride(self):
    t = i = tf.keras.Input([])
    t = as_dict_v2(t, 't')
    model = tf.keras.Model(i, t)

    self.assertEqual(model.layers[1].name, 'as_dict')
    self.assertIsInstance(model.layers[1], kt.TFGNNOpLambda)
    result = model(tf.convert_to_tensor(1.0))
    self.assertIn('t', result)
    self.assertEqual(result['t'], 1.0)

  def testDisallowKerasTensors(self):
    self.assertEqual(size(tf.convert_to_tensor([1, 2, 3])), 3)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Calling `size()` using the Keras Functional API is not supported.'
        ' Consider calling this function from the `call()` method of a custom'
        ' Keras Layer',
    ):
      size(tf.keras.Input([]))
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Calling `size_v2()` using the Keras Functional API is not supported.'
        ' Consider calling this function from the `call()` method of a custom'
        ' Keras Layer or tf.shape(t)[0] as an alternative.',
    ):
      size_v2(tf.keras.Input([]))

  @parameterized.named_parameters(_OPS_TEST_CASES)
  def testOps(self, layer_name: str, transformation):
    inputs = tf.keras.Input(type_spec=_TEST_GRAPH_TENSOR.spec)

    outputs = transformation(inputs)
    model = tf.keras.Model(inputs, outputs)
    self.assertIsInstance(model.get_layer(layer_name), kt.TFGNNOpLambda)

    self.assertAllClose(
        model(_TEST_GRAPH_TENSOR), transformation(_TEST_GRAPH_TENSOR)
    )


class DocStringsDelegationTest(_TestBase):

  def testDelegateKerasTensors(self):
    @kt.delegate_keras_tensors
    def test_fn(t: tf.Tensor, a: int, *, b: str):
      """Delegates Keras tensors."""
      del t, a, b

    signature = inspect.signature(test_fn, follow_wrapped=True)
    self.assertSequenceEqual(list(signature.parameters.keys()), ['t', 'a', 'b'])
    self.assertEqual(inspect.getdoc(test_fn), 'Delegates Keras tensors.')

  def testDisallowKerasTensors(self):
    @kt.disallow_keras_tensors
    def test_fn(t1: tf.Tensor, t2: tf.Tensor, *, x: str):
      """Deprecates Keras tensors."""
      del t1, t2, x

    signature = inspect.signature(test_fn, follow_wrapped=True)
    self.assertSequenceEqual(
        list(signature.parameters.keys()), ['t1', 't2', 'x']
    )
    self.assertEqual(inspect.getdoc(test_fn), 'Deprecates Keras tensors.')


class WrappedOpsSavingTest(_SaveAndLoadTestBase):

  @tf.keras.utils.register_keras_serializable()
  class _Pack(tf.keras.layers.Layer):

    def call(self, inputs):
      return tf.nest.pack_sequence_as(_TEST_GRAPH_TENSOR.spec, inputs,
                                      expand_composites=True)

  def setUp(self):
    super().setUp()
    tf.keras.backend.clear_session()

  def _get_test_model(self, transformation):
    fields = tf.nest.flatten(_TEST_GRAPH_TENSOR, expand_composites=True)
    field_specs = [tf.TensorSpec(field.shape, field.dtype, name=f'field_{i}')
                   for i, field in enumerate(fields)]
    inputs = [tf.keras.Input(type_spec=spec) for spec in field_specs]
    restored_graph_tensor = self._Pack()(inputs)
    result = transformation(restored_graph_tensor)
    outputs = tf.keras.layers.Layer(name='result')(result)
    return tf.keras.Model(inputs, outputs)

  @parameterized.named_parameters(
      ('Baseline', tftu.ModelReloading.SKIP),
      ('SavedModel', tftu.ModelReloading.SAVED_MODEL),
      ('Keras', tftu.ModelReloading.KERAS),
      ('KerasV3', tftu.ModelReloading.KERAS_V3))
  def testSimpleKerasModelSaving(self, model_reloading):
    t = i = tf.keras.Input([])
    t = as_dict_v2(t, key='t')
    model = tf.keras.Model(i, t)
    restored_model = tftu.maybe_reload_model(self, model, model_reloading,
                                             'wrapped-ops-simple-model')
    if tftu.is_keras_model_reloading(model_reloading):
      self.assertEqual(restored_model.layers[1].name, 'as_dict')
      self.assertIsInstance(restored_model.layers[1], kt.TFGNNOpLambda)
    result = restored_model(tf.convert_to_tensor([1.0]))
    self.assertIn('t', result)
    self.assertEqual(result['t'], [1.0])

  @parameterized.named_parameters(
      ('Baseline', tftu.ModelReloading.SKIP),
      ('SavedModel', tftu.ModelReloading.SAVED_MODEL),
      ('Keras', tftu.ModelReloading.KERAS),
      ('KerasV3', tftu.ModelReloading.KERAS_V3))
  def testKerasModelSaving(self, model_reloading):
    def run_test(transformation, layer_name):
      tf.keras.backend.clear_session()
      model = self._get_test_model(transformation)
      self.assertIsInstance(
          model.get_layer(layer_name),
          kt.TFGNNOpLambda,
      )
      model = tftu.maybe_reload_model(self, model, model_reloading,
                                      'wrapped-ops-model')
      if tftu.is_keras_model_reloading(model_reloading):
        self.assertIsInstance(
            model.get_layer(layer_name),
            kt.TFGNNOpLambda,
        )
      self.assertAllClose(
          model(tf.nest.flatten(_TEST_GRAPH_TENSOR, expand_composites=True)),
          transformation(_TEST_GRAPH_TENSOR),
      )

    for case_name, layer_name, transformation in _OPS_TEST_CASES:
      with self.subTest(case_name):
        run_test(transformation, layer_name)

  @parameterized.parameters([True, False])
  def testSimpleSavingForInference(self, use_legacy_model_save: bool):
    t = i = tf.keras.Input([], name='x')
    t = plus_one(t)
    t = plus_one(t)
    t = tf.keras.layers.Layer(name='y')(t)
    model = tf.keras.Model(i, t)
    restored_model = self._save_and_load_inference_model(
        model, use_legacy_model_save)
    signature = restored_model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    self.assertAllClose(
        signature(x=tf.convert_to_tensor([0.0, 10.0]))['y'], [2.0, 12.0]
    )

  @parameterized.parameters([True, False])
  def testSavingForInference(self, use_legacy_model_save: bool):
    def run_test(transformation):
      model = self._get_test_model(transformation)
      restored_model = self._save_and_load_inference_model(
          model, use_legacy_model_save)
      signature = restored_model.signatures[
          tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
      test_input_list = tf.nest.flatten(
          _TEST_GRAPH_TENSOR, expand_composites=True)
      test_input_dict = {f'field_{i}': field
                         for i, field in enumerate(test_input_list)}
      actual = signature(**test_input_dict)['result']
      expected = transformation(_TEST_GRAPH_TENSOR)
      self.assertAllClose(expected, actual)

    for case_name, _, transformation in _OPS_TEST_CASES:
      with self.subTest(case_name):
        run_test(transformation)


if __name__ == '__main__':
  tf.test.main()
