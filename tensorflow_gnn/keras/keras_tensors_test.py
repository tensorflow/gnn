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

import os

from typing import Mapping

from absl.testing import parameterized
import numpy as np

import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import broadcast_ops
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import pool_ops
from tensorflow_gnn.keras import keras_tensors as kt

as_tensor = tf.convert_to_tensor
as_ragged = tf.ragged.constant


class _TestBase(tf.test.TestCase, parameterized.TestCase):

  def assertFieldsEqual(self, actual: const.Fields, expected: const.Fields):
    self.assertIsInstance(actual, Mapping)
    self.assertAllEqual(actual.keys(), expected.keys())
    for key in actual.keys():
      self.assertAllEqual(actual[key], expected[key], msg=f'feature={key}')


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
      ('ragged', as_tensor([2]), as_ragged([[1], [], [1, 2, 3]])),
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

  def _get_test_model(self):
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

    return tf.keras.Model([source, target, num_nodes], graph)

  @parameterized.parameters(
      dict(save_format='tf', include_optimizer=False, save_traces=False),
      dict(save_format='tf', include_optimizer=True, save_traces=True),
      dict(save_format='keras_v3'),
  )
  def testKerasModelSaving(self, save_format, **save_args):

    def _save_and_load(model: tf.keras.Model) -> tf.keras.Model:
      ext = '.keras' if save_format == 'keras_v3' else ''
      path = os.path.join(self.get_temp_dir(), f'model.{ext}')
      model.save(path, save_format=save_format, **save_args)
      return tf.keras.models.load_model(path)

    # TODO(b/276291104): Remove when TF 2.13+ is required by all of TFGNN
    if any(tf.__version__.startswith(f'2.{v}') for v in [10, 11, 12]):
      self.skipTest(
          f'Skipping test for Keras V3 format for TF {tf.__version__} < 2.13'
      )

    model = self._get_test_model()
    model = _save_and_load(model)
    graph = model([as_tensor([0, 1]), as_tensor([0, 0]), as_tensor(2)])
    self.assertAllEqual(graph.node_sets['node'].sizes, [2])
    adjacency = graph.edge_sets['edge'].adjacency
    self.assertAllEqual(adjacency.source, [0, 1])
    self.assertAllEqual(adjacency.target, [0, 0])

  @parameterized.parameters([True, False])
  def testSavingForInference(self, use_saved_model: bool):
    def _save_and_load(model: tf.keras.Model) -> tf.keras.Model:
      path = os.path.join(self.get_temp_dir(), 'model')
      if use_saved_model:
        tf.saved_model.save(model, path)
      else:
        model.save(path, include_optimizer=False, save_traces=True)
      return tf.saved_model.load(path)

    model = self._get_test_model()
    outputs = tf.reduce_sum(
        model.output.node_sets['node'].sizes
    ) + tf.reduce_sum(model.output.edge_sets['edge'].sizes)
    inference_model = tf.keras.Model(inputs=model.inputs, outputs=outputs)
    restored_model = _save_and_load(inference_model)
    total_sizes = restored_model([
        as_tensor([0, 1], tf.int64),
        as_tensor([0, 0], tf.int64),
        as_tensor(2, tf.int64),
    ])
    self.assertAllEqual(total_sizes, 2 + 2)


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
          adjacency=adj.Adjacency.from_indices(('node', as_tensor([0, 1])),
                                               ('node', as_tensor([0, 0]))))
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
        result.features, {
            's': as_tensor([0 + 1 + 2, 1]),
            'v': as_tensor([[0], [1]]),
            'r': as_ragged([[1, 2], []]),
        })

  @parameterized.parameters([
      gt.Context.from_fields(sizes=as_tensor([1, 1]),),
      gt.NodeSet.from_fields(
          features={},
          sizes=as_tensor([1, 1]),
      ),
      gt.EdgeSet.from_fields(
          features={},
          sizes=as_tensor([1, 1]),
          adjacency=adj.Adjacency.from_indices(('node', as_tensor([0, 1])),
                                               ('node', as_tensor([0, 0]))))
  ])
  def testSizes(self, example):
    piece_input = tf.keras.Input(type_spec=example.spec)
    sizes = tf.keras.Model(piece_input, piece_input.sizes)(example)
    self.assertAllEqual(sizes, [1, 1])

    total_size = tf.keras.Model(piece_input, piece_input.total_size)(example)
    self.assertAllEqual(total_size, 2)

  @parameterized.parameters([
      gt.Context.from_fields(sizes=as_tensor([[1], [2]]),),
      gt.NodeSet.from_fields(
          features={},
          sizes=as_tensor([[2], [1]]),
      ),
      gt.EdgeSet.from_fields(
          features={},
          sizes=as_tensor([[1], [1]]),
          adjacency=adj.Adjacency.from_indices(
              ('node', as_tensor([[0], [1]])),
              ('node', as_tensor([[0], [0]])))),
      gt.GraphTensor.from_pieces(node_sets={
          'node':
              gt.NodeSet.from_fields(
                  features={},
                  sizes=as_tensor([[2], [1]]),
              )
      }),
  ])
  def testNumComponents(self, example):
    piece_input = tf.keras.Input(type_spec=example.spec)
    num_components_model = tf.keras.Model(piece_input,
                                          piece_input.num_components)
    self.assertAllEqual(num_components_model(example), [1, 1])

    total_num_components_model = tf.keras.Model(
        piece_input, piece_input.total_num_components)
    self.assertAllEqual(total_num_components_model(example), 2)

  def testAdjacency(self):
    adjacency = adj.Adjacency.from_indices(('node.a', as_tensor([0, 1])),
                                           ('node.b', as_tensor([0, 0])))
    adjacency_input = tf.keras.Input(type_spec=adjacency.spec)
    self.assertEqual(adjacency.source_name, 'node.a')
    self.assertEqual(adjacency.target_name, 'node.b')
    self.assertAllEqual(
        tf.keras.Model(adjacency_input, adjacency_input.source)(adjacency),
        [0, 1])
    self.assertAllEqual(
        tf.keras.Model(adjacency_input, adjacency_input.target)(adjacency),
        [0, 0])
    self.assertIsInstance(adjacency_input.get_indices_dict(), dict)

  def testHyperAdjacency(self):
    adjacency = adj.HyperAdjacency.from_indices(
        {0: ('node', as_tensor([0, 1]))})
    adjacency_input = tf.keras.Input(type_spec=adjacency.spec)
    self.assertEqual(adjacency.node_set_name(0), 'node')
    self.assertAllEqual(
        tf.keras.Model(adjacency_input, adjacency_input[0])(adjacency), [0, 1])
    self.assertIsInstance(adjacency_input.get_indices_dict(), dict)

  # TODO(b/283404258): Remove {broadcast,pool}_*() support for KeradGraphTensor.
  # Users are meant to call tfgnn.keras.layers.Broadcast and Pool instead.
  def testGraphTensorOps(self):
    example = gt.GraphTensor.from_pieces(
        gt.Context.from_fields(features={'f': as_tensor([1])}),
        node_sets={
            'node':
                gt.NodeSet.from_fields(
                    features={'f': as_tensor([1, 2])},
                    sizes=as_tensor([2]),
                )
        },
        edge_sets={
            'edge':
                gt.EdgeSet.from_fields(
                    features={'f': as_tensor([1, 2, 3])},
                    sizes=as_tensor([3]),
                    adjacency=adj.Adjacency.from_indices(
                        ('node', as_tensor([0, 1, 1])),
                        ('node', as_tensor([0, 0, 1]))))
        })

    graph_input = tf.keras.Input(type_spec=example.spec)
    self.assertIsInstance(graph_input, kt.GraphKerasTensor)
    graph_output = graph_input.replace_features(
        context={
            'f':
                pool_ops.pool_edges_to_context(
                    graph_input, 'edge', feature_name='f') +
                pool_ops.pool_nodes_to_context(
                    graph_input, 'node', feature_name='f')
        },
        node_sets={
            'node': {
                'f':
                    broadcast_ops.broadcast_context_to_nodes(
                        graph_input, 'node', feature_name='f') +
                    pool_ops.pool_edges_to_node(
                        graph_input, 'edge', const.TARGET, feature_name='f')
            }
        },
        edge_sets={
            'edge': {
                'f':
                    broadcast_ops.broadcast_context_to_edges(
                        graph_input, 'edge', feature_name='f') +
                    broadcast_ops.broadcast_node_to_edges(
                        graph_input, 'edge', const.SOURCE, feature_name='f')
            }
        },
    )
    self.assertIsInstance(graph_output, kt.GraphKerasTensor)
    features_output = {
        'context.f': graph_output.context['f'],
        'node.f': graph_output.node_sets['node']['f'],
        'edge.f': graph_output.edge_sets['edge']['f']
    }
    result = tf.keras.Model(graph_input, features_output)(example)

    self.assertFieldsEqual(
        result, {
            'context.f': [(1 + 2 + 1) + (2 + 3)],
            'node.f': [1 + (1 + 2), 1 + (3)],
            'edge.f': [1 + 1, 1 + 2, 1 + 2]
        })

  def testRemoveFeatures(self):
    example = gt.GraphTensor.from_pieces(
        gt.Context.from_fields(
            features={'f': as_tensor([1]), 'c': as_tensor([10])}),
        node_sets={
            'node': gt.NodeSet.from_fields(
                features={'f': as_tensor([2, 3]), 'n': as_tensor([11, 12])},
                sizes=as_tensor([2]))},
        edge_sets={
            'edge': gt.EdgeSet.from_fields(
                features={'f': as_tensor([4]), 'e': as_tensor([13])},
                sizes=as_tensor([1]),
                adjacency=adj.Adjacency.from_indices(
                    ('node', as_tensor([0])),
                    ('node', as_tensor([1]))))})

    graph_input = tf.keras.Input(type_spec=example.spec)
    self.assertIsInstance(graph_input, kt.GraphKerasTensor)
    graph_output = graph_input.remove_features(
        context=['c'],
        node_sets={'node': ['n']},
        edge_sets={'edge': ['e']})
    self.assertIsInstance(graph_output, kt.GraphKerasTensor)
    result = tf.keras.Model(graph_input, graph_output)(example)

    self.assertFieldsEqual(result.context.features, {'f': [1]})
    self.assertFieldsEqual(result.node_sets['node'].features, {'f': [2, 3]})
    self.assertFieldsEqual(result.edge_sets['edge'].features, {'f': [4]})

  @parameterized.parameters([True, False])
  def testModelSaving(self, static_shapes):
    # A GraphTensorSpec for a homogeneous graph with an indeterminate number
    # of components flattened into a scalar graph (suitable for model
    # computations). Each node has a state of shape [2] and each edge has a
    # weight of shape [1].
    # Test the model (original and restored) on a graph with one component:
    #
    #           /--  0.5 -->>
    #    [10, 0]             [12, 0]
    #           <<-- -0.5 --/
    def create_graph_tensor(factor):
      factor = tf.cast(factor, tf.int32)

      def tile(tensor, factor):
        assert tensor.shape.rank in (1, 2)
        return tf.tile(tensor,
                       [factor] if tensor.shape.rank == 1 else [factor, 1])

      return gt.GraphTensor.from_pieces(
          edge_sets={
              'edge':
                  gt.EdgeSet.from_fields(
                      features={
                          'edge_weight':
                              tile(
                                  as_tensor([[0.5], [-0.5]], tf.float32),
                                  factor)
                      },
                      sizes=as_tensor([2]) * factor,
                      adjacency=adj.HyperAdjacency.from_indices(
                          indices={
                              const.SOURCE: ('node',
                                             tile(as_tensor([0, 1]), factor)),
                              const.TARGET: ('node',
                                             tile(as_tensor([1, 0]), factor)),
                          }))
          },
          node_sets={
              'node':
                  gt.NodeSet.from_fields(
                      features={
                          'state':
                              tile(
                                  as_tensor([[10, 0.], [12., 0.]], tf.float32),
                                  factor)
                      },
                      sizes=as_tensor([2]) * factor)
          })

    def get_input_spec():
      if static_shapes:
        spec = create_graph_tensor(1).spec
        # Check that dataset spec has static component dimensions.
        self.assertAllEqual(spec.edge_sets_spec['edge']['edge_weight'],
                            tf.TensorSpec(tf.TensorShape([2, 1]), tf.float32))
        return spec

      ds = tf.data.Dataset.range(1, 3).map(create_graph_tensor)
      spec = ds.element_spec
      # Check that dataset spec has relaxed component dimensions.
      self.assertAllEqual(spec.edge_sets_spec['edge']['edge_weight'],
                          tf.TensorSpec(tf.TensorShape([None, 1]), tf.float32))
      return spec

    # A Keras Model that inputs and outputs such a GraphTensor.
    fnn = tf.keras.layers.Dense(
        units=2,
        name='swap_node_state_coordinates',
        use_bias=False,
        kernel_initializer=tf.keras.initializers.Constant([[0., 1.], [1., 0.]]))

    inputs = tf.keras.layers.Input(type_spec=get_input_spec())
    graph = inputs

    weight = graph.edge_sets['edge']['edge_weight']
    node_state = graph.node_sets['node']['state']
    source_value = broadcast_ops.broadcast_node_to_edges(
        graph, 'edge', const.SOURCE, feature_name='state')
    message = tf.multiply(weight, source_value)
    pooled_message = pool_ops.pool_edges_to_node(
        graph, 'edge', const.TARGET, feature_value=message)
    node_updates = fnn(pooled_message)
    node_state += node_updates
    outputs = graph.replace_features(node_sets={'node': {'state': node_state}})
    model = tf.keras.Model(inputs, outputs)
    # Save and restore the model.
    export_dir = os.path.join(self.get_temp_dir(), 'graph-model')
    tf.saved_model.save(model, export_dir)
    restored_model = tf.saved_model.load(export_dir)

    @tf.function
    def readout(graph):
      return graph.node_sets['node']['state']

    expected_1 = as_tensor([[10., -6.], [12., 5.]], tf.float32)
    graph_1 = create_graph_tensor(1)
    self.assertAllClose(readout(model(graph_1)), expected_1)
    self.assertAllClose(readout(restored_model(graph_1)), expected_1)


if __name__ == '__main__':
  tf.test.main()
