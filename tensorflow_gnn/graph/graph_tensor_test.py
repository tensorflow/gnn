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
"""Tests for GraphTensor  (go/tf-gnn-api)."""

import collections
from typing import Mapping, Optional

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import graph_tensor_test_utils as tu

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import type_spec
# pylint: enable=g-direct-tensorflow-import

as_tensor = tf.convert_to_tensor
as_ragged = tf.ragged.constant


class CreationTest(tu.GraphTensorTestBase):
  """Tests for context, node sets and edge sets creation."""

  def assertFieldsEqual(self, actual: const.Fields, expected: const.Fields):
    self.assertIsInstance(actual, Mapping)
    self.assertAllEqual(actual.keys(), expected.keys())
    for key in actual.keys():
      self.assertAllEqual(actual[key], expected[key], msg=f'feature={key}')

  @parameterized.parameters([
      dict(features={}, shape=[]),
      dict(
          features={
              'a': as_tensor([1, 2, 3]),
              'b': as_ragged([[1, 2], [3], []])
          },
          shape=[]),
      dict(
          features={
              'a': as_tensor([[1., 2.], [3., 4.]]),
              'b': as_ragged([[[1, 2], []], [[], [3]]])
          },
          shape=[2])
  ])
  def testContext(self, features, shape):
    context = gt.Context.from_fields(features=features, shape=shape)
    self.assertAllEqual(context.shape, shape)
    self.assertFieldsEqual(context.features, features)
    self.assertFieldsEqual(context.get_features_dict(), features)
    if features:
      self.assertAllEqual(context['a'], features['a'])
      self.assertAllEqual(context.spec['a'],
                          type_spec.type_spec_from_value(features['a']))

  def testCreationChain(self):
    source = gt.Context.from_fields(features={'x': as_tensor([1.])})
    copy1 = gt.Context.from_fields(features=source.features)
    copy2 = gt.Context.from_fields(features=source.get_features_dict())
    self.assertFieldsEqual(source.features, copy1.features)
    self.assertFieldsEqual(source.features, copy2.features)

  def testFieldsImmutability(self):

    def set_x_to_2(features):
      features['x'] = as_tensor([2.])

    source = gt.Context.from_fields(features={'x': as_tensor([1.])})
    features = source.features
    self.assertRaisesRegex(Exception, 'does not support item assignment',
                           lambda: set_x_to_2(features))
    fields_copy = source.get_features_dict()
    set_x_to_2(fields_copy)
    self.assertFieldsEqual(fields_copy, {'x': as_tensor([2.])})
    self.assertFieldsEqual(source.features, {'x': as_tensor([1.])})

  @parameterized.parameters([
      dict(features={}, sizes=as_tensor([3]), expected_shape=[]),
      dict(
          features={
              'a': as_tensor([1., 2., 3.]),
              'b': as_ragged([[1, 2], [3], []])
          },
          sizes=as_tensor([1]),
          expected_shape=[]),
      dict(
          features={
              'a': as_ragged([[1., 2., 3.], [1., 2.]]),
              'b': as_ragged([[[1], [], [3]], [[], [2]]])
          },
          sizes=as_tensor([[3], [2]]),
          expected_shape=[2])
  ])
  def testNodeSet(self, features, sizes, expected_shape):
    node_set = gt.NodeSet.from_fields(features=features, sizes=sizes)
    self.assertAllEqual(node_set.shape, expected_shape)
    self.assertAllEqual(node_set.sizes, sizes)
    self.assertFieldsEqual(node_set.features, features)
    if features:
      self.assertAllEqual(node_set['a'], features['a'])
      self.assertAllEqual(node_set.spec['a'],
                          type_spec.type_spec_from_value(features['a']))

  @parameterized.parameters([
      dict(
          features={},
          sizes=as_tensor([2]),
          adjacency=adj.HyperAdjacency.from_indices({
              const.SOURCE: ('node', as_tensor([0, 1])),
              const.TARGET: ('node', as_tensor([1, 2])),
          }),
          expected_shape=[]),
      dict(
          features={'a': as_ragged([[1., 2.], [3.]])},
          sizes=as_ragged([[1, 1], [1]]),
          adjacency=adj.HyperAdjacency.from_indices({
              const.SOURCE: ('node.a', as_ragged([[0, 1], [0]])),
              const.TARGET: ('node.b', as_ragged([[1, 2], [0]])),
          }),
          expected_shape=[2]),
  ])
  def testEdgeSet(self, features, sizes, adjacency, expected_shape):
    edge_set = gt.EdgeSet.from_fields(
        features=features, sizes=sizes, adjacency=adjacency)
    self.assertAllEqual(edge_set.shape, expected_shape)
    self.assertAllEqual(edge_set.sizes, sizes)
    self.assertFieldsEqual(edge_set.features, features)
    if features:
      self.assertAllEqual(edge_set['a'], features['a'])
      self.assertAllEqual(edge_set.spec['a'],
                          type_spec.type_spec_from_value(features['a']))

    self.assertAllEqual(edge_set.adjacency.shape, expected_shape)
    self.assertAllEqual(edge_set.adjacency[const.SOURCE],
                        adjacency[const.SOURCE])

  def testEmptyGraphTensor(self):
    result = gt.GraphTensor.from_pieces()
    self.assertEqual(result.shape, [])
    self.assertEqual(result.context.shape, [])
    self.assertEmpty(result.context.features)
    self.assertEmpty(result.node_sets)
    self.assertEmpty(result.edge_sets)
    self.assertEqual(
        ('GraphTensor(\n'
         '  context=Context('
         'features={}, sizes=[], shape=(), indices_dtype=tf.int32),\n'
         '  node_set_names=[],\n'
         '  edge_set_names=[])'), repr(result))

  def testGraphTensor(self):
    result = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(
            features={'label': as_tensor([['X'], ['Y']])}),
        node_sets={
            'a':
                gt.NodeSet.from_fields(
                    features={}, sizes=as_tensor([[1], [1]])),
            'b':
                gt.NodeSet.from_fields(
                    features={}, sizes=as_tensor([[2], [1]])),
        },
        edge_sets={
            'a->b':
                gt.EdgeSet.from_fields(
                    features={'weight': as_ragged([[1., 2.], [3.]])},
                    sizes=as_tensor([[2], [1]]),
                    adjacency=adj.Adjacency.from_indices(
                        source=('a', as_ragged([[0, 1], [0]])),
                        target=('b', as_ragged([[1, 2], [0]])),
                    )),
        },
    )
    self.assertEqual(result.shape, [2])
    self.assertEqual(result.context.shape, [2])
    self.assertEqual(result.edge_sets['a->b'].shape, [2])
    self.assertEqual(result.edge_sets['a->b'].adjacency.shape, [2])
    self.assertEqual(result.node_sets['a'].shape, [2])
    self.assertEqual(result.node_sets['b'].shape, [2])

    self.assertAllEqual(result.context['label'], [['X'], ['Y']])
    self.assertAllEqual(result.node_sets['a'].sizes, [[1], [1]])
    self.assertAllEqual(result.node_sets['b'].sizes, [[2], [1]])
    self.assertAllEqual(result.edge_sets['a->b'].sizes, [[2], [1]])
    self.assertAllEqual(result.edge_sets['a->b']['weight'],
                        as_ragged([[1., 2.], [3.]]))

    self.assertEqual(result.spec.context_spec['label'],
                     tf.TensorSpec([2, 1], tf.string))

    edge_spec = result.spec.edge_sets_spec['a->b']
    self.assertEqual(edge_spec['weight'],
                     tf.RaggedTensorSpec([2, None], tf.float32, 1, tf.int64))
    self.assertEqual(edge_spec.adjacency_spec.node_set_name(const.SOURCE), 'a')
    self.assertEqual(edge_spec.adjacency_spec.node_set_name(const.TARGET), 'b')
    self.assertEqual(
        ''.join(('GraphTensor('
                 'context=Context('
                 'features={\'label\':<tf.Tensor:shape=(2,1),'
                 'dtype=tf.string>},'
                 'sizes=[[1][1]],'
                 'shape=(2,),'
                 'indices_dtype=tf.int32),'
                 "node_set_names=['a', 'b'],"
                 "edge_set_names=['a->b'])").split()),
        # Easy way to get rid of whitespace
        ''.join(repr(result).split()))


class HomogeneousTest(tu.GraphTensorTestBase):
  """Tests for homogeneous(...)."""

  def testNodeSizesIfNotEdges(self):
    src = tf.constant([0, 3])
    tgt = tf.constant([1, 2])
    self.assertRaisesRegex(ValueError, 'node_set_sizes must be provided',
                           lambda: gt.homogeneous(source=src, target=tgt))

  @parameterized.named_parameters(
      (
          'homogeneous_no_features',
          gt.homogeneous(
              source=tf.constant([0, 3]),
              target=tf.constant([1, 2]),
              node_set_sizes=tf.constant([4])),
          gt.GraphTensor.from_pieces(
              context=gt.Context.from_fields(features={}, sizes=None),
              node_sets={
                  const.NODES:
                      gt.NodeSet.from_fields(
                          features={}, sizes=tf.constant([4]))
              },
              edge_sets={
                  const.EDGES:
                      gt.EdgeSet.from_fields(
                          features={},
                          sizes=tf.constant([2]),
                          adjacency=adj.Adjacency.from_indices(
                              source=(const.NODES, tf.constant([0, 3])),
                              target=(const.NODES, tf.constant([1, 2])),
                          )),
              },
          ),
      ),
      (
          'homogeneous_node_features_only',
          gt.homogeneous(
              source=tf.constant([0, 3]),
              target=tf.constant([1, 2]),
              node_features=tf.eye(4),
          ),
          gt.GraphTensor.from_pieces(
              context=gt.Context.from_fields(features={}, sizes=None),
              node_sets={
                  const.NODES:
                      gt.NodeSet.from_fields(
                          features={const.HIDDEN_STATE: tf.eye(4)},
                          sizes=tf.constant([4]))
              },
              edge_sets={
                  const.EDGES:
                      gt.EdgeSet.from_fields(
                          features={},
                          sizes=tf.constant([2]),
                          adjacency=adj.Adjacency.from_indices(
                              source=(const.NODES, tf.constant([0, 3])),
                              target=(const.NODES, tf.constant([1, 2])),
                          )),
              },
          ),
      ),
      (
          'homogeneous_multiple_components',
          gt.homogeneous(
              source=tf.constant([0, 3, 4, 5]),
              target=tf.constant([1, 2, 6, 4]),
              node_features=tf.eye(7),
              node_set_sizes=tf.constant([4, 3]),
              edge_set_sizes=tf.constant([2, 2]),
          ),
          gt.GraphTensor.from_pieces(
              context=gt.Context.from_fields(
                  features={}, sizes=tf.constant([1, 1])),
              node_sets={
                  const.NODES:
                      gt.NodeSet.from_fields(
                          features={const.HIDDEN_STATE: tf.eye(7)},
                          sizes=tf.constant([4, 3]))
              },
              edge_sets={
                  const.EDGES:
                      gt.EdgeSet.from_fields(
                          features={},
                          sizes=tf.constant([2, 2]),
                          adjacency=adj.Adjacency.from_indices(
                              source=(const.NODES, tf.constant([0, 3, 4, 5])),
                              target=(const.NODES, tf.constant([1, 2, 6, 4])),
                          )),
              },
          ),
      ),
      (
          'homogeneous_single_component_implied_sizes',
          gt.homogeneous(
              source=tf.constant([0, 3, 4, 5]),
              target=tf.constant([1, 2, 6, 4]),
              node_features=tf.eye(7),
              edge_features=tf.ones([4, 3]),
              context_features=tf.zeros(5),
          ),
          gt.GraphTensor.from_pieces(
              context=gt.Context.from_fields(
                  features={const.HIDDEN_STATE: tf.zeros(5)},
                  sizes=tf.constant([1])),
              node_sets={
                  const.NODES:
                      gt.NodeSet.from_fields(
                          features={const.HIDDEN_STATE: tf.eye(7)},
                          sizes=tf.constant([7]))
              },
              edge_sets={
                  const.EDGES:
                      gt.EdgeSet.from_fields(
                          features={const.HIDDEN_STATE: tf.ones([4, 3])},
                          sizes=tf.constant([4]),
                          adjacency=adj.Adjacency.from_indices(
                              source=(const.NODES, tf.constant([0, 3, 4, 5])),
                              target=(const.NODES, tf.constant([1, 2, 6, 4])),
                          )),
              },
          ),
      ),
      (
          'homogeneous_single_component_named_features',
          gt.homogeneous(
              source=tf.constant([0, 3, 4, 5]),
              target=tf.constant([1, 2, 6, 4]),
              node_features={'onehots': tf.eye(7)},
              edge_features={'floats': tf.ones([4, 3])},
              context_features={'labels': tf.zeros(5)},
          ),
          gt.GraphTensor.from_pieces(
              context=gt.Context.from_fields(
                  features={'labels': tf.zeros(5)},
                  sizes=tf.constant([1])),
              node_sets={
                  const.NODES:
                      gt.NodeSet.from_fields(
                          features={'onehots': tf.eye(7)},
                          sizes=tf.constant([7]))
              },
              edge_sets={
                  const.EDGES:
                      gt.EdgeSet.from_fields(
                          features={'floats': tf.ones([4, 3])},
                          sizes=tf.constant([4]),
                          adjacency=adj.Adjacency.from_indices(
                              source=(const.NODES, tf.constant([0, 3, 4, 5])),
                              target=(const.NODES, tf.constant([1, 2, 6, 4])),
                          )),
              },
          ),
      ),
  )
  def testHomogeneous(self, actual, expected):
    """Tests for homogeneous()."""
    self.assertGraphTensorsEqual(actual, expected)

  def testInGraphMode(self):

    @tf.function
    def create(source, target, node_features, edge_features, context_features):
      return gt.homogeneous(
          source=source,
          target=target,
          node_features=node_features,
          edge_features=edge_features,
          context_features=context_features,
      )
    actual = create(
        source=tf.constant([0, 3, 4, 5]),
        target=tf.constant([1, 2, 6, 4]),
        node_features={'onehots': tf.eye(7)},
        edge_features={'floats': tf.ones([4, 3])},
        context_features={'labels': tf.zeros(5)},
    )
    expected = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(
            features={'labels': tf.zeros(5)}, sizes=tf.constant([1])
        ),
        node_sets={
            const.NODES: gt.NodeSet.from_fields(
                features={'onehots': tf.eye(7)}, sizes=tf.constant([7])
            )
        },
        edge_sets={
            const.EDGES: gt.EdgeSet.from_fields(
                features={'floats': tf.ones([4, 3])},
                sizes=tf.constant([4]),
                adjacency=adj.Adjacency.from_indices(
                    source=(const.NODES, tf.constant([0, 3, 4, 5])),
                    target=(const.NODES, tf.constant([1, 2, 6, 4])),
                ),
            ),
        },
    )
    self.assertGraphTensorsEqual(actual, expected)

  def assertGraphTensorsEqual(self, actual, expected):
    self.assertFieldsEqual(
        actual.node_sets['nodes'].features,
        expected.node_sets['nodes'].features,
    )
    self.assertAllEqual(actual.node_sets['nodes'].sizes,
                        expected.node_sets['nodes'].sizes)
    self.assertFieldsEqual(
        actual.edge_sets['edges'].features,
        expected.edge_sets['edges'].features,
    )
    self.assertAllEqual(actual.edge_sets['edges'].sizes,
                        expected.edge_sets['edges'].sizes)
    self.assertFieldsEqual(actual.context.features, expected.context.features)
    self.assertAllEqual(actual.context.sizes, expected.context.sizes)


class ReplaceFeaturesTest(tu.GraphTensorTestBase):
  """Tests for replace_features()."""

  @parameterized.parameters([
      dict(features={'a': as_tensor([2])}),
      dict(features={
          'a': as_tensor([2]),
          'b': as_ragged([[1, 2]])
      })
  ])
  def testContext(self, features):
    context = gt.Context.from_fields(features={'a': as_tensor([1])})
    result = context.replace_features(features)
    self.assertFieldsEqual(result.features, features)

  @parameterized.parameters([
      dict(features={'a': as_tensor([2., 1.])}),
      dict(features={
          'a': as_tensor([2., 1.]),
          'b': as_ragged([[1, 2], []])
      })
  ])
  def testNodeSet(self, features):
    node_set = gt.NodeSet.from_fields(
        features={'a': as_tensor([1., 2.])}, sizes=as_tensor([2]))
    result = node_set.replace_features(features)
    self.assertFieldsEqual(result.features, features)
    self.assertAllEqual(result.sizes, [2])

  @parameterized.parameters([
      dict(features={'a': as_tensor([2., 1.])}),
      dict(features={
          'a': as_tensor([2., 1.]),
          'b': as_ragged([[1, 2], []])
      })
  ])
  def testEdgeSet(self, features):
    edge_set = gt.EdgeSet.from_fields(
        features={'a': as_tensor([1., 2.])},
        sizes=as_tensor([2]),
        adjacency=adj.HyperAdjacency.from_indices({0: ('a', as_tensor([0,
                                                                       1]))}))
    result = edge_set.replace_features(features)
    self.assertFieldsEqual(result.features, features)
    self.assertAllEqual(result.sizes, [2])
    self.assertAllEqual(result.adjacency[0], [0, 1])

  def testGraphTensor(self):
    source = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(
            features={'label': as_tensor([['X'], ['Y']])}),
        node_sets={
            'a':
                gt.NodeSet.from_fields(
                    features={}, sizes=as_tensor([[1], [1]])),
            'b':
                gt.NodeSet.from_fields(
                    features={}, sizes=as_tensor([[2], [1]])),
        },
        edge_sets={
            'a->b':
                gt.EdgeSet.from_fields(
                    features={'weight': as_ragged([[1., 2.], [3.]])},
                    sizes=as_tensor([[2], [1]]),
                    adjacency=adj.HyperAdjacency.from_indices({
                        const.SOURCE: ('a', as_ragged([[0, 1], [0]])),
                        const.TARGET: ('b', as_ragged([[1, 2], [0]])),
                    })),
        },
    )
    result1 = source.replace_features(
        context={'label': as_tensor([['A'], ['B']])})
    self.assertAllEqual(list(result1.node_sets.keys()), ['a', 'b'])
    self.assertAllEqual(list(result1.edge_sets.keys()), ['a->b'])
    self.assertAllEqual(result1.context['label'], [['A'], ['B']])
    self.assertEmpty(result1.node_sets['a'].features)
    self.assertEmpty(result1.node_sets['b'].features)

    result2 = source.replace_features(
        edge_sets={
            'a->b': {
                'f0': as_ragged([[0., 0.], [0.]]),
                'f1': as_ragged([[1., 1.], [1.]])
            }
        })
    self.assertAllEqual(list(result2.node_sets.keys()), ['a', 'b'])
    self.assertFieldsEqual(result2.edge_sets['a->b'].features, {
        'f0': as_ragged([[0., 0.], [0.]]),
        'f1': as_ragged([[1., 1.], [1.]])
    })

    result3 = source.replace_features(
        node_sets={'a': {
            'f': as_ragged([[0.], [0.]])
        }})
    self.assertAllEqual(list(result3.node_sets.keys()), ['a', 'b'])
    self.assertAllEqual(list(result3.edge_sets.keys()), ['a->b'])
    self.assertFieldsEqual(result3.node_sets['a'].features,
                           {'f': as_ragged([[0.], [0.]])})
    self.assertEmpty(result1.node_sets['b'].features)

    with self.assertRaisesWithLiteralMatch(
        ValueError, ('Some node sets in the `node_sets` are not present'
                     ' in the graph tensor: [\'x\']')):
      features = {'f': as_ragged([[0.], [0.]])}
      source.replace_features(node_sets={
          'a': features,
          'x': features,
      })

    with self.assertRaisesWithLiteralMatch(
        ValueError, ('Some edge sets in the `edge_sets` are not present'
                     ' in the graph tensor: [\'a->x\', \'a->y\']')):
      features = {'f': as_ragged([[1., 1.], [1.]])}
      source.replace_features(edge_sets={
          'a->x': features,
          'a->y': features,
      })


class RemoveFeaturesTest(tu.GraphTensorTestBase):
  """Tests for remove_features()."""

  def _make_test_graph(self):
    return gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(features={
            'fc': as_tensor([10]),
            'f2': as_tensor([20])
        }),
        node_sets={
            'a':
                gt.NodeSet.from_fields(
                    features={
                        'fa': as_tensor([10]),
                        'f2': as_tensor([20])
                    },
                    sizes=as_tensor([1])),
            'b':
                gt.NodeSet.from_fields(
                    features={
                        'fb': as_tensor([10]),
                        'f2': as_tensor([20])
                    },
                    sizes=as_tensor([1]))
        },
        edge_sets={
            'ab':
                gt.EdgeSet.from_fields(
                    features={
                        'fab': as_tensor([10]),
                        'f2': as_tensor([20])
                    },
                    sizes=as_tensor([1]),
                    adjacency=adj.HyperAdjacency.from_indices({
                        const.SOURCE: ('a', as_tensor([0])),
                        const.TARGET: ('b', as_tensor([0]))
                    })),
            'ba':
                gt.EdgeSet.from_fields(
                    features={
                        'fba': as_tensor([10]),
                        'f2': as_tensor([20])
                    },
                    sizes=as_tensor([1]),
                    adjacency=adj.HyperAdjacency.from_indices({
                        const.SOURCE: ('b', as_tensor([0])),
                        const.TARGET: ('a', as_tensor([0]))
                    }))
        })

  @parameterized.named_parameters(
      ('None', [], [], [], [], []),
      ('OneFromContext', ['f2'], [], [], [], []),
      ('OneFromNode', [], [], ['f2'], [], []),
      ('OneFromEdge', [], [], [], [], ['f2']),
      ('OneFromAll', ['fc'], ['fa'], ['fb'], ['fab'], ['fba']),
      ('RepeatedFromAll', ['fc'] * 2, ['fa'] * 2, ['fb'] * 2, ['fab'] * 2,
       ['fba'] * 2),
      ('TwoFromEachType', ['fc', 'f2'], ['fa', 'f2'], [], [], ['fba', 'f2']),
      ('All', ['fc', 'f2'], ['fa', 'f2'], ['fb', 'f2'], ['fab', 'f2'
                                                        ], ['fba', 'f2']),
  )
  def testRemove(self, rm_context, rm_a, rm_b, rm_ab, rm_ba):
    graph = self._make_test_graph()

    kwargs = collections.defaultdict(dict)
    if rm_context:
      kwargs['context'] = rm_context
    if rm_a:
      kwargs['node_sets']['a'] = rm_a
    if rm_b:
      kwargs['node_sets']['b'] = rm_b
    if rm_ab:
      kwargs['edge_sets']['ab'] = rm_ab
    if rm_ba:
      kwargs['edge_sets']['ba'] = rm_ba

    result = graph.remove_features(**kwargs)
    self.assertCountEqual(result.context.features,
                          {'fc', 'f2'} - set(rm_context))
    self.assertCountEqual(result.node_sets['a'].features,
                          {'fa', 'f2'} - set(rm_a))
    self.assertCountEqual(result.node_sets['b'].features,
                          {'fb', 'f2'} - set(rm_b))
    self.assertCountEqual(result.edge_sets['ab'].features,
                          {'fab', 'f2'} - set(rm_ab))
    self.assertCountEqual(result.edge_sets['ba'].features,
                          {'fba', 'f2'} - set(rm_ba))

  def testRemoveNonexistantFromContext(self):
    graph = self._make_test_graph()
    with self.assertRaisesRegex(
        ValueError, 'GraphTensor has no feature context\\[\'xyz\'\\]'):
      _ = graph.remove_features(context=['fc', 'xyz', 'f2'])

  def testRemoveNonexistantFromNodeSet(self):
    graph = self._make_test_graph()
    with self.assertRaisesRegex(
        ValueError,
        'GraphTensor has no feature node_sets\\[\'b\'\\]\\[\'xyz\'\\]'):
      _ = graph.remove_features(node_sets={
          'a': ['fa', 'f2'],
          'b': ['fb', 'xyz', 'f2']
      })

  def testRemoveNonexistantFromEdgeSet(self):
    graph = self._make_test_graph()
    with self.assertRaisesRegex(
        ValueError,
        'GraphTensor has no feature edge_sets\\[\'ba\'\\]\\[\'xyz\'\\]'):
      _ = graph.remove_features(edge_sets={
          'ab': ['fab', 'f2'],
          'ba': ['fba', 'xyz', 'f2']
      })


class ResolveValueTest(tu.GraphTensorTestBase):
  """Tests for resolve_value()."""

  @parameterized.named_parameters(
      ('Context', gt.Context.from_fields(
          sizes=as_tensor([1, 1]),
          features={
              'feat': as_tensor([[1, 2], [3, 4]]),
              'other': as_tensor([[9], [9]]),
          })),
      ('NodeSet', gt.NodeSet.from_fields(
          sizes=as_tensor([1, 1]),
          features={
              'feat': as_tensor([[1, 2], [3, 4]]),
              'other': as_tensor([[9], [9]]),
          })),
      ('EdgeSet', gt.EdgeSet.from_fields(
          sizes=as_tensor([1, 1]),
          features={
              'feat': as_tensor([[1, 2], [3, 4]]),
              'other': as_tensor([[9], [9]]),
          },
          adjacency=adj.HyperAdjacency.from_indices({
              const.SOURCE: ('a', as_tensor([0, 1])),
              const.TARGET: ('b', as_tensor([1, 0]))
          }))))
  def test(self, graph_piece):
    self.assertAllEqual(
        [[1, 2], [3, 4]],
        gt.resolve_value(graph_piece, feature_name='feat'))
    feature_value = as_tensor([[5, 6], [7, 8]])
    self.assertAllEqual(
        [[5, 6], [7, 8]],
        gt.resolve_value(graph_piece, feature_value=feature_value))
    with self.assertRaisesRegex(ValueError, r'One of'):
      gt.resolve_value(graph_piece)
    with self.assertRaisesRegex(ValueError, r'One of'):
      gt.resolve_value(graph_piece,
                       feature_name='feat', feature_value=feature_value)


class ElementsCountsTest(tf.test.TestCase, parameterized.TestCase):

  def testEmpty(self):
    graph = gt.GraphTensor.from_pieces()
    self.assertEqual(graph.total_num_components, 0)
    self.assertEqual(graph.spec.total_num_components, 0)

  def testContextOnly(self):
    graph1 = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(
            features={'label': as_tensor([['X'], ['Y'], ['Z']])}, shape=[]))
    self.assertEqual(graph1.spec.total_num_components, 3)

    graph2 = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(
            features={'f': as_tensor([[1., 2.], [3., 4.], [5., 6.]])},
            shape=[]))
    self.assertEqual(graph2.spec.total_num_components, 3)

  def testNodeSetsOnly(self):
    graph = gt.GraphTensor.from_pieces(
        node_sets={
            'node':
                gt.NodeSet.from_fields(
                    features={
                        'f':
                            as_ragged([[1., 0.], [1., 0.], [1., 0.], [1., 0.]]),
                    },
                    sizes=as_tensor([1, 2, 1])),
        })
    self.assertEqual(graph.spec.total_num_components, 3)
    self.assertEqual(graph.node_sets['node'].spec.total_size, 4)
    self.assertEqual(graph.node_sets['node'].total_size, 1 + 2 + 1)

  def testStaticlyShaped(self):
    graph = gt.GraphTensor.from_pieces(
        node_sets={
            'node':
                gt.NodeSet.from_fields(
                    features={
                        'f':
                            as_tensor([[1., 0.], [1., 0.], [1., 0.], [1., 0.]]),
                    },
                    sizes=as_tensor([1, 2, 1])),
        },
        edge_sets={
            'edge':
                gt.EdgeSet.from_fields(
                    features={'weight': as_tensor([1., 2., 3.])},
                    sizes=as_tensor([2, 0, 1]),
                    adjacency=adj.HyperAdjacency.from_indices({
                        const.SOURCE: ('node', as_tensor([0, 1, 2])),
                        const.TARGET: ('node', as_tensor([1, 2, 0])),
                    })),
        },
    )
    self.assertEqual(graph.spec.total_num_components, 1 + 1 + 1)
    self.assertIsInstance(graph.node_sets['node'].spec.total_size, int)
    self.assertEqual(graph.node_sets['node'].spec.total_size, 4)
    self.assertEqual(graph.node_sets['node'].total_size, 1 + 2 + 1)
    self.assertIsInstance(graph.edge_sets['edge'].spec.total_size, int)
    self.assertEqual(graph.edge_sets['edge'].spec.total_size, 3)
    self.assertEqual(graph.edge_sets['edge'].total_size, 2 + 0 + 1)

  def testRank1(self):
    graph = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(
            features={'label': as_tensor([['1', '2'], ['3', '4']])}),
        node_sets={
            'a':
                gt.NodeSet.from_fields(
                    features={'f': as_ragged([[1., 2.], [3., 4.]])},
                    sizes=as_tensor([[1, 1], [1, 1]])),
        },
        edge_sets={
            'a->a':
                gt.EdgeSet.from_fields(
                    features={},
                    sizes=as_tensor([[2, 0], [0, 1]]),
                    adjacency=adj.HyperAdjacency.from_indices({
                        const.SOURCE: ('a', as_ragged([[0, 1], [0]])),
                        const.TARGET: ('a', as_ragged([[1, 2], [0]])),
                    })),
        },
    )
    self.assertEqual(graph.spec.total_num_components, 2 + 2)
    self.assertIsNone(graph.node_sets['a'].spec.total_size)
    self.assertEqual(graph.node_sets['a'].total_size, 1 + 1 + 1 + 1)
    self.assertIsNone(graph.edge_sets['a->a'].spec.total_size)
    self.assertEqual(graph.edge_sets['a->a'].total_size, 2 + 0 + 0 + 1)

  def testRank1StaticlyShaped(self):
    graph = gt.GraphTensor.from_pieces(
        node_sets={
            'a':
                gt.NodeSet.from_fields(
                    features={'f': as_tensor([[1., 2.], [3., 4.]])},
                    sizes=as_tensor([[1, 1], [1, 1]])),
        },
        edge_sets={
            'a->a':
                gt.EdgeSet.from_fields(
                    features={},
                    sizes=as_tensor([[2, 0], [0, 2]]),
                    adjacency=adj.HyperAdjacency.from_indices({
                        const.SOURCE: ('a', as_tensor([[0, 1], [0, 0]])),
                        const.TARGET: ('a', as_tensor([[1, 2], [0, 0]])),
                    })),
        },
    )
    self.assertEqual(graph.spec.total_num_components, 2 + 2)
    self.assertIsInstance(graph.node_sets['a'].spec.total_size, int)
    self.assertEqual(graph.node_sets['a'].spec.total_size, 4)
    self.assertEqual(graph.node_sets['a'].total_size, 1 + 1 + 1 + 1)
    self.assertIsInstance(graph.edge_sets['a->a'].spec.total_size, int)
    self.assertEqual(graph.edge_sets['a->a'].spec.total_size, 4)
    self.assertEqual(graph.edge_sets['a->a'].total_size, 2 + 0 + 0 + 2)


class TfFunctionTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for GraphTensor and TfFunction interaction."""

  def testTfFunctionForNodeSets(self):

    @tf.function
    def add(node_set, value):
      features = node_set.features.copy()
      features['x'] += value
      return gt.NodeSet.from_fields(features=features, sizes=node_set.sizes)

    node_set = gt.NodeSet.from_fields(
        features={'x': as_tensor([1, 2, 3])}, sizes=as_tensor([3]))

    node_set = add(node_set, 1)
    node_set = add(node_set, 1)
    node_set = add(node_set, 1)
    node_set = add(node_set, 1)
    self.assertEqual(add.experimental_get_tracing_count(), 1)
    self.assertAllEqual(node_set['x'], as_tensor([5, 6, 7]))

  def testFieldsMappingIsAccepted(self):

    @tf.function
    def create_context(features):
      return gt.Context.from_fields(features=features)

    source = gt.Context.from_fields(features={'x': as_tensor([1, 2, 3])})
    result1 = create_context(source.features)
    result2 = create_context(result1.features)

    self.assertEqual(create_context.experimental_get_tracing_count(), 1)
    self.assertAllEqual(result1['x'], source['x'])
    self.assertAllEqual(result2['x'], source['x'])

  def testTfFunctionTracing(self):

    @tf.function
    def add(node_set, value):
      features = node_set.features.copy()
      features['x'] += value
      return gt.NodeSet.from_fields(features=features, sizes=node_set.sizes)

    node_set = gt.NodeSet.from_fields(
        features={'x': as_tensor([1, 2, 3])}, sizes=as_tensor([3]))
    node_set = add(node_set, 1)
    self.assertEqual(add.experimental_get_tracing_count(), 1)
    node_set = add(node_set, 1)
    self.assertEqual(add.experimental_get_tracing_count(), 1)
    self.assertAllEqual(node_set['x'], as_tensor([1 + 2, 2 + 2, 3 + 2]))
    node_set = gt.NodeSet.from_fields(
        features={'x': as_tensor([1, 2, 3, 4])}, sizes=as_tensor([4]))
    node_set = add(node_set, 1)
    self.assertEqual(add.experimental_get_tracing_count(), 2)
    self.assertAllEqual(node_set['x'], as_tensor([1 + 1, 2 + 1, 3 + 1, 4 + 1]))

  def testTfFunctionWithGraph(self):

    @tf.function
    def concat(graph_a, graph_b):
      a, b = graph_a.edge_sets['edge'], graph_b.edge_sets['edge']
      join = lambda a, b: tf.concat([a, b], 0)
      return gt.GraphTensor.from_pieces(
          edge_sets={
              'edge':
                  gt.EdgeSet.from_fields(
                      features={'f': join(a['f'], b['f'])},
                      sizes=a.sizes + b.sizes,
                      adjacency=adj.HyperAdjacency.from_indices(
                          indices={
                              const.SOURCE: ('node',
                                             join(a.adjacency[const.SOURCE],
                                                  b.adjacency[const.SOURCE])),
                              const.TARGET: ('node',
                                             join(a.adjacency[const.TARGET],
                                                  b.adjacency[const.TARGET])),
                          }))
          })

    def gen_graph(features):
      features = as_tensor(features)
      return gt.GraphTensor.from_pieces(
          edge_sets={
              'edge':
                  gt.EdgeSet.from_fields(
                      features={'f': features},
                      sizes=tf.reshape(tf.size(features), [1]),
                      adjacency=adj.HyperAdjacency.from_indices(
                          indices={
                              const.SOURCE: ('node',
                                             tf.zeros_like(features, tf.int64)),
                              const.TARGET: ('node',
                                             tf.ones_like(features, tf.int64)),
                          }))
          })

    graph = concat(gen_graph([1, 2]), gen_graph([3, 4]))
    self.assertEqual(concat.experimental_get_tracing_count(), 1)
    graph = concat(gen_graph([5, 6]), gen_graph([7, 8]))
    self.assertEqual(concat.experimental_get_tracing_count(), 1)
    graph = concat(graph, graph)
    self.assertEqual(concat.experimental_get_tracing_count(), 2)
    graph = concat(graph, graph)
    self.assertEqual(concat.experimental_get_tracing_count(), 3)
    graph = concat(gen_graph([2, 1]), gen_graph([4, 3]))
    self.assertEqual(concat.experimental_get_tracing_count(), 3)


class BatchingUnbatchingMergingTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for Graph Tensor specification."""

  def testVarSizeBatching(self):

    @tf.function
    def generate(num_nodes):
      return gt.Context.from_fields(
          features={
              'x':
                  tf.range(num_nodes),
              'r':
                  tf.RaggedTensor.from_row_lengths(
                      tf.ones(tf.stack([num_nodes], 0), dtype=tf.float32),
                      tf.stack([0, num_nodes, 0], 0)),
          })

    ds = tf.data.Dataset.range(0, 9)
    ds = ds.map(generate)
    ds = ds.batch(1)
    ds = ds.unbatch()
    ds = ds.batch(3, True)
    ds = ds.batch(2)

    itr = iter(ds)
    element = next(itr)
    self.assertAllEqual(
        element['x'],
        as_ragged([
            [[], [0], [0, 1]],
            [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]],
        ]))
    self.assertAllEqual(
        element['r'],
        as_ragged([
            [[[], [], []], [[], [1], []], [[], [1, 1], []]],
            [[[], [1, 1, 1], []], [[], [1, 1, 1, 1], []],
             [[], [1, 1, 1, 1, 1], []]],
        ]))

    self.assertAllEqual(
        type_spec.type_spec_from_value(element['x']),
        tf.RaggedTensorSpec(
            shape=[2, 3, None],
            dtype=tf.int64,
            ragged_rank=2,
            row_splits_dtype=tf.int32))

    element = next(itr)
    self.assertAllEqual(
        element['x'],
        as_ragged([
            [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6],
             [0, 1, 2, 3, 4, 5, 6, 7]],
        ]))
    self.assertAllEqual(
        type_spec.type_spec_from_value(element['x']),
        tf.RaggedTensorSpec(
            shape=[1, 3, None],
            dtype=tf.int64,
            ragged_rank=2,
            row_splits_dtype=tf.int32))

  def testFixedSizeBatching(self):

    @tf.function
    def generate(i):
      return gt.Context.from_fields(features={'i': tf.stack([i, i + 1], 0)})

    ds = tf.data.Dataset.range(0, 6)
    ds = ds.map(generate)
    ds = ds.batch(2, drop_remainder=True)
    ds = ds.batch(3, drop_remainder=True)
    ds = ds.unbatch()
    ds = ds.unbatch()
    ds = ds.batch(1, drop_remainder=True)
    ds = ds.batch(2, drop_remainder=True)

    element = next(iter(ds))
    self.assertAllEqual(element['i'], [[[0, 1]], [[1, 2]]])
    self.assertAllEqual(
        type_spec.type_spec_from_value(element['i']),
        tf.TensorSpec(shape=[2, 1, 2], dtype=tf.int64))

  def testFixedSizeBatchingSpecs(self):

    @tf.function
    def generate(i):
      return gt.Context.from_fields(
          features={'i': tf.stack([i, i + 1, i + 3], 0)})

    ds = tf.data.Dataset.range(0, 6)
    ds = ds.map(generate)
    ds = ds.batch(2, drop_remainder=True)
    ds = ds.batch(3, drop_remainder=True)
    ds = ds.unbatch()
    ds = ds.unbatch()
    ds = ds.batch(2, drop_remainder=True)
    ds = ds.batch(1, drop_remainder=True)
    self.assertEqual(ds.element_spec['i'],
                     tf.TensorSpec(shape=[1, 2, 3], dtype=tf.int64))

  def testFixedSizeBatchingWithRaggedFeature(self):

    @tf.function
    def generate(i):
      return gt.Context.from_fields(
          features={
              'r':
                  tf.cond(i == 0, lambda: as_ragged([[1, 2, 3], [4]]),
                          lambda: as_ragged([[1], [2, 3, 4], [5]]))
          },
          indices_dtype=tf.int64)

    ds = tf.data.Dataset.range(0, 2)
    ds = ds.map(generate)
    ds = ds.batch(2, drop_remainder=True)
    self.assertAllEqual(
        ds.element_spec['r'],
        tf.RaggedTensorSpec(
            shape=[2, None, None],
            dtype=tf.int32,
            ragged_rank=2,
            row_splits_dtype=tf.int64))

    element = next(iter(ds))
    self.assertAllEqual(element['r'],
                        as_ragged([[[1, 2, 3], [4]], [[1], [2, 3, 4], [5]]]))

  def testGraphTensorFixedSizeBatching(self):

    @tf.function
    def generate(seed):
      edge_count = 2
      node_count = 2
      edge_set = gt.EdgeSet.from_fields(
          features={'f': tf.range(start=seed, limit=seed + edge_count)},
          sizes=tf.expand_dims(edge_count, 0),
          adjacency=adj.HyperAdjacency.from_indices(
              indices={
                  const.SOURCE: ('node',
                                 (tf.zeros([edge_count], dtype=tf.int64))),
                  const.TARGET: ('node',
                                 (tf.ones([edge_count], dtype=tf.int64))),
              }))
      node_set = gt.NodeSet.from_fields(
          features={'f': tf.range(start=seed, limit=seed + node_count)},
          sizes=tf.expand_dims(node_count, 0))

      return gt.GraphTensor.from_pieces(
          edge_sets={'edge': edge_set}, node_sets={'node': node_set})

    ds = tf.data.Dataset.range(1, 3)
    ds = ds.map(generate)
    ds = ds.batch(2, drop_remainder=True)
    graph = next(iter(ds))
    self.assertAllEqual(graph.node_sets['node'].sizes, [[2], [2]])
    self.assertAllEqual(graph.node_sets['node']['f'], [[1., 2.], [2., 3.]])
    self.assertAllEqual(graph.edge_sets['edge'].sizes, [[2], [2]])
    self.assertAllEqual(graph.edge_sets['edge']['f'], [[1., 2.], [2., 3.]])
    ds = ds.map(lambda g: g.merge_batch_to_components())
    graph = next(iter(ds))
    self.assertAllEqual(graph.node_sets['node'].sizes, [2, 2])
    self.assertAllEqual(graph.node_sets['node']['f'], [1., 2., 2., 3.])

    edge = graph.edge_sets['edge']
    self.assertAllEqual(edge.sizes, [2, 2])
    self.assertAllEqual(edge['f'], [1., 2., 2., 3.])
    self.assertAllEqual(edge.adjacency[const.SOURCE], [0, 0, 2, 2])
    self.assertAllEqual(edge.adjacency[const.TARGET], [1, 1, 3, 3])

  def testGraphTensorVarSizeBatching(self):

    @tf.function
    def generate(seed):
      edge_count = seed
      node_count = seed + 1
      edge_set = gt.EdgeSet.from_fields(
          features={'f': tf.range(edge_count, dtype=tf.float32)},
          sizes=tf.expand_dims(edge_count, 0),
          adjacency=adj.HyperAdjacency.from_indices(
              indices={
                  const.SOURCE: ('node',
                                 (tf.zeros([edge_count], dtype=tf.int64))),
                  const.TARGET: ('node',
                                 (tf.zeros([edge_count], dtype=tf.int64))),
              }))
      node_set = gt.NodeSet.from_fields(
          features={'f': tf.range(node_count, dtype=tf.float32)},
          sizes=tf.expand_dims(node_count, 0))

      return gt.GraphTensor.from_pieces(
          edge_sets={'edge': edge_set}, node_sets={'node': node_set})

    ds = tf.data.Dataset.range(0, 3)
    ds = ds.map(generate)
    ds = ds.batch(2, drop_remainder=True)
    graph = next(iter(ds))
    self.assertAllEqual(graph.node_sets['node']['f'], as_ragged([[0.], [0.,
                                                                        1.]]))
    self.assertAllEqual(graph.node_sets['node'].sizes, [[1], [2]])
    self.assertAllEqual(graph.edge_sets['edge']['f'], as_ragged([[], [0.]]))
    self.assertAllEqual(graph.edge_sets['edge'].sizes, [[0], [1]])
    ds = ds.map(lambda g: g.merge_batch_to_components())
    graph = next(iter(ds))
    self.assertAllEqual(graph.edge_sets['edge'].sizes, [0, 1])
    self.assertAllEqual(graph.edge_sets['edge']['f'], [0.])
    edge = graph.edge_sets['edge']
    self.assertAllEqual(edge.sizes, [0, 1])
    self.assertAllEqual(edge['f'], [0.])
    self.assertAllEqual(edge.adjacency[const.SOURCE], [1])
    self.assertAllEqual(edge.adjacency[const.TARGET], [1])

  def testGraphTensorEmptyValue(self):

    @tf.function
    def generate(seed):
      edge_count = seed
      node_count = seed + 1
      edge_set = gt.EdgeSet.from_fields(
          features={'f': tf.range(edge_count, dtype=tf.float32)},
          sizes=tf.expand_dims(edge_count, 0),
          adjacency=adj.HyperAdjacency.from_indices(
              indices={
                  const.SOURCE: ('node',
                                 (tf.zeros([edge_count], dtype=tf.int64))),
                  const.TARGET: ('node',
                                 (tf.zeros([edge_count], dtype=tf.int64))),
              }))
      node_set = gt.NodeSet.from_fields(
          features={'f': tf.range(node_count, dtype=tf.float32)},
          sizes=tf.expand_dims(node_count, 0))

      return gt.GraphTensor.from_pieces(
          edge_sets={'edge': edge_set}, node_sets={'node': node_set})

    ds = tf.data.Dataset.range(0, 3)
    ds = ds.map(generate)
    ds = ds.batch(2, drop_remainder=False)
    spec = ds.element_spec
    empty_value = spec._create_empty_value()
    self.assertTrue(spec.is_compatible_with(empty_value))

    self.assertAllEqual(empty_value.node_sets['node'].sizes,
                        tf.constant([], shape=(0, 1)))
    self.assertAllEqual(empty_value.node_sets['node']['f'],
                        as_ragged([], dtype=tf.float32, ragged_rank=1))
    edge = empty_value.edge_sets['edge']
    self.assertAllEqual(edge.sizes, tf.constant([], shape=(0, 1)))
    self.assertAllEqual(edge['f'], as_ragged([],
                                             dtype=tf.float32,
                                             ragged_rank=1))
    self.assertAllEqual(edge.adjacency[const.SOURCE],
                        as_ragged([], dtype=tf.int64, ragged_rank=1))

  def testTFLite(self):
    test_graph_dict = {
        'node_features': tf.constant([[1.], [2.], [3.]], tf.float32),
        'node_row_lengths': tf.constant([2, 1], tf.int32),
        'edge_row_lengths': tf.constant([2, 1], tf.int32),
        'edge_source': tf.constant([0, 1, 0], tf.int32),
        'edge_target': tf.constant([1, 0, 0], tf.int32),
    }
    inputs = {
        'node_features': tf.keras.Input([1], None, 'node_features', tf.float32),
        'node_row_lengths': tf.keras.Input(
            [], None, 'node_row_lengths', tf.int32),
        'edge_source': tf.keras.Input([], None, 'edge_source', tf.int32),
        'edge_target': tf.keras.Input([], None, 'edge_target', tf.int32),
        'edge_row_lengths': tf.keras.Input(
            [], None, 'edge_row_lengths', tf.int32),
    }
    outputs = _MakeGraphTensorMerged()(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # The other unit tests should verify that this is correct
    expected = [0, 1, 2]

    # TODO(b/276291104): Remove when TF 2.11+ is required by all of TFGNN
    if tf.__version__.startswith('2.10.'):
      self.skipTest('GNN models are unsupported in TFLite until TF 2.11 but '
                    f'got TF {tf.__version__}')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    model_content = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=model_content)
    signature_runner = interpreter.get_signature_runner('serving_default')
    obtained = signature_runner(
        **test_graph_dict)['private__make_graph_tensor_merged']
    self.assertAllClose(expected, obtained)


# TODO(b/274779989): Replace this layer with a more standard representation
# of GraphTensor as a dict of plain Tensors.
class _MakeGraphTensorMerged(tf.keras.layers.Layer):
  """Makes a homogeneous GraphTensor of rank 0 with two components."""

  def call(self, inputs):
    node_features = tf.RaggedTensor.from_row_lengths(
        inputs['node_features'], inputs['node_row_lengths'])
    edge_source = tf.RaggedTensor.from_row_lengths(
        inputs['edge_source'], inputs['edge_row_lengths'])
    edge_target = tf.RaggedTensor.from_row_lengths(
        inputs['edge_target'], inputs['edge_row_lengths'])
    graph = gt.GraphTensor.from_pieces(
        node_sets={
            'nodes':
                gt.NodeSet.from_fields(
                    sizes=tf.expand_dims(inputs['node_row_lengths'], -1),
                    features={const.HIDDEN_STATE: node_features})
        },
        edge_sets={
            'edges':
                gt.EdgeSet.from_fields(
                    sizes=tf.expand_dims(inputs['edge_row_lengths'], -1),
                    adjacency=adj.Adjacency.from_indices(
                        ('nodes', edge_source),
                        ('nodes', edge_target)),)
        }).merge_batch_to_components()
    return graph.edge_sets['edges'].adjacency.source


class NumComponentsTest(tu.GraphTensorTestBase):
  """Tests for GraphTensor tranformations."""

  @parameterized.parameters([
      dict(features={}, shape=[], expected=0),
      dict(features={'a': as_tensor([2])}, shape=[], expected=1),
      dict(features={'a': as_tensor([1, 2, 3])}, shape=[], expected=3),
      dict(
          features={'a': as_tensor([[1], [2], [3]])},
          shape=[None],
          expected=[1, 1, 1]),
      dict(
          features={'a': as_tensor([[1, 1], [2, 2]])},
          shape=[2],
          expected=[2, 2]),
      dict(features={'a': as_ragged([[1], [2, 2]])}, shape=[], expected=2),
      dict(
          features={'a': as_ragged([[1], [2, 2]])}, shape=[1], expected=[1, 2]),
      dict(
          features={'a': as_ragged([[[1], [2]], [[3]], [[]], []])},
          shape=[1],
          expected=[2, 1, 1, 0]),
      dict(
          features={
              'a': as_tensor([2]),
              'b': as_ragged([[1, 2]])
          },
          shape=[],
          expected=1)
  ])
  def testContext(self, features, shape, expected):
    context = gt.Context.from_fields(features=features, shape=shape)
    expected = as_tensor(expected)
    self.assertAllEqual(context.num_components, expected)
    self.assertAllEqual(context.total_num_components, tf.reduce_sum(expected))
    graph = gt.GraphTensor.from_pieces(context=context)
    self.assertAllEqual(graph.num_components, expected)
    self.assertAllEqual(graph.total_num_components, tf.reduce_sum(expected))

  @parameterized.parameters([
      dict(
          features={},
          sizes=as_tensor([2]),
          adjacency=adj.HyperAdjacency.from_indices({
              const.SOURCE: ('node', as_tensor([0, 1])),
              const.TARGET: ('node', as_tensor([1, 2])),
          }),
          expected=1),
      dict(
          features={'a': as_ragged([[1., 2.], [3.]])},
          sizes=as_ragged([[1, 1], [1]]),
          adjacency=adj.HyperAdjacency.from_indices({
              const.SOURCE: ('node.a', as_ragged([[0, 1], [0]])),
              const.TARGET: ('node.b', as_ragged([[1, 2], [0]])),
          }),
          expected=[2, 1]),
  ])
  def testEdgeAndNodeSets(self, features, sizes, adjacency, expected):
    node_set = gt.NodeSet.from_fields(features=features, sizes=sizes)
    edge_set = gt.EdgeSet.from_fields(
        features=features, sizes=sizes, adjacency=adjacency)

    expected = as_tensor(expected)
    for case_index, piece in enumerate([
        node_set, edge_set,
        gt.GraphTensor.from_pieces(edge_sets={'edge': edge_set}),
        gt.GraphTensor.from_pieces(node_sets={'node': node_set}),
        gt.GraphTensor.from_pieces(
            node_sets={'node': node_set}, edge_sets={'edge': edge_set})
    ]):
      self.assertAllEqual(
          piece.num_components, expected, msg=f'case_index={case_index}')
      self.assertAllEqual(
          piece.total_num_components,
          tf.reduce_sum(expected),
          msg=f'case_index={case_index}')

  @parameterized.parameters([
      dict(features={}, sizes=as_tensor([]), expected=0),
      dict(features={}, sizes=as_tensor([2]), expected=1),
      dict(features={}, sizes=as_tensor([[1], [1]]), expected=[1, 1]),
      dict(
          features={'a': as_ragged([[1., 2.], [3.], [4.]])},
          sizes=as_ragged([[1, 1], [1], [0]]),
          expected=[2, 1, 1]),
  ])
  def testContextUpdate(self, features, sizes, expected):
    context = gt.Context.from_fields()
    node_set = gt.NodeSet.from_fields(features=features, sizes=sizes)
    self.assertAllEqual(context.num_components, 0)
    graph = gt.GraphTensor.from_pieces(context, node_sets={'node': node_set})
    self.assertAllEqual(graph.context.num_components, expected)
    self.assertAllEqual(graph.context.total_num_components,
                        tf.reduce_sum(expected))


@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.int32)])
def node_set_sizes_to_test_results(inputs):
  context = gt.Context.from_fields(sizes=tf.ones_like(inputs))
  nodes = gt.NodeSet.from_fields(sizes=inputs)
  graph = gt.GraphTensor.from_pieces(context=context,
                                     node_sets={'nodes': nodes})
  return tf.stack([
      graph.total_num_components,
      graph.num_components,
      graph.node_sets['nodes'].total_size,
  ], axis=0)


class NodeSetSizesToTestResults(tf.keras.layers.Layer):

  def call(self, sizes):
    return node_set_sizes_to_test_results(sizes)


class SizesTFLiteTest(tf.test.TestCase):

  def testTFLite(self):
    test_tensor = tf.constant([2, 2])
    inputs = tf.keras.Input([], None, 'node_sizes', tf.int32)
    outputs = NodeSetSizesToTestResults()(inputs)
    model = tf.keras.Model(inputs, outputs)

    # The other unit tests should verify that this is correct
    expected = [2, 2, 4]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    model_content = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=model_content)
    signature_runner = interpreter.get_signature_runner('serving_default')
    obtained = signature_runner(
        node_sizes=test_tensor)['node_set_sizes_to_test_results']
    self.assertAllClose(expected, obtained)


class CheckScalarGraphTensorTest(tf.test.TestCase):

  def testSuccess(self):
    graph_tensor = gt.GraphTensor.from_pieces(node_sets={
        'nodes': gt.NodeSet.from_fields(sizes=[1], features={'f': [[1.]]})
    })
    gt.check_scalar_graph_tensor(graph_tensor)  # Doesn't raise.

  def testFailure(self):
    graph_tensor = gt.GraphTensor.from_pieces(node_sets={
        'nodes': gt.NodeSet.from_fields(sizes=[[1]], features={'f': [[[1.]]]})
    })
    with self.assertRaisesRegex(ValueError,
                                r'My test code requires.*got `rank=1`'):
      gt.check_scalar_graph_tensor(graph_tensor, 'My test code')


class GetAuxTypePrefixTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      ('_readout', '_readout'),
      ('_readout/source/1', '_readout'),
      ('_readout:train', '_readout'),
      ('_readout:train/seed', '_readout'),
      ('#reserved/x', '#reserved'),
      ('!reserved/x', '!reserved'),
      ('%reserved/x', '%reserved'),
      ('.reserved/x', '.reserved'),
      ('^reserved/x', '^reserved'),
      ('~reserved/x', '~reserved'),
  )
  def testType(self, set_name, expected):
    actual = gt.get_aux_type_prefix(set_name)
    self.assertEqual(expected, actual)

  @parameterized.parameters(
      ('video',),
      ('User',),
      ('49ers',),
      ('-dashed-',),
  )
  def testNone(self, set_name):
    actual = gt.get_aux_type_prefix(set_name)
    self.assertIsNone(actual)


class CheckHomogeneousGraphTensorTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('', False), ('WithReadout', True))
  def testSuccess(self, add_readout):
    graph = _make_test_graph_from_num_pieces(1, 1, add_readout=add_readout)
    expected = ('atoms_0', 'bonds_0')
    # These calls don't raise.
    self.assertEqual(expected,
                     gt.get_homogeneous_node_and_edge_set_name(graph))
    self.assertEqual(expected,
                     gt.get_homogeneous_node_and_edge_set_name(graph.spec))

  @parameterized.named_parameters(
      ('ZeroNodeSets', 0, 0),
      ('ThreeNodeSets', 3, 1),
      ('ZeroEdgeSetsButReadout', 1, 0, True),
      ('SevenEdgeSets', 1, 7),
      ('ManyPieces', 4, 5),
      ('ManyPiecesAndReadout', 4, 5, True))
  def testFailure(self, num_node_sets, num_edge_sets, add_readout=False):
    graph = _make_test_graph_from_num_pieces(num_node_sets, num_edge_sets,
                                             add_readout)
    with self.assertRaisesRegex(ValueError,
                                r'a graph with 1 node set and 1 edge set'):
      gt.get_homogeneous_node_and_edge_set_name(graph)
    with self.assertRaisesRegex(ValueError,
                                r'a graph with 1 node set and 1 edge set'):
      gt.get_homogeneous_node_and_edge_set_name(graph.spec)


def _make_test_graph_from_num_pieces(num_node_sets, num_edge_sets, add_readout):
  # pylint: disable=g-complex-comprehension
  node_sets = {
      f'atoms_{i}': gt.NodeSet.from_fields(
          sizes=[2, 1], features={'f': [[1.], [2.], [3.]]})
      for i in range(num_node_sets)
  }
  edge_sets = {
      f'bonds_{j}': gt.EdgeSet.from_fields(
          sizes=[2, 0],
          adjacency=adj.Adjacency.from_indices(
              source=('atoms_0', [0, 2]),
              target=('atoms_0', [1, 2])))
      for j in range(num_edge_sets)
  }
  if add_readout:
    node_sets['_readout'] = gt.NodeSet.from_fields(sizes=[1, 1])
    edge_sets['_readout/seed'] = gt.EdgeSet.from_fields(
        sizes=[1, 1],
        adjacency=adj.Adjacency.from_indices(
            source=('atoms_0', [1, 2]),
            target=('atoms_0', [0, 2])))

  return gt.GraphTensor.from_pieces(node_sets=node_sets, edge_sets=edge_sets)


class SpecRelaxationTest(tu.GraphTensorTestBase):

  def _get_context_spec(self, num_components: Optional[int]) -> gt.ContextSpec:
    return gt.ContextSpec.from_field_specs(
        features_spec={
            'v': tf.TensorSpec(shape=(num_components,), dtype=tf.int16),
            'm': tf.TensorSpec(shape=(num_components, 3), dtype=tf.int32),
        })

  def _get_node_set_spec(self, num_components: Optional[int],
                         num_nodes: Optional[int]) -> gt.NodeSetSpec:
    return gt.NodeSetSpec.from_field_specs(
        sizes_spec=tf.TensorSpec(shape=(num_components,), dtype=tf.int64),
        features_spec={
            'id':
                tf.TensorSpec(shape=(num_nodes,), dtype=tf.int32),
            'words':
                tf.RaggedTensorSpec(
                    shape=(num_nodes, None),
                    dtype=tf.string,
                    ragged_rank=1,
                    row_splits_dtype=tf.int64),
        })

  def _get_edge_set_spec(self, num_components: Optional[int],
                         num_edges: Optional[int]) -> gt.EdgeSetSpec:
    return gt.EdgeSetSpec.from_field_specs(
        sizes_spec=tf.TensorSpec(shape=(num_components,), dtype=tf.int64),
        features_spec={
            'weight': tf.TensorSpec(shape=(num_edges,), dtype=tf.float32),
        },
        adjacency_spec=adj.AdjacencySpec.from_incident_node_sets(
            'a',
            'b',
            tf.TensorSpec(shape=(num_edges,), dtype=tf.int64),
        ))

  def _get_graph_tensor_spec(self, num_components: Optional[int],
                             num_nodes: Optional[int],
                             num_edges: Optional[int]) -> gt.GraphTensorSpec:
    return gt.GraphTensorSpec.from_piece_specs(
        context_spec=self._get_context_spec(num_components),
        node_sets_spec={
            'a': self._get_node_set_spec(num_components, num_nodes),
            'b': self._get_node_set_spec(num_components, num_nodes)
        },
        edge_sets_spec={
            'a->b': self._get_edge_set_spec(num_components, num_edges)
        })

  def testContext(self):
    expected = self._get_context_spec(None)
    self.assertEqual(
        self._get_context_spec(3).relax(num_components=True), expected)
    self.assertEqual(
        self._get_context_spec(3).relax(num_components=True).relax(
            num_components=True), expected)

  @parameterized.product(num_components=[True, False], num_nodes=[True, False])
  def testNodeSet(self, num_components, num_nodes):
    original = self._get_node_set_spec(1, 3)
    expected = self._get_node_set_spec(None if num_components else 1,
                                       None if num_nodes else 3)
    relaxed1 = original.relax(
        num_components=num_components, num_nodes=num_nodes)
    relaxed2 = relaxed1.relax(
        num_components=num_components, num_nodes=num_nodes)

    self.assertEqual(relaxed1, expected)
    self.assertEqual(relaxed2, expected)

  @parameterized.product(num_components=[True, False], num_edges=[True, False])
  def testEdgeSet(self, num_components, num_edges):
    original = self._get_edge_set_spec(1, 5)
    expected = self._get_edge_set_spec(None if num_components else 1,
                                       None if num_edges else 5)
    relaxed1 = original.relax(
        num_components=num_components, num_edges=num_edges)
    relaxed2 = relaxed1.relax(
        num_components=num_components, num_edges=num_edges)

    self.assertEqual(relaxed1, expected)
    self.assertEqual(relaxed2, expected)

  @parameterized.product(
      num_components=[True, False],
      num_nodes=[True, False],
      num_edges=[True, False])
  def testGraphTensor(self, num_components, num_nodes, num_edges):
    original = self._get_graph_tensor_spec(1, 5, 7)
    expected = self._get_graph_tensor_spec(None if num_components else 1,
                                           None if num_nodes else 5,
                                           None if num_edges else 7)
    relaxed1 = original.relax(
        num_components=num_components, num_nodes=num_nodes, num_edges=num_edges)
    relaxed2 = relaxed1.relax(
        num_components=num_components, num_nodes=num_nodes, num_edges=num_edges)

    self.assertEqual(relaxed1, expected)
    self.assertEqual(relaxed2, expected)


if __name__ == '__main__':
  tf.test.main()
