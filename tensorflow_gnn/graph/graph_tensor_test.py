"""Tests for GraphTensor  (go/tf-gnn-api)."""

from typing import Mapping

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
    self.assertRaisesRegex(Exception,
                           'does not support item assignment',
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
    edge_set = gt.EdgeSet.from_fields(features=features,
                                      sizes=sizes,
                                      adjacency=adjacency)
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

  def testEmpyGraphTensor(self):
    result = gt.GraphTensor.from_pieces()
    self.assertEqual(result.shape, [])
    self.assertEqual(result.context.shape, [])
    self.assertEmpty(result.context.features)
    self.assertEmpty(result.node_sets)
    self.assertEmpty(result.edge_sets)

  def testGraphTensor(self):
    result = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(
            features={'label': as_tensor([['X'], ['Y']])}),
        node_sets={
            'a': gt.NodeSet.from_fields(features={},
                                        sizes=as_tensor([[1], [1]])),
            'b': gt.NodeSet.from_fields(features={},
                                        sizes=as_tensor([[2], [1]])),
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
    self.assertEqual(
        edge_spec.adjacency_spec.node_set_name(const.SOURCE), 'a')
    self.assertEqual(
        edge_spec.adjacency_spec.node_set_name(const.TARGET), 'b')


class ReplaceFieldsTest(tu.GraphTensorTestBase):
  """Tests for GraphTensor tranformations."""

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
    node_set = gt.NodeSet.from_fields(features={'a': as_tensor([1., 2.])},
                                      sizes=as_tensor([2]))
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
    edge_set = gt.EdgeSet.from_fields(features={'a': as_tensor([1., 2.])},
                                      sizes=as_tensor([2]),
                                      adjacency=adj.HyperAdjacency.from_indices(
                                          {0: ('a', as_tensor([0, 1]))}))
    result = edge_set.replace_features(features)
    self.assertFieldsEqual(result.features, features)
    self.assertAllEqual(result.sizes, [2])
    self.assertAllEqual(result.adjacency[0], [0, 1])

  def testGraphTensor(self):
    source = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(
            features={'label': as_tensor([['X'], ['Y']])}),
        node_sets={
            'a': gt.NodeSet.from_fields(features={},
                                        sizes=as_tensor([[1], [1]])),
            'b': gt.NodeSet.from_fields(features={},
                                        sizes=as_tensor([[2], [1]])),
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


class ElementsCountsTest(tf.test.TestCase):

  def testEmpty(self):
    graph = gt.GraphTensor.from_pieces()
    self.assertIsNone(graph.spec.total_num_components)

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
            'a': gt.NodeSet.from_fields(
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
            'a': gt.NodeSet.from_fields(
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
      return gt.NodeSet.from_fields(features=features,
                                    sizes=node_set.sizes)

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
      return gt.NodeSet.from_fields(features=features,
                                    sizes=node_set.sizes)

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
                              const.SOURCE: (
                                  'node',
                                  join(a.adjacency[const.SOURCE],
                                       b.adjacency[const.SOURCE])),
                              const.TARGET: (
                                  'node',
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
                              const.SOURCE: (
                                  'node', tf.zeros_like(features, tf.int64)),
                              const.TARGET: (
                                  'node', tf.ones_like(features, tf.int64)),
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

    ds = tf.data.Dataset.range(0, 7)
    ds = ds.map(generate)
    ds = ds.batch(1)
    ds = ds.unbatch()
    ds = ds.batch(3)
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
            shape=[2, None, None],
            dtype=tf.int64,
            ragged_rank=2,
            row_splits_dtype=tf.int32))

    element = next(itr)
    self.assertAllEqual(element['x'], as_ragged([
        [[0, 1, 2, 3, 4, 5]],
    ]))
    self.assertAllEqual(
        type_spec.type_spec_from_value(element['x']),
        tf.RaggedTensorSpec(
            shape=[1, None, None],
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


if __name__ == '__main__':
  tf.test.main()
