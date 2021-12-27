"""Tests for batching_utils_test.py."""
from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import batching_utils
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import graph_tensor_test_utils as tu
from tensorflow_gnn.graph import preprocessing_common as preprocessing

as_tensor = tf.convert_to_tensor
as_ragged = tf.ragged.constant


class DynamicBatchTest(tu.GraphTensorTestBase):
  """Tests for context, node sets and edge sets creation."""

  @parameterized.parameters([
      dict(target_num_components=100, features={
          'id': as_tensor([1]),
      }),
      dict(
          target_num_components=3,
          features={
              'f1': as_tensor([1.]),
              'f2': as_tensor([[1., 2.]]),
              'i3': as_tensor([[[1, 2], [3, 4]]]),
              'r1': as_ragged([[], ['a', 'b']]),
          }),
  ])
  def testFeaturesBatching(self, target_num_components: int,
                           features: gt.Fields):
    source = gt.GraphTensor.from_pieces(
        gt.Context.from_fields(shape=[], features=features))
    dataset = tf.data.Dataset.from_tensors(source)
    dataset = dataset.repeat(target_num_components)
    dataset = batching_utils.dynamic_batch(
        dataset,
        preprocessing.SizesConstraints(
            total_num_components=target_num_components,
            total_num_nodes={},
            total_num_edges={}))
    result = list(dataset)
    self.assertLen(result, 1)
    result = result[0]
    self.assertAllEqual(result.shape, tf.TensorShape([target_num_components]))
    self.assertAllEqual(result.total_num_components, target_num_components)
    self.assertAllEqual(result.context.sizes, [[1]] * target_num_components)

    expected_features = tf.nest.map_structure(
        lambda f: tf.stack([f] * target_num_components, axis=0), features)
    self.assertFieldsEqual(result.context.features, expected_features)

  @parameterized.parameters([tf.data.UNKNOWN_CARDINALITY, 5])
  def testDynamicBatching1(self, cardinality):

    def generate(num_components):
      sizes = tf.ones([num_components], dtype=tf.int64)
      features = {'f': tf.fill([num_components, 2], value=.5)}
      return gt.GraphTensor.from_pieces(
          gt.Context.from_fields(features=features, sizes=sizes))

    dataset = tf.data.Dataset.from_tensor_slices([1, 0, 2, 3, 2])
    dataset = dataset.map(generate)
    if cardinality == tf.data.UNKNOWN_CARDINALITY:
      dataset = dataset.filter(lambda _: True)
    self.assertEqual(dataset.cardinality(), cardinality)

    dataset = batching_utils.dynamic_batch(
        dataset,
        preprocessing.SizesConstraints(
            total_num_components=4, total_num_nodes={}, total_num_edges={}))
    self.assertEqual(dataset.cardinality(), tf.data.UNKNOWN_CARDINALITY)
    result = list(dataset)
    self.assertLen(result, 3)
    self.assertAllEqual(result[0].num_components, [1, 0, 2])
    self.assertAllEqual(
        result[0].context.features['f'],
        as_ragged([[[.5, .5]], [], [[.5, .5], [.5, .5]]], ragged_rank=1))
    self.assertAllEqual(result[1].num_components, [3])
    self.assertAllEqual(
        result[1].context.features['f'],
        as_ragged([[[.5, .5], [.5, .5], [.5, .5]]], ragged_rank=1))
    self.assertAllEqual(result[2].num_components, [2])
    self.assertAllEqual(result[2].context.features['f'],
                        as_ragged([[[.5, .5], [.5, .5]]], ragged_rank=1))

  @parameterized.parameters([tf.data.UNKNOWN_CARDINALITY, 6])
  def testDynamicBatching2(self, cardinality):

    def generate(num_components):
      sizes = tf.ones([num_components], dtype=tf.int64)
      features = {'f': tf.fill([num_components, 2], value=.5)}
      return gt.GraphTensor.from_pieces(
          gt.Context.from_fields(features=features, sizes=sizes))

    dataset = tf.data.Dataset.from_tensor_slices([0, 1, 2, 3, 2, 2])
    dataset = dataset.map(generate)
    if cardinality == tf.data.UNKNOWN_CARDINALITY:
      dataset = dataset.filter(lambda _: True)
    self.assertEqual(dataset.cardinality(), cardinality)

    dataset = batching_utils.dynamic_batch(
        dataset,
        preprocessing.SizesConstraints(
            total_num_components=4, total_num_nodes={}, total_num_edges={}))
    self.assertEqual(dataset.cardinality(), tf.data.UNKNOWN_CARDINALITY)
    result = list(dataset)
    self.assertLen(result, 3)
    self.assertAllEqual(result[0].num_components, [0, 1, 2])
    self.assertAllEqual(
        result[0].context.features['f'],
        as_ragged([[], [[.5, .5]], [[.5, .5], [.5, .5]]], ragged_rank=1))
    self.assertAllEqual(result[1].num_components, [3])
    self.assertAllEqual(
        result[1].context.features['f'],
        as_ragged([[[.5, .5], [.5, .5], [.5, .5]]], ragged_rank=1))
    self.assertAllEqual(result[2].num_components, [2, 2])
    self.assertAllEqual(
        result[2].context.features['f'],
        as_ragged([[[.5, .5]] * 2, [[.5, .5]] * 2], ragged_rank=1))

  @parameterized.parameters(
      [tf.data.UNKNOWN_CARDINALITY, tf.data.INFINITE_CARDINALITY])
  def testInfiniteDataset(self, cardinality):

    def generate(num_components):
      sizes = tf.ones([num_components], dtype=tf.int64)
      return gt.GraphTensor.from_pieces(gt.Context.from_fields(sizes=sizes))

    dataset = tf.data.Dataset.from_tensor_slices([1, 2, 1])
    dataset = dataset.map(generate)
    dataset = dataset.repeat()
    if cardinality == tf.data.UNKNOWN_CARDINALITY:
      dataset = dataset.filter(lambda _: True)
    self.assertEqual(dataset.cardinality(), cardinality)

    dataset = batching_utils.dynamic_batch(
        dataset,
        preprocessing.SizesConstraints(
            total_num_components=3, total_num_nodes={}, total_num_edges={}))

    self.assertEqual(dataset.cardinality(), cardinality)
    dataset = dataset.map(lambda g: g.total_num_components)
    dataset = dataset.take(3 * 100)
    # (1 2) (1 1) (2 1) (1 2) (1 1) (2 1) ..
    self.assertEqual(list(dataset.as_numpy_iterator()), [3, 2, 3] * 100)

  test_a2b4_ab3_graph = gt.GraphTensor.from_pieces(
      node_sets={
          'a': gt.NodeSet.from_fields(features={'f': [1., 2.]}, sizes=[2]),
          'b': gt.NodeSet.from_fields(features={}, sizes=[4]),
      },
      edge_sets={
          'a->b':
              gt.EdgeSet.from_fields(
                  features={'f': as_tensor([1., 2., 3.])},
                  sizes=as_tensor([3]),
                  adjacency=adj.Adjacency.from_indices(
                      ('a', as_tensor([0, 1, 1])),
                      ('b', as_tensor([0, 1, 3])),
                  )),
      },
  )
  test_a1b1_ab1_graph = gt.GraphTensor.from_pieces(
      node_sets={
          'a': gt.NodeSet.from_fields(features={'f': [3.]}, sizes=[1]),
          'b': gt.NodeSet.from_fields(features={}, sizes=[1]),
      },
      edge_sets={
          'a->b':
              gt.EdgeSet.from_fields(
                  features={'f': as_tensor([4.])},
                  sizes=as_tensor([1]),
                  adjacency=adj.Adjacency.from_indices(
                      ('a', as_tensor([0])),
                      ('b', as_tensor([0])),
                  )),
      },
  )

  def testGraphBatching(self):

    def generate(index):
      return tf.cond(
          index <= 1,
          lambda: self.test_a1b1_ab1_graph,
          lambda: self.test_a2b4_ab3_graph,
      )

    dataset = tf.data.Dataset.range(5)
    dataset = dataset.map(generate)
    dataset = batching_utils.dynamic_batch(
        dataset,
        preprocessing.SizesConstraints(
            total_num_components=4,
            total_num_nodes={
                'a': 5,
                'b': 7
            },
            total_num_edges={'a->b': 6}))
    result = list(dataset)
    # [(1,1,1), (1,1,1), (2,4,3)], [(2,4,3)], [(2,4,3)]
    self.assertLen(result, 3)
    self.assertAllEqual(result[0].num_components, [1, 1, 1])
    self.assertAllEqual(result[0].node_sets['a'].sizes, [[1], [1], [2]])
    self.assertAllEqual(result[0].node_sets['a']['f'],
                        as_ragged([[3.], [3.], [1., 2.]]))
    self.assertAllEqual(result[0].node_sets['b'].sizes, [[1], [1], [4]])
    self.assertAllEqual(result[0].edge_sets['a->b'].sizes, [[1], [1], [3]])
    self.assertAllEqual(result[0].edge_sets['a->b']['f'],
                        as_ragged([[4.], [4.], [1., 2., 3.]]))

    def check_equal(x, y):
      self.assertAllEqual(x, y)
      return x

    self.assertAllEqual(result[1].num_components, [1])
    tf.nest.map_structure(
        check_equal,
        result[1].merge_batch_to_components(),
        self.test_a2b4_ab3_graph,
        expand_composites=True)

    self.assertAllEqual(result[2].num_components, [1])
    tf.nest.map_structure(
        check_equal,
        result[2].merge_batch_to_components(),
        self.test_a2b4_ab3_graph,
        expand_composites=True)

  def testRaisesOnInvalidConfig(self):

    dataset = tf.data.Dataset.from_tensors(self.test_a1b1_ab1_graph)

    def batch(dataset, constrains):
      return batching_utils.dynamic_batch(dataset, constrains)

    no_a_node = preprocessing.SizesConstraints(
        total_num_components=1,
        total_num_nodes={'b': 100},
        total_num_edges={'a->b': 100})
    self.assertRaisesRegex(
        ValueError,
        ('The maximum total number of <a> nodes must be specified as'
         r' `constraints.total_num_nodes\[<a>\]`'),
        lambda: batch(dataset, no_a_node))

    no_edge = preprocessing.SizesConstraints(
        total_num_components=1,
        total_num_nodes={
            'a': 100,
            'b': 100
        },
        total_num_edges={'?': 200})
    self.assertRaisesRegex(
        ValueError,
        ('The maximum total number of <a->b> edges must be specified as'
         r' `constraints.total_num_edges\[<a->b>\]`'),
        lambda: batch(dataset, no_edge))

  @parameterized.parameters([True, False])
  def testRaisesOnImpossibleBatching(self, repeat):

    def generate(index):
      return tf.cond(
          index <= 2,
          lambda: self.test_a1b1_ab1_graph,
          lambda: self.test_a2b4_ab3_graph,
      )

    dataset = tf.data.Dataset.range(5)
    dataset = dataset.map(generate)
    if repeat:
      dataset = dataset.repeat()

    def batch(dataset, constrains):
      dataset = batching_utils.dynamic_batch(dataset, constrains)
      dataset = dataset.take(5)
      return list(dataset)

    components_overflow = preprocessing.SizesConstraints(
        total_num_components=0,
        total_num_nodes={
            'a': 100,
            'b': 100
        },
        total_num_edges={'a->b': 100})
    self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        ('Could not pad graph as it already has more graph components'
         ' then it is allowed by `total_sizes.total_num_components`'),
        lambda: batch(dataset, components_overflow))

    nodes_overflow = preprocessing.SizesConstraints(
        total_num_components=2,
        total_num_nodes={
            'a': 100,
            'b': 2
        },
        total_num_edges={'a->b': 100})
    self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                           ('Could not pad <b> as it already has more nodes'
                            ' then it is allowed by the'
                            r' `total_sizes.total_num_nodes\[<b>\]`'),
                           lambda: batch(dataset, nodes_overflow))

    edges_overflow = preprocessing.SizesConstraints(
        total_num_components=2,
        total_num_nodes={
            'a': 100,
            'b': 100
        },
        total_num_edges={'a->b': 2})
    self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                           ('Could not pad <a->b> as it already has more edges'
                            ' then it is allowed by the'
                            r' `total_sizes.total_num_edges\[<a->b>\]'),
                           lambda: batch(dataset, edges_overflow))


if __name__ == '__main__':
  tf.test.main()
