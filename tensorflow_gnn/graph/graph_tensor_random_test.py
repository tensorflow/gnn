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
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as gc
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import graph_tensor_random as gr
from tensorflow_gnn.graph import schema_utils
from tensorflow_gnn.proto import graph_schema_pb2 as schema_pb2
from tensorflow_gnn.utils import test_utils

dt = tf.convert_to_tensor
rt = tf.ragged.constant


class TestRandomRaggedTensor(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    gc.enable_graph_tensor_validation_at_runtime()

  @parameterized.parameters([
      tf.bool,
      tf.int8,
      tf.uint8,
      tf.int16,
      tf.uint16,
      tf.int32,
      tf.uint32,
      tf.int32,
      tf.uint32,
      tf.int64,
      tf.uint64,
      tf.bfloat16,
      tf.float16,
      tf.float32,
      tf.float64,
      tf.string,
  ])
  def test_typed_random_values(self, dtype):
    value = gr.typed_random_values(1_000, dtype)
    if dtype.is_floating:
      self.assertAllGreaterEqual(value, 0.0)
      self.assertAllLess(value, 1.0)
    self.assertEqual(dtype, value.dtype)
    self.assertEqual(1_000, value.shape[0])

  @parameterized.parameters(([4],),
                            ([3, None],),
                            ([None, 3],),
                            ([5, None, 4, None, 3],),
                            ([None, 4, None, 3, None],))
  def test_random_ragged_tensor_shapes(self, shape_list):
    tensor = gr.random_ragged_tensor(shape_list, tf.float32)
    tensor.shape.assert_is_compatible_with(shape_list)

  def test_random_ragged_tensor_mixed_dynamic(self):
    shape_list = [tf.random.uniform((), 2, 9, tf.int32), 4, None]
    tensor = gr.random_ragged_tensor(shape_list, tf.float32)
    tensor.shape.assert_is_compatible_with(shape_list)

  @parameterized.parameters(tf.int32,
                            tf.int64,
                            tf.float32,
                            tf.float64,
                            tf.string)
  def test_random_ragged_tensor_types(self, dtype):
    shape_list = [4, None, 3]
    tensor = gr.random_ragged_tensor(shape_list, dtype)
    self.assertIs(tensor.dtype, dtype)
    tensor.shape.assert_is_compatible_with(shape_list)

  def test_random_ragged_tensor_sample_values(self):
    primo = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    tensor = gr.random_ragged_tensor([4, None, 3], tf.int32,
                                     sample_values=primo)
    self.assertTrue(all(value in primo
                        for value in tf.reshape(tensor.flat_values, [-1])))

  def test_random_ragged_tensor_row_lengths_range(self):
    tensor = gr.random_ragged_tensor([100, None], tf.int32,
                                     row_lengths_range=[10, 15])
    self.assertTrue(all(10 <= value < 15
                        for value in tensor.row_lengths()))

  @parameterized.parameters(tf.int32, tf.int64)
  def test_random_ragged_tensor_row_splits_dtype(self, dtype):
    tensor = gr.random_ragged_tensor([4, None, 3], tf.float32,
                                     row_splits_dtype=dtype)
    self.assertIs(dtype, tensor.row_splits.dtype)


class TestRandomEdgeIndicesTensor(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    gc.enable_graph_tensor_validation_at_runtime()

  @parameterized.parameters(tf.int32, tf.int64)
  def test_empty(self, dtype):
    empty = tf.convert_to_tensor([], dtype)
    result = gr._random_edge_indices(empty, empty)
    self.assertAllEqual(result, empty)

  @parameterized.parameters(
      dict(num_edges=[0]), dict(num_edges=[0, 0]), dict(num_edges=[0, 0, 0])
  )
  def test_no_edges(self, num_edges):
    num_edges = tf.convert_to_tensor(num_edges, tf.int64)
    num_nodes = tf.ones_like(num_edges)
    result = gr._random_edge_indices(num_edges, num_nodes)
    self.assertAllEqual(result, tf.convert_to_tensor([], tf.int64))

  def test_single_node(self):
    self.assertAllEqual(
        gr._random_edge_indices(dt([1]), dt([1])),
        dt([0], tf.int64),
    )
    self.assertAllEqual(
        gr._random_edge_indices(dt([1, 1]), dt([1, 1])),
        dt([0, 1], tf.int64),
    )
    self.assertAllEqual(
        gr._random_edge_indices(dt([2]), dt([1])),
        dt([0, 0], tf.int64),
    )
    self.assertAllEqual(
        gr._random_edge_indices(dt([3, 2, 1]), dt([1, 1, 1])),
        dt([0, 0, 0, 1, 1, 2], tf.int64),
    )
    self.assertAllEqual(
        gr._random_edge_indices(dt([1, 0, 0, 1, 0, 1]), dt([1, 1, 1, 1, 1, 1])),
        dt([0, 3, 5], tf.int64),
    )

  def test_components_split(self):
    results = []
    for _ in range(1_00):
      results.append(gr._random_edge_indices(dt([1, 2]), dt([2, 3])))
    result = tf.stack(results, axis=0)
    c1, c2 = result[:, 0], result[:, 1:]
    self.assertBetween(
        tf.math.reduce_mean(tf.cast(c1 == 0, tf.float32)), 0.4, 0.6
    )
    self.assertBetween(
        tf.math.reduce_mean(tf.cast(c1 == 1, tf.float32)), 0.4, 0.6
    )
    self.assertAllInRange(c1, 0, 2, open_upper_bound=True)
    self.assertBetween(
        tf.math.reduce_mean(tf.cast(c2 == 2, tf.float32)), 0.3, 0.5
    )
    self.assertAllInRange(c2, 2, 5, open_upper_bound=True)


class TestRandomGraphTensor(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    gc.enable_graph_tensor_validation_at_runtime()
    self.schema = test_utils.get_proto_resource(
        'testdata/feature_repr.pbtxt', schema_pb2.GraphSchema())
    self.spec = schema_utils.create_graph_spec_from_schema_pb(self.schema)

  def test_random_graph_tensor(self):
    gtensor = gr.random_graph_tensor(self.spec, row_lengths_range=[4, 32])
    self.assertEqual(gtensor.shape, ())

    self.assertEqual(set(gtensor.context.features), {'rankings'})
    rankings = gtensor.context['rankings']
    self.assertEqual(rankings.shape, [1, 4])

    self.assertEqual(set(gtensor.node_sets), {'items', 'persons'})
    items = gtensor.node_sets['items']
    self.assertEqual(items['category'].shape, items.sizes)
    self.assertIsInstance(items['amounts'], tf.RaggedTensor)
    items['amounts'].shape.assert_is_compatible_with(
        tf.TensorShape(items.sizes) + tf.TensorShape([None]))
    persons = gtensor.node_sets['persons']
    self.assertEqual(persons['name'].shape, persons.sizes)
    self.assertEqual(persons['age'].shape, persons.sizes)
    self.assertEqual(persons['country'].shape, persons.sizes)

    self.assertEqual(set(gtensor.edge_sets), {'purchased', 'is-friend'})
    purchased = gtensor.edge_sets['purchased']
    self.assertEqual(purchased.adjacency.source.shape, purchased.sizes)
    self.assertEqual(purchased.adjacency.target.shape, purchased.sizes)
    isfriend = gtensor.edge_sets['is-friend']
    self.assertEqual(isfriend.adjacency.source.shape, isfriend.sizes)
    self.assertEqual(isfriend.adjacency.target.shape, isfriend.sizes)

  def test_generate_random_graph_tensor(self):
    # This form of generation does not require a spec, and works fine.
    ds = tf.data.Dataset.range(1).repeat().map(
        lambda _: gr.random_graph_tensor(self.spec))
    for graph in ds.take(4):
      self.assertIsInstance(graph, gt.GraphTensor)

  def test_generate_random_graph_tensor_from_generator(self):
    def random_graph_tensor_generator(spec) -> tf.data.Dataset:
      def generator():
        while True:
          yield gr.random_graph_tensor(spec)
      return tf.data.Dataset.from_generator(generator, output_signature=spec)

    for graph in random_graph_tensor_generator(self.spec).take(4):
      self.assertIsInstance(graph, gt.GraphTensor)


class TestSpecConstraints(tf.test.TestCase):
  example = gt.GraphTensor.from_pieces(
      context=gt.Context.from_fields(features={'label': dt(['X', 'Y'])}),
      node_sets={
          'a': gt.NodeSet.from_fields(
              features={'f': rt([[1, 2], [3]])}, sizes=dt([1, 1])
          ),
          'b': gt.NodeSet.from_fields(
              features={'f': dt([[1, 0], [2, 0], [3, 0]])}, sizes=dt([2, 1])
          ),
      },
      edge_sets={
          'a->b': gt.EdgeSet.from_fields(
              sizes=dt([2, 1]),
              adjacency=adj.Adjacency.from_indices(
                  source=('a', dt([0, 0, 1])),
                  target=('b', dt([0, 1, 2])),
              ),
          ),
      },
  )

  def setUp(self):
    super().setUp()
    gc.enable_graph_tensor_validation_at_runtime()

  def test_components_fixed(self):
    a_sizes, b_sizes, ab_sizes = [], [], []
    for _ in range(100):
      spec = self.example.spec
      result = gr.random_graph_tensor(
          spec,
          row_lengths_range=(0, 4),
          num_components_range=(
              spec.total_num_components,
              spec.total_num_components + 1,
          ),
      )
      self.assertAllEqual(
          result.spec.total_num_components, spec.total_num_components
      )

      self.assertTrue(
          result.spec.is_compatible_with(
              spec.relax(num_nodes=True, num_edges=True)
          )
      )
      self.assertAllEqual(
          tf.size(result.node_sets['a']['f'].row_lengths()),
          tf.math.reduce_sum(result.node_sets['a'].sizes),
      )
      self.assertAllEqual(
          tf.shape(result.node_sets['b']['f']),
          [tf.math.reduce_sum(result.node_sets['b'].sizes), 2],
      )
      ab = result.edge_sets['a->b']
      for node_set, index in [
          (result.node_sets['a'], ab.adjacency.source),
          (result.node_sets['b'], ab.adjacency.target),
      ]:
        self.assertLess(
            tf.math.reduce_max(index[: ab.sizes[0]]),
            node_set.sizes[0],
        )
        self.assertGreaterEqual(
            tf.math.reduce_min(index[ab.sizes[0] :]),
            node_set.sizes[0],
        )
        self.assertLess(
            tf.math.reduce_max(index[ab.sizes[0] :]),
            node_set.sizes[0] + node_set.sizes[1],
        )

      self.assertAllEqual(result.num_components, 2)
      a_sizes.extend(result.node_sets['a'].sizes.numpy())
      b_sizes.extend(result.node_sets['b'].sizes.numpy())
      ab_sizes.extend(result.edge_sets['a->b'].sizes.numpy())
    self.assertSetEqual({0, 1, 2, 3}, set(a_sizes))
    self.assertSetEqual({0, 1, 2, 3}, set(b_sizes))
    self.assertContainsSubset({0, 1, 2, 3, 4}, set(ab_sizes))

  def test_components_relaxed(self):
    num_components = []
    for _ in range(100):
      spec = self.example.spec
      result = gr.random_graph_tensor(spec, num_components_range=(0, 3))
      self.assertTrue(
          result.spec.is_compatible_with(
              spec.relax(num_components=True, num_nodes=True, num_edges=True)
          )
      )
      self.assertAllEqual(
          tf.size(result.node_sets['a']['f'].row_lengths()),
          tf.math.reduce_sum(result.node_sets['a'].sizes),
      )
      self.assertAllEqual(
          tf.shape(result.node_sets['b']['f']),
          [tf.math.reduce_sum(result.node_sets['b'].sizes), 2],
      )
      num_components.append(result.num_components.numpy())
    self.assertSetEqual({0, 1, 2}, set(num_components))


if __name__ == '__main__':
  tf.test.main()
