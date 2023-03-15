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
import numpy
import tensorflow as tf

from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import graph_tensor_random as gr
from tensorflow_gnn.graph import schema_utils
from tensorflow_gnn.proto import graph_schema_pb2 as schema_pb2
from tensorflow_gnn.utils import test_utils


class TestRandomRaggedTensor(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([(tf.int32, numpy.int32),
                             (tf.int64, numpy.int64),
                             (tf.float32, numpy.float32),
                             (tf.float64, numpy.float64),
                             (tf.string, object)])
  def test_typed_random_values(self, dtype, ptype):
    value = gr.typed_random_values(5, dtype)
    self.assertEqual(5, value.shape[0])

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


class TestRandomGraphTensor(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
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


if __name__ == '__main__':
  tf.test.main()
