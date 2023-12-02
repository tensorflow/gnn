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
"""Tests for GraphPiece extension type."""

from typing import Mapping, Tuple, Union

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_gnn.graph import graph_piece as gp
from tensorflow_gnn.graph import tf_internal
from tensorflow_gnn.utils import tf_test_utils as tftu

TestValue = Union[np.ndarray, tf.RaggedTensor, tf.Tensor, 'TestPiece']


class TestPiece(gp.GraphPieceBase):
  """Graph piece implementation for a simple value as a nest."""

  @classmethod
  def from_value(
      cls,
      value: Mapping[str, TestValue],
      shape_dims: Tuple[int, ...] = (),
      *,
      metadata=None,
      indices_dtype: tf.dtypes.DType = tf.int32,
      row_splits_dtype: tf.dtypes.DType = tf.int64,
      check_consistent_indices_dtype: bool = True,
      check_consistent_row_splits_dtype: bool = True
  ) -> 'TestPiece':
    def convert_fn(field):
      assert isinstance(
          field, (np.ndarray, tf.Tensor, tf.RaggedTensor, TestPiece)), field
      return tf.convert_to_tensor(field) if isinstance(field,
                                                       np.ndarray) else field

    return cls._from_data(
        tf.nest.map_structure(convert_fn, value),
        shape=tf.TensorShape(shape_dims),
        indices_dtype=indices_dtype,
        row_splits_dtype=row_splits_dtype,
        metadata=metadata,
        check_consistent_indices_dtype=check_consistent_indices_dtype,
        check_consistent_row_splits_dtype=check_consistent_row_splits_dtype,
        validate=True
    )

  @property
  def value(self):
    return self._data

  @staticmethod
  def _type_spec_cls():
    return TestPieceSpec


@tf_internal.type_spec_register('tensorflow_gnn.TestPieceSpec')
class TestPieceSpec(gp.GraphPieceSpecBase):
  """Graph piece specification."""

  @staticmethod
  def _value_type():
    return TestPiece

  @property
  def value_spec(self):
    return self._data_spec


class NestingTest(tf.test.TestCase, parameterized.TestCase):

  def testCreation(self):
    piece = TestPiece.from_value(
        {
            'a': np.array([1, 2, 3]),
            'b': {
                '1': tf.ragged.constant([[1, 2], [3], []])
            }
        }, (),
        metadata={
            'count': 1,
            'dtype': tf.int32
        })
    self.assertAllEqual(piece.value['a'], [1, 2, 3])
    self.assertAllEqual(piece.value['b']['1'],
                        tf.ragged.constant([[1, 2], [3], []]))
    self.assertAllEqual(piece.spec._metadata, {'count': 1, 'dtype': tf.int32})

  @parameterized.parameters([((3,),), ((None,),), ((3, 2),), ((3, None),),
                             ((None, None),)])
  def testShapeUpdate(self, batch_shape):
    leaf = TestPiece.from_value(
        np.array([[[1], [0]], [[2], [0]], [[3], [0]]]), shape_dims=())
    self.assertAllEqual(leaf.shape, tf.TensorShape([]))
    root = TestPiece.from_value(TestPiece.from_value(leaf), batch_shape)
    self.assertTrue(root.shape.is_compatible_with(batch_shape))
    self.assertTrue(root.value.value.shape.is_compatible_with(batch_shape))

  @parameterized.parameters([((1,),), ((2,),), ((4,),), ((3, 1),), ((3, 3),),
                             ((None, 3),), ((2, None),)])
  def testInvalidShapeUpdate(self, batch_shape):
    leaf = TestPiece.from_value(
        np.array([[[1], [0]], [[2], [0]], [[3], [0]]]), shape_dims=())
    self.assertAllEqual(leaf.shape, tf.TensorShape([]))
    err_msg = ('Fields have batch dimensions that are not compatible'
               ' with the GraphPiece shape')
    self.assertRaisesRegex(
        ValueError, err_msg,
        lambda: TestPiece.from_value(TestPiece.from_value(leaf), batch_shape))

  def testMetadata(self):
    leaf = TestPiece.from_value(
        np.array([1]), metadata={'foo': 1, 'bar': 2, 'baz': 3}
    )
    root = TestPiece.from_value(leaf, metadata={'foo': 10, 'bar': 20})
    self.assertAllEqual(root.spec._metadata, {'foo': 10, 'bar': 20})
    self.assertAllEqual(
        root.value.spec._metadata, {'foo': 1, 'bar': 2, 'baz': 3}
    )

  def testNoMetadataUpdate(self):
    leaf = TestPiece.from_value(np.array([1]))
    root = TestPiece.from_value(leaf, metadata={'ignore': 'y'})
    self.assertIsNone(root.value.spec._metadata)

  def testNoMetadataUpdate2(self):
    leaf = TestPiece.from_value(np.array([1]), metadata={'ignore': 'y'})
    root = TestPiece.from_value(leaf)
    self.assertAllEqual(root.value.spec._metadata, {'ignore': 'y'})


class TfFunctionTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for graph piece with TF Function."""

  @parameterized.parameters([
      (np.array([0]), np.array([1])),
      (np.array([0, 1]), np.array([1, 2])),
      (np.array([[0], [1]]), np.array([[1], [2]])),
      (tf.ragged.constant([[1., 2.], [3.]]), tf.ragged.constant([[], []])),
  ])
  def testNoRetracing(self, value1, value2):

    @tf.function
    def add1(piece):
      return TestPiece.from_value(piece.value + 1)

    piece1 = TestPiece.from_value(value1)
    piece2 = TestPiece.from_value(value2)

    piece1 = add1(piece1)
    piece1 = add1(piece1)
    piece2 = add1(piece2)
    piece2 = add1(piece2)
    piece1 = add1(piece1)
    piece2 = add1(piece2)
    self.assertEqual(add1.experimental_get_tracing_count(), 1)
    self.assertAllEqual(piece1.value, value1 + 3)
    self.assertAllEqual(piece2.value, value2 + 3)

  @parameterized.parameters([
      ({
          'x': np.array([0]),
          'y': np.array([0, 1])
      }, {
          'x': np.array([1]),
          'y': np.array([1, 0])
      }),
  ])
  def testNoRetracingDict(self, value1, value2):

    @tf.function
    def add1(piece):
      return TestPiece.from_value(
          {
              'x': piece.value['x'] + 1,
              'y': piece.value['y'] + 1
          },
          metadata={'ttl': 1})

    piece1 = TestPiece.from_value(value1, metadata={'ttl': 1})
    piece2 = TestPiece.from_value(value2, metadata={'ttl': 1})

    piece1 = add1(piece1)
    piece1 = add1(piece1)
    piece2 = add1(piece2)
    piece2 = add1(piece2)
    piece1 = add1(piece1)
    piece2 = add1(piece2)
    self.assertEqual(add1.experimental_get_tracing_count(), 1)
    self.assertAllEqual(piece1.value['x'], value1['x'] + 3)
    self.assertAllEqual(piece2.value['y'], value2['y'] + 3)

  @parameterized.parameters([
      (np.array([0]), np.array([1, 2])),
      (np.array([0, 1]), np.array([[1, 2]])),
      (np.array([[1, 2], [3, 4]]), np.array([[2, 1], [4, 3], [5, 4]])),
      (tf.ragged.constant([[1., 2.], [3.]]), tf.ragged.constant([[], [], []])),
  ])
  def testRetracing(self, value1, value2):

    @tf.function
    def add1(piece):
      return TestPiece.from_value(piece.value + 1)

    piece1 = TestPiece.from_value(value1)
    piece2 = TestPiece.from_value(value2)

    piece2 = add1(piece2)
    piece2 = add1(piece2)
    piece1 = add1(piece1)
    piece1 = add1(piece1)
    piece1 = add1(piece1)
    piece2 = add1(piece2)
    self.assertEqual(add1.experimental_get_tracing_count(), 2)
    self.assertAllEqual(piece1.value, value1 + 3)
    self.assertAllEqual(piece2.value, value2 + 3)

  def testRetracingWithMetadata(self):

    @tf.function
    def inc_metadata(piece):
      return TestPiece.from_value(
          piece.value, metadata={'ttl': (piece.spec._metadata['ttl'] + 1)})

    piece = TestPiece.from_value(np.array([1]), metadata={'ttl': 0})
    piece = inc_metadata(piece)
    piece = inc_metadata(piece)
    piece = inc_metadata(piece)
    self.assertEqual(inc_metadata.experimental_get_tracing_count(), 3)
    self.assertAllEqual(piece.spec._metadata['ttl'], 3)

  @parameterized.parameters([
      (np.array([0]), np.array([1])),
      (np.array([0, 1]), np.array([1, 2])),
      (np.array([[0], [1]]), np.array([[1], [2]])),
      (tf.ragged.constant([[1, 2], [3]]), tf.ragged.constant([[1], [2, 3]])),
  ])
  def testRetracingWithNesting(self, value1, value2):

    @tf.function
    def nest(piece):
      return TestPiece.from_value(piece)

    piece1 = TestPiece.from_value(value1)
    piece1 = nest(piece1)
    piece1 = nest(piece1)
    self.assertEqual(nest.experimental_get_tracing_count(), 2)
    self.assertAllEqual(piece1.value.value.value, value1)

    piece2 = TestPiece.from_value(value2)
    piece2 = nest(piece2)
    piece2 = nest(piece2)
    self.assertEqual(nest.experimental_get_tracing_count(), 2)
    self.assertAllEqual(piece2.value.value.value, value2)

  @parameterized.parameters([
      (0, np.array([0]), np.array([1])),
      (1, np.array([[0], [1]]), np.array([[1], [2], [3]])),
      (2, np.array([[[0]], [[1]]]), np.array([[[1]], [[2]], [[3]]])),
      (1, tf.ragged.constant([[1, 2], [3]]), tf.ragged.constant([[1], [2, 3]])),
  ])
  def testShapesInEagerMode(self, batch_rank, true_value, false_value):
    shape_dims = [None] * batch_rank

    @tf.function
    def cond_fn(pred: tf.Tensor) -> tf.Tensor:
      return tf.cond(
          pred,  #
          lambda: TestPiece.from_value(true_value, shape_dims=shape_dims),
          lambda: TestPiece.from_value(false_value, shape_dims=shape_dims))

    true_piece = cond_fn(tf.convert_to_tensor(True))
    self.assertEqual(true_piece.shape.rank, batch_rank)
    self.assertTrue(true_piece.shape.is_fully_defined())
    self.assertEqual(true_piece.shape, true_value.shape[:batch_rank])

    false_piece = cond_fn(tf.convert_to_tensor(False))
    self.assertEqual(false_piece.shape.rank, batch_rank)
    self.assertTrue(false_piece.shape.is_fully_defined())
    self.assertEqual(false_piece.shape, false_piece.shape[:batch_rank])

    self.assertEqual(cond_fn.experimental_get_tracing_count(), 1)

  @parameterized.parameters([
      (0, np.array([0]), np.array([1])),
      (1, np.array([[0], [1]]), np.array([[1], [2], [3]])),
      (2, np.array([[[0]], [[1]]]), np.array([[[1]], [[2]], [[3]]])),
      (1, tf.ragged.constant([[1, 2], [3]]), tf.ragged.constant([[1], [2, 3]])),
  ])
  def testShapesRelaxation(self, batch_rank, true_value, false_value):
    shape_dims = [None] * batch_rank

    @tf.function
    def readout(value: TestPiece) -> tf.Tensor:
      return value.value

    @tf.function
    def cond_fn(pred: tf.Tensor) -> tf.Tensor:
      result = tf.cond(
          pred,  #
          lambda: TestPiece.from_value(true_value, shape_dims=shape_dims),
          lambda: TestPiece.from_value(false_value, shape_dims=shape_dims))

      return readout(result)

    self.assertAllEqual(true_value, cond_fn(tf.convert_to_tensor(True)))
    self.assertAllEqual(false_value, cond_fn(tf.convert_to_tensor(False)))

    self.assertEqual(readout.experimental_get_tracing_count(), 1)
    self.assertEqual(cond_fn.experimental_get_tracing_count(), 1)


class BatchingUnbatchingTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for graph piece with TF Dataset batching/unbatching."""

  def testStaticSpecs(self):

    @tf.function
    def generate(_):
      return TestPiece.from_value(value=np.array([1, 2, 3]))

    ds = tf.data.Dataset.range(0, 7)
    ds = ds.map(generate)
    ds = ds.batch(3, True)

    self.assertAllEqual(ds.element_spec._data_spec,
                        tf.TensorSpec(shape=[3, 3], dtype=tf.int64))

    ds = ds.batch(2)
    self.assertAllEqual(ds.element_spec._data_spec,
                        tf.TensorSpec(shape=[None, 3, 3], dtype=tf.int64))

  @parameterized.parameters([tf.int32, tf.int64])
  def testDynamicSpecs(self, row_splits_dtype: tf.DType):
    @tf.function
    def generate(num_nodes):
      return TestPiece.from_value(
          value=tf.range(num_nodes), row_splits_dtype=row_splits_dtype
      )

    ds = tf.data.Dataset.range(0, 7)
    ds = ds.map(generate)
    ds = ds.batch(3, True)

    self.assertAllEqual(
        ds.element_spec._data_spec,
        tf.RaggedTensorSpec(
            shape=[3, None],
            dtype=tf.int64,
            ragged_rank=1,
            row_splits_dtype=row_splits_dtype))

    ds = ds.batch(2)
    self.assertAllEqual(
        ds.element_spec._data_spec,
        tf.RaggedTensorSpec(
            shape=[None, 3, None],
            dtype=tf.int64,
            ragged_rank=2,
            row_splits_dtype=row_splits_dtype))

  @parameterized.parameters([tf.int32, tf.int64])
  def testRaggedSpecs(self, row_splits_dtype: tf.DType):
    @tf.function
    def generate(num_nodes):
      return TestPiece.from_value(
          value={
              'r':
                  tf.RaggedTensor.from_row_lengths(
                      tf.ones(tf.stack([num_nodes], 0), dtype=tf.float32),
                      tf.stack([0, num_nodes, 0], 0)),
          }, row_splits_dtype=row_splits_dtype)

    ds = tf.data.Dataset.range(0, 9, output_type=row_splits_dtype)
    ds = ds.map(generate)
    ds = ds.batch(1)
    ds = ds.unbatch()
    ds = ds.batch(3, True)
    ds = ds.batch(2)

    itr = iter(ds)
    element = next(itr)
    self.assertAllEqual(
        tf.type_spec_from_value(element.value['r']),
        tf.RaggedTensorSpec(
            shape=[2, 3, 3, None],
            dtype=tf.float32,
            ragged_rank=3,
            row_splits_dtype=row_splits_dtype))

    element = next(itr)
    self.assertAllEqual(
        tf.type_spec_from_value(element.value['r']),
        tf.RaggedTensorSpec(
            shape=[1, 3, 3, None],
            dtype=tf.float32,
            ragged_rank=3,
            row_splits_dtype=row_splits_dtype,
        ),
    )

  def testSpecsWithDropRemainder(self):
    @tf.function
    def generate(_):
      return TestPiece.from_value(value=np.array([1, 2, 3]))

    ds = tf.data.Dataset.range(0, 7)
    ds = ds.map(generate)
    ds = ds.batch(2, drop_remainder=True)

    self.assertAllEqual(ds.element_spec._data_spec,
                        tf.TensorSpec(shape=[2, 3], dtype=tf.int64))

    ds = ds.batch(1, drop_remainder=True)
    self.assertAllEqual(ds.element_spec._data_spec,
                        tf.TensorSpec(shape=[1, 2, 3], dtype=tf.int64))

  def testBatchShapes(self):

    def generate(offset):
      value = tf.range(offset)
      return TestPiece.from_value(value=value)

    ds = tf.data.Dataset.range(0, 3)
    ds = ds.map(generate)
    ds = ds.batch(2)
    self.assertAllEqual(ds.element_spec.shape, tf.TensorShape([None]))
    itr = iter(ds)
    self.assertAllEqual(next(itr).shape, tf.TensorShape([2]))
    self.assertAllEqual(next(itr).shape, tf.TensorShape([1]))

  def testShapesRelaxation(self):

    def cond_fn(pred):
      return tf.cond(pred, lambda: TestPiece.from_value(np.array([0])),
                     lambda: TestPiece.from_value(np.array([0, 1])))

    ds = tf.data.Dataset.from_tensor_slices([True, False])
    ds = ds.map(cond_fn)
    self.assertAllEqual(ds.element_spec.shape, tf.TensorShape([]))
    itr = iter(ds)
    self.assertAllEqual(next(itr).value, np.array([0]))
    self.assertAllEqual(next(itr).value, np.array([0, 1]))

    ds = ds.batch(2, drop_remainder=True)
    self.assertAllEqual(ds.element_spec.shape, tf.TensorShape([2]))
    self.assertAllEqual(next(iter(ds)).value, tf.ragged.constant([[0], [0, 1]]))
    ds = ds.unbatch()
    itr = iter(ds)
    self.assertAllEqual(next(itr).value, np.array([0]))
    self.assertAllEqual(next(itr).value, np.array([0, 1]))

  def testFixedSize(self):

    @tf.function
    def generate(offset):
      value = tf.range(offset, offset + 2)
      value = tf.ensure_shape(value, [2])
      return TestPiece.from_value(value=value, metadata={'fixed': True})

    ds = tf.data.Dataset.range(0, 7)
    ds = ds.map(generate)
    ds = ds.batch(3, True)
    element = next(iter(ds))
    self.assertAllEqual(element.value, np.array([[0, 1], [1, 2], [2, 3]]))
    ds = ds.unbatch().batch(2, True).batch(2, True)

    itr = iter(ds)
    element = next(itr)
    self.assertAllEqual(element.value,
                        np.array([[[0, 1], [1, 2]], [[2, 3], [3, 4]]]))
    self.assertAllEqual(element.spec._metadata, {'fixed': True})
    self.assertRaises(StopIteration, lambda: next(itr))

  def testDynamic(self):

    @tf.function
    def generate(num_nodes):
      return TestPiece.from_value(value=tf.range(num_nodes),)

    ds = tf.data.Dataset.range(0, 6)
    ds = ds.map(generate)
    ds = ds.batch(3)
    element = next(iter(ds))
    self.assertAllEqual(element.value, tf.ragged.constant([[], [0], [0, 1]]))
    ds = ds.unbatch().batch(2, True).batch(2)

    itr = iter(ds)
    element = next(itr)
    self.assertAllEqual(element.value,
                        tf.ragged.constant([[[], [0]], [[0, 1], [0, 1, 2]]]))
    element = next(itr)
    self.assertAllEqual(element.value,
                        tf.ragged.constant([[[0, 1, 2, 3], [0, 1, 2, 3, 4]]]))
    self.assertRaises(StopIteration, lambda: next(itr))

  def testNestedPieces(self):

    @tf.function
    def generate(num_nodes):
      value = TestPiece.from_value(value=tf.range(num_nodes))
      value = TestPiece.from_value(value=value)
      return TestPiece.from_value(value=value)

    ds = tf.data.Dataset.range(0, 7)
    ds = ds.map(generate)
    ds = ds.batch(3)
    element = next(iter(ds))
    self.assertAllEqual(element.value.value.value,
                        tf.ragged.constant([[], [0], [0, 1]]))
    ds = ds.unbatch().batch(2, True).batch(2)
    element = next(iter(ds))
    self.assertAllEqual(element.value.value.value,
                        tf.ragged.constant([[[], [0]], [[0, 1], [0, 1, 2]]]))

  def testFixedSizeNestedPieces(self):

    @tf.function
    def generate(offset):
      value = tf.range(offset, offset + 2)
      value = tf.ensure_shape(value, [2])
      return TestPiece.from_value(TestPiece.from_value(value=value))

    ds = tf.data.Dataset.range(0, 7)
    ds = ds.map(generate)
    ds = ds.batch(3, True)
    element = next(iter(ds))
    self.assertAllEqual(element.value.value, np.array([[0, 1], [1, 2], [2, 3]]))
    ds = ds.unbatch().batch(2, True).batch(2, True)
    element = next(iter(ds))
    self.assertAllEqual(element.value.value,
                        np.array([[[0, 1], [1, 2]], [[2, 3], [3, 4]]]))

  @parameterized.parameters([tf.int32, tf.int64])
  def testRagged(self, row_splits_dtype: tf.dtypes.DType):
    @tf.function
    def generate(num_nodes):
      return TestPiece.from_value(
          value={
              'x':
                  tf.range(num_nodes, dtype=tf.int32),
              'r':
                  tf.RaggedTensor.from_row_lengths(
                      tf.ones(tf.stack([num_nodes], 0), dtype=tf.float32),
                      tf.stack([0, num_nodes, 0], 0)),
          }, row_splits_dtype=row_splits_dtype)

    ds = tf.data.Dataset.range(0, 9, output_type=row_splits_dtype)
    ds = ds.map(generate)
    ds = ds.batch(1)
    ds = ds.unbatch()
    ds = ds.batch(3, True)
    ds = ds.batch(2)

    self.assertEqual(ds.element_spec.row_splits_dtype, row_splits_dtype)

    itr = iter(ds)
    element = next(itr)
    self.assertAllEqual(
        element.value['x'],
        tf.ragged.constant([
            [[], [0], [0, 1]],
            [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]],
        ], row_splits_dtype=row_splits_dtype))
    self.assertAllEqual(
        element.value['r'],
        tf.ragged.constant([
            [[[], [], []], [[], [1], []], [[], [1, 1], []]],
            [[[], [1, 1, 1], []], [[], [1, 1, 1, 1], []],
             [[], [1, 1, 1, 1, 1], []]],
        ], row_splits_dtype=row_splits_dtype))

    element = next(itr)
    self.assertAllEqual(element.value['x'],
                        tf.ragged.constant([
                            [[0, 1, 2, 3, 4, 5],
                             [0, 1, 2, 3, 4, 5, 6],
                             [0, 1, 2, 3, 4, 5, 6, 7]],
                        ], row_splits_dtype=row_splits_dtype))
    self.assertAllEqual(
        tf.type_spec_from_value(element.value['x']),
        tf.RaggedTensorSpec(
            shape=[1, 3, None],
            dtype=tf.int32,
            ragged_rank=2,
            row_splits_dtype=row_splits_dtype))


class ShapeInvariantsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name='scalar.int32',
          field=tf.convert_to_tensor([1]),
          batch_rank=0,
          dtype=tf.int32,
          expected_result=tf.convert_to_tensor([], tf.int32),
      ),
      dict(
          testcase_name='scalar.int64',
          field=tf.convert_to_tensor([1]),
          batch_rank=0,
          dtype=tf.int64,
          expected_result=tf.convert_to_tensor([], tf.int64),
      ),
      dict(
          testcase_name='dense.rank1',
          field=tf.convert_to_tensor([1, 2, 3]),
          batch_rank=1,
          dtype=tf.int32,
          expected_result=tf.convert_to_tensor([3], tf.int32),
      ),
      dict(
          testcase_name='dense.rank2',
          field=tf.convert_to_tensor([[[1], [2], [3]], [[4], [5], [6]]]),
          batch_rank=2,
          dtype=tf.int64,
          expected_result=tf.convert_to_tensor([2, 3], tf.int64),
      ),
      dict(
          testcase_name='ragged.rank1',
          field=tf.ragged.constant([[], [1], [2, 3]]),
          batch_rank=1,
          dtype=tf.int32,
          expected_result=tf.convert_to_tensor([3], tf.int32),
      ),
      dict(
          testcase_name='ragged.rank3',
          field=tf.RaggedTensor.from_tensor(
              tf.zeros([3, 4, 2, 5]), ragged_rank=3
          ),
          batch_rank=3,
          dtype=tf.int32,
          expected_result=tf.convert_to_tensor([3, 4, 2], tf.int32),
      ),
      dict(
          testcase_name='ragged.rank3.uniform_inner',
          field=tf.RaggedTensor.from_tensor(
              tf.zeros([3, 4, 2, 5]), ragged_rank=1
          ),
          batch_rank=3,
          dtype=tf.int32,
          expected_result=tf.convert_to_tensor([3, 4, 2], tf.int32),
      ),
      dict(
          testcase_name='ragged.rank2',
          field=tf.RaggedTensor.from_tensor(
              tf.ones([2, 3, 1]), ragged_rank=2
          ).with_values(tf.ragged.constant([[], [1], [2, 3], [4], [], [5]])),
          batch_rank=2,
          dtype=tf.int32,
          expected_result=tf.convert_to_tensor([2, 3], tf.int32),
      ),
  ])
  def testBatchShapeTensorForFields(
      self,
      field: gp.Field,
      batch_rank: int,
      dtype: tf.dtypes.DType,
      expected_result: tf.Tensor,
  ):
    self.assertAllEqual(
        expected_result, gp._get_batch_shape_tensor(field, batch_rank, dtype)
    )

  def testBatchShapeTensorForPieces(self):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
        ]
    )
    def get_shape(f):
      piece = TestPiece.from_value(
          TestPiece.from_value(
              {
                  'f': f,
              },
              shape_dims=(None,),
          ),
          shape_dims=(None,),
      )
      return gp.get_shape_tensor(piece), gp.get_shape_tensor(piece.value)

    shape1, shape2 = get_shape(tf.convert_to_tensor([[1], [2], [3]]))

    self.assertAllEqual([3], shape1)
    self.assertAllEqual(shape1, shape2)

  def testRaisesOnDenseFieldsShapeMismatch(self):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
            tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
        ]
    )
    def build(f1, f2):
      return TestPiece.from_value(
          {
              'f1': f1,
              'f2': f2,
          },
          shape_dims=(None,),
      )

    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError, 'Fields have different batch dimensions'
    ):
      build(
          tf.convert_to_tensor([[1], [2]]),
          tf.convert_to_tensor([[1], [2], [3]]),
      )

  def testRaisesOnRaggedFieldsShapeMismatch(self):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
            tf.RaggedTensorSpec(
                shape=[None, None], dtype=tf.int32, ragged_rank=1
            ),
        ]
    )
    def build(d, r):
      TestPiece.from_value(
          {
              'd': d,
              'r': r,
          },
          shape_dims=(None,),
      )

    _ = build(
        tf.convert_to_tensor([[1], [2]]),
        tf.ragged.constant([[1], [2]]),
    )

    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError, 'Fields have different batch dimensions'
    ):
      build(
          tf.convert_to_tensor([[1], [2]]),
          tf.ragged.constant([[1], [2], [3]]),
      )

  def testRaisesOnComponentsShapeMismatch(self):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
            tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
        ]
    )
    def build(f1, f2):
      return TestPiece.from_value(
          {
              'f1': TestPiece.from_value(f1, shape_dims=(None,)),
              'f2': TestPiece.from_value(f2, shape_dims=(None,)),
          },
          shape_dims=(None,),
      )

    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError, 'Fields have different batch dimensions'
    ):
      build(
          tf.convert_to_tensor([[1], [2]]),
          tf.convert_to_tensor([[1], [2], [3]]),
      )

  def testRaisesOnDenseInnerNoneDimension(self):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
        ]
    )
    def build(t):
      return TestPiece.from_value(
          t,
          shape_dims=(None, None),
      )

    with self.assertRaisesRegex(
        ValueError,
        'All shape dimensions except the outermost must be fully defined',
    ):
      build(
          tf.zeros([3, 3, 3]),
      )

  def testRaisesOnRaggedInnerNoneDimension(self):
    @tf.function(
        input_signature=[
            tf.RaggedTensorSpec(
                shape=[None, None, None], dtype=tf.int32, ragged_rank=2
            ),
        ]
    )
    def build(r):
      return TestPiece.from_value(
          r,
          shape_dims=(None, None),
      )

    with self.assertRaisesRegex(
        ValueError,
        'All shape dimensions except the outermost must be fully defined',
    ):
      build(
          tf.ragged.constant([[[1]], [[2]], [[3]]]),
      )


class MergeBatchToComponentsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for graph piece with TF Dataset batching/unbatching."""

  @parameterized.named_parameters([
      dict(
          testcase_name='vector',
          dim_size=2,
          source=np.array([[1, 2], [3, 4]]),
          expected=np.array([1, 2, 3, 4])),
      dict(
          testcase_name='matrix',
          dim_size=2,
          source=np.array([[[1], [2]], [[3], [4]]]),
          expected=np.array([[1], [2], [3], [4]])),
      dict(
          testcase_name='ragged rank-1, scalar value',
          dim_size=None,
          source=tf.ragged.constant([[1, 2], [], [3], [4]]),
          expected=np.array([1, 2, 3, 4])),
      dict(
          testcase_name='ragged rank-1, vector value',
          dim_size=None,
          source=tf.ragged.constant([[[1], [2]], [], [[3]], [[4]]],
                                    ragged_rank=1,
                                    inner_shape=(1,)),
          expected=np.array([[1], [2], [3], [4]])),
      dict(
          testcase_name='ragged rank-2',
          dim_size=None,
          source=tf.ragged.constant([[['a', 'b']], [], [['c']], [['d']], []]),
          expected=tf.ragged.constant([['a', 'b'], ['c'], ['d']])),
  ])
  def testRank1Plain(self, dim_size: int, source, expected):
    piece = TestPiece.from_value(source, shape_dims=(dim_size,))
    result = piece._merge_batch_to_components()
    self.assertAllEqual(result.shape, [])
    self.assertAllEqual(result.value, expected)

  @parameterized.named_parameters([
      dict(
          testcase_name='fixed size scalar',
          batch_dims=(2, 2),
          source=np.array([[[1], [2]], [[3], [4]]]),
          expected=np.array([1, 2, 3, 4])),
      dict(
          testcase_name='variable size scalar',
          batch_dims=(None, 4),
          source=tf.RaggedTensor.from_uniform_row_length(
              tf.ragged.constant([['a', 'b'], [], ['c', 'd'], []]),
              uniform_row_length=4),
          expected=np.array(['a', 'b', 'c', 'd'])),
      dict(
          testcase_name='variable size vector',
          batch_dims=(None, 2),
          source=tf.RaggedTensor.from_uniform_row_length(
              tf.ragged.constant([[['a', '1'], ['b', '2']], [], [['c', '3']],
                                  [['d', '4']], [], []],
                                 ragged_rank=1,
                                 inner_shape=(2,)),
              uniform_row_length=2),
          expected=np.array([['a', 1], ['b', 2], ['c', 3], ['d', 4]])),
  ])
  def testRank2Plain(self, batch_dims: Tuple[int, int], source, expected):
    piece = TestPiece.from_value(source, shape_dims=batch_dims)
    result = piece._merge_batch_to_components()
    self.assertAllEqual(result.shape, [])
    self.assertAllEqual(result.value, expected)

  def testRank1Nested(self):
    piece = TestPiece.from_value(
        {
            'sizes':
                np.array([[3], [0], [2], [0]]),
            'field':
                TestPiece.from_value(
                    tf.ragged.constant([['a', 'b', 'c'], [], ['d', 'e'], []]))
        },
        shape_dims=(None,))
    result = piece._merge_batch_to_components()
    self.assertAllEqual(result.shape, [])
    self.assertAllEqual(result.value['sizes'], [3, 0, 2, 0])
    self.assertAllEqual(result.value['field'].value, ['a', 'b', 'c', 'd', 'e'])


@tf.keras.utils.register_keras_serializable(package='GNNtesting')
class SwapXY(tf.keras.layers.Layer):
  """[x, y] -> [y, x]."""

  def call(self, piece: TestPiece):
    return TestPiece.from_value({'x': piece.value['y'], 'y': piece.value['x']})


class KerasModelSavingLoadingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (tftu.ModelReloading.SKIP,),
      (tftu.ModelReloading.SAVED_MODEL,),
      (tftu.ModelReloading.KERAS,))
  def testTrivialModelSaving(self, model_reloading):

    def add1(p: TestPiece) -> TestPiece:
      return TestPiece.from_value(p.value + 1)

    value = TestPiece.from_value(np.array([0, 1, 2]))
    inputs = tf.keras.layers.Input(type_spec=value.spec)
    outputs = tf.keras.layers.Lambda(add1)(inputs)
    model = tf.keras.Model(inputs, outputs)
    restored_model = tftu.maybe_reload_model(self, model, model_reloading,
                                             'trivial-model')

    def readout_x(piece):
      return piece.value

    expected = np.array([1, 2, 3])
    self.assertAllClose(readout_x(model(value)), expected)
    self.assertAllClose(readout_x(restored_model(value)), expected)

  @parameterized.product(
      [dict(x=np.array([0]), y=np.array([1])),
       dict(x=np.array([0, 1]), y=np.array([1, 2])),
       dict(x=np.array([[0], [1]]), y=np.array([[1], [2], [3]])),
       dict(x=tf.ragged.constant([[1, 2], [3]]),
            y=tf.ragged.constant([[1], [], []]))],
      model_reloading=[
          tftu.ModelReloading.SAVED_MODEL,
          tftu.ModelReloading.KERAS],
  )
  def testFlatModelSaving(self, x, y, model_reloading):

    value = TestPiece.from_value({'x': x, 'y': y})
    inputs = tf.keras.layers.Input(type_spec=value.spec)
    outputs = SwapXY()(inputs)
    model = tf.keras.Model(inputs, outputs)
    restored_model = tftu.maybe_reload_model(self, model, model_reloading,
                                             'flat-model')

    def readout_x(piece):
      return piece.value['x']

    self.assertAllClose(readout_x(model(value)), y)
    self.assertAllClose(readout_x(restored_model(value)), y)

  @parameterized.product(
      [dict(x=np.array([0]), y=np.array([1])),
       dict(x=np.array([0, 1]), y=np.array([1, 2])),
       dict(x=np.array([[0], [1]]), y=np.array([[1], [2], [3]])),
       dict(x=tf.ragged.constant([[1, 2], [3]]),
            y=tf.ragged.constant([[1], [], []]))],
      model_reloading=[
          tftu.ModelReloading.SAVED_MODEL,
          tftu.ModelReloading.KERAS],
  )
  def testNestedModelSaving(self, x, y, model_reloading):

    value = TestPiece.from_value({
        'x': TestPiece.from_value(TestPiece.from_value(x)),
        'y': TestPiece.from_value(TestPiece.from_value(y))
    })
    inputs = tf.keras.layers.Input(type_spec=value.spec)
    outputs = SwapXY()(inputs)
    model = tf.keras.Model(inputs, outputs)
    restored_model = tftu.maybe_reload_model(self, model, model_reloading,
                                             'nested-model')

    def readout_x(piece):
      return piece.value['x'].value.value

    self.assertAllClose(readout_x(model(value)), y)
    self.assertAllClose(readout_x(restored_model(value)), y)

  @parameterized.product(
      indices_dtype=[tf.int32, tf.int64],
      row_splits_dtype=[tf.int32, tf.int64],
      model_reloading=[
          tftu.ModelReloading.SAVED_MODEL,
          tftu.ModelReloading.KERAS])
  def testAttributesSaving(
      self, indices_dtype, row_splits_dtype, model_reloading
  ):
    value = TestPiece.from_value(
        TestPiece.from_value(
            TestPiece.from_value(
                tf.ragged.constant(
                    [[1, 2], [3]],
                    dtype=indices_dtype,
                    row_splits_dtype=row_splits_dtype,
                ),
                indices_dtype=indices_dtype,
                row_splits_dtype=row_splits_dtype,
            ),
            indices_dtype=indices_dtype,
            row_splits_dtype=row_splits_dtype,
        ),
        indices_dtype=indices_dtype,
        row_splits_dtype=row_splits_dtype,
    )
    self.assertEqual(value.spec.indices_dtype, indices_dtype)
    self.assertEqual(value.spec.row_splits_dtype, row_splits_dtype)
    self.assertEqual(value.value.spec.indices_dtype, indices_dtype)
    self.assertEqual(value.value.spec.row_splits_dtype, row_splits_dtype)

    inputs = tf.keras.layers.Input(type_spec=value.spec)
    model = tf.keras.Model(inputs, inputs)
    restored_model = tftu.maybe_reload_model(self, model, model_reloading,
                                             'attribute-saving')

    result = restored_model(value)
    self.assertEqual(result.spec.indices_dtype, indices_dtype)
    self.assertEqual(result.spec.row_splits_dtype, row_splits_dtype)
    self.assertEqual(result.value.spec.indices_dtype, indices_dtype)
    self.assertEqual(result.value.spec.row_splits_dtype, row_splits_dtype)


class AttributesSettersTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([tf.int32, tf.int64])
  def testWithRowSplitsDType(self, row_splits_dtype: tf.DType):
    value = TestPiece.from_value(
        TestPiece.from_value(
            tf.ragged.constant(
                [[1, 2], [3]],
                row_splits_dtype=row_splits_dtype,
            ),
            row_splits_dtype=row_splits_dtype,
        ),
        row_splits_dtype=row_splits_dtype,
    )
    self.assertEqual(value.row_splits_dtype, row_splits_dtype)
    self.assertEqual(value.value.row_splits_dtype, row_splits_dtype)
    self.assertEqual(value.value.value.row_splits.dtype, row_splits_dtype)
    self.assertEqual(value.spec.row_splits_dtype, row_splits_dtype)
    self.assertEqual(value.value.spec.row_splits_dtype, row_splits_dtype)

    new_row_splits_dtype = (
        tf.int64 if row_splits_dtype == tf.int32 else tf.int32
    )

    value = value.with_row_splits_dtype(new_row_splits_dtype)
    self.assertEqual(value.row_splits_dtype, new_row_splits_dtype)
    self.assertEqual(value.value.row_splits_dtype, new_row_splits_dtype)
    self.assertEqual(value.value.value.row_splits.dtype, new_row_splits_dtype)
    self.assertEqual(value.spec.row_splits_dtype, new_row_splits_dtype)
    self.assertEqual(value.value.spec.row_splits_dtype, new_row_splits_dtype)

    spec = value.spec.with_row_splits_dtype(new_row_splits_dtype)
    self.assertEqual(spec.row_splits_dtype, new_row_splits_dtype)
    self.assertEqual(spec.value_spec.row_splits_dtype, new_row_splits_dtype)
    self.assertEqual(
        spec.value_spec.value_spec.row_splits_dtype, new_row_splits_dtype
    )

  @parameterized.parameters([tf.int32, tf.int64])
  def testWithIndicesDType(self, indices_dtype: tf.DType):
    value = TestPiece.from_value(
        TestPiece.from_value(
            tf.constant(
                [[1], [3]],
            ),
            indices_dtype=indices_dtype,
        ),
        indices_dtype=indices_dtype,
    )
    self.assertEqual(value.indices_dtype, indices_dtype)
    self.assertEqual(value.value.indices_dtype, indices_dtype)
    self.assertEqual(value.spec.indices_dtype, indices_dtype)
    self.assertEqual(value.value.spec.indices_dtype, indices_dtype)

    new_indices_dtype = tf.int64 if indices_dtype == tf.int32 else tf.int32
    value = value.with_indices_dtype(new_indices_dtype)
    self.assertEqual(value.indices_dtype, new_indices_dtype)
    self.assertEqual(value.value.indices_dtype, new_indices_dtype)
    self.assertEqual(value.spec.indices_dtype, new_indices_dtype)
    self.assertEqual(value.value.spec.indices_dtype, new_indices_dtype)

    spec = value.spec.with_indices_dtype(new_indices_dtype)
    self.assertEqual(spec.indices_dtype, new_indices_dtype)
    self.assertEqual(spec.value_spec.indices_dtype, new_indices_dtype)

  def testRaisesOnUnsupportedRowSplitsType(self):
    with self.assertRaisesRegex(ValueError, 'must be int32 or int64'):
      _ = TestPiece.from_value(
          tf.constant(
              [],
          ),
          row_splits_dtype=tf.int16,
      )
    with self.assertRaisesRegex(ValueError, 'must be int32 or int64'):
      _ = TestPiece.from_value(
          tf.constant(
              [],
          ),
      ).with_row_splits_dtype(tf.float32)

  def testRaisesOnIncompatibleRowSplits(self):
    with self.assertRaisesRegex(ValueError, 'row splits dtype'):
      _ = TestPiece.from_value(
          tf.ragged.constant(
              [[1, 2], [3]],
              row_splits_dtype=tf.int64,
          ),
          row_splits_dtype=tf.int32,
      )

  def testRaisesOnUnsupportedIndicesType(self):
    with self.assertRaisesRegex(ValueError, 'must be int32 or int64'):
      _ = TestPiece.from_value(
          tf.constant(
              [[1], [3]],
          ),
          indices_dtype=tf.int16,
      )
    with self.assertRaisesRegex(ValueError, 'must be int32 or int64'):
      _ = TestPiece.from_value(
          tf.constant(
              [[1], [3]],
          ),
      ).with_indices_dtype(tf.float32)

  def testRaisesOnIncompatibleIndices(self):
    with self.assertRaisesRegex(ValueError, 'indices dtype'):
      _ = TestPiece.from_value(
          TestPiece.from_value(
              tf.constant(
                  [[1], [3]],
              ),
              indices_dtype=tf.int32,
          ),
          indices_dtype=tf.int64,
      )


class CreateEmptyValueTest(tf.test.TestCase, parameterized.TestCase):
  """Tests dummy values creation for DistributedStrategies."""

  @parameterized.parameters([tf.int32, tf.int64])
  def testPlainDynamic(self, row_splits_dtype: tf.DType):
    def generate(num_nodes):
      shape = tf.stack([num_nodes], 0)
      ragged = tf.RaggedTensor.from_row_lengths(
          tf.ones((1 + 3) * shape, dtype=tf.float32),
          tf.concat(
              [
                  tf.ones(shape, row_splits_dtype),
                  3 * tf.ones(shape, row_splits_dtype),
              ],
              0,
          ),
      )
      ragged = tf.RaggedTensor.from_uniform_row_length(ragged, 4)
      matrix = tf.stack(
          [tf.range(num_nodes), num_nodes - tf.range(num_nodes)], 1
      )
      return TestPiece.from_value(
          value={
              'v': tf.range(num_nodes),
              'm': matrix,
              'r': ragged,
          },
          row_splits_dtype=row_splits_dtype,
      )

    ds = tf.data.Dataset.range(0, 3).map(generate)
    ds = ds.batch(2)
    spec = ds.element_spec
    result = spec._create_empty_value()

    self.assertTrue(spec.is_compatible_with(result))

    self.assertAllEqual(result.value['v'],
                        tf.ragged.constant([], dtype=tf.int32, ragged_rank=1))
    self.assertAllEqual(
        result.value['m'],
        tf.ragged.constant([], dtype=tf.int32, ragged_rank=1, inner_shape=(2,)))
    self.assertAllEqual(result.value['r'],
                        tf.ragged.constant([], dtype=tf.float32, ragged_rank=3))

  def testPlainStatic(self):

    def generate(num_nodes):
      range3 = tf.range(num_nodes, num_nodes + 3)
      range3 = tf.ensure_shape(range3, [3])
      return TestPiece.from_value(
          value={
              'm':
                  tf.stack([range3, num_nodes - range3], 1),
              'r':
                  tf.RaggedTensor.from_row_lengths(
                      tf.ones(tf.stack([num_nodes], 0), dtype=tf.float32),
                      tf.stack([0, num_nodes, 0], 0)),
          })

    ds = tf.data.Dataset.range(0, 3).map(generate)
    ds = ds.batch(2)
    spec = ds.element_spec
    result = spec._create_empty_value()

    self.assertTrue(spec.is_compatible_with(result))

    self.assertAllEqual(result.value['m'], tf.zeros([0, 3, 2], tf.int32))
    self.assertAllEqual(result.value['r'],
                        tf.ragged.constant([], dtype=tf.float32, ragged_rank=2))

  @parameterized.product(
      indices_dtype=[tf.int32, tf.int64], row_splits_dtype=[tf.int32, tf.int64]
  )
  def testNestedDynamic(
      self, indices_dtype: tf.DType, row_splits_dtype: tf.DType
  ):
    def generate(num_nodes):
      return TestPiece.from_value(
          TestPiece.from_value(
              tf.range(num_nodes),
              indices_dtype=indices_dtype,
              row_splits_dtype=row_splits_dtype,
          ),
          indices_dtype=indices_dtype,
          row_splits_dtype=row_splits_dtype,
      )

    ds = tf.data.Dataset.range(0, 3).map(generate)
    ds = ds.batch(2)
    spec = ds.element_spec
    result = spec._create_empty_value()

    self.assertTrue(spec.is_compatible_with(result))

    self.assertAllEqual(result.value.value,
                        tf.ragged.constant([], dtype=tf.int32, ragged_rank=1))


class BackwardInconsistentIndexSupportTest(
    tf.test.TestCase, parameterized.TestCase
):
  """Tests that Keras models saved before b/285269757 could be loaded back.

  Checks that saved Keras models having GraphTenors with inconsistent indices
  could be loaded back and manipulated.
  """

  def _save_and_load(self, piece: TestPiece) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=())
    outputs = tf.keras.layers.Lambda(lambda _: piece)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    restored_model = tftu.maybe_reload_model(
        self, model, tftu.ModelReloading.KERAS, 'test-piece-model')

    return restored_model(tf.convert_to_tensor([0.0]))

  def _flatten_unflatten(self, piece: TestPiece) -> TestPiece:
    spec = piece.spec
    return tf.nest.pack_sequence_as(
        spec,
        tf.nest.flatten(piece, expand_composites=True),
        expand_composites=True,
    )

  def testIncosistentRowSplitsLoadable(self):
    test_piece = TestPiece.from_value(
        {'f': tf.ragged.constant([[1, 2], [3], []], row_splits_dtype=tf.int64)},
        row_splits_dtype=tf.int32,
        check_consistent_row_splits_dtype=False,
    )

    restored_piece1 = self._save_and_load(test_piece)
    self.assertAllEqual(restored_piece1.row_splits_dtype, tf.int32)
    self.assertAllEqual(restored_piece1.value['f'].row_splits.dtype, tf.int64)

    restored_piece2 = self._save_and_load(restored_piece1)
    self.assertAllEqual(restored_piece2.row_splits_dtype, tf.int32)
    self.assertAllEqual(restored_piece2.value['f'].row_splits.dtype, tf.int64)

    restored_piece3 = self._flatten_unflatten(restored_piece2)
    self.assertAllEqual(restored_piece3.spec, restored_piece2.spec)

  def testIncosistentRowSplitsFixable(self):
    test_piece = TestPiece.from_value(
        {'f': tf.ragged.constant([[1, 2], [3], []], row_splits_dtype=tf.int64)},
        row_splits_dtype=tf.int32,
        check_consistent_row_splits_dtype=False,
    )

    fixed = test_piece.with_indices_dtype(tf.int32).with_row_splits_dtype(
        tf.int32
    )
    self.assertAllEqual(fixed.row_splits_dtype, tf.int32)
    self.assertAllEqual(fixed.value['f'].row_splits.dtype, tf.int32)

  def testIncosistentIndicesLoadable(self):
    test_piece = TestPiece.from_value(
        {'f': TestPiece.from_value({}, indices_dtype=tf.int32)},
        indices_dtype=tf.int64,
        check_consistent_indices_dtype=False,
    )

    restored_piece1 = self._save_and_load(test_piece)
    self.assertAllEqual(restored_piece1.indices_dtype, tf.int64)
    self.assertAllEqual(restored_piece1.value['f'].indices_dtype, tf.int32)

    restored_piece2 = self._save_and_load(restored_piece1)
    self.assertAllEqual(restored_piece2.indices_dtype, tf.int64)
    self.assertAllEqual(restored_piece2.value['f'].indices_dtype, tf.int32)

    restored_piece3 = self._flatten_unflatten(restored_piece2)
    self.assertAllEqual(restored_piece3.spec, restored_piece2.spec)

  def testIncosistentIndicesFixable(self):
    test_piece = TestPiece.from_value(
        {'f': TestPiece.from_value({}, indices_dtype=tf.int32)},
        indices_dtype=tf.int64,
        check_consistent_indices_dtype=False,
    )

    fixed = test_piece.with_indices_dtype(tf.int32).with_row_splits_dtype(
        tf.int32
    )
    self.assertAllEqual(fixed.indices_dtype, tf.int32)
    self.assertAllEqual(fixed.value['f'].indices_dtype, tf.int32)


if __name__ == '__main__':
  tf.test.main()
