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

import os
from typing import Mapping, Tuple, Union

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_gnn.graph import graph_piece as gp
from tensorflow_gnn.graph import tf_internal

TestValue = Union[np.ndarray, tf.RaggedTensor, tf.Tensor, 'TestPiece']


class TestPiece(gp.GraphPieceBase):
  """Graph piece implementation for a simple value as a nest."""

  @classmethod
  def from_value(cls,
                 value: Mapping[str, TestValue],
                 shape_dims: Tuple[int, ...] = (),
                 *,
                 metadata=None,
                 indices_dtype: tf.dtypes.DType = tf.int32) -> 'TestPiece':

    def convert_fn(field):
      assert isinstance(
          field, (np.ndarray, tf.Tensor, tf.RaggedTensor, TestPiece)), field
      return tf.convert_to_tensor(field) if isinstance(field,
                                                       np.ndarray) else field

    return cls._from_data(
        tf.nest.map_structure(convert_fn, value),
        shape=tf.TensorShape(shape_dims),
        indices_dtype=indices_dtype,
        metadata=metadata)

  @property
  def value(self):
    return self._data

  @staticmethod
  def _type_spec_cls():
    return TestPieceSpec


@tf_internal.type_spec_register('tensorflow_gnn.TestPieceSpec')
class TestPieceSpec(gp.GraphPieceSpecBase):
  """Graph piece specification."""

  @property
  def value_type(self):
    return TestPiece


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

  def testMetadataUpdate(self):
    leaf = TestPiece.from_value(
        np.array([1]), metadata={
            'to_update': 0,
            'do_not_update': 'x'
        })
    root = TestPiece.from_value(leaf, metadata={'to_update': 3, 'ignore': 'y'})
    self.assertAllEqual(root.value.spec._metadata, {
        'to_update': 3,
        'do_not_update': 'x'
    })

  def testMetadataUpdateChain(self):
    leaf = TestPiece.from_value(
        np.array([1]), metadata={
            'a': 1,
            'b': 1,
            'c': 1
        })
    node = TestPiece.from_value(leaf, metadata={'a': 2, 'c': 2})
    root = TestPiece.from_value(node, metadata={'a': 3, 'b': 3})
    self.assertAllEqual(root.value.value.spec._metadata, {
        'a': 3,
        'b': 1,
        'c': 2
    })

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

  def testDynamicSpecs(self):

    @tf.function
    def generate(num_nodes):
      return TestPiece.from_value(value=tf.range(num_nodes))

    ds = tf.data.Dataset.range(0, 7)
    ds = ds.map(generate)
    ds = ds.batch(3, True)

    self.assertAllEqual(
        ds.element_spec._data_spec,
        tf.RaggedTensorSpec(
            shape=[3, None],
            dtype=tf.int64,
            ragged_rank=1,
            row_splits_dtype=tf.int32))

    ds = ds.batch(2)
    self.assertAllEqual(
        ds.element_spec._data_spec,
        tf.RaggedTensorSpec(
            shape=[None, 3, None],
            dtype=tf.int64,
            ragged_rank=2,
            row_splits_dtype=tf.int32))

  def testRaggedSpecs(self):

    @tf.function
    def generate(num_nodes):
      return TestPiece.from_value(
          value={
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
        tf.type_spec_from_value(element.value['r']),
        tf.RaggedTensorSpec(
            shape=[2, 3, 3, None],
            dtype=tf.float32,
            ragged_rank=3,
            row_splits_dtype=tf.int64))

    element = next(itr)
    self.assertAllEqual(
        tf.type_spec_from_value(element.value['r']),
        tf.RaggedTensorSpec(
            shape=[1, 3, 3, None],
            dtype=tf.float32,
            ragged_rank=3,
            row_splits_dtype=tf.int64))

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

  def testRaisesOnVarSizeBatching(self):

    @tf.function
    def generate(num_nodes):
      return TestPiece.from_value(value=tf.range(num_nodes),)

    ds = tf.data.Dataset.range(0, 6)
    ds = ds.map(generate)
    ds = ds.batch(3)
    self.assertRaises(NotImplementedError, lambda: ds.batch(2))

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

  def testRagged(self):

    @tf.function
    def generate(num_nodes):
      return TestPiece.from_value(
          value={
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
        element.value['x'],
        tf.ragged.constant([
            [[], [0], [0, 1]],
            [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]],
        ]))
    self.assertAllEqual(
        element.value['r'],
        tf.ragged.constant([
            [[[], [], []], [[], [1], []], [[], [1, 1], []]],
            [[[], [1, 1, 1], []], [[], [1, 1, 1, 1], []],
             [[], [1, 1, 1, 1, 1], []]],
        ]))

    element = next(itr)
    self.assertAllEqual(element.value['x'],
                        tf.ragged.constant([
                            [[0, 1, 2, 3, 4, 5],
                             [0, 1, 2, 3, 4, 5, 6],
                             [0, 1, 2, 3, 4, 5, 6, 7]],
                        ]))
    self.assertAllEqual(
        tf.type_spec_from_value(element.value['x']),
        tf.RaggedTensorSpec(
            shape=[1, 3, None],
            dtype=tf.int64,
            ragged_rank=2,
            row_splits_dtype=tf.int32))


class MergeBatchToComponentsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for graph piece with TF Dataset batching/unbatching."""

  @parameterized.parameters([
      dict(
          description='vector',
          dim_size=2,
          source=np.array([[1, 2], [3, 4]]),
          expected=np.array([1, 2, 3, 4])),
      dict(
          description='matrix',
          dim_size=2,
          source=np.array([[[1], [2]], [[3], [4]]]),
          expected=np.array([[1], [2], [3], [4]])),
      dict(
          description='ragged rank-1, scalar value',
          dim_size=None,
          source=tf.ragged.constant([[1, 2], [], [3], [4]]),
          expected=np.array([1, 2, 3, 4])),
      dict(
          description='ragged rank-1, vector value',
          dim_size=None,
          source=tf.ragged.constant([[[1], [2]], [], [[3]], [[4]]],
                                    ragged_rank=1,
                                    inner_shape=(1,)),
          expected=np.array([[1], [2], [3], [4]])),
      dict(
          description='ragged rank-2',
          dim_size=None,
          source=tf.ragged.constant([[['a', 'b']], [], [['c']], [['d']], []]),
          expected=tf.ragged.constant([['a', 'b'], ['c'], ['d']])),
  ])
  def testRank1Plain(self, description: str, dim_size: int, source, expected):
    piece = TestPiece.from_value(source, shape_dims=(dim_size,))
    result = piece._merge_batch_to_components()
    self.assertAllEqual(result.shape, [])
    self.assertAllEqual(result.value, expected)

  @parameterized.parameters([
      dict(
          description='fixed size scalar',
          batch_dims=(2, 2),
          source=np.array([[[1], [2]], [[3], [4]]]),
          expected=np.array([1, 2, 3, 4])),
      dict(
          description='variable size scalar',
          batch_dims=(None, 4),
          source=tf.RaggedTensor.from_uniform_row_length(
              tf.ragged.constant([['a', 'b'], [], ['c', 'd'], []]),
              uniform_row_length=4),
          expected=np.array(['a', 'b', 'c', 'd'])),
      dict(
          description='variable size vector',
          batch_dims=(None, 2),
          source=tf.RaggedTensor.from_uniform_row_length(
              tf.ragged.constant([[['a', '1'], ['b', '2']], [], [['c', '3']],
                                  [['d', '4']], [], []],
                                 ragged_rank=1,
                                 inner_shape=(2,)),
              uniform_row_length=2),
          expected=np.array([['a', 1], ['b', 2], ['c', 3], ['d', 4]])),
  ])
  def testRank2Plain(self, description: str, batch_dims: Tuple[int, int],
                     source, expected):
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


class SwapXY(tf.keras.layers.Layer):
  """[x, y] -> [y, x]."""

  def call(self, piece: TestPiece):
    return TestPiece.from_value({'x': piece.value['y'], 'y': piece.value['x']})


class KerasModelSavingLoadingTest(tf.test.TestCase, parameterized.TestCase):

  def _save_and_restore_model(self, model: tf.keras.Model) -> tf.keras.Model:
    export_dir = os.path.join(self.get_temp_dir(), 'graph-model')
    tf.saved_model.save(model, export_dir)
    return tf.saved_model.load(export_dir)

  def testTrivialModelSaving(self):

    def add1(p: TestPiece) -> TestPiece:
      return TestPiece.from_value(p.value + 1)

    value = TestPiece.from_value(np.array([0, 1, 2]))
    inputs = tf.keras.layers.Input(type_spec=value.spec)
    outputs = tf.keras.layers.Lambda(add1)(inputs)
    model = tf.keras.Model(inputs, outputs)
    restored_model = self._save_and_restore_model(model)

    def readout_x(piece):
      return piece.value

    expected = np.array([1, 2, 3])
    self.assertAllClose(readout_x(model(value)), expected)
    self.assertAllClose(readout_x(restored_model(value)), expected)

  @parameterized.parameters([
      (np.array([0]), np.array([1])),
      (np.array([0, 1]), np.array([1, 2])),
      (np.array([[0], [1]]), np.array([[1], [2], [3]])),
      (tf.ragged.constant([[1, 2], [3]]), tf.ragged.constant([[1], [], []])),
  ])
  def testModelSaving(self, x, y):

    value = TestPiece.from_value({'x': x, 'y': y})
    inputs = tf.keras.layers.Input(type_spec=value.spec)
    outputs = SwapXY()(inputs)
    model = tf.keras.Model(inputs, outputs)
    restored_model = self._save_and_restore_model(model)

    def readout_x(piece):
      return piece.value['x']

    self.assertAllClose(readout_x(model(value)), y)
    self.assertAllClose(readout_x(restored_model(value)), y)

  @parameterized.parameters([
      (np.array([0]), np.array([1])),
      (np.array([0, 1]), np.array([1, 2])),
      (np.array([[0], [1]]), np.array([[1], [2], [3]])),
      (tf.ragged.constant([[1, 2], [3]]), tf.ragged.constant([[1], [], []])),
  ])
  def testNestedModelSaving(self, x, y):

    value = TestPiece.from_value({
        'x': TestPiece.from_value(TestPiece.from_value(x)),
        'y': TestPiece.from_value(TestPiece.from_value(y))
    })
    inputs = tf.keras.layers.Input(type_spec=value.spec)
    outputs = SwapXY()(inputs)
    model = tf.keras.Model(inputs, outputs)
    restored_model = self._save_and_restore_model(model)

    def readout_x(piece):
      return piece.value['x'].value.value

    self.assertAllClose(readout_x(model(value)), y)
    self.assertAllClose(readout_x(restored_model(value)), y)


class CreateEmptyValueTest(tf.test.TestCase, parameterized.TestCase):
  """Tests dummy values creation for DistributedStrategies."""

  def testPlainDynamic(self):

    def generate(num_nodes):
      shape = tf.stack([num_nodes], 0)
      ragged = tf.RaggedTensor.from_row_lengths(
          tf.ones((1 + 3) * shape, dtype=tf.float32),
          tf.concat([tf.ones(shape, tf.int32), 3 * tf.ones(shape, tf.int32)],
                    0))
      ragged = tf.RaggedTensor.from_uniform_row_length(ragged, 4)
      matrix = tf.stack([tf.range(num_nodes), num_nodes - tf.range(num_nodes)],
                        1)
      return TestPiece.from_value(value={
          'v': tf.range(num_nodes),
          'm': matrix,
          'r': ragged,
      })

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

  def testNestedDynamic(self):

    def generate(num_nodes):
      return TestPiece.from_value(TestPiece.from_value(tf.range(num_nodes)))

    ds = tf.data.Dataset.range(0, 3).map(generate)
    ds = ds.batch(2)
    spec = ds.element_spec
    result = spec._create_empty_value()

    self.assertTrue(spec.is_compatible_with(result))

    self.assertAllEqual(result.value.value,
                        tf.ragged.constant([], dtype=tf.int32, ragged_rank=1))


if __name__ == '__main__':
  tf.test.main()
