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
"""Tests for tensor_utils."""

import math
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_gnn.graph import tensor_utils as utils


as_tensor = tf.convert_to_tensor
as_ragged = tf.ragged.constant


class TensorUtilsTest(tf.test.TestCase):

  def testIsDenseTensor(self):
    self.assertTrue(utils.is_dense_tensor(as_tensor(1.)))
    self.assertTrue(utils.is_dense_tensor(as_tensor([1., 2.])))
    self.assertTrue(utils.is_dense_tensor(as_tensor([[1., 2., 3.]])))
    self.assertTrue(
        utils.is_dense_tensor(
            tf.keras.layers.Input(
                type_spec=tf.TensorSpec(shape=[None], dtype=tf.float32))))
    self.assertTrue(
        utils.is_dense_tensor(
            tf.keras.layers.Input(shape=[None], dtype=tf.float32)))

    self.assertFalse(utils.is_dense_tensor(list([1., 2.])))
    self.assertFalse(utils.is_dense_tensor(np.array([[1.]])))
    self.assertFalse(utils.is_dense_tensor(as_ragged([[1.], [1., 2.]])))
    self.assertFalse(
        utils.is_dense_tensor(tf.RaggedTensor.from_row_lengths([1., 2.], [2])))
    self.assertFalse(
        utils.is_dense_tensor(
            tf.SparseTensor(
                indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])))

    self.assertFalse(
        utils.is_dense_tensor(
            tf.keras.layers.Input(
                type_spec=tf.RaggedTensorSpec(
                    shape=[None, None], dtype=tf.float32, ragged_rank=1))))

  def testIsRaggedTensor(self):

    self.assertTrue(utils.is_ragged_tensor(as_ragged([[1.], [1., 2.]])))
    self.assertTrue(
        utils.is_ragged_tensor(tf.RaggedTensor.from_row_lengths([1., 2.], [2])))
    self.assertTrue(
        utils.is_ragged_tensor(
            tf.RaggedTensor.from_nested_row_lengths([1], [[1], [1]])))
    self.assertTrue(
        utils.is_ragged_tensor(
            tf.keras.layers.Input(
                type_spec=tf.RaggedTensorSpec(
                    shape=[None, None], dtype=tf.float32, ragged_rank=1))))
    self.assertTrue(
        utils.is_ragged_tensor(
            tf.keras.layers.Input(
                type_spec=tf.RaggedTensorSpec(
                    shape=[None, None, None], dtype=tf.int32, ragged_rank=2))))

    self.assertFalse(utils.is_ragged_tensor(as_tensor([1., 2.])))
    self.assertFalse(utils.is_ragged_tensor(list([[1.], [2., 3.]])))
    self.assertFalse(utils.is_ragged_tensor(np.array([[1.]])))
    self.assertFalse(
        utils.is_ragged_tensor(
            tf.SparseTensor(
                indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])))

  def testOnesLikeLeadingDimsDense(self):
    self.assertAllEqual(
        utils.ones_like_leading_dims(as_tensor([0]), 1, tf.int32), [1])
    self.assertAllEqual(
        utils.ones_like_leading_dims(as_tensor([1, 2]), 1, tf.int32), [1, 1])
    self.assertAllEqual(
        utils.ones_like_leading_dims(as_tensor([[0]]), 1, tf.float32), [1])
    self.assertAllEqual(
        utils.ones_like_leading_dims(as_tensor([[0]]), 2, tf.bool), [[1]])
    self.assertAllEqual(
        utils.ones_like_leading_dims(as_tensor([[0, 1], [2, 3]]), 1, tf.int32),
        [1, 1])
    self.assertAllEqual(
        utils.ones_like_leading_dims(as_tensor([[0, 1], [2, 3]]), 2, tf.int32),
        [[1, 1], [1, 1]])

  def testOnesLikeLeadingDimsRagged(self):
    self.assertAllEqual(
        utils.ones_like_leading_dims(as_ragged([[0], [0]]), 1, tf.int32),
        as_tensor([1, 1]))
    self.assertAllEqual(
        utils.ones_like_leading_dims(as_ragged([[0], [0]]), 2, tf.int32),
        as_ragged([[1], [1]]))
    self.assertAllEqual(
        utils.ones_like_leading_dims(as_ragged([[0, 1], [2]]), 1, tf.float32),
        as_tensor([1, 1]))
    self.assertAllEqual(
        utils.ones_like_leading_dims(as_ragged([[0, 1], [2]]), 2, tf.bool),
        as_ragged([[1, 1], [1]]))
    self.assertAllEqual(
        utils.ones_like_leading_dims(
            as_ragged([[[0], [1]], [[2]]]), 1, tf.int32), as_tensor([1, 1]))
    self.assertAllEqual(
        utils.ones_like_leading_dims(
            as_ragged([[[0], [1]], [[2]]]), 2, tf.int32),
        as_ragged([[1, 1], [1]]))
    self.assertAllEqual(
        utils.ones_like_leading_dims(
            as_ragged([[[0], [1]], [[2]]]), 3, tf.int32),
        as_ragged([[[1], [1]], [[1]]]))

  def testEnsureStaticNRowsForDense(self):

    @tf.function
    def transform(value):
      return tf.concat([value, value], 0)

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32)])
    def dense_case(value):
      self.assertIsNone(value.shape[0])
      value = utils.ensure_static_nrows(value, 2)
      value = transform(value)
      # Note: returning value prevents TF from stipping graph operations.
      return dict(dim=value.shape[0], value=value)

    self.assertDictEqual(
        dense_case(as_tensor([[], []], tf.float32)),
        dict(dim=4, value=as_tensor([[], [], [], []], tf.float32)))
    self.assertDictEqual(
        dense_case(as_tensor([[1.], [3.]], tf.float32)),
        dict(dim=4, value=as_tensor([[1.], [3.], [1.], [3.]])))
    self.assertDictEqual(
        dense_case(as_tensor([[1., 2.], [3., 4.]], tf.float32)),
        dict(dim=4, value=as_tensor([[1., 2.], [3., 4.], [1., 2.], [3., 4.]])))

    self.assertEqual(dense_case.experimental_get_tracing_count(), 1)

    self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                           'is not compatible with expected shape',
                           lambda: dense_case(as_tensor([[1.]], tf.float32)))

  def testEnsureStaticNRowsForRagged(self):

    @tf.function
    def transform(value):
      return tf.reduce_sum(value, axis=-1)

    @tf.function(input_signature=[
        tf.RaggedTensorSpec([None, None], dtype=tf.int32, ragged_rank=1)
    ])
    def ragged_case(value):
      self.assertIsNone(value.shape[0])
      self.assertIsNone(value.row_splits.shape[0])
      value = utils.ensure_static_nrows(value, 2)
      value = transform(value)
      # Note: returning value prevents TF from stipping graph operations.
      return dict(dim=value.shape[0], value=value)

    self.assertDictEqual(
        ragged_case(as_ragged([[1, 2], [3]], tf.int32)),
        dict(dim=2, value=as_tensor([3, 3], tf.int32)))
    self.assertDictEqual(
        ragged_case(as_ragged([[], []], tf.int32)),
        dict(dim=2, value=as_tensor([0, 0], tf.int32)))

    self.assertEqual(ragged_case.experimental_get_tracing_count(), 1)

    self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                           'is not compatible with expected shape',
                           lambda: ragged_case(as_ragged([[]], tf.int32)))

  def testFill(self):
    self.assertAllEqual(
        utils.fill(tf.TensorSpec([None], tf.float32), 0, 1.),
        as_tensor([], tf.float32))
    self.assertAllEqual(
        utils.fill(tf.TensorSpec([None], tf.string), 3, b'?'),
        as_tensor([b'?', b'?', b'?']))
    self.assertAllEqual(
        utils.fill(tf.TensorSpec([None, 1, 2], tf.int32), 1, 2),
        as_tensor([[[2, 2]]]))

    self.assertAllEqual(
        utils.fill(tf.RaggedTensorSpec([None, None], tf.string), 0, b'?'),
        as_ragged([], tf.string, ragged_rank=1))
    self.assertAllEqual(
        utils.fill(tf.RaggedTensorSpec([None, None], tf.float32), 4, 1.),
        as_ragged([[], [], [], []], tf.float32))
    self.assertAllEqual(
        utils.fill(
            tf.RaggedTensorSpec([None, 3, None], tf.float32, ragged_rank=2), 2,
            1.), as_ragged([[[], [], []], [[], [], []]], tf.float32))
    self.assertAllEqual(
        utils.fill(
            tf.RaggedTensorSpec([None, 3, 2, 1, None],
                                tf.float32,
                                ragged_rank=4), 1, 1.),
        as_ragged([[[[[]], [[]]], [[[]], [[]]], [[[]], [[]]]]], tf.float32))

  def testPadToNRows(self):
    self.assertAllEqual(
        utils.pad_to_nrows(as_tensor([1, 2], tf.int32), 2, 0),
        as_tensor([1, 2], tf.int32))
    self.assertAllEqual(
        utils.pad_to_nrows(as_tensor([b'a', b'b']), 3, '?'),
        as_tensor([b'a', b'b', b'?']))
    self.assertAllEqual(
        utils.pad_to_nrows(as_tensor([[1, 2], [2, 3]], tf.int32), 2, 8),
        as_tensor([[1, 2], [2, 3]], tf.int32))
    self.assertAllEqual(
        utils.pad_to_nrows(as_tensor([[1, 2]], tf.int32), 3, 8),
        as_tensor([[1, 2], [8, 8], [8, 8]], tf.int32))

    self.assertAllEqual(
        utils.pad_to_nrows(as_ragged([[1.]], ragged_rank=1), 3, 0.),
        as_ragged([[1.], [], []], ragged_rank=1))
    self.assertAllEqual(
        utils.pad_to_nrows(
            as_ragged([[[], ['b', 'c']], [], [['d']]], ragged_rank=2), 5, ''),
        as_ragged([[[], ['b', 'c']], [], [['d']], [], []], ragged_rank=2))
    value = tf.RaggedTensor.from_row_lengths(['a', 'b', 'c'],
                                             [0, 0, 1, 0, 2, 0])
    value = tf.RaggedTensor.from_uniform_row_length(value, 3)
    self.assertAllEqual(utils.pad_to_nrows(value, 2, ''), value)
    self.assertAllEqual(
        utils.pad_to_nrows(value, 3, ''),
        tf.RaggedTensor.from_uniform_row_length(
            as_ragged([[], [], ['a'], [], ['b', 'c'], [], [], [], []]), 3))

  def testPadToNRowsEmptyTensors(self):
    self.assertAllEqual(utils.pad_to_nrows(as_tensor([]), 0, 0), as_tensor([]))
    self.assertAllEqual(
        utils.pad_to_nrows(tf.constant([], shape=[0, 2]), 0, 0),
        tf.constant([], shape=[0, 2]))

    for ragged_rank in range(1, 5):
      self.assertAllEqual(
          utils.pad_to_nrows(as_ragged([], ragged_rank=ragged_rank), 0, 0),
          as_ragged([], ragged_rank=ragged_rank),
          msg=str(ragged_rank))

  def testPadToNRowsInvarianceToMultipleCalls(self):
    value = utils.pad_to_nrows(as_tensor([[1, 2]], tf.int32), 5, 8)
    self.assertAllEqual(utils.pad_to_nrows(value, 5, 8), value)

    value = tf.RaggedTensor.from_row_lengths(['a', 'b', 'c'],
                                             [0, 0, 1, 0, 2, 0])
    value = tf.RaggedTensor.from_uniform_row_length(value, 3)
    self.assertAllEqual(utils.pad_to_nrows(value, 2, ''), value)
    value = tf.RaggedTensor.from_uniform_row_length(value, 1)
    self.assertAllEqual(utils.pad_to_nrows(value, 2, ''), value)
    value = tf.RaggedTensor.from_row_lengths(
        value, tf.ones([value.nrows()], value.row_splits.dtype))
    self.assertAllEqual(utils.pad_to_nrows(value, 2, ''), value)


_SEED = 42


class RaggedShuffleTest(tf.test.TestCase, parameterized.TestCase):

  def testEmpty(self):
    result = utils.segment_random_index_shuffle(
        segment_ids=tf.convert_to_tensor([], tf.int32), seed=_SEED)
    self.assertAllEqual(result, tf.convert_to_tensor([], tf.int32))

  def testDeterminism(self):
    tf.random.set_seed(_SEED)
    result1 = utils.segment_random_index_shuffle(
        segment_ids=[0, 0, 1, 1], seed=_SEED)
    for _ in range(30):
      tf.random.set_seed(_SEED)
      result2 = utils.segment_random_index_shuffle(
          segment_ids=[0, 0, 1, 1], seed=_SEED)
      self.assertAllEqual(result1, result2)

  @parameterized.parameters(_SEED, None)
  def testRespectsSegmentIds(self, seed):
    segment_ids = [1, 1, 1, 2, 2, 3, 5, 5, 5, 7]
    for _ in range(30):
      result = utils.segment_random_index_shuffle(
          segment_ids=segment_ids, seed=seed)
      self.assertAllEqual(tf.sort(result[0:3]), [0, 1, 2])
      self.assertAllEqual(tf.sort(result[3:5]), [3, 4])
      self.assertAllEqual(tf.sort(result[5:6]), [5])
      self.assertAllEqual(tf.sort(result[6:9]), [6, 7, 8])
      self.assertAllEqual(tf.sort(result[9:10]), [9])

  @parameterized.parameters(7, 61)
  def testDistribution(self, seed):
    segment_ids = [0, 0, 1, 1, 1, 2, 2, 2, 2]
    distr = [0] * len(segment_ids)
    for _ in range(1_000):
      result = utils.segment_random_index_shuffle(
          segment_ids=tf.convert_to_tensor(segment_ids, tf.int32), seed=seed)
      distr[result[3].numpy()] += 1
    distr = tf.cast(distr, tf.float32) / 1_000.0
    std = math.sqrt(0.333 * 0.667 / 1_000.0)
    self.assertAllEqual(distr[:2], [0., 0.])
    self.assertAllClose(distr[2:5], [0.333] * 3, atol=5. * std, rtol=0.0)
    self.assertAllEqual(distr[5:], [0., 0., 0., 0.])


if __name__ == '__main__':
  tf.test.main()
