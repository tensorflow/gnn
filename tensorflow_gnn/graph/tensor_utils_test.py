"""Tests for tensor_utils."""

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


if __name__ == '__main__':
  tf.test.main()
