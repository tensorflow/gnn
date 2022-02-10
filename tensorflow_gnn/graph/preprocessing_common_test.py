"""Tests for preprocessing_common."""

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.graph import preprocessing_common

as_tensor = tf.convert_to_tensor


class ReduceMeanTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('Tensor', as_tensor([1., 2.])),
      ('Tuple', (as_tensor([1.]), as_tensor([[2.]]))),
      ('Dict', {
          'm': as_tensor([[1], [2]]),
          'v': as_tensor([1]),
          's': as_tensor(2.),
      }),
  )
  def testEmpty(self, value):
    ds = tf.data.Dataset.from_tensors(value)
    ds = ds.take(0)
    result = preprocessing_common.compute_basic_stats(ds)
    tf.nest.map_structure(
        self.assertAllEqual, result.mean,
        tf.nest.map_structure(lambda t: tf.fill(tf.shape(t), 0.0), value))
    tf.nest.map_structure(
        self.assertAllEqual, result.minimum,
        tf.nest.map_structure(lambda t: tf.fill(tf.shape(t), t.dtype.max),
                              value))
    tf.nest.map_structure(
        self.assertAllEqual, result.maximum,
        tf.nest.map_structure(lambda t: tf.fill(tf.shape(t), t.dtype.min),
                              value))

  @parameterized.named_parameters(
      ('Tensor', as_tensor([1., 2.])),
      ('Tuple', as_tensor([1.])),
      ('Dict', {
          'm': as_tensor([[1], [2]]),
          'v': as_tensor([1]),
          's': as_tensor(2.),
      }),
  )
  def testSingleElement(self, value):
    ds = tf.data.Dataset.from_tensors(value)
    result = preprocessing_common.compute_basic_stats(ds)
    tf.nest.map_structure(
        self.assertAllEqual, result.minimum,
        tf.nest.map_structure(lambda t: tf.cast(t, tf.float32), value))
    tf.nest.map_structure(self.assertAllEqual, result.maximum, value)
    tf.nest.map_structure(self.assertAllEqual, result.minimum, value)

  def testMeanComputation(self):

    def generator(x):
      return {
          'x': x,
          '2x': 2 * x,
      }

    ds = tf.data.Dataset.range(100).map(generator)
    result = preprocessing_common.compute_basic_stats(ds)
    self.assertAllClose(result.mean['x'], (99.0 + 0.0) * 0.5)
    self.assertAllClose(result.mean['2x'], 2.0 * (99.0 + 0.0) * 0.5)
    self.assertAllClose(result.minimum['x'], 0)
    self.assertAllClose(result.minimum['2x'], 0)
    self.assertAllClose(result.maximum['x'], 99)
    self.assertAllClose(result.maximum['2x'], 99 * 2)


if __name__ == '__main__':
  tf.test.main()
