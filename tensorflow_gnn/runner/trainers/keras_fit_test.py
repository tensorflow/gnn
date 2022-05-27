"""Tests for keras_fit."""
import tensorflow as tf

from tensorflow_gnn.runner import orchestration
from tensorflow_gnn.runner.trainers import keras_fit


class KerasFitTest(tf.test.TestCase):

  def test_protocol(self):
    self.assertIsInstance(keras_fit.KerasTrainer, orchestration.Trainer)


if __name__ == "__main__":
  tf.test.main()
