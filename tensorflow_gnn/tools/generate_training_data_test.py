"""Unit tests for generate training data test."""

from os import path

from absl import flags
import tensorflow as tf
from tensorflow_gnn.tools import generate_training_data
from tensorflow_gnn.utils import test_utils


FLAGS = flags.FLAGS


class GenerateDataTest(tf.test.TestCase):

  def test_generate_training_data(self):
    schema_filename = test_utils.get_resource("examples/schemas/mpnn.pbtxt")
    output_filename = path.join(FLAGS.test_tmpdir, "examples.tfrecords")
    generate_training_data.generate_training_data(
        schema_filename, output_filename, "tfrecord", 64)
    self.assertTrue(path.exists(output_filename))


if __name__ == "__main__":
  tf.test.main()
