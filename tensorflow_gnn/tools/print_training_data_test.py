"""Unit tests for printer."""

from os import path

from absl import flags
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.graph import graph_tensor_test_utils as tu
from tensorflow_gnn.tools import print_training_data
from tensorflow_gnn.utils import test_utils


FLAGS = flags.FLAGS


class PrintDataTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    schema_filename = test_utils.get_resource('examples/schemas/mpnn.pbtxt')
    schema = tfgnn.read_schema(schema_filename)
    examples_filename = path.join(FLAGS.test_tmpdir, 'examples.tfrecords')
    tu.generate_random_data_files(schema, examples_filename,
                                  num_shards=1, num_examples=32)

    # TODO(blais): Support sharded input in the print tool.
    FLAGS.graph_schema = schema_filename
    FLAGS.examples = examples_filename + '-00000-of-00001'
    self.assertTrue(path.exists(FLAGS.examples))
    FLAGS.file_format = 'tfrecord'
    FLAGS.num_examples = 16
    FLAGS.batch_size = 4

  def test_print_python(self):
    FLAGS.mode = 'python'
    print_training_data.app_main([])

  def test_print_json(self):
    FLAGS.mode = 'json'
    print_training_data.app_main([])

  def test_print_textproto(self):
    FLAGS.mode = 'textproto'
    print_training_data.app_main([])


if __name__ == '__main__':
  print_training_data.define_flags()
  tf.test.main()
