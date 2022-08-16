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
"""Unit tests for sampled stats."""

from os import path

from absl import flags
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.graph import graph_tensor_test_utils as tu
from tensorflow_gnn.tools import sampled_stats
from tensorflow_gnn.utils import test_utils


FLAGS = flags.FLAGS


class SampleStatsTest(tf.test.TestCase):

  def test_basic_sample_stats_run_on_random_data(self):
    schema = test_utils.get_proto_resource('examples/schemas/mpnn.pbtxt',
                                           tfgnn.GraphSchema())

    num_shards = 8
    examples_filename = path.join(FLAGS.test_tmpdir, 'examples.tfrecords')
    examples_pattern = f'{examples_filename}-?????-of-{num_shards:05d}'
    tu.generate_random_data_files(schema,
                                  examples_filename,
                                  num_shards=num_shards,
                                  num_examples=128)

    stats_filename = path.join(FLAGS.test_tmpdir, 'stats.pbtxt')
    sampled_stats.run_pipeline(examples_pattern, 'tfrecord',
                               schema, 10, stats_filename)

    with open(stats_filename) as statsfile:
      contents = statsfile.read()
      self.assertRegex(contents, 'feature_stats')


if __name__ == '__main__':
  tf.test.main()
