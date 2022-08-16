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
"""Print a stream of graph tensor example protos, for inspection and debugging.

You can use this to check out your data. This tool can convert the data in a
`GraphTensor` instance to a pretty-printed Python format (`python`), JSON format
(`json`), or the text-formatted protocol buffer message ('textproto').
"""

import functools
import json
import pprint

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.data import unigraph


FLAGS = flags.FLAGS


def define_flags():
  """Define the program flags."""

  flags.DEFINE_string('graph_schema', None,
                      'Filename containing text-formatted schema.')

  flags.DEFINE_string('examples', None,
                      'Filename of TFRecord file to read.')

  flags.DEFINE_enum('file_format', 'tfrecord',
                    ['tfrecord', 'recordio', 'sstable'],
                    'The format of the input data.')

  flags.DEFINE_enum('mode', 'python', ['python', 'json', 'textproto'],
                    'Select how to convert the tensor for printing.')

  flags.DEFINE_integer('num_examples', 0,
                       'Maximum number of examples (or batches) to print out '
                       '(default is to print all)')

  flags.DEFINE_integer('batch_size', 0,
                       'The size of each mini-batch (the default prints the '
                       'tensors unbatched')


def get_dataset(pattern: str, file_format: str) -> tf.data.Dataset:
  """Create a dataset from the given filenames."""
  fn_dataset = tf.data.Dataset.list_files(pattern)
  if file_format == 'tfrecord':
    dataset = fn_dataset.interleave(tf.data.TFRecordDataset)
  # Placeholder for Google-internal file format datasets
  return dataset


def app_main(_):
  """Read some graph tensor training subgraph examples and print them."""

  schema = tfgnn.read_schema(FLAGS.graph_schema)
  spec = tfgnn.create_graph_spec_from_schema_pb(schema)

  # Read the input Example protos.
  file_format = FLAGS.file_format or unigraph.guess_file_format(FLAGS.examples)
  dataset = get_dataset(FLAGS.examples, file_format)

  # Optionally batch the examples.
  if FLAGS.batch_size and FLAGS.mode != 'textproto':
    dataset = dataset.batch(FLAGS.batch_size)
    parser = functools.partial(tfgnn.parse_example, spec)
  else:
    parser = functools.partial(tfgnn.parse_single_example, spec)

  # Optionally cap the number of examples.
  if FLAGS.num_examples:
    dataset = dataset.take(FLAGS.num_examples)

  # Pretty-format the values for each of the examples and print them.
  if FLAGS.mode in {'python', 'json'}:
    dataset = dataset.map(parser)
    for graph in dataset:
      graph_data = tfgnn.graph_tensor_to_values(graph)
      if FLAGS.mode in 'json':
        print(json.dumps(graph_data, sort_keys=True, indent=2))
      else:
        pprint.pprint(graph_data)
  elif FLAGS.mode == 'textproto':
    for example_str in dataset:
      example = tf.train.Example()
      example.ParseFromString(example_str.numpy())
      print(example)


def main():
  define_flags()
  flags.mark_flag_as_required('graph_schema')
  flags.mark_flag_as_required('examples')
  app.run(app_main)


if __name__ == '__main__':
  main()
