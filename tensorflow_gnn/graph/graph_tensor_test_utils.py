"""Set of utility functions for GraphTensor tests."""

from typing import Mapping, Text

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor_encode as ge
from tensorflow_gnn.graph import graph_tensor_random as gr
from tensorflow_gnn.graph import schema_utils as su
from tensorflow_gnn.proto import graph_schema_pb2 as schema_pb2


class GraphTensorTestBase(tf.test.TestCase, parameterized.TestCase):
  """Base class for GraphTensor tests."""

  def assertFieldsEqual(self, actual: const.Fields, expected: const.Fields):
    self.assertIsInstance(actual, Mapping)
    self.assertAllEqual(actual.keys(), expected.keys())
    for key in actual.keys():
      self.assertAllEqual(actual[key], expected[key], msg=f'feature={key}')


def generate_random_data_files(schema: schema_pb2.GraphSchema,
                               filebase: Text,
                               num_shards: int,
                               num_examples: int):
  """Write some random data to a file.

  Args:
    schema: A GraphSchema instance.
    filebase: A string, base filename.
    num_shards: The number of shards to generate.
    num_examples: The number of examples to produce.
  """
  filenames = ['{}-{:05d}-of-{:05d}'.format(filebase, shard, num_shards)
               for shard in range(num_shards)]
  num_base_examples = num_examples // len(filenames)
  remainder = num_examples - num_base_examples * len(filenames)

  spec = su.create_graph_spec_from_schema_pb(schema)
  for findex, filename in enumerate(filenames):
    with tf.io.TFRecordWriter(filename) as file_writer:
      num_shard_examples = num_base_examples + (1 if findex < remainder else 0)
      for _ in range(num_shard_examples):
        graph = gr.random_graph_tensor(spec)
        example = ge.write_example(graph)
        file_writer.write(example.SerializeToString())
