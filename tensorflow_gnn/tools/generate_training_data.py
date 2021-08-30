"""Given a schema, produce some sample tf.Example data.

The purpose of this script is to provide some example encoded data for a
developer to inspect and get a sense of how to produce the correct encoding. The
data contents are fairly bogus, but the shape of the data is correct. You can
inspect them with something like gqui.
"""

from typing import Iterator

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_gnn as tfgnn


FLAGS = flags.FLAGS


def define_flags():
  """Define the program flags."""

  flags.DEFINE_string("graph_schema", None,
                      ("A filename to a text-formatted schema proto of the "
                       "available graph features."))

  flags.DEFINE_string("examples", None,
                      "A filename to write out random graph data to.")

  flags.DEFINE_enum("file_format", "tfrecord",
                    ["tfrecord", "recordio", "sstable"],
                    "The format of the input data.")

  flags.DEFINE_integer("num_examples", 100,
                       "The number of examples to generate.")

  flags.mark_flag_as_required("graph_schema")
  flags.mark_flag_as_required("examples")


def write_stream(generator: Iterator[str],
                 filename: str,
                 file_format: str):
  """Write examples produced by a generator to a file."""

  if file_format == "tfrecord":
    with tf.io.TFRecordWriter(filename) as writer:
      for example in generator:
        writer.write(example)

  else:
    raise ValueError(f"File format '{file_format}' not supported")


def generate_training_data(schema_filename: str,
                           examples_filename: str,
                           file_format: str,
                           num_examples: int):
  """Generate some training data. See flags for details."""
  schema = tfgnn.read_schema(schema_filename)
  spec = tfgnn.create_graph_spec_from_schema_pb(schema)
  def generate_random_examples() -> tf.train.Example:
    for _ in range(num_examples):
      graph = tfgnn.random_graph_tensor(spec)
      example = tfgnn.write_example(graph)
      yield example.SerializeToString()
  write_stream(generate_random_examples(), examples_filename, file_format)


def app_main(_):
  generate_training_data(FLAGS.graph_schema,
                         FLAGS.examples,
                         FLAGS.file_format,
                         FLAGS.num_examples)


def main():
  define_flags()
  app.run(app_main)


if __name__ == "__main__":
  main()
