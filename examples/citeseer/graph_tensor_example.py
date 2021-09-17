#!/usr/bin/env python3
"""Graph tensor example.

This is a simple example using the GraphTensor type to parse, represent and
perform simple operations on. It is intended as a most basic usage of
GraphTensor to read some data from files.

TODO(blais): blais - turn this into something either simpler or more useful.
This was the code at the top of the original `graph_tensor.py` implementation.
"""

import functools
import pprint

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_gnn as tfgnn


FLAGS = flags.FLAGS

flags.DEFINE_string("citeseer_root", "/tmp/citeseer",
                    "Path to the encoded Citeseer dataset example directory.")


@tf.function
def transform_fn(gt: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
  values = gt.values
  # Remove string fields.
  del values.context["seed_id"]
  del values.nodes["paper"]["category"]
  return gt.with_values(values)


def main(_):
  # Parameters of the examples.
  root = FLAGS.citeseer_root
  data_paths = tf.data.Dataset.list_files(
      f"{root}/citeseer-2hops-directed.tfrecord")
  schema_path = f"{root}/citeseer.pbtxt"
  training = True
  batch_size = 5
  num_replicas_in_sync = 10

  # Read Citeseer example files and batch them.
  ds = tf.data.TFRecordDataset(data_paths)
  if training:
    ds = ds.repeat()
    ds = ds.shuffle(1000)
  ds = ds.batch(batch_size, True)

  # Parse batches of TF examples.
  schema = tfgnn.read_schema(schema_path)
  spec = tfgnn.create_graph_spec_from_schema_pb(schema)
  ds = ds.map(functools.partial(tfgnn.parse_example, spec))

  # Transform features
  ds = ds.map(transform_fn)

  # Convert `batch_size` graphs to single graph with `batch_size` sub-graphs
  ds = ds.map(lambda graph: graph.merge_batch_to_components())

  # Batch by the number of worker replicas in sync (e.g. for Mirrored strategy).
  ds = ds.batch(num_replicas_in_sync, True)

  # Check results:
  for gt in ds:
    pprint.pprint(gt.edge_sets["citation"]["words"])


if __name__ == "__main__":
  app.run(main)
