#!/usr/bin/env python3
"""Graph tensor example.

This is a simple example using the GraphTensor type to parse, represent and
perform simple operations on. It is intended as a most basic usage of
GraphTensor to read some data from files.

TODO(blais): blais - turn this into something either simpler or more useful.
This was the code at the top of the original `graph_tensor.py` implementation.
"""

import functools

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_gnn as gnn


FLAGS = flags.FLAGS

flags.DEFINE_string("citeseer_root", "/tmp/citeseer",
                    "Path to the encoded Citeseer dataset example directory.")


@tf.function
def flatten_fn(gt: gnn.GraphTensor) -> gnn.GraphTensor:
  return gt.flatten()


@tf.function
def transform_fn(gt: gnn.GraphTensor) -> gnn.GraphTensor:
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

  # See Citeseer example.
  ds = tf.data.TFRecordDataset(data_paths)
  if training:
    ds = ds.repeat()
    ds = ds.shuffle(1000)

  ds = ds.batch(batch_size, True)

  # Parse multiple TF example
  schema = gnn.read_schema(schema_path)
  gtspec = gnn.create_graph_spec_from_schema_pb(schema)

  ds = ds.map(functools.partial(gnn.parse_example, gtspec))

  # Transform features
  ds = ds.map(transform_fn)

  # Convert `batch_size` graphs to single graph with `batch_size` sub-graphs
  ds = ds.map(flatten_fn)

  # Batch by the number of worker replicas in sync (e.g. for Mirrored strategy).
  ds = ds.batch(num_replicas_in_sync, True)

  # Check results:
  for gt in ds:
    flat_gt = flatten_fn(gt)
    features = gnn.edge_gather_source_value(flat_gt,
                                            edge_set="citation",
                                            source_field_name="words")
    print(features)


if __name__ == "__main__":
  app.run(main)
