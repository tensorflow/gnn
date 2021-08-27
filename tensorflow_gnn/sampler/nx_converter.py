"""Generate random graphs from NetworkX to various data formats.

Note: This does not produce Unigraph format. This tool has been built mainly to
generate random inputs for the nx_sampler tool.
"""

import collections
import csv
import os
from os import path
from typing import Any, Dict, Optional, Set

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.sampler import nx_io

from google.protobuf import text_format

FLAGS = flags.FLAGS


def define_flags():
  """Create command-line flags."""

  flags.DEFINE_string(
      "input_graph", None,
      ("Single input file in one of the various formats supported by "
       "NetworkX (non-sharded). See --output_format for formats."))
  flags.DEFINE_enum_class(
      "file_format", None, nx_io.NxFileFormat,
      ("The input file in one of the various formats supported by NetworkX. "
       "See --file_format for formats."))

  flags.DEFINE_string(
      "output_dir", None,
      "Output directory where this program will write the Unigraph file.")

  flags.mark_flags_as_required(["input_graph", "file_format", "output_dir"])


def collect_attribute_stats(nodes_or_edges) -> Dict[str, Set[tf.dtypes.DType]]:
  """Iterate over attributes dicts and infer names and types."""
  attr_stats = collections.defaultdict(set)
  for key in nodes_or_edges:
    attr = nodes_or_edges[key]
    for name, value in attr.items():
      dtype = get_dtype(value)
      if dtype is not None:
        attr_stats[name].add(dtype)
  return attr_stats


def get_dtype(value: Any) -> Optional[tf.dtypes.DType]:
  """Convert Python type to TF type."""
  ptype = type(value)
  if ptype in {str, bytes}:
    return tf.string
  elif ptype is int:
    return tf.int64
  elif ptype is float:
    return tf.float32
  elif ptype in {list, tuple}:
    raise NotImplementedError("Features which are lists aren't supported in "
                              "this converter yet.")
  else:
    return None


def write_ids_and_attributes(view, stats, filename: str, is_edges: bool):
  """Write node or edge ids and attributes to a CSV file."""
  with tf.io.gfile.GFile(filename, "w") as csvfile:
    writer = csv.writer(csvfile)
    header = ["source", "target"] if is_edges else ["id"]
    header.extend(sorted(stats.keys()))
    writer.writerow(header)
    for key in view:
      attr = view[key]
      row = list(key) if is_edges else [key]
      row.extend(attr.get(key, "") for key in sorted(stats.keys()))
      writer.writerow(row)


def app_main(unused_argv):
  """Main program."""
  try:
    graph = nx_io.read_graph(FLAGS.input_graph, FLAGS.file_format)
  except ValueError as exc:
    raise SystemExit("Writing failed: {}".format(exc))

  output_dir = FLAGS.output_dir
  os.makedirs(output_dir, exist_ok=True)

  # All graphs that are in NetworkX format are by default homogeneous graphs. If
  # you'd like a heterogeneous graph, see the generate_training_data tool that
  # can be leveraged to produce random data for one of these and then can be
  # parsed by the sampler.
  schema = tfgnn.GraphSchema()
  nodes = schema.node_sets["nodes"]
  nodes.metadata.filename = "nodes.csv"
  edges = schema.edge_sets["edges"]
  edges.metadata.filename = "edges.csv"

  # Figure out the set of features from our nodes.
  node_stats = collect_attribute_stats(graph.nodes)
  for key, dtypes in sorted(node_stats.items()):
    if len(dtypes) != 1:
      logging.warn("Multiple dtypes for key %s; skipping.", key)
      continue
    feature = nodes.features[key]
    # pytype: disable=attribute-error
    feature.dtype = next(iter(dtypes)).as_datatype_enum
    # pytype: enable=attribute-error

  # Figure out the set of features from our edges.
  edge_stats = collect_attribute_stats(graph.edges)
  for key, dtypes in sorted(edge_stats.items()):
    if len(dtypes) != 1:
      logging.warn("Multiple dtypes for key %s; skipping.", key)
      continue
    feature = edges.features[key]
    # pytype: disable=attribute-error
    feature.dtype = next(iter(dtypes)).as_datatype_enum
    # pytype: enable=attribute-error

  # Write out a schema file.
  with tf.io.gfile.GFile(path.join(output_dir, "schema.pbtxt"), "w") as outfile:
    outfile.write(text_format.MessageToString(schema))

  # Write out nodes and edges files as CSV.
  write_ids_and_attributes(graph.nodes, node_stats,
                           path.join(output_dir, nodes.metadata.filename),
                           False)
  write_ids_and_attributes(graph.edges, edge_stats,
                           path.join(output_dir, edges.metadata.filename),
                           True)


def main():
  define_flags()
  app.run(app_main)


if __name__ == "__main__":
  main()
