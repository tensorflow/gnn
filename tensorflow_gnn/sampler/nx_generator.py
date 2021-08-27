"""Generate random graphs from NetworkX to various data formats.

Note: This does not produce Unigraph format. This tool has been built mainly to
generate random inputs for the nx_sampler tool.
"""

import random

from absl import app
from absl import flags
import networkx as nx

from tensorflow_gnn.sampler import nx_io


FLAGS = flags.FLAGS


def define_flags():
  """Create command-line flags."""

  # Note(blais): You could accept a graph schema and generate random features,
  # if any of the formats support them. This is probably uncommon for those
  # formats, except perhaps for edge weights.

  flags.DEFINE_string(
      "output_graph", None,
      ("Single output file in one of the various formats supported by "
       "NetworkX (non-sharded). See --output_format for formats."))

  flags.DEFINE_enum_class(
      "file_format", None, nx_io.NxFileFormat,
      ("The output file in one of the various formats supported by NetworkX. "
       "See --file_format for formats."))

  flags.DEFINE_integer(
      "num_nodes", 2**12,
      "Number of nodes to generate in the random graph.")
  flags.DEFINE_float(
      "edge_probability", 2**-9,
      "Probability of creating an edge to any other node in the random graph.")

  flags.mark_flags_as_required(["output_graph", "file_format"])


def generate_random_graph(directed: bool = True) -> nx.Graph:
  """Generate a random graph."""
  graph = nx.fast_gnp_random_graph(FLAGS.num_nodes, FLAGS.edge_probability,
                                   directed=directed)
  for edge in graph.edges:
    graph.edges[edge]["weight"] = random.random()
  return graph


def app_main(unused_argv):
  """Main program."""
  graph = generate_random_graph()
  try:
    nx_io.write_graph(graph, FLAGS.output_graph, FLAGS.file_format)
  except ValueError as exc:
    raise SystemExit("Writing failed: {}".format(exc))


def main():
  define_flags()
  app.run(app_main)


if __name__ == "__main__":
  main()
