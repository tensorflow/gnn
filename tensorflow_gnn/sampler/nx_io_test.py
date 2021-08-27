"""Tests for open source graph sampler."""

from os import path

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import networkx as nx

from tensorflow_gnn.sampler import nx_io


FLAGS = flags.FLAGS


class TestNxIO(parameterized.TestCase):

  @parameterized.parameters(
      (nx_io.NxFileFormat.ADJACENCY_LIST, True),
      (nx_io.NxFileFormat.MULTILINE_ADJACENCY_LIST, True),
      (nx_io.NxFileFormat.EDGE_LIST, True),
      (nx_io.NxFileFormat.GEXF, True),
      (nx_io.NxFileFormat.GML, True),
      (nx_io.NxFileFormat.PICKLE, True),
      (nx_io.NxFileFormat.GRAPHML, True),
      (nx_io.NxFileFormat.GRAPH6, False),
      (nx_io.NxFileFormat.SPARSE6, False),
      (nx_io.NxFileFormat.PAJEK, True),
  )
  def test_round_trip_graph(self, file_format, directed):
    graph = nx.fast_gnp_random_graph(2**12, 2**-9, directed=directed)
    filename = path.join(FLAGS.test_tmpdir,
                         "graph.{}".format(file_format.name))
    nx_io.write_graph(graph, filename, file_format)
    rtt_graph = nx_io.read_graph(filename, file_format)
    self.assertEqual(len(graph.nodes), len(rtt_graph.nodes))

  @parameterized.parameters(
      (nx_io.NxFileFormat.LEDA,),
  )
  def test_write_graph_fail(self, file_format):
    graph = nx.fast_gnp_random_graph(2**12, 2**-9, directed=True)
    filename = path.join(FLAGS.test_tmpdir,
                         "graph.{}".format(file_format.name))
    with self.assertRaises(ValueError):
      nx_io.write_graph(graph, filename, file_format)


if __name__ == "__main__":
  absltest.main()
