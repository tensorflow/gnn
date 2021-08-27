"""Simple I/O library for various file formats supported by NetworkX.

This just abstracts all the file formats so we can build command-line converter
programs.
"""

import enum

import networkx as nx


class NxFileFormat(enum.Enum):
  """Supported NetworkX file format."""

  ADJACENCY_LIST = "AdjacencyList"
  MULTILINE_ADJACENCY_LIST = "MultilineAdjacencyList"
  EDGE_LIST = "EdgeList"
  GEXF = "GEXF"
  GML = "GML"
  PICKLE = "Pickle"
  GRAPHML = "GraphML"
  LEDA = "LEDA"
  GRAPH6 = "Graph6"
  SPARSE6 = "Sparse6"
  PAJEK = "Pajek"


def read_graph(filename: str, file_format: NxFileFormat) -> nx.Graph:
  """Read a Nx graph in one of the supported formats."""
  if file_format == NxFileFormat.ADJACENCY_LIST:
    graph = nx.read_adjlist(filename)
  elif file_format == NxFileFormat.MULTILINE_ADJACENCY_LIST:
    graph = nx.read_multiline_adjlist(filename)
  elif file_format == NxFileFormat.EDGE_LIST:
    graph = nx.read_weighted_edgelist(filename)
  elif file_format == NxFileFormat.GEXF:
    graph = nx.read_gexf(filename)
  elif file_format == NxFileFormat.GML:
    graph = nx.read_gml(filename)
  elif file_format == NxFileFormat.PICKLE:
    graph = nx.read_gpickle(filename)
  elif file_format == NxFileFormat.GRAPHML:
    graph = nx.read_graphml(filename)
  elif file_format == NxFileFormat.LEDA:
    graph = nx.read_leda(filename)
  elif file_format == NxFileFormat.GRAPH6:
    graph = nx.read_graph6(filename)
  elif file_format == NxFileFormat.SPARSE6:
    graph = nx.read_sparse6(filename)
  elif file_format == NxFileFormat.PAJEK:
    graph = nx.read_pajek(filename)
  else:
    raise NotImplementedError(
        "File format '{}' not supported.".format(file_format))
  return graph


def write_graph(graph: nx.Graph, filename: str, file_format: NxFileFormat):
  """Write a Nx graph in one of the supported formats."""
  if file_format == NxFileFormat.ADJACENCY_LIST:
    nx.write_adjlist(graph, filename)
  elif file_format == NxFileFormat.MULTILINE_ADJACENCY_LIST:
    nx.write_multiline_adjlist(graph, filename)
  elif file_format == NxFileFormat.EDGE_LIST:
    nx.write_weighted_edgelist(graph, filename)
  elif file_format == NxFileFormat.GEXF:
    nx.write_gexf(graph, filename)
  elif file_format == NxFileFormat.GML:
    nx.write_gml(graph, filename)
  elif file_format == NxFileFormat.PICKLE:
    nx.write_gpickle(graph, filename)
  elif file_format == NxFileFormat.GRAPHML:
    nx.write_graphml(graph, filename)
  elif file_format == NxFileFormat.GRAPH6:
    nx.write_graph6(graph, filename)
  elif file_format == NxFileFormat.SPARSE6:
    nx.write_sparse6(graph, filename)
  elif file_format == NxFileFormat.PAJEK:
    nx.write_pajek(graph, filename)
  elif file_format == NxFileFormat.LEDA:
    raise ValueError(
        "File format '{}' not supported for writing.".format(file_format))
  else:
    raise NotImplementedError(
        "File format '{}' not supported.".format(file_format))
