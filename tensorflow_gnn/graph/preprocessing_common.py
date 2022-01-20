"""Common data classes and functions for all preprocessing operations."""

from typing import Optional, Mapping, NamedTuple, Union
import tensorflow as tf
from tensorflow_gnn.graph import graph_constants as const


class SizesConstraints(NamedTuple):
  """Constraints on the number of entities in the graph."""
  total_num_components: Union[int, tf.Tensor]
  total_num_nodes: Mapping[const.NodeSetName, Union[int, tf.Tensor]]
  total_num_edges: Mapping[const.EdgeSetName, Union[int, tf.Tensor]]


class FeatureDefaultValues(NamedTuple):
  """Default values for graph context, node sets and edge sets features."""
  context: Optional[const.Fields] = None
  node_sets: Optional[Mapping[const.NodeSetName, const.Fields]] = None
  edge_sets: Optional[Mapping[const.EdgeSetName, const.Fields]] = None
