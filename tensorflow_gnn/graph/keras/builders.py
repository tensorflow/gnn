"""Utility functions to simplify construction of GNN layers."""

import collections
from typing import Any, Callable, Mapping, Optional, Set

import tensorflow as tf

from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph.keras.layers import graph_update as graph_update_lib
from tensorflow_gnn.graph.keras.layers import next_state as next_state_lib


class ConvGNNBuilder:
  """Factory of layers that do convolutions on a graph.

  ConvGNNBuilder object constructs `GraphUpdate` layers, that apply arbitrary
  convolutions and updates on nodes of a graph. The convolutions (created by the
  `convolutions_factory`) propagate information to the incident edges of the
  graph. The results of the convolution together with the current nodes states
  are used to update the nodes, using a layer created by
  `nodes_next_state_factory`.

  Layers created by ConvGNNBuilder can be (re-)used in any order.

  Example:
    # Model hyper-parameters:
    h_dims = {'a': 64, 'b': 32, 'c': 32}
    m_dims = {'a->b': 64, 'b->c': 32, 'c->a': 32}

    # ConvGNNBuilder initialization:
    gnn = tfgnn.ConvGNNBuilder(
      lambda edge_set_name: tfgnn.SimpleConvolution(
         tf.keras.layers.Dense(m_dims[edge_set_name])),
      lambda node_set_name: tfgnn.NextStateFromConcat(
         tf.keras.layers.Dense(h_dims[node_set_name]))
    )

    # Two rounds of message passing to target node sets:
    model = tf.keras.models.Sequential([
        gnn.Convolve({"a", "b", "c"}),  # equivalent to gnn.Convolve()
        gnn.Convolve({"c"}),
    ])

  Init args:
    convolutions_factory: callable that takes as an input edge set name and
      returns graph convolution as EdgesToNodePooling layer.
    nodes_next_state_factory: callable that takes as an input node set name and
      returns node set next state as NextStateForNodeSet layer.
  """

  def __init__(
      self, convolutions_factory: Callable[
          [const.EdgeSetName], graph_update_lib.EdgesToNodePoolingLayer],
      nodes_next_state_factory: Callable[[const.NodeSetName],
                                         next_state_lib.NextStateForNodeSet]):
    self._convolutions_factory = convolutions_factory
    self._nodes_next_state_factory = nodes_next_state_factory

  def Convolve(
      self,
      node_sets: Optional[Set[const.NodeSetName]] = None
  ) -> tf.keras.layers.Layer:
    """Constructs GraphUpdate layer for the set of target nodes.

    This method contructs NodeSetUpdate layers from convolutions and next state
    factories (specified during the class construction) for the target node
    sets. The resulting node set update layers are combined and returned as a
    GraphUpdate layer.

    Args:
      node_sets: optional set of node set names to be updated. Not setting this
        parameter is equivalent to updating all node sets.

    Returns:
      GraphUpdate layer wrapped with OncallBuilder for delayed building.
    """

    def _Init(graph_spec: gt.GraphTensorSpec) -> Mapping[str, Any]:
      target_to_inputs = collections.defaultdict(dict)
      target_node_sets = set(
          graph_spec.node_sets_spec if node_sets is None else node_sets)
      for edge_set_name, edge_set_spec in graph_spec.edge_sets_spec.items():
        if not isinstance(edge_set_spec.adjacency_spec, adj.AdjacencySpec):
          raise ValueError('Unsupported adjacency type {}'.format(
              type(edge_set_spec.adjacency_spec).__name__))
        target_node_set = edge_set_spec.adjacency_spec.target_name
        if target_node_set in target_node_sets:
          target_to_inputs[target_node_set][
              edge_set_name] = self._convolutions_factory(edge_set_name)

      node_set_updates = dict()
      for node_set in target_node_sets:
        next_state = self._nodes_next_state_factory(node_set)
        node_set_updates[node_set] = graph_update_lib.NodeSetUpdate(
            edge_set_inputs=target_to_inputs[node_set], next_state=next_state)
      return dict(node_sets=node_set_updates)

    return graph_update_lib.GraphUpdate(deferred_init_callback=_Init)
