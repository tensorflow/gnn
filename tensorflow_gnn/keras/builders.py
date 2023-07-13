# Copyright 2021 The TensorFlow GNN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility functions to simplify construction of GNN layers."""

import collections
from typing import Any, Callable, Collection, Mapping, Optional

import tensorflow as tf

from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.keras.layers import graph_update as graph_update_lib
from tensorflow_gnn.keras.layers import next_state as next_state_lib


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

  ```python
  # Model hyper-parameters:
  h_dims = {'a': 64, 'b': 32, 'c': 32}
  m_dims = {'a->b': 64, 'b->c': 32, 'c->a': 32}

  # ConvGNNBuilder initialization:
  gnn = tfgnn.keras.ConvGNNBuilder(
    lambda edge_set_name, receiver_tag: tfgnn.keras.layers.SimpleConv(
        tf.keras.layers.Dense(m_dims[edge_set_name]),
        receiver_tag=receiver_tag),
    lambda node_set_name: tfgnn.keras.layers.NextStateFromConcat(
        tf.keras.layers.Dense(h_dims[node_set_name])),
    receiver_tag=tfgnn.TARGET)

  # Two rounds of message passing to target node sets:
  model = tf.keras.models.Sequential([
      gnn.Convolve({"a", "b", "c"}),  # equivalent to gnn.Convolve()
      gnn.Convolve({"c"}),
  ])
  ```

  Advanced users can pass additional callbacks to further customize the creation
  of node set updates and the complete graph updates. The default values of
  those callbacks are equivalent to

  ```python
  def node_set_update_factory(node_set_name, edge_set_inputs, next_state):
    del node_set_name  # Unused by default.
    return tfgnn.keras.layers.NodeSetUpdate(edge_set_inputs, next_state)

  def graph_update_factory(deferred_init_callback, name):
    return tfgnn.keras.layers.GraphUpdate(
        deferred_init_callback=deferred_init_callback, name=name)
  ```

  Init args:
    convolutions_factory: called as
      `convolutions_factory(edge_set_name, receiver_tag=receiver_tag)`
      to return the convolution layer for the edge set towards the specified
      receiver. The `receiver_tag` kwarg is omitted from the call if it is
      omitted from the init args (but that usage is deprecated).
    nodes_next_state_factory: called as
      `nodes_next_state_factory(node_set_name)` to return the next-state layer
      for the respectve NodeSetUpdate.
    receiver_tag: Set this to `tfgnn.TARGET` or `tfgnn.SOURCE` to choose which
      incident node of each edge receives the convolution result.
      DEPRECATED: This used to be optional and effectively default to TARGET.
      New code is expected to set it in any case.
    node_set_update_factory: If set, called as
      `node_set_update_factory(node_set_name, edge_set_inputs, next_state)`
      to return the node set update for the given `node_set_name`. The
      remaining arguments are as expected by `tfgnn.keras.layers.NodeSetUpdate`.
    graph_update_factory: If set, called as
      `graph_update_factory(deferred_init_callback, name)` to return the graph
      update. The arguments are as expected by `tfgnn.keras.layers.GraphUpdate`.
  """

  def __init__(
      self,
      convolutions_factory: Callable[
          ..., graph_update_lib.EdgesToNodePoolingLayer],
      nodes_next_state_factory: Callable[[const.NodeSetName],
                                         next_state_lib.NextStateForNodeSet],
      *,
      receiver_tag: Optional[const.IncidentNodeTag] = None,
      node_set_update_factory: Optional[Callable[
          ..., graph_update_lib.NodeSetUpdateLayer]] = None,
      graph_update_factory: Optional[Callable[
          ..., tf.keras.layers.Layer]] = None,
    ):
    self._convolutions_factory = convolutions_factory
    self._nodes_next_state_factory = nodes_next_state_factory
    self._receiver_tag = receiver_tag

    if node_set_update_factory is None:
      self._node_set_update_factory = _default_node_set_update_factory
    else:
      self._node_set_update_factory = node_set_update_factory
    if graph_update_factory is None:
      self._graph_update_factory = _default_graph_update_factory
    else:
      self._graph_update_factory = graph_update_factory

  def Convolve(  # To be called like a class initializer.  pylint: disable=invalid-name
      self,
      node_sets: Optional[Collection[const.NodeSetName]] = None,
      name: Optional[str] = None,
  ) -> tf.keras.layers.Layer:
    """Constructs GraphUpdate layer for the set of receiver node sets.

    This method contructs NodeSetUpdate layers from convolutions and next state
    factories (specified during the class construction) for the given receiver
    node sets. The resulting node set update layers are combined and returned
    as one GraphUpdate layer. Auxiliary node sets (e.g., as needed for
    `tfgnn.keras.layers.NamedReadout`) are ignored.

    Args:
      node_sets: By default, the result updates all node sets that receive from
        at least one edge set. Optionally, this argument can specify a subset
        of those node sets. It is not allowed to include node sets that do not
        receive messages from any edge set. It is also not allowed to include
        auxiliary node sets.
      name: Optionally, a name for the returned GraphUpate layer.

    Returns:
      A GraphUpdate layer, with building deferred to the first call.
    """
    if node_sets is not None:
      node_sets = set(node_sets)
      for node_set_name in node_sets:
        if gt.get_aux_type_prefix(node_set_name):
          # This is not allowed, because it makes no sense for the aux node sets
          # known so far (March 2023). How would aux or non-aux edge sets be
          # selected uniformly for convolving into aux and non-aux node sets?
          raise ValueError(
              f"Convolution requested towards aux node set '{node_set_name}'.")

    def _init(graph_spec: gt.GraphTensorSpec) -> Mapping[str, Any]:
      if self._receiver_tag is None:
        receiver_tag = const.TARGET
        receiver_tag_specified = False
      else:
        receiver_tag = self._receiver_tag
        receiver_tag_specified = True

      receiver_to_inputs = collections.defaultdict(dict)

      for edge_set_name, edge_set_spec in graph_spec.edge_sets_spec.items():
        if gt.get_aux_type_prefix(edge_set_name):
          continue
        if not isinstance(edge_set_spec.adjacency_spec, adj.HyperAdjacencySpec):
          raise ValueError("Unsupported adjacency type {}".format(
              type(edge_set_spec.adjacency_spec).__name__))
        receiver_node_set = edge_set_spec.adjacency_spec.node_set_name(
            receiver_tag)
        if node_sets is not None and receiver_node_set not in node_sets:
          continue
        if gt.get_aux_type_prefix(receiver_node_set):
          # This cannot happen for the aux node sets known so far (March 2023)
          # and likely indicates the accidental use of an auxiliary name.
          raise ValueError(
              f"Node set '{receiver_node_set}' is auxiliary but the incident "
              f"edge set '{edge_set_name}' (at tag {receiver_tag}) is not.")
        if receiver_tag_specified:
          conv = self._convolutions_factory(edge_set_name,
                                            receiver_tag=receiver_tag)
        else:
          conv = self._convolutions_factory(edge_set_name)
        receiver_to_inputs[receiver_node_set][edge_set_name] = conv

      if node_sets is None:
        receiver_node_sets = receiver_to_inputs.keys()
      else:
        non_receiver_node_sets = [
            node_set_name for node_set_name in node_sets
            if node_set_name not in receiver_to_inputs]
        if non_receiver_node_sets:
          raise ValueError(
              "A GraphUpdate was requested to include the following node sets "
              "that do not receive inputs from any edge set: "
              f"{non_receiver_node_sets}. The receiver tag is {receiver_tag}, "
              f"where SOURCE={const.SOURCE} and TARGET={const.TARGET}. "
              "In convolutional or message-passing GNNs, this is likely an "
              "error and hence unsupported by `tfgnn.keras.ConvGNNBuilder` and "
              "the GraphUpdate classes built by it. "
          )
        receiver_node_sets = node_sets

      node_set_updates = dict()
      for node_set in receiver_node_sets:
        next_state = self._nodes_next_state_factory(node_set)
        node_set_updates[node_set] = self._node_set_update_factory(
            node_set, receiver_to_inputs[node_set], next_state)
      return dict(node_sets=node_set_updates)

    return self._graph_update_factory(deferred_init_callback=_init, name=name)


def _default_node_set_update_factory(
    node_set_name: const.NodeSetName,
    edge_set_inputs: Mapping[const.EdgeSetName,
                             graph_update_lib.EdgesToNodePoolingLayer],
    next_state: next_state_lib.NextStateForNodeSet,
) -> graph_update_lib.NodeSetUpdateLayer:
  del node_set_name  # Unused.
  return graph_update_lib.NodeSetUpdate(edge_set_inputs, next_state)


def _default_graph_update_factory(
    deferred_init_callback: Callable[[gt.GraphTensorSpec], Mapping[str, Any]],
    name: str,
)-> tf.keras.layers.Layer:
  return graph_update_lib.GraphUpdate(
      deferred_init_callback=deferred_init_callback, name=name)
