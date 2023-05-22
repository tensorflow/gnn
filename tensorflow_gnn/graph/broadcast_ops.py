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
"""The broadcast operations on a GraphTensor."""
from __future__ import annotations
from typing import Optional, Sequence, Union

import tensorflow as tf

from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import tag_utils
from tensorflow_gnn.graph import tensor_utils as utils

Field = const.Field
FieldName = const.FieldName
NodeSetName = const.NodeSetName
EdgeSetName = const.EdgeSetName
IncidentNodeTag = const.IncidentNodeTag
IncidentNodeOrContextTag = const.IncidentNodeOrContextTag
GraphTensor = gt.GraphTensor


def broadcast_node_to_edges(graph_tensor: GraphTensor,
                            edge_set_name: EdgeSetName,
                            node_tag: IncidentNodeTag,
                            *,
                            feature_value: Optional[Field] = None,
                            feature_name: Optional[FieldName] = None) -> Field:
  """Broadcasts values from nodes to incident edges.

  Given a particular edge set (identified by `edge_set_name` name), this
  operation collects node features from the specific incident node of each edge
  (as indicated by `node_tag`). For example, setting `node_tag=tfgnn.SOURCE` and
  `reduce_type='sum'` gathers the source node features over each edge. (See the
  corresponding `pool_edges_to_node()` mirror operation).

  The feature to fetch node values from is provided either by name (using
  `feature_name`) and found in the graph tensor itself, or provided explicitly
  (using `feature_value`) in which case its shape has to be compatible with the
  shape prefix of the node set being gathered from. One of `feature_value`
  or `feature_name` must be specified.

  Args:
    graph_tensor: A scalar GraphTensor.
    edge_set_name: The name of the edge set to which values are broadcast.
    node_tag: The incident side of each edge from which values are broadcast,
      specified by its tag in the edge set (e.g. `tfgnn.SOURCE`,
      `tfgnn.TARGET`).
    feature_value: A ragged or dense source node feature values. Has a shape
      `[num_nodes, *feature_shape]`, where `num_nodes` is the number of nodes in
      the incident node set and feature_shape is the shape of the feature value
      for each node.
    feature_name: A source node feature name.

  Returns:
    Source node value broadcast to corresponding edges. Has a shape `[num_edges,
    *feature_shape]`, where `num_edges` is the number of edges in the
    `edge_set_name` edge set and `feature_shape` is not affected.
  """
  gt.check_scalar_graph_tensor(graph_tensor, "tfgnn.broadcast_node_to_edges()")
  adjacency = graph_tensor.edge_sets[edge_set_name].adjacency
  node_name = adjacency.node_set_name(node_tag)
  node_value = gt.resolve_value(
      graph_tensor.node_sets[node_name],
      feature_value=feature_value,
      feature_name=feature_name)
  return tf.gather(node_value, adjacency[node_tag])


def broadcast_context_to_nodes(
    graph_tensor: GraphTensor,
    node_set_name: NodeSetName,
    *,
    feature_value: Optional[Field] = None,
    feature_name: Optional[FieldName] = None) -> Field:
  """Broadcasts a context value to the `node_set` nodes.

  Given a particular node set (as identified by `node_set_name`), this operation
  collects context features from the corresponding graphs to each node. See the
  corresponding `pool_nodes_to_context()` mirror operation).

  The context feature to fetch values from is provided either by name (using
  `feature_name`) and found in the graph tensor itself, or provided explicitly
  (using `feature_value`) in which case its shape has to be compatible with the
  shape prefix of the node set being gathered from. One of `feature_value` or
  `feature_name` must be specified.

  Args:
    graph_tensor: A scalar GraphTensor.
    node_set_name: A node set name.
    feature_value: A ragged or dense graph context feature value. Has a shape
      `[num_components, *feature_shape]`, where `num_components` is the number
      of components in a graph and `feature_shape` is the shape of the feature
      value for each component.
    feature_name: A context feature name.

  Returns:
    Graph context value broadcast to the `node_set` nodes. Has a shape
    `[num_nodes, *feature_shape]`, where `num_nodes` is the number of nodes in
    the `node_set_name` node set and `feature_shape` is not affected.
  """
  return _broadcast_context(
      graph_tensor,
      graph_tensor.node_sets[node_set_name],
      feature_value=feature_value,
      feature_name=feature_name)


def broadcast_context_to_edges(
    graph_tensor: GraphTensor,
    edge_set_name: EdgeSetName,
    *,
    feature_value: Optional[Field] = None,
    feature_name: Optional[FieldName] = None) -> Field:
  """Broadcasts a context value to the `edge_set` edges.

  Given a particular edge set (as identified by `edge_set_name`), this operation
  collects context features from the corresponding graphs to each edge. See the
  corresponding `pool_edges_to_context()` mirror operation).

  The context feature to fetch values from is provided either by name (using
  `feature_name`) and found in the graph tensor itself, or provided explicitly
  (using `feature_value`) in which case its shape has to be compatible with the
  shape prefix of the node set being gathered from. One of `feature_value` or
  `feature_name` must be specified.

  Args:
    graph_tensor: A scalar GraphTensor.
    edge_set_name: An edge set name.
    feature_value: A ragged or dense graph context feature value. Has a shape
      `[num_components, *feature_shape]`, where `num_components` is the number
      of components in a graph and `feature_shape` is the shape of the feature
      value for each component.
    feature_name: A context feature name.

  Returns:
    Graph context value broadcast to the `edge_set` edges. Has a shape
    `[num_edges, *feature_shape]`, where `num_edges` is the number of edges in
    the `edge_set_name` edge set and `feature_shape` is not affected.
  """
  return _broadcast_context(
      graph_tensor,
      graph_tensor.edge_sets[edge_set_name],
      feature_value=feature_value,
      feature_name=feature_name)


def _broadcast_context(graph_tensor: GraphTensor,
                       node_or_edge_set: Union[gt.NodeSet, gt.EdgeSet],
                       *,
                       feature_value: Optional[Field] = None,
                       feature_name: Optional[FieldName] = None) -> Field:
  """Broadcasts context value to graph node or edge."""

  gt.check_scalar_graph_tensor(graph_tensor, "tfgnn.broadcast_context_to_*()")

  context_value = gt.resolve_value(
      graph_tensor.context,
      feature_value=feature_value,
      feature_name=feature_name)

  # TODO(b/184021442): cache result.
  return utils.repeat(
      context_value,
      node_or_edge_set.sizes,
      repeats_sum_hint=node_or_edge_set.spec.total_size)


def broadcast_v2(
    graph_tensor: GraphTensor,
    from_tag: IncidentNodeOrContextTag,
    *,
    edge_set_name: Union[Sequence[EdgeSetName], EdgeSetName, None] = None,
    node_set_name: Union[Sequence[NodeSetName], NodeSetName, None] = None,
    feature_value: Optional[Field] = None,
    feature_name: Optional[FieldName] = None) -> Union[list[Field], Field]:
  """Broadcasts values from nodes to edges, or from context to nodes or edges.

  This function broadcasts a feature value from context to nodes or edges if
  called with `from_tag=tfgnn.CONTEXT`, or from incident nodes to edges if
  called with `from_tag` set to an ordinary node tag like `tfgnn.SOURCE` or
  `tfgnn.TARGET`.

  The `edge_set_name` (or `node_set_name`, when broadcasting from context)
  can be set to the name of a single destination, or to a list of names of
  multiple destinations.

  Functionally, there is no difference to calling the underlying functions
  `broadcast_node_to_edges()`, `broadcast_context_to_nodes()`, or
  `broadcast_context_to_edges()` directly on individual edge sets or node sets.
  However, the more generic API of this function provides the proper mirror
  image of `tfgnn.pool()`, which comes in handy for some algorithms.

  Args:
    graph_tensor: A scalar GraphTensor.
    from_tag: Values are broadcast from context if this is `tfgnn.CONTEXT` or
      from the incident node on each edge with this tag.
    edge_set_name: The name of the edge set to which values are broadcast, or
      a non-empty sequence of such names. Unless `from_tag=tfgnn.CONTEXT`,
      all named edge sets must have the same incident node set at the given tag.
    node_set_name: The name of the node set to which values are broadcast,
      or a non-empty sequence of such names. Can only be passed together with
      `from_tag=tfgnn.CONTEXT`. Exactly one of edge_set_name or node_set_name
      must be set.
    feature_value: A tensor of shape `[num_items, *feature_shape]` from which
      the broadcast values are taken. The first dimension indexes the items
      from which the broadcast is done (that is, the nodes of the common node
      set identified by `from_tag`, or the graph components in the context).
    feature_name: The name of a feature stored in the graph, for use instead of
      feature_value. Exactly one of feature_name or feature_value must be set.

  Returns:
    The result of broadcasting to the specified edge set(s) or node set(s).
    If a single name was specified, the result is is a single tensor.
    If a list of names was specified, the result is a list of tensors,
    with parallel indices.
  """
  gt.check_scalar_graph_tensor(graph_tensor, "broadcast()")
  edge_set_names, node_set_names, got_sequence_args = (
      tag_utils.get_edge_or_node_set_name_args_for_tag(
          graph_tensor.spec, from_tag,
          edge_set_name=edge_set_name, node_set_name=node_set_name,
          function_name="broadcast()"))
  del edge_set_name, node_set_name  # Replaced by their cleaned-up versions.
  if (feature_value is None) == (feature_name is None):
    raise ValueError(
        "broadcast() requires exactly one of feature_name of feature_value.")
  feature_kwargs = dict(feature_value=feature_value, feature_name=feature_name)

  if from_tag == const.CONTEXT:
    if edge_set_names is not None:
      result = [broadcast_context_to_edges(graph_tensor, name, **feature_kwargs)
                for name in edge_set_names]
    else:
      result = [broadcast_context_to_nodes(graph_tensor, name, **feature_kwargs)
                for name in node_set_names]
  else:
    result = [
        broadcast_node_to_edges(graph_tensor, name, from_tag, **feature_kwargs)
        for name in edge_set_names]

  if got_sequence_args:
    return result
  else:
    assert len(result) == 1
    return result[0]
