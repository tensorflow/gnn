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
"""Defines normalization operations over a GraphTensor."""

import functools
from typing import cast, Optional, Sequence, Union

import tensorflow as tf
from tensorflow_gnn.graph import broadcast_ops
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import pool_ops


Field = const.Field
FieldName = const.FieldName
NodeSetName = const.NodeSetName
EdgeSetName = const.EdgeSetName
IncidentNodeTag = const.IncidentNodeTag
IncidentNodeOrContextTag = const.IncidentNodeOrContextTag
GraphTensor = gt.GraphTensor


def softmax(
    graph_tensor: GraphTensor,
    per_tag: IncidentNodeOrContextTag,
    *,
    edge_set_name: Union[Sequence[EdgeSetName], EdgeSetName, None] = None,
    node_set_name: Union[Sequence[NodeSetName], NodeSetName, None] = None,
    feature_value: Union[Sequence[Field], Field, None] = None,
    feature_name: Optional[FieldName] = None) -> Union[Sequence[Field], Field]:
  """Computes softmax over a many-to-one relationship in a GraphTensor.

  This function can be used to compute a softmax normalization...

    * of edge values, across the edges with a common incident node at `per_tag`
      (e.g., SOURCE or TARGET);
    * of node values, across all the nodes in the same graph component;
    * of edge values, across all the edges in the same graph component.

  For non-scalar values, the softmax function is applied element-wise.

  Args:
    graph_tensor: A scalar GraphTensor.
    per_tag: tfgnn.CONTEXT for normalization per graph component, or an incident
      node tag (e.g., `tfgnn.SOURCE` or `tfgnn.TARGET`) for normalization per
      common incident node.
    edge_set_name: The name of the edge set on which values are normalized,
      or a non-empty sequence of such names. Unless `from_tag=tfgnn.CONTEXT`,
      all named edge sets must have the same incident node set at the given tag.
    node_set_name: The name of the node set on which values are normalized,
      or a non-empty sequence of such names. Can only be passed together with
      `from_tag=tfgnn.CONTEXT`. Exactly one of edge_set_name or node_set_name
      must be set.
    feature_value: A tensor or list of tensors, parallel to the node_set_names
      or edge_set_names, to supply the input values of softmax. Each tensor
      has shape `[num_items, *feature_shape]`, where `num_items` is the number
      of edges in the given edge set or nodes in the given node set, and
      `*feature_shape` is the same across all inputs.
    feature_name: The name of a feature stored on each graph piece from which
      pooling is done, for use instead of an explicity passed feature_value.
      Exactly one of feature_name or feature_value must be set.

  Returns:
    A tensor or a list of tensors with the softmaxed values. The dimensions of
    the tensors and the length of the list do not change from the input.
  """
  # Set up a list of `values` to be softmaxed with `pool` and `broadcast` calls.
  edge_set_names, node_set_names, values, got_sequence_args = (
      pool_ops.get_pool_args_as_sequences(
          graph_tensor, per_tag,
          edge_set_name=edge_set_name, node_set_name=node_set_name,
          feature_value=feature_value, feature_name=feature_name,
          function_name="softmax()"))
  pool = functools.partial(
      pool_ops.pool_v2, graph_tensor, per_tag,
      edge_set_name=edge_set_names, node_set_name=node_set_names)
  broadcast = functools.partial(
      broadcast_ops.broadcast_v2, graph_tensor, per_tag,
      edge_set_name=edge_set_names, node_set_name=node_set_names)

  # Compute softmax. Subtract the maxes for numerical stability.
  # Some segment_maxes may be -inf, but that's broadcast nowhere.
  segment_maxes = pool(reduce_type="max", feature_value=values)
  maxes = broadcast(feature_value=segment_maxes)
  exp_values = [tf.exp(v - m) for v, m in _zip_strict(values, maxes)]
  sum_exp_values = broadcast(feature_value=pool(reduce_type="sum",
                                                feature_value=exp_values))
  result = [ev / sev for ev, sev in _zip_strict(exp_values, sum_exp_values)]

  # Return result with the same nesting as the inputs.
  if got_sequence_args:
    return result
  else:
    assert len(result) == 1
    return result[0]


# For Python 3.10+, replace by zip(..., strict=True).
def _zip_strict(*args):
  arg_lens = {len(arg) for arg in args}
  if len(arg_lens) > 1:
    raise ValueError(f"zip() arguments have unequal lengths {arg_lens}")
  return zip(*args)


def softmax_edges_per_node(
    graph_tensor: GraphTensor,
    edge_set_name: EdgeSetName,
    node_tag: IncidentNodeTag,
    *,
    feature_value: Optional[Field] = None,
    feature_name: Optional[FieldName] = None) -> Field:
  """Returns softmax() of edge values per common `node_tag` node."""
  return cast(Field,  # Not a Sequence for non-Sequence inputs.
              softmax(graph_tensor, node_tag, edge_set_name=edge_set_name,
                      feature_value=feature_value, feature_name=feature_name))


def softmax_edges_per_component(
    graph_tensor: GraphTensor,
    edge_set_name: EdgeSetName,
    *,
    feature_value: Optional[Field] = None,
    feature_name: Optional[FieldName] = None) -> Field:
  """Returns softmax() of edge values per graph component."""
  return cast(Field,  # Not a Sequence for non-Sequence inputs.
              softmax(graph_tensor, const.CONTEXT, edge_set_name=edge_set_name,
                      feature_value=feature_value, feature_name=feature_name))


def softmax_nodes_per_component(
    graph_tensor: GraphTensor,
    node_set_name: NodeSetName,
    *,
    feature_value: Optional[Field] = None,
    feature_name: Optional[FieldName] = None) -> Field:
  """Returns softmax() of node values per graph component."""
  return cast(Field,  # Not a Sequence for non-Sequence inputs.
              softmax(graph_tensor, const.CONTEXT, node_set_name=node_set_name,
                      feature_value=feature_value, feature_name=feature_name))
