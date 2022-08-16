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
from typing import Optional

import tensorflow as tf
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import graph_tensor_ops as ops


def softmax(
    graph_tensor: gt.GraphTensor,
    per_tag: const.IncidentNodeOrContextTag,
    *,
    edge_set_name: Optional[const.EdgeSetName] = None,
    node_set_name: Optional[const.NodeSetName] = None,
    feature_value: Optional[gt.Field] = None,
    feature_name: Optional[gt.FieldName] = None) -> gt.Field:
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
    edge_set_name: The name of the edge set on which values are normalized
      Exactly one of edge_set_name and node_set_name must be set.
    node_set_name: The name of the node set on which values are normalized,
      allowed only if per_tag is `tfgnn.CONTEXT`. See also edge_set_name.
    feature_value: A ragged or dense tensor with the value; cf. feature_name.
    feature_name: The name of the feature to be used as input value.
      Exactly one of feature_value or feature_name must be set.

  Raises:
    ValueError: if `graph_tensor` does not contain an edge set or node set
      of the given name.

  Returns:
    The softmaxed values. The dimensions do not change from the input.
  """
  # Set up the `value` to be softmaxed with generic `pool` and `broadcast`.
  if bool(edge_set_name is None) + bool(node_set_name is None) != 1:
    raise ValueError("Must pass exactly one of edge_set_name, node_set_name.")
  if edge_set_name:
    value = ops.resolve_value(
        graph_tensor.edge_sets[edge_set_name],
        feature_value=feature_value, feature_name=feature_name)
    pool = functools.partial(
        ops.pool, graph_tensor, per_tag, edge_set_name=edge_set_name)
    broadcast = functools.partial(
        ops.broadcast, graph_tensor, per_tag, edge_set_name=edge_set_name)
  else:
    value = ops.resolve_value(
        graph_tensor.node_sets[node_set_name],
        feature_value=feature_value, feature_name=feature_name)
    pool = functools.partial(
        ops.pool, graph_tensor, per_tag, node_set_name=node_set_name)
    broadcast = functools.partial(
        ops.broadcast, graph_tensor, per_tag, node_set_name=node_set_name)

  # Compute softmax. Subtract the maxes for numerical stability.
  # Some segment_maxes may be -inf, but that's broadcast nowhere.
  segment_maxes = pool(reduce_type="max", feature_value=value)
  maxes = broadcast(feature_value=segment_maxes)
  exp_edge_value = tf.exp(value - maxes)
  sum_exp_value = pool(reduce_type="sum", feature_value=exp_edge_value)
  return exp_edge_value / broadcast(feature_value=sum_exp_value)


def softmax_edges_per_node(
    graph_tensor: gt.GraphTensor,
    edge_set_name: gt.EdgeSetName,
    node_tag: const.IncidentNodeTag,
    *,
    feature_value: Optional[gt.Field] = None,
    feature_name: Optional[gt.FieldName] = None) -> gt.Field:
  """Returns softmax() of edge values per common `node_tag` node."""
  return softmax(graph_tensor, node_tag, edge_set_name=edge_set_name,
                 feature_value=feature_value, feature_name=feature_name)


def softmax_edges_per_component(
    graph_tensor: gt.GraphTensor,
    edge_set_name: gt.EdgeSetName,
    *,
    feature_value: Optional[gt.Field] = None,
    feature_name: Optional[gt.FieldName] = None) -> gt.Field:
  """Returns softmax() of edge values per graph component."""
  return softmax(graph_tensor, const.CONTEXT, edge_set_name=edge_set_name,
                 feature_value=feature_value, feature_name=feature_name)


def softmax_nodes_per_component(
    graph_tensor: gt.GraphTensor,
    node_set_name: gt.NodeSetName,
    *,
    feature_value: Optional[gt.Field] = None,
    feature_name: Optional[gt.FieldName] = None) -> gt.Field:
  """Returns softmax() of node values per graph component."""
  return softmax(graph_tensor, const.CONTEXT, node_set_name=node_set_name,
                 feature_value=feature_value, feature_name=feature_name)
