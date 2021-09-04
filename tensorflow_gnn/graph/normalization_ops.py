"""Defines normalization operations over a GraphTensor."""

from typing import Optional

import tensorflow as tf
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import graph_tensor_ops as ops


def softmax_edges_per_node(
    graph_tensor: gt.GraphTensor,
    edge_set_name: gt.EdgeSetName,
    node_tag: const.IncidentNodeTag,
    *,
    feature_value: Optional[gt.Field] = None,
    feature_name: Optional[gt.FieldName] = None) -> gt.Field:
  """Softmaxes all the edges in the graph over their incident nodes.

  This function performs a per-edge softmax operation, grouped by all
  the edges per node the direction of `node_tag`.

  Args:
    graph_tensor: A scalar GraphTensor.
    edge_set_name: The name of the edge set from which values are pooled.
    node_tag: The incident node of each edge at which values are aggregated,
      identified by its tag in the edge set.
    feature_value: A ragged or dense edge feature value.
    feature_name: An edge feature name.

  Raises:
    ValueError is `edge_set_name` is not in the `graph_tensor` edges.

  Returns:
    The edge values softmaxed per incident node. The dimensions do not change.
  """
  edge_value = ops.resolve_value(
      graph_tensor.edge_sets[edge_set_name],
      feature_value=feature_value,
      feature_name=feature_name)

  # Subtract the maxes for numerical stability.
  segment_maxes = ops.pool_edges_to_node(
      graph_tensor, edge_set_name, node_tag, 'max', feature_value=edge_value)
  maxes = ops.broadcast_node_to_edges(
      graph_tensor, edge_set_name, node_tag, feature_value=segment_maxes)
  exp_edge_value = tf.exp(edge_value - maxes)

  sum_exp_edge_value = ops.pool_edges_to_node(
      graph_tensor,
      edge_set_name,
      node_tag,
      'sum',
      feature_value=exp_edge_value)
  return exp_edge_value / ops.broadcast_node_to_edges(
      graph_tensor, edge_set_name, node_tag, feature_value=sum_exp_edge_value)
