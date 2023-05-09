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
"""The original ("v1") pool operations on a GraphTensor."""

from typing import Callable, List, Optional, Union

import tensorflow as tf

from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import tensor_utils as utils

Field = const.Field
FieldName = const.FieldName
NodeSetName = const.NodeSetName
EdgeSetName = const.EdgeSetName
IncidentNodeTag = const.IncidentNodeTag
GraphTensor = gt.GraphTensor

# Unsorted reduce operation (for a variable number of graph items).
#
# Could be any of tf.math.unsorted_segment_{sum|min|max|mean|...}.
#
# Args:
#  feature: GraphTensor feature (tensor or ragged tensor).
#  segment_ids: rank 1 integer tensor with segment ids for each feature value.
#    Its shape need is a prefix of `feature`.
#  num_segments: the total number of segments. Should be either scalar tensor or
#    python integer (to enable XLA support).
#
# Returns:
#   Field values aggregated over segments_ids.
UnsortedReduceOp = Callable[[Field, tf.Tensor, Union[tf.Tensor, int]], Field]


# TODO(b/265760014): Move to pool_ops.py and rewrite to wrap pool_v2().
def pool_edges_to_node(graph_tensor: GraphTensor,
                       edge_set_name: EdgeSetName,
                       node_tag: IncidentNodeTag,
                       reduce_type: str = 'sum',
                       *,
                       feature_value: Optional[Field] = None,
                       feature_name: Optional[FieldName] = None) -> Field:
  """Aggregates (pools) edge values to incident nodes.

  Given a particular edge set (identified by `edge_set_name` name), this
  operation reduces edge features at the specific incident node of each edge (as
  indicated by `node_tag`). For example, setting `node_tag=tfgnn.TARGET` and
  `reduce_type='sum'` computes the sum over the incoming edge features at each
  node. (See the corresponding `broadcast_node_to_edges()` mirror operation).

  The feature to fetch edge values from is provided either by name (using
  `feature_name`) and found in the graph tensor itself, or provided explicitly
  (using `feature_value`) in which case its shape has to be compatible with the
  shape prefix of the edge set being gathered from. One of `feature_value`
  or `feature_name` must be specified.

  (Note that in most cases the `feature_value` form will be used, because in a
  regular convolution, we will first broadcast over edges and combine the result
  of that with this function.)

  Args:
    graph_tensor: A scalar GraphTensor.
    edge_set_name: The name of the edge set from which values are pooled.
    node_tag: The incident node of each edge at which values are aggregated,
      identified by its tag in the edge set.
    reduce_type: A pooling operation name, like 'sum', 'mean' or 'max'. For the
      list of supported values use `get_registered_reduce_operation_names()`.
    feature_value: A ragged or dense edge feature value. Has a shape
      `[num_edges, *feature_shape]`, where `num_edges` is the number of edges in
      the `edge_set_name` edge set and `feature_shape` is the shape of the
      feature value for each edge.
    feature_name: An edge feature name.

  Returns:
    The edge values pooled to each incident node. Has a shape `[num_nodes,
    *feature_shape]`, where `num_nodes` is the number of nodes in the incident
    node set and `feature_shape` is not affected.
  """
  gt.check_scalar_graph_tensor(graph_tensor, 'tfgnn.pool_edges_to_node()')
  unsorted_reduce_op = _resolve_reduce_op(reduce_type)

  edge_value = gt.resolve_value(
      graph_tensor.edge_sets[edge_set_name],
      feature_value=feature_value,
      feature_name=feature_name)

  adjacency = graph_tensor.edge_sets[edge_set_name].adjacency
  node_set = graph_tensor.node_sets[adjacency.node_set_name(node_tag)]
  total_node_count = node_set.spec.total_size
  if total_node_count is None:
    total_node_count = node_set.total_size
  return unsorted_reduce_op(edge_value, adjacency[node_tag], total_node_count)


# TODO(b/265760014): Move to pool_ops.py and rewrite to wrap pool_v2().
def pool_nodes_to_context(graph_tensor: GraphTensor,
                          node_set_name: NodeSetName,
                          reduce_type: str = 'sum',
                          *,
                          feature_value: Optional[Field] = None,
                          feature_name: Optional[FieldName] = None) -> Field:
  """Aggregates (pools) node values to graph context.

  Given a particular node set (identified by `node_set_name`), this operation
  reduces node features to their corresponding graph component. For example,
  setting `reduce_type='sum'` computes the sum over the node features of each
  graph. (See the corresponding `broadcast_context_to_nodes()` mirror
  operation).

  The feature to fetch node values from is provided either by name (using
  `feature_name`) and found in the graph tensor itself, or provided explicitly
  (using `feature_value`). One of `feature_value` or `feature_name` must be
  specified.

  Args:
    graph_tensor: A scalar GraphTensor.
    node_set_name: A node set name.
    reduce_type: A pooling operation name, like 'sum', 'mean' or 'max'. For the
      list of supported values use `get_registered_reduce_operation_names()`.
    feature_value: A ragged or dense node feature value. Has a shape
      `[num_nodes, *feature_shape]`, where `num_nodes` is the number of nodes in
      the `node_set_name` node set and `feature_shape` is the shape of the
      feature value for each node.
    feature_name: A node feature name.

  Returns:
    Node value pooled to graph context. Has a shape `[num_components,
    *feature_shape]`, where `num_components` is the number of components in a
    graph and `feature_shape` is not affected.
  """
  return _pool_to_context(
      graph_tensor,
      graph_tensor.node_sets[node_set_name],
      reduce_type,
      feature_value=feature_value,
      feature_name=feature_name)


# TODO(b/265760014): Move to pool_ops.py and rewrite to wrap pool_v2().
def pool_edges_to_context(graph_tensor: GraphTensor,
                          edge_set_name: EdgeSetName,
                          reduce_type: str = 'sum',
                          *,
                          feature_value: Optional[Field] = None,
                          feature_name: Optional[FieldName] = None) -> Field:
  """Aggregates (pools) edge values to graph context.

  Given a particular edge set (identified by `edge_set_name`), this operation
  reduces edge features to their corresponding graph component. For example,
  setting `reduce_type='sum'` computes the sum over the edge features of each
  graph. (See the corresponding `broadcast_context_to_edges()` mirror
  operation).

  The feature to fetch edge values from is provided either by name (using
  `feature_name`) and found in the graph tensor itself, or provided explicitly
  (using `feature_value`). One of `feature_value` or `feature_name` must be
  specified.

  (Note that in most cases the `feature_value` form will be used, because in a
  regular convolution, we will first broadcast over edges and combine the result
  of that with this function or a pooling over the nodes.)

  Args:
    graph_tensor: A scalar GraphTensor.
    edge_set_name: An edge set name.
    reduce_type: A pooling operation name, like 'sum', 'mean' or 'max'. For the
      list of supported values use `get_registered_reduce_operation_names()`.
    feature_value: A ragged or dense edge feature value. Has a shape
      `[num_edges, *feature_shape]`, where `num_edges` is the number of edges in
      the `edge_set_name` edge set and `feature_shape` is the shape of the
      feature value for each edge.
    feature_name: An edge feature name.

  Returns:
    A node value pooled to graph context. Has a shape `[num_components,
    *feature_shape]`, where `num_components` is the number of components in a
    graph and `feature_shape` is not affected.
  """
  return _pool_to_context(
      graph_tensor,
      graph_tensor.edge_sets[edge_set_name],
      reduce_type,
      feature_value=feature_value,
      feature_name=feature_name)


# TODO(b/265760014): Remove in favor of pool_v2().
# The difference is that v2 supports multiple node/edge sets,
# composite reduce types, and no longer uses the _REGISTERED_REDUCE_OPS here.
def pool_v1(graph_tensor: GraphTensor,
            to_tag: const.IncidentNodeOrContextTag,
            *,
            edge_set_name: Optional[EdgeSetName] = None,
            node_set_name: Optional[NodeSetName] = None,
            reduce_type: str = 'sum',
            feature_value: Optional[Field] = None,
            feature_name: Optional[FieldName] = None) -> Field:
  """Pools values from edges to nodes, or from nodes or edges to context.

  This function pools to context if `to_tag=tfgnn.CONTEXT` and pools from edges
  to incident nodes if `to_tag` is an ordinary node tag like `tfgnn.SOURCE` or
  `tfgnn.TARGET`. Most user code will not need this flexibility and can directly
  call one of the underlying functions pool_edges_to_node, pool_nodes_to_context
  or pool_edges_to_context.

  Args:
    graph_tensor: A scalar GraphTensor.
    to_tag: Values are pooled to context if this is `tfgnn.CONTEXT` or to the
      incident node on each edge with this tag.
    edge_set_name: The name of the edge set from which values are pooled.
    node_set_name: The name of the node set from which values are pooled.
      Can only be set with `to_tag=tfgnn.CONTEXT`. Either edge_set_name or
      node_set_name must be set.
    reduce_type: As for the underlying pool_*() function: a pooling operation
      name. Defaults to 'sum'.
    feature_value: As for the underlying pool_*() function.
    feature_name: As for the underlying pool_*() function.
      Exactly one of feature_name or feature_value must be set.

  Returns:
    The result of the underlying pool_*() function.
  """
  _validate_names_and_tag(
      to_tag, edge_set_name=edge_set_name, node_set_name=node_set_name)
  if to_tag == const.CONTEXT:
    if node_set_name is not None:
      return pool_nodes_to_context(
          graph_tensor, reduce_type=reduce_type, node_set_name=node_set_name,
          feature_value=feature_value, feature_name=feature_name)
    else:
      return pool_edges_to_context(
          graph_tensor, reduce_type=reduce_type, edge_set_name=edge_set_name,
          feature_value=feature_value, feature_name=feature_name)
  else:
    return pool_edges_to_node(
        graph_tensor, reduce_type=reduce_type, edge_set_name=edge_set_name,
        node_tag=to_tag,
        feature_value=feature_value, feature_name=feature_name)


# TODO(b/265760014): Remove together with pool_v1().
def _validate_names_and_tag(tag, *, edge_set_name, node_set_name):
  """Helper for pool_v1()."""
  if tag == const.CONTEXT:
    num_names = bool(edge_set_name is None) + bool(node_set_name is None)
    if num_names != 1:
      raise ValueError('With tag CONTEXT, must pass exactly 1 of '
                       f'edge_set_name, node_set_name; got {num_names}.')
  else:
    if edge_set_name is None or node_set_name is not None:
      raise ValueError('Must pass edge_set_name but not node_set_name '
                       'for a tag other than CONTEXT.')


# TODO(b/265760014): Remove together with pool_v1().
def with_empty_set_value(reduce_op: UnsortedReduceOp,
                         empty_set_value) -> UnsortedReduceOp:
  """Wraps `reduce_op` so that `empty_set_value` is used to fill empty segments.

  This helper function allows to customize the value that will be used to fill
  empty segments in the result of `reduce_op`. Some standard unsorted segment
  operations may result in -infinity or infinity values for empty segments (e.g.
  -infinity for `tf.math.unsorted_segment_max`). Although the use of these
  extreme values is mathematically grounded, they are not neural nets friendly
  and could lead to numerical overflow. So in practice it is better to use some
  safer default for empty segments and let a NN learn how to condition on that.

  Args:
    reduce_op: unsorted reduce operation to wrap (e.g.
      tf.math.unsorted_segment_{min|max|mean|sum|..}).
    empty_set_value: scalar value to fill empty segments in `reduce_op` result.

  Returns:
    Wrapped `reduce_op`.
  """

  def wrapped_reduce_op(data, segment_ids, num_segments):
    result = reduce_op(data, segment_ids, num_segments)
    mask_dims = [utils.outer_dimension_size(data)] + [1] * (
        result.shape.rank - 1)
    mask = tf.math.unsorted_segment_sum(
        tf.ones(mask_dims, segment_ids.dtype), segment_ids, num_segments)
    mask = tf.logical_not(tf.cast(mask, tf.bool))

    empty_set_overwrite = tf.convert_to_tensor(
        empty_set_value, dtype=result.dtype)
    if empty_set_overwrite.shape.rank != 0:
      raise ValueError('Expected scalar `empty_set_value`,'
                       f' got shape={empty_set_overwrite.shape}.')
    return _where_scalar_or_field(mask, empty_set_overwrite, result)

  return wrapped_reduce_op


# TODO(b/265760014): Remove together with pool_v1().
def with_minus_inf_replaced(reduce_op: UnsortedReduceOp,
                            replacement_value) -> UnsortedReduceOp:
  """Wraps `reduce_op` so that `replacement_value` replaces '-inf', `dtype.min`.

  This helper function replaces all '-inf' and `dtype.min` from the `reduce_op`
  output with the `replacement_value`.  The standard reduce max operations may
  result in minimum possible values if used for empty sets (e.g. `dtype.min` is
  used by `tf.math.unsorted_segment_max` and '-inf' - by `tf.math.reduce_max`).
  Those values are not NN friendly and could lead to numerical overflow. In
  practice it is better to use some safer default to mark empty sets and let a
  NN learn how to condition on that.


  NOTE: If you need to differentiate infinities coming from the pooled data and
  those created by the empty sets, consider using `with_empty_set_value()`.

  Args:
    reduce_op: unsorted reduce operation to wrap (e.g.
      tf.math.unsorted_segment_max).
    replacement_value: scalar value to replace '-inf' and `dtype.min` in the
      `reduce_op` output.

  Returns:
    Wrapped `reduce_op`.
  """

  def wrapped_reduce_op(data, segment_ids, num_segments):
    result = reduce_op(data, segment_ids, num_segments)
    inf_overwrite = tf.convert_to_tensor(replacement_value, dtype=result.dtype)
    if inf_overwrite.shape.rank != 0:
      raise ValueError('Expected scalar `replacement_value`,'
                       f' got shape={inf_overwrite.shape}.')
    return _where_scalar_or_field(
        tf.less_equal(result, result.dtype.min), inf_overwrite, result)

  return wrapped_reduce_op


# TODO(b/265760014): Remove together with pool_v1().
def with_plus_inf_replaced(reduce_op: UnsortedReduceOp,
                           replacement_value) -> UnsortedReduceOp:
  """Wraps `reduce_op` so that `replacement_value` replaces '+inf', `dtype.max`.

  This helper function replaces all '+inf' and `dtype.max` from the `reduce_op`
  output with the `replacement_value`.  The standard reduce min operations may
  result in maximum possible values if used for empty sets (e.g. `dtype.max` is
  used by `tf.math.unsorted_segment_min` and '+inf' - by `tf.math.reduce_min`).
  Those values are not NN friendly and could lead to numerical overflow. In
  practice it is better to use some safer default to mark empty sets and let a
  NN learn how to condition on that.

  NOTE: If you need to differentiate infinities coming from the pooled data and
  those created by the empty sets, consider using `with_empty_set_value()`.

  Args:
    reduce_op: unsorted reduce operation to wrap (e.g.
      tf.math.unsorted_segment_min).
    replacement_value: scalar value to replace '+inf' and `dtype.max` in the
      `reduce_op` output.

  Returns:
    Wrapped `reduce_op`.
  """

  def wrapped_reduce_op(data, segment_ids, num_segments):
    result = reduce_op(data, segment_ids, num_segments)
    inf_overwrite = tf.convert_to_tensor(replacement_value, dtype=result.dtype)
    if inf_overwrite.shape.rank != 0:
      raise ValueError('Expected scalar `replacement_value`,'
                       f' got shape={inf_overwrite.shape}.')

    return _where_scalar_or_field(
        tf.greater_equal(result, result.dtype.max), inf_overwrite, result)

  return wrapped_reduce_op


# TODO(b/265760014): Remove together with pool_v1().
_REGISTERED_REDUCE_OPS = {
    'sum': tf.math.unsorted_segment_sum,
    'mean': tf.math.unsorted_segment_mean,
    'max': tf.math.unsorted_segment_max,
    'max_no_inf': with_minus_inf_replaced(tf.math.unsorted_segment_max, 0),
    'min': tf.math.unsorted_segment_min,
    'min_no_inf': with_plus_inf_replaced(tf.math.unsorted_segment_min, 0),
    'prod': tf.math.unsorted_segment_prod,
}


# TODO(b/265760014): Remove together with pool_v1().
def _resolve_reduce_op(reduce_type: str) -> UnsortedReduceOp:
  try:
    return _REGISTERED_REDUCE_OPS[reduce_type]
  except KeyError:
    raise ValueError(  # pylint: disable=raise-missing-from
        f'Unknown reduce type {reduce_type}. '
        f'Known reduce types are: {get_registered_reduce_operation_names()}')


# TODO(b/265760014): Move to pool_ops.py and rewrite for pool_v2().
def get_registered_reduce_operation_names() -> List[str]:
  """Returns the registered list of supported reduce operation names."""
  return list(_REGISTERED_REDUCE_OPS.keys())


# TODO(b/265760014): Remove together with pool_v1().
def _pool_to_context(graph_tensor: GraphTensor,
                     node_or_edge_set: Union[gt.NodeSet, gt.EdgeSet],
                     reduce_type: str,
                     *,
                     feature_value: Optional[gt.Field] = None,
                     feature_name: Optional[str] = None) -> gt.Field:
  """Aggregates (pools) node or edge value to graph context."""
  gt.check_scalar_graph_tensor(graph_tensor, 'tfgnn.pool_*_to_context()')
  assert feature_name is None or isinstance(feature_name, str)

  value = gt.resolve_value(
      node_or_edge_set, feature_value=feature_value, feature_name=feature_name)

  sizes = node_or_edge_set.sizes
  unsorted_reduce_op = _resolve_reduce_op(reduce_type)
  # TODO(b/184021442): cache result.
  return unsorted_reduce_op(
      value,
      utils.row_lengths_to_row_ids(
          sizes, sum_row_lengths_hint=node_or_edge_set.spec.total_size),
      utils.outer_dimension_size(sizes))


# TODO(b/265760014): Remove together with pool_v1().
def _where_scalar_or_field(condition: const.Field, true_scalar_value: tf.Tensor,
                           false_value: const.Field) -> const.Field:
  """Optimized tf.where for the scalar false side."""
  assert true_scalar_value.shape.rank == 0
  if utils.is_ragged_tensor(false_value):
    # tf.where specialization for the ragged tensors does not support scalar
    # inputs broadcasting in generic cases. As a workaround, we create the
    # ragged tensor with the same type spec as the false side but filled with
    # `true_scalar_value` values.
    # TODO(b/216278499): remove this workaround after fixing.
    true_flat_values = tf.fill(
        utils.dims_list(false_value.flat_values), true_scalar_value)
    true_value = false_value.with_flat_values(true_flat_values)
  else:
    true_value = true_scalar_value
  return tf.where(condition, true_value, false_value)


