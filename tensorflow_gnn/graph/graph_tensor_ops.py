"""Broadcasts and pools features between node sets, edge sets and context."""
import functools
from typing import Any, Callable, List, Mapping, Optional, Union
import tensorflow as tf

from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import tensor_utils as utils
from tensorflow_gnn.keras import keras_tensors as kt

Field = const.Field
FieldName = const.FieldName
NodeSetName = const.NodeSetName
EdgeSetName = const.EdgeSetName
IncidentNodeTag = const.IncidentNodeTag
GraphTensor = gt.GraphTensor
GraphKerasTensor = kt.GraphKerasTensor

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


# Combine operation (for a fixed list of tensors).
#
# Could be tf.math.add_n(....) or tf.concat(..., axis=-1)
#
# Args:
#  inputs: a list of Tensors or RaggedTensors.
#
# Returns:
#   A Tensor or RaggedTensor.
CombineOp = Callable[[List[Field]], Field]


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
  gt.check_scalar_graph_tensor(graph_tensor, 'tfgnn.broadcast_node_to_edges()')
  adjacency = graph_tensor.edge_sets[edge_set_name].adjacency
  node_name = adjacency.node_set_name(node_tag)
  node_value = resolve_value(
      graph_tensor.node_sets[node_name],
      feature_value=feature_value,
      feature_name=feature_name)
  return tf.gather(node_value, adjacency[node_tag])


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
      You may use `register_reduce_operation()` to register new ops.
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

  edge_value = resolve_value(
      graph_tensor.edge_sets[edge_set_name],
      feature_value=feature_value,
      feature_name=feature_name)

  adjacency = graph_tensor.edge_sets[edge_set_name].adjacency
  node_set = graph_tensor.node_sets[adjacency.node_set_name(node_tag)]
  total_node_count = node_set.spec.total_size
  if total_node_count is None:
    total_node_count = node_set.total_size
  return unsorted_reduce_op(edge_value, adjacency[node_tag], total_node_count)


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
      You may `register_reduce_operation()` to register new ops.
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
      You may `register_reduce_operation()` to register new ops.
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


def broadcast(graph_tensor: GraphTensor,
              from_tag: const.IncidentNodeOrContextTag,
              *,
              edge_set_name: Optional[EdgeSetName] = None,
              node_set_name: Optional[NodeSetName] = None,
              feature_value: Optional[Field] = None,
              feature_name: Optional[FieldName] = None) -> Field:
  """Broadcasts values from nodes to edges, or from context to nodes or edges.

  This function broadcasts from context if `from_tag=tfgnn.CONTEXT` and
  broadcasts from incident nodes to edges if `from_tag` is an ordinary node tag
  like `tfgnn.SOURCE` or `tfgnn.TARGET`. Most user code will not need this
  flexibility and can directly call one of the underlying functions
  `broadcast_node_to_edges()`, `broadcast_context_to_nodes()`, or
  `broadcast_context_to_edges()`.

  Args:
    graph_tensor: A scalar GraphTensor.
    from_tag: Values are broadcast from context if this is `tfgnn.CONTEXT` or
      from the incident node on each edge with this tag.
    edge_set_name: The name of the edge set to which values are broadcast.
    node_set_name: The name of the node set to which values are broadcast.
      Can only be set with `from_tag=tfgnn.CONTEXT`. Either edge_set_name or
      node_set_name must be set.
    feature_value: As for the underlying broadcast_*() function.
    feature_name: As for the underlying broadcast_*() function.
      Exactly one of feature_name or feature_value must be set.

  Returns:
    The result of the underlying broadcast_*() function.
  """
  _validate_names_and_tag(
      from_tag, edge_set_name=edge_set_name, node_set_name=node_set_name)
  if from_tag == const.CONTEXT:
    if node_set_name is not None:
      return broadcast_context_to_nodes(
          graph_tensor, node_set_name=node_set_name,
          feature_value=feature_value, feature_name=feature_name)
    else:
      return broadcast_context_to_edges(
          graph_tensor, edge_set_name=edge_set_name,
          feature_value=feature_value, feature_name=feature_name)
  else:
    return broadcast_node_to_edges(
        graph_tensor, edge_set_name=edge_set_name, node_tag=from_tag,
        feature_value=feature_value, feature_name=feature_name)


def pool(graph_tensor: GraphTensor,
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


def _validate_names_and_tag(tag, *, edge_set_name, node_set_name):
  """Helper for generic broadcast() and pool()."""
  if tag == const.CONTEXT:
    num_names = bool(edge_set_name is None) + bool(node_set_name is None)
    if num_names != 1:
      raise ValueError('With tag CONTEXT, must pass exactly 1 of '
                       f'edge_set_name, node_set_name; got {num_names}.')
  else:
    if edge_set_name is None or node_set_name is not None:
      raise ValueError('Must pass edge_set_name but not node_set_name '
                       'for a tag other than CONTEXT.')


def gather_first_node(graph_tensor: GraphTensor,
                      node_set_name: NodeSetName,
                      *,
                      feature_value: Optional[Field] = None,
                      feature_name: Optional[FieldName] = None) -> Field:
  """Gathers feature value from the first node of each graph component.

  Given a particular node set (identified by `node_set_name`), this operation
  will gather the given feature from the first node of each graph component.

  This is often used for rooted graphs created by sampling around the
  neighborhoods of seed nodes in a large graph: by convention, each seed node is
  the first node of its component in the respective node set, and this operation
  reads out the information it has accumulated there. (In other node sets, the
  first node may be arbitrary -- or nonexistant, in which case this operation
  must not be used and may raise an error at runtime.)

  The feature to fetch node values from is provided either by name (using
  `feature_name`) and found in the graph tensor itself, or provided explicitly
  (using `feature_value`) in which case its shape has to be compatible with the
  shape prefix of the node set being gathered from. One of `feature_value`
  or `feature_name` must be specified.

  Args:
    graph_tensor: A scalar GraphTensor.
    node_set_name: A seed node set name.
    feature_value: A ragged or dense node feature value. Has a shape
      `[num_nodes, *feature_shape]`, where `num_nodes` is the number of nodes in
      the `node_set_name` node set and `feature_shape` is the shape of the
      feature value for each node.
    feature_name: A node feature name.

  Returns:
    A tensor of gathered feature values, one for each graph component, like a
    context feature. Has a shape `[num_components, *feature_shape]`, where
    `num_components` is the number of components in a graph and `feature_shape`
    is not affected.
  """
  gt.check_scalar_graph_tensor(graph_tensor, 'tfgnn.gather_first_node()')
  node_set = graph_tensor.node_sets[node_set_name]
  node_value = resolve_value(
      node_set, feature_value=feature_value, feature_name=feature_name)

  sizes = node_set.sizes
  assert_positive_sizes = tf.debugging.assert_positive(
      sizes,
      message=f'tfgnn.gather_first_node(..., node_set_name={node_set_name}) '
      'called for a graph in which one or more components contain no nodes.')
  with tf.control_dependencies([assert_positive_sizes]):
    components_starts = tf.math.cumsum(sizes, exclusive=True)
    return tf.gather(node_value, components_starts)


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


_REGISTERED_REDUCE_OPS = {
    'sum': tf.math.unsorted_segment_sum,
    'mean': tf.math.unsorted_segment_mean,
    'max': tf.math.unsorted_segment_max,
    'max_no_inf': with_minus_inf_replaced(tf.math.unsorted_segment_max, 0),
    'min': tf.math.unsorted_segment_min,
    'min_no_inf': with_plus_inf_replaced(tf.math.unsorted_segment_min, 0),
    'prod': tf.math.unsorted_segment_prod,
}


def _resolve_reduce_op(reduce_type: str) -> UnsortedReduceOp:
  try:
    return _REGISTERED_REDUCE_OPS[reduce_type]
  except KeyError:
    raise ValueError(  # pylint: disable=raise-missing-from
        f'Unknown reduce type {reduce_type}. '
        f'Known reduce types are: {get_registered_reduce_operation_names()}')


def get_registered_reduce_operation_names() -> List[str]:
  """Returns the registered list of supported reduce operation names."""
  return list(_REGISTERED_REDUCE_OPS.keys())


def register_reduce_operation(reduce_type: str,
                              *,
                              unsorted_reduce_op: UnsortedReduceOp,
                              allow_override: bool = False) -> None:
  """Register a new reduction operation for pooling.

  This function can be used to insert a reduction operation in the supported
  list of `reduce_type` aggregations for all the pooling operations defined in
  this module.

  Args:
    reduce_type: A pooling operation name. This name must not conflict with the
      existing pooling operations in the registry, except if `allow_override` is
      set. For the full list of supported values use
      `get_registered_reduce_operation_names()`.
    unsorted_reduce_op: The TensorFlow op for reduction. This op does not rely
      on sorting over edges.
    allow_override: A boolean flag to allow overwriting the existing registry of
      operations. Use this with care.
  """
  if not allow_override and reduce_type in _REGISTERED_REDUCE_OPS:
    raise ValueError(
        'Reduce operation with that name already exists.'
        ' Call `get_registered_reduce_operation_names()` for the full'
        ' list of registered operations or set `override_existing=True`'
        ' to allow to redefine existing operations.')
  assert callable(unsorted_reduce_op)
  _REGISTERED_REDUCE_OPS[reduce_type] = unsorted_reduce_op


def combine_values(inputs: List[Field], combine_type: str) -> Field:
  """Combines a list of tensors into one (by concatenation or otherwise).

  This is a convenience wrapper around standard TensorFlow operations, to
  provide standard names for common types of combining.

  Args:
    inputs: a list of Tensors or RaggedTensors, with shapes and types that are
      compatible for the selected combine_type.
    combine_type: one of the following string values, to select the method for
      combining the inputs:

        * "sum": The input tensors are added. Their dtypes and shapes must
          match.
        * "concat": The input tensors are concatenated along the last axis.
          Their dtypes and shapes must match, except for the number of elements
          along the last axis.

  Returns:
    A tensor with the combined value of the inputs.
  """
  combine_op = _resolve_combine_op(combine_type)
  try:
    result = combine_op(inputs)
  except tf.errors.InvalidArgumentError as e:
    raise tf.errors.InvalidArgumentError(
        e.node_def, e.op,
        f'tfgnn.combine_values() failed for combine_type="{combine_type}". '
        'Please check that all inputs have matching structures, except '
        'as allowed by the combine_type. (The last dimension may differ for '
        '"concat" but not for "sum").\n'
        f'Inputs were: {inputs}'
    ) from e
  return result


_COMBINE_OPS = {
    'sum': tf.math.add_n,
    'concat': functools.partial(tf.concat, axis=-1),
}


def _resolve_combine_op(combine_type: str) -> CombineOp:
  try:
    return _COMBINE_OPS[combine_type]
  except KeyError:
    raise ValueError(  # pylint: disable=raise-missing-from
        f'Unknown combine type {combine_type}. '
        f'Known combine types are: {list(_COMBINE_OPS.keys())}')


def is_graph_tensor(value: Any) -> bool:
  """Returns whether `value` is a GraphTensor (possibly wrapped for Keras)."""
  return isinstance(value, (GraphTensor, GraphKerasTensor))


def shuffle_scalar_components(graph_tensor: GraphTensor,
                              *,
                              seed: Optional[int] = None) -> GraphTensor:
  """Shuffles context, node set and edge set features across components.

  Args:
    graph_tensor: A scalar GraphTensor.
    seed: A seed for random uniform shuffle.

  Returns:
    A scalar GraphTensor with its component's features shuffled.
  """
  gt.check_scalar_graph_tensor(graph_tensor,
                               'tfgnn.shuffle_scalar_components()')

  context = _shuffle_features(graph_tensor.context.features, seed=seed)
  node_sets, edge_sets = {}, {}

  for node_set_name, node_set in graph_tensor.node_sets.items():
    node_sets[node_set_name] = _shuffle_features(node_set.features, seed=seed)

  for edge_set_name, edge_set in graph_tensor.edge_sets.items():
    edge_sets[edge_set_name] = _shuffle_features(edge_set.features, seed=seed)

  return graph_tensor.replace_features(context, node_sets, edge_sets)


def _shuffle_features(features: Mapping[FieldName, Field],
                      *,
                      seed: Optional[int] = None) -> Mapping[FieldName, Field]:
  """Shuffles dense or ragged features.

  Args:
    features: A mapping FieldName to Field.
    seed: A seed for random uniform shuffle.

  Returns:
    A mapping FieldName to Field with Field shuffled across its outer dimension.
  """

  def _shuffle(feature_name):
    feature_value = features[feature_name]
    if utils.is_dense_tensor(feature_value):
      return tf.random.shuffle(feature_value, seed=seed)
    elif utils.is_ragged_tensor(feature_value):
      value_rowids = feature_value.value_rowids()
      shuffled_value_rowids = tf.random.shuffle(value_rowids, seed=seed)
      values = feature_value.values
      new_values = tf.gather(values, tf.argsort(shuffled_value_rowids))
      return feature_value.with_values(new_values)
    else:
      raise ValueError(
          'Operation is currently supported only for dense or ragged tensors, '
          f'got feature_name={feature_name} and feature_value={feature_value}.')

  return {
      feature_name: _shuffle(feature_name) for feature_name in features.keys()
  }


def resolve_value(values_map: Any,
                  *,
                  feature_value: Optional[Field] = None,
                  feature_name: Optional[str] = None) -> Field:
  """Resolves feature value by its name or provided value."""
  if (feature_value is None) == (feature_name is None):
    raise ValueError('One of feature name of feature value must be specified.')

  if feature_value is not None:
    # TODO(b/189087785): check that value shape is valid.
    return feature_value
  if feature_name is not None:
    return values_map[feature_name]
  assert False, 'This should never happen, please file a bug with TF-GNN.'


def _broadcast_context(graph_tensor: GraphTensor,
                       node_or_edge_set: Union[gt.NodeSet, gt.EdgeSet],
                       *,
                       feature_value: Optional[Field] = None,
                       feature_name: Optional[FieldName] = None) -> Field:
  """Broadcasts context value to graph node or edge."""

  gt.check_scalar_graph_tensor(graph_tensor, 'tfgnn.broadcast_context_to_*()')

  context_value = resolve_value(
      graph_tensor.context,
      feature_value=feature_value,
      feature_name=feature_name)

  # TODO(b/184021442): cache result.
  return utils.repeat(
      context_value,
      node_or_edge_set.sizes,
      repeats_sum_hint=node_or_edge_set.spec.total_size)


def _pool_to_context(graph_tensor: GraphTensor,
                     node_or_edge_set: Union[gt.NodeSet, gt.EdgeSet],
                     reduce_type: str,
                     *,
                     feature_value: Optional[gt.Field] = None,
                     feature_name: Optional[str] = None) -> gt.Field:
  """Aggregates (pools) node or edge value to graph context."""
  gt.check_scalar_graph_tensor(graph_tensor, 'tfgnn.pool_*_to_context()')
  assert feature_name is None or isinstance(feature_name, str)

  value = resolve_value(
      node_or_edge_set, feature_value=feature_value, feature_name=feature_name)

  sizes = node_or_edge_set.sizes
  unsorted_reduce_op = _resolve_reduce_op(reduce_type)
  # TODO(b/184021442): cache result.
  return unsorted_reduce_op(
      value,
      utils.row_lengths_to_row_ids(
          sizes, sum_row_lengths_hint=node_or_edge_set.spec.total_size),
      utils.outer_dimension_size(sizes))


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
