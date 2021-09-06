"""Broadcasts and pools features between node sets, edge sets and graph context.

See the `operations.md` document in the guide for an explanation of how to use
this.
"""
from typing import Any, Callable, List, Optional, Union
import tensorflow as tf

from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import tensor_utils as utils
from tensorflow_gnn.graph.keras import keras_tensors as kt

Field = const.Field
FieldName = const.FieldName
NodeSetName = const.NodeSetName
EdgeSetName = const.EdgeSetName
IncidentNodeTag = const.IncidentNodeTag
GraphTensor = gt.GraphTensor
GraphKerasTensor = kt.GraphKerasTensor

# Unsorted reduce operation.
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


def broadcast_node_to_edges(graph_tensor: GraphTensor,
                            edge_set_name: EdgeSetName,
                            node_tag: IncidentNodeTag,
                            *,
                            feature_value: Optional[Field] = None,
                            feature_name: Optional[FieldName] = None) -> Field:
  """Broadcasts values from nodes to incident edges.

  Given a particular edge set (identified by `edge_set_name` name), this
  operation collects node features from the specific incident node of each edge
  (as indicated by `node_tag`). For example, setting `node_tag=SOURCE` and
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
      specified by its tag in the edge set (e.g. `gt.SOURCE`, `gt.TARGET`).
    feature_value: A ragged or dense source node feature values. Has a shape
      `[n, f1..fk]`, where `n` index source nodes; `f1..fk` features inner
      dimensions.
    feature_name: A source node feature name.

  Returns:
    Source node value broadcast to corresponding edges, with shape `[e,
    f1..fk]`, where `e` indexes edges; the inner `f1..fk` feature dimensions are
    not affected.
  """
  _assert_scalar_graph_tensor(graph_tensor)
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
  indicated by `node_tag`). For example, setting `node_tag=TARGET` and
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
      You may use `register_reduce_operation(..)` to register new ops.
    feature_value: A ragged or dense edge feature value.
    feature_name: An edge feature name.

  Returns:
    The edge values pooled to each incident node. The first dimension size
    represents the number of nodes; the further dimensions do not change.
  """
  _assert_scalar_graph_tensor(graph_tensor)
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
    feature_value: A ragged or dense graph context feature value.
    feature_name: A context feature name.

  Returns:
    Graph context value broadcast to the `node_set` nodes. The first dimension
    size equals to the number of nodes, the higher dimensions do not change.
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
    feature_value: A ragged or dense graph context feature value.
    feature_name: A context feature name.

  Returns:
    Graph context value broadcast to the `edge_set` edges. The first dimension
    size equals to the number of nodes, the higher dimensions do not change.
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
      You may `register_reduce_operation(..)` to register new ops.
    feature_value: A ragged or dense node feature value.
    feature_name: A node feature name.

  Returns:
    Node value pooled to graph context. The first dimension size equals to the
    first context dimension (number of graph components), the higher dimensions
    do not change.
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
      You may `register_reduce_operation(..)` to register new ops.
    feature_value: A ragged or dense edge feature value.
    feature_name: An edge feature name.

  Returns:
    A node value pooled to graph context. The first dimension size equals to the
    first context dimension (number of graph components), the higher dimensions
    do not change.
  """
  return _pool_to_context(
      graph_tensor,
      graph_tensor.edge_sets[edge_set_name],
      reduce_type,
      feature_value=feature_value,
      feature_name=feature_name)


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
    feature_value: A ragged or dense node feature value.
    feature_name: A node feature name.

  Returns:
    A tensor of gathered feature values, one for each graph component, like a
    context feature.
  """
  _assert_scalar_graph_tensor(graph_tensor)
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


# TODO(b/184021014): provide min/max pooling with NN friendly default values
# (currently this is inf and -inf).
# See https://www.tensorflow.org/api_docs/python/tf/math/unsorted_segment_min
_REGISTERED_REDUCE_OPS = {
    'sum': tf.math.unsorted_segment_sum,
    'mean': tf.math.unsorted_segment_mean,
    'max': tf.math.unsorted_segment_max,
    'min': tf.math.unsorted_segment_min,
    'prod': tf.math.unsorted_segment_prod,
}


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


def is_graph_tensor(value: Any) -> bool:
  """Returns whether `value` is a GraphTensor (possibly wrapped for Keras)."""
  return isinstance(value, (GraphTensor, GraphKerasTensor))


def _resolve_reduce_op(reduce_type: str) -> UnsortedReduceOp:
  try:
    return _REGISTERED_REDUCE_OPS[reduce_type]
  except KeyError:
    raise ValueError(
        f'Unknown reduce type {reduce_type}.'
        ' Call `get_registered_reduce_operation_names()` for the full'
        ' list of registered operations.')


def _assert_scalar_graph_tensor(graph_tensor: GraphTensor) -> None:
  if graph_tensor.rank != 0:
    raise ValueError((
        'Operation is currently supported only for scalar (rank=0) GraphTensor,'
        f' got rank={graph_tensor.rank}.'
        ' Use graph_tensor.merge_batch_to_components() to convert an arbitrary'
        ' graph tensor to a scalar graph tensor.'))


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

  _assert_scalar_graph_tensor(graph_tensor)

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
  _assert_scalar_graph_tensor(graph_tensor)
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
