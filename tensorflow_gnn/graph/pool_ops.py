# Copyright 2023 The TensorFlow GNN Authors. All Rights Reserved.
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
"""The pool operations on a GraphTensor."""

from __future__ import annotations
import abc
import functools
from typing import Optional, Sequence, Union

import tensorflow as tf

from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import tag_utils
from tensorflow_gnn.graph import tensor_utils as utils
from tensorflow_gnn.keras import keras_tensors as kt

Field = const.Field
FieldName = const.FieldName
NodeSetName = const.NodeSetName
EdgeSetName = const.EdgeSetName
IncidentNodeTag = const.IncidentNodeTag
IncidentNodeOrContextTag = const.IncidentNodeOrContextTag
GraphTensor = gt.GraphTensor
GraphTensorSpec = gt.GraphTensorSpec


def pool_edges_to_node(graph_tensor: GraphTensor,
                       edge_set_name: EdgeSetName,
                       node_tag: IncidentNodeTag,
                       reduce_type: str = "sum",
                       *,
                       feature_value: Optional[Field] = None,
                       feature_name: Optional[FieldName] = None) -> Field:
  """Aggregates (pools) edge values to incident nodes.

  Given a particular edge set (identified by `edge_set_name` name), this
  operation reduces edge features at the specific incident node of each edge (as
  indicated by `node_tag`). For example, setting `node_tag=tfgnn.TARGET` and
  `reduce_type="sum"` computes the sum over the incoming edge features at each
  node, while `reduce_type="sum|mean"` would compute the concatenation of their
  sum and mean along the innermost axis, in this order.

  For the converse operation of broadcasting from nodes to incident edges,
  see `tfgnn.broadcast_node_to_edges()`. For a generalization beyond a single
  edge set, see `tfgnn.pool()`.

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
    reduce_type: A pooling operation name like `"sum"` or `"mean"`, or a
      `|`-separated combination of these; see `tfgnn.pool()`.
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
  gt.check_scalar_graph_tensor(graph_tensor, "tfgnn.pool_edges_to_node()")
  if not isinstance(edge_set_name, str):
    raise TypeError(
        "pool_edges_to_node() requires edge_set_name to be a string, "
        f"got {edge_set_name}.")
  return pool_v2(graph_tensor, node_tag,
                 edge_set_name=edge_set_name, reduce_type=reduce_type,
                 feature_value=feature_value, feature_name=feature_name)


def pool_nodes_to_context(graph_tensor: GraphTensor,
                          node_set_name: NodeSetName,
                          reduce_type: str = "sum",
                          *,
                          feature_value: Optional[Field] = None,
                          feature_name: Optional[FieldName] = None) -> Field:
  """Aggregates (pools) node values to graph context.

  Given a particular node set (identified by `node_set_name`), this operation
  reduces node features to their corresponding graph component. For example,
  setting `reduce_type="sum"` computes the sum over the node features of each
  graph, while `reduce_type="sum|mean"` would compute the concatenation of their
  sum and mean along the innermost axis, in this order.

  For the converse operation of broadcasting from context to nodes, see
  `tfgnn.broadcast_context_to_nodes()`. For a generalization beyond a single
  node set, see `tfgnn.pool()`.

  The feature to fetch node values from is provided either by name (using
  `feature_name`) and found in the graph tensor itself, or provided explicitly
  (using `feature_value`). One of `feature_value` or `feature_name` must be
  specified.

  Args:
    graph_tensor: A scalar GraphTensor.
    node_set_name: A node set name.
    reduce_type: A pooling operation name, like `"sum"` or `"mean"`, or a
      `|`-separated combination of these; see `tfgnn.pool()`.
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
  gt.check_scalar_graph_tensor(graph_tensor, "tfgnn.pool_nodes_to_context()")
  if not isinstance(node_set_name, str):
    raise TypeError(
        "pool_nodes_to_context() requires node_set_name to be a string, "
        f"got {node_set_name}.")
  return pool_v2(graph_tensor,
                 const.CONTEXT,
                 node_set_name=node_set_name,
                 reduce_type=reduce_type,
                 feature_value=feature_value,
                 feature_name=feature_name)


def pool_edges_to_context(graph_tensor: GraphTensor,
                          edge_set_name: EdgeSetName,
                          reduce_type: str = "sum",
                          *,
                          feature_value: Optional[Field] = None,
                          feature_name: Optional[FieldName] = None) -> Field:
  """Aggregates (pools) edge values to graph context.

  Given a particular edge set (identified by `edge_set_name`), this operation
  reduces edge features to their corresponding graph component. For example,
  setting `reduce_type="sum"` computes the sum over the edge features of each
  graph, while `reduce_type="sum|mean"` would compute the concatenation of their
  sum and mean along the innermost axis, in this order.

  For the converse operation of broadcasting from context to edges, see
  `tfgnn.broadcast_context_to_edges()`. For a generalization beyond a single
  edge set, see `tfgnn.pool()`.

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
    reduce_type: A pooling operation name, like `"sum"` or `"mean"`, or a 
      `|`-separated combination of these; see `tfgnn.pool()`.
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
  gt.check_scalar_graph_tensor(graph_tensor, "tfgnn.pool_edges_to_context()")
  if not isinstance(edge_set_name, str):
    raise TypeError(
        "pool_edges_to_context() requires edge_set_name to be a string, "
        f"got {edge_set_name}.")
  return pool_v2(graph_tensor,
                 const.CONTEXT,
                 edge_set_name=edge_set_name,
                 reduce_type=reduce_type,
                 feature_value=feature_value,
                 feature_name=feature_name)


def pool_v2(
    graph_tensor: GraphTensor,
    to_tag: IncidentNodeOrContextTag,
    *,
    edge_set_name: Union[Sequence[EdgeSetName], EdgeSetName, None] = None,
    node_set_name: Union[Sequence[NodeSetName], NodeSetName, None] = None,
    reduce_type: str,
    feature_value: Union[Sequence[Field], Field, None] = None,
    feature_name: Optional[FieldName] = None) -> Field:
  """Pools values from edges to nodes, or from nodes or edges to context.

  This function pools to context if `to_tag=tfgnn.CONTEXT` and pools from edges
  to incident nodes if `to_tag` is an ordinary node tag like `tfgnn.SOURCE` or
  `tfgnn.TARGET`.

  The `edge_set_name` (or `node_set_name`, when pooling to context) can be set
  to a single name, or to a non-empty list of names. Pooling is done as if
  all named edge sets (or node sets) were concatenated into a single edge set
  (or node set).
  For example, `pool(reduce_type="mean", edge_sets=["a", "b"], ...)` will form
  the sum over all edges in "a" and "b" and divide by their total number, giving
  equal weight to each.

  The following choices of `reduce_type` are supported:

  | `reduce_type` | Description                                               |
  |---------------|-----------------------------------------------------------|
  | `"sum"`       | element-wise sum of input tensors                         |
  | `"prod"`      | element-wise product of input tensors (beware of overflow)|
  | `"mean"`      | element-wise mean (average), or zero for no inputs        |
  | `"max"`       | element-wise maximum, or `-inf` for no inputs             |
  | `"max_no_inf"`| element-wise maximum, or zero for no inputs               |
  | `"min"`       | element-wise minimum, or `-inf` for no inputs             |
  | `"min_no_inf"`| element-wise minimum, or zero for no inputs               |

  The helper function `tfgnn.get_registered_reduce_operation_names()` returns
  a list of these values.

  Moreover, `reduce_type` can be set to a `|`-separated list of reduce types,
  such as `reduce_type="mean|sum"`, which will return the concatenation of
  their individual results along the innermost axis in the order of appearance.

  TODO(b/286005254): pool() from multiple edge sets (or node sets) does not yet
  support RaggedTensors.

  Args:
    graph_tensor: A scalar GraphTensor.
    to_tag: Values are pooled to context if this is `tfgnn.CONTEXT` or to the
      incident node on each edge with this tag.
    edge_set_name: The name of the edge set from which values are pooled, or
      a non-empty sequence of such names. Unless `to_tag=tfgnn.CONTEXT`,
      all named edge sets must have the same incident node set at the given tag.
    node_set_name: The name of the node set from which values are pooled,
      or a non-empty sequence of such names. Can only be set with
      `to_tag=tfgnn.CONTEXT`. Exactly one of edge_set_name or node_set_name
      must be set.
    reduce_type: A string with the name of a pooling operation, or multiple ones
      separated by `|`. See the table above for the known names.
    feature_value: A tensor or list of tensors, parallel to the node_set_names
      or edge_set_names, to supply the input values of pooling. Each tensor
      has shape `[num_items, *feature_shape]`, where `num_items` is the number
      of edges in the given edge set or nodes in the given node set, and
      `*feature_shape` is the same across all inputs. The `*feature_shape` may
      contain ragged dimensions. All the ragged values that are reduced onto
      any one item of the graph must have the same ragged index structure,
      so that a result can be computed from them.
    feature_name: The name of a feature stored on each graph piece from which
      pooling is done, for use instead of an explicity passed feature_value.
      Exactly one of feature_name or feature_value must be set.

  Returns:
    A tensor with the result of pooling from the conceptual concatenation of the
    named edge set(s) or node set(s) to the destination selected by `to_tag`.
    Its shape is `[num_items, *feature_shape]`, where `num_items` is the number
    of destination nodes (or graph components if `to_tag=tfgnn.CONTEXT`)
    and `*feature_shape` is as for all the inputs.
  """
  gt.check_scalar_graph_tensor(graph_tensor, "pool()")

  edge_set_names, node_set_names, feature_values, _ = (
      get_pool_args_as_sequences(
          graph_tensor, to_tag,
          edge_set_name=edge_set_name, node_set_name=node_set_name,
          feature_value=feature_value, feature_name=feature_name,
          function_name="pool()"))
  del edge_set_name, node_set_name, feature_value  # Use canonicalized forms.

  if len(feature_values) > 1 and any(
      utils.is_ragged_tensor(fv) for fv in feature_values):
    raise ValueError(
        "TODO(b/286005254): pool() from multiple edge sets (or node sets) "
        "does not (yet?) support RaggedTensors.")

  if not reduce_type:
    raise ValueError("pool() requires one more more reduce types, "
                     f"separated by '|', but got '{reduce_type}'.")

  # Catch incompatible input shapes early, and with a clear message.
  # GraphTensor forbids None in feature dims, except for ragged dimensions.
  feature_shapes = [fv.shape[1:] for fv in feature_values]
  if not all(feature_shapes[0] == feature_shapes[i]
             for i in range(1, len(feature_shapes))):
    if feature_name:
      msg_lines = [
          f"Cannot pool() incompatible shapes of feature '{feature_name}':"]
    else:
      msg_lines = ["Cannot pool() incompatible feature shapes:"]
    if edge_set_names is not None:
      msg_lines.extend([
          f"  edge set '{name}' has feature shape {shape.as_list()}"
          for name, shape in zip(edge_set_names, feature_shapes)])
    else:
      msg_lines.extend([
          f"  node set '{name}' has feature shape {shape.as_list()}"
          for name, shape in zip(node_set_names, feature_shapes)])
    raise ValueError("\n".join(msg_lines))

  return _pool_internal(
      graph_tensor, to_tag,
      edge_set_names=edge_set_names, node_set_names=node_set_names,
      reduce_type=reduce_type, feature_values=feature_values)


def _pool_internal(
    graph: GraphTensor,
    to_tag: IncidentNodeOrContextTag,
    *,
    edge_set_names: Optional[Sequence[EdgeSetName]] = None,
    node_set_names: Optional[Sequence[NodeSetName]] = None,
    reduce_type: str,
    feature_values: Sequence[Field]) -> Field:
  """Returns pool() result from canonicalized args."""

  # Decide how to compute each requested reduce_type.
  reduce_types = reduce_type.split("|")
  if len(feature_values) != 1:
    # In the general case, all outputs are computed by MultiReducers.
    reduce_types_multi = set(reduce_types)
    reduce_types_single = set()
  else:
    # In case of reducing from a single graph piece, outputs could mostly be
    # computed directly by GraphPieceReducers, but some special cases benefit
    # from using a MultiReducer instead.
    reduce_types_multi = set()
    reduce_types_single = set(reduce_types)
    def _promote_from_single_to_multi(rt):
      reduce_types_single.remove(rt)
      reduce_types_multi.add(rt)
    # If "sum" is computed anyways, compute "mean" from "sum" and "_count".
    if "mean" in reduce_types and "sum" in reduce_types:
      _promote_from_single_to_multi("mean")
      _promote_from_single_to_multi("sum")
    # "max_no_inf" and "min_no_inf" are only available as MultiReducers.
    # TODO(b/286007330): Automate this kind of fallback.
    if "max_no_inf" in reduce_types:
      _promote_from_single_to_multi("max_no_inf")
    if "min_no_inf" in reduce_types:
      _promote_from_single_to_multi("min_no_inf")

  # Find the MultiReducers to be evaluated.
  multi_reducers = {rt: _MULTI_REDUCER_CLASSES[rt]()
                    for rt in reduce_types_multi}

  # Find the names of PieceReducers to be evaluated.
  piece_reducer_names = reduce_types_single.copy()
  for multi_reducer in multi_reducers.values():
    piece_reducer_names.update(multi_reducer.get_piece_reducer_names())

  # For each named PieceReducer, compute the list of its reduction results
  # on the list of input graph pieces.
  reduced_pieces = {}
  for piece_reducer_name in piece_reducer_names:
    piece_reducer = _GRAPH_PIECE_REDUCER_CLASSES[piece_reducer_name]()
    piece_reducer_results = []
    for i in range(len(feature_values)):
      piece_reducer_results.append(piece_reducer.reduce(
          graph, to_tag,
          feature_value=feature_values[i],
          edge_set_name=edge_set_names[i] if edge_set_names else None,
          node_set_name=node_set_names[i] if node_set_names else None))
    reduced_pieces[piece_reducer_name] = piece_reducer_results

  # For each requested reduce_type, in the user-requested order,
  # get its result according to how it was computed.
  reductions = []
  for rt in reduce_types:
    if rt in reduce_types_multi:
      multi_reducer = multi_reducers[rt]
      reductions.append(multi_reducer.compute_from_pieces(reduced_pieces))
    else:
      assert rt in reduce_types_single  # Only if reducing a single graph piece.
      (reduced_piece,) = reduced_pieces[rt]  # Unpack.
      reductions.append(reduced_piece)

  # Return the concatenated results.
  if len(reductions) == 1:
    return reductions[0]
  else:
    ranks = [r.shape.rank for r in reductions]
    unique_ranks = set(ranks)
    if len(unique_ranks) > 1 or next(iter(unique_ranks)) in [None, 0, 1]:
      raise ValueError(
          f"pool() with multiple reduce_types {reduce_type} requires all their "
          "results to have the same known rank >= 2, to allow concatenation "
          f"along the last feature axis, but got ranks {ranks}")
    return tf.concat(reductions, axis=-1)


def get_pool_args_as_sequences(
    graph: GraphTensor,
    tag: IncidentNodeOrContextTag,
    *,
    edge_set_name: Union[Sequence[EdgeSetName], EdgeSetName, None] = None,
    node_set_name: Union[Sequence[NodeSetName], NodeSetName, None] = None,
    feature_value: Union[Sequence[Field], Field, None] = None,
    feature_name: Optional[FieldName] = None,
    function_name: str = "This operation",
) -> tuple[Optional[Sequence[EdgeSetName]],
           Optional[Sequence[NodeSetName]],
           Sequence[Field],
           bool]:
  """Returns pool()-style args checked and with canonicalized types.

  Args:
    graph: The `GraphTensor`, as for `pool()`.
    tag: Same as for `pool()`.
    edge_set_name: As for `pool()`, can be set to a name or sequence of names.
    node_set_name: As for `pool()`, can be set to a name or sequence of names.
    feature_value: As for `pool()`, can be set to a value or sequence of values.
    feature_name: As for `pool()`, can be set to a feature name.
    function_name: Optionally, the user-visible name of the function whose args
      are processed.

  Returns:
    Tuple `(edge_set_names, node_set_names, feature_values, got_sequence_args)`
    with exactly one of `edge_set_names, node_set_names` being a list and
    the other being `None`. `feature_values` is a list of matching length,
    possibly looked by `feature_name` if not originally given.
    `got_sequence_args` is set to False if original non-sequence args have
    been converted to lists of length 1.

  Raises:
    ValueError: if not exactly one of edge_set_name, node_set_name is set.
    ValueError: if node_set_name is set for a `tag != tfgnn.CONTEXT`.
    ValueError: if the given edge_set_names have different endpoints at
      the given `tag != tfgnn.CONTEXT`.
    ValueError: if not exactly one of feature_value, feature_name is set.
    ValueError: if feature_value and node/edge_set name disagree about being
      a single value or a sequence of the same length.
  """
  edge_set_names, node_set_names, got_sequence_args = (
      tag_utils.get_edge_or_node_set_name_args_for_tag(
          graph.spec, tag,
          edge_set_name=edge_set_name, node_set_name=node_set_name,
          function_name=function_name))

  if (feature_value is None) == (feature_name is None):
    raise ValueError(
        f"{function_name} requires exactly one of feature_name, feature_value.")
  if feature_value is not None:
    if isinstance(feature_value, Sequence) != got_sequence_args:
      raise ValueError(
          f"{function_name} allows a Sequence as a feature_values kwarg "
          "if and only if the edge/node_set_names are also a Sequence ")
  if isinstance(feature_value, Sequence):
    feature_values = feature_value
  elif feature_value is not None:
    feature_values = [feature_value]
  elif edge_set_names is not None:
    feature_values = [graph.edge_sets[edge_set_name][feature_name]
                      for edge_set_name in edge_set_names]
  else:
    feature_values = [graph.node_sets[node_set_name][feature_name]
                      for node_set_name in node_set_names]
  if edge_set_names is not None:
    _check_same_length("edge_set_names", edge_set_names, feature_values)
  else:
    _check_same_length("node_set_names", node_set_names, feature_values)

  return edge_set_names, node_set_names, feature_values, got_sequence_args


def _check_same_length(
    piece_arg_name: str,
    piece_names: Sequence[str],
    feature_values: Sequence[Field]) -> None:
  if len(piece_names) != len(feature_values):
    raise ValueError(
        f"pool() requires the same number of {piece_arg_name} and "
        f"feature_values but got {len(piece_names)} and {len(feature_values)}.")


class GraphPieceReducer(abc.ABC):
  """Base class to implement pool() for one reduce_type from one graph piece.

  This base class implements a `reduce()` method for pooling from one
  graph piece (one edge set into a node set, or one node/edge set into
  context) by dispatching onto the kind of TF op that is suitable for the
  adjacency structure at hand, say, `unsorted_segment_{sum,max,...}`.

  Subclasses implement methods like `unsorted_segment_op()` to supply the
  actual TF ops for their respective operation (sum, max, ...).
  Subclasses are usually looked up in_GRAPH_PIECE_REDUCER_CLASSES.

  Note that calling pool() on multiple graph pieces and/or with multiple
  reduce types may translate non-trivially into reductions of individual
  graph pieces. For example, "mean" across multiple edge sets translates
  to "sum" and "_count" on all of them.
  """

  ##
  ## PUBLIC INTERFACE for use by pool() etc.
  ##
  def reduce(
      self,
      graph: GraphTensor,
      to_tag: IncidentNodeOrContextTag,
      *,
      edge_set_name: Optional[EdgeSetName] = None,
      node_set_name: Optional[NodeSetName] = None,
      feature_value: Field) -> Field:
    """Returns pooled feature values of the given graph piece."""
    gt.check_scalar_graph_tensor(graph)

    # Pooling to context.
    if to_tag == const.CONTEXT:
      if edge_set_name is not None:
        node_or_edge_set = graph.edge_sets[edge_set_name]
      else:
        node_or_edge_set = graph.node_sets[node_set_name]
      sizes = node_or_edge_set.sizes
      return self.unsorted_segment_op(
          feature_value,
          utils.row_lengths_to_row_ids(
              sizes, sum_row_lengths_hint=node_or_edge_set.spec.total_size),
          utils.outer_dimension_size(sizes))

    # Pooling from edges to node.
    adjacency = graph.edge_sets[edge_set_name].adjacency
    if isinstance(adjacency, (kt.HyperAdjacencyKerasTensor,  # TODO(b/283404258)
                              adj.HyperAdjacency)):
      node_set = graph.node_sets[adjacency.node_set_name(to_tag)]
      total_node_count = node_set.spec.total_size
      if total_node_count is None:
        total_node_count = node_set.total_size
      return self.unsorted_segment_op(feature_value,
                                      adjacency[to_tag],
                                      total_node_count)
    else:
      raise ValueError(f"Edge set '{edge_set_name}' has unknown "
                       f"adjacency type {type(adjacency).__name__}")

  ##
  ## SUBCLASS INTERFACE
  ##
  @abc.abstractmethod
  def unsorted_segment_op(
      self,
      values: Field,
      segment_ids: tf.Tensor,
      num_segments: tf.Tensor)-> Field:
    raise NotImplementedError("To be implemented by op-specific subclass.")


class CountGraphPieceReducer(GraphPieceReducer):
  """Implements count-pooling from one graph piece."""

  def unsorted_segment_op(self,
                          values: Field,
                          segment_ids: tf.Tensor,
                          num_segments: tf.Tensor)-> Field:
    """Implements subclass API."""
    ones = tf.ones(tf.shape(values)[0], dtype=values.dtype)
    return tf.math.unsorted_segment_sum(ones, segment_ids, num_segments)


class MaxGraphPieceReducer(GraphPieceReducer):
  """Implements max-pooling from one graph piece."""

  def unsorted_segment_op(self,
                          values: Field,
                          segment_ids: tf.Tensor,
                          num_segments: tf.Tensor) -> Field:
    """Implements subclass API."""
    return tf.math.unsorted_segment_max(values, segment_ids, num_segments)


class MeanGraphPieceReducer(GraphPieceReducer):
  """Implements mean-pooling from one graph piece."""

  def unsorted_segment_op(self,
                          values: Field,
                          segment_ids: tf.Tensor,
                          num_segments: tf.Tensor) -> Field:
    """Implements subclass API."""
    return tf.math.unsorted_segment_mean(values, segment_ids, num_segments)


class MinGraphPieceReducer(GraphPieceReducer):
  """Implements min-pooling from one graph piece."""

  def unsorted_segment_op(self,
                          values: Field,
                          segment_ids: tf.Tensor,
                          num_segments: tf.Tensor) -> Field:
    """Implements subclass API."""
    return tf.math.unsorted_segment_min(values, segment_ids, num_segments)


class SumGraphPieceReducer(GraphPieceReducer):
  """Implements sum-pooling from one graph piece."""

  def unsorted_segment_op(self,
                          values: Field,
                          segment_ids: tf.Tensor,
                          num_segments: tf.Tensor) -> Field:
    """Implements subclass API."""
    return tf.math.unsorted_segment_sum(values, segment_ids, num_segments)


class ProdGraphPieceReducer(GraphPieceReducer):
  """Implements prod-pooling from one graph piece."""

  def unsorted_segment_op(self,
                          values: Field,
                          segment_ids: tf.Tensor,
                          num_segments: tf.Tensor) -> Field:
    """Implements subclass API."""
    return tf.math.unsorted_segment_prod(values, segment_ids, num_segments)


# IMPORTANT: When adding a public reduce_type, don't forget to add the matching
# entry to MULTI_REDUCER_CLASSES.
_GRAPH_PIECE_REDUCER_CLASSES = {
    "_count": CountGraphPieceReducer,   # For internal use only.
    "max": MaxGraphPieceReducer,
    "mean": MeanGraphPieceReducer,
    "min": MinGraphPieceReducer,
    "prod": ProdGraphPieceReducer,
    "sum": SumGraphPieceReducer,
}


class MultiReducer(abc.ABC):
  """Base class to define pool() for one reduce_type from multiple graph pieces.

  A MultiReducer defines the computation of `tfgnn.pool()` for one reduce_type
  from one or more graph pieces (that is, from edge sets into a node set, or
  from node/edge sets into context) in two steps:

    * Method `get_piece_reducer_names()` decides which `GraphPieceReducer`s
      to run on each graph piece and returns their keys for lookup in
      _GRAPH_PIECE_REDUCER_CLASSES.
    * Method `compute_from_pieces()` receives a dict of `GraphPieceReducer`
      results and computes the final result from them.

  Subclasses implement both methods as appropriate for the respective operation
  they provide.

  The two-step approach enables to de-duplicate the evlauation of
  `GraphPieceReducer`s between several `MultiReducer`s. As a typical example,
  consider how `pool(reduce_type="sum|mean")` needs to compute sums and counts
  already for `"mean"`, but sums are also useful for `"sum"`.
  """

  @abc.abstractmethod
  def get_piece_reducer_names(self) -> list[str]:
    """Returns list of GraphPieceReducer names used by compute_from_pieces()."""
    raise NotImplementedError("To be implemented by op-specific subclass")

  @abc.abstractmethod
  def compute_from_pieces(self,
                          pieces: dict[str, list[Field]]) -> Field:
    raise NotImplementedError("To be implemented by op-specific subclass")


class MaxMultiReducer(MultiReducer):
  """Implements max-pooling from one or more graph pieces."""

  def get_piece_reducer_names(self) -> list[str]:
    """Implements subclass API."""
    return ["max"]

  def compute_from_pieces(self,
                          pieces: dict[str, list[Field]]) -> Field:
    """Implements subclass API."""
    return functools.reduce(tf.math.maximum, pieces["max"])


class MaxNoInfMultiReducer(MultiReducer):

  def get_piece_reducer_names(self) -> list[str]:
    return ["max"]

  def compute_from_pieces(self,
                          pieces: dict[str, list[Field]]) -> Field:
    result = functools.reduce(tf.math.maximum, pieces["max"])
    return _where_scalar_or_field(tf.less_equal(result, result.dtype.min),
                                  tf.zeros([], dtype=result.dtype),
                                  result)


def _where_scalar_or_field(condition: const.Field, true_scalar_value: tf.Tensor,
                           false_value: const.Field) -> const.Field:
  """Fixed tf.where for a scalar true side and possibly ragged false side."""
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


class MeanMultiReducer(MultiReducer):
  """Implements mean-pooling from one or more graph pieces."""

  def get_piece_reducer_names(self) -> list[str]:
    """Implements subclass API."""
    return ["sum", "_count"]

  def compute_from_pieces(self,
                          pieces: dict[str, list[Field]]) -> Field:
    """Implements subclass API."""
    sum_ = tf.add_n(pieces["sum"])
    count = tf.add_n(pieces["_count"])
    return tf.math.divide_no_nan(
        sum_, _expand_count_to_rank(count, sum_.shape.rank))


def _expand_count_to_rank(count, rank):
  assert count.shape.rank == 1, "Internal error in counting"
  assert rank >= 1, "Feature rank lost in summation"
  while count.shape.rank < rank:
    count = tf.expand_dims(count, axis=-1)
  return count


class MinMultiReducer(MultiReducer):
  """Implements min-pooling from one or more graph pieces."""

  def get_piece_reducer_names(self) -> list[str]:
    """Implements subclass API."""
    return ["min"]

  def compute_from_pieces(self,
                          pieces: dict[str, list[Field]]) -> Field:
    """Implements subclass API."""
    return functools.reduce(tf.math.minimum, pieces["min"])


class MinNoInfMultiReducer(MultiReducer):

  def get_piece_reducer_names(self) -> list[str]:
    return ["min"]

  def compute_from_pieces(self,
                          pieces: dict[str, list[Field]]) -> Field:
    result = functools.reduce(tf.math.minimum, pieces["min"])
    return _where_scalar_or_field(tf.greater_equal(result, result.dtype.max),
                                  tf.zeros([], dtype=result.dtype),
                                  result)


class SumMultiReducer(MultiReducer):
  """Implements sum-pooling from one or more graph pieces."""

  def get_piece_reducer_names(self) -> list[str]:
    """Implements subclass API."""
    return ["sum"]

  def compute_from_pieces(self,
                          pieces: dict[str, list[Field]]) -> Field:
    """Implements subclass API."""
    return tf.add_n(pieces["sum"])


class ProdMultiReducer(MultiReducer):
  """Implements prod-pooling from one or more graph pieces."""

  def get_piece_reducer_names(self) -> list[str]:
    """Implements subclass API."""
    return ["prod"]

  def compute_from_pieces(self,
                          pieces: dict[str, list[Field]]) -> Field:
    """Implements subclass API."""
    return functools.reduce(tf.math.multiply, pieces["prod"])


# IMPORTANT: Keep in sync with the docstring of pool().
_MULTI_REDUCER_CLASSES = {
    "max": MaxMultiReducer,
    "max_no_inf": MaxNoInfMultiReducer,
    "mean": MeanMultiReducer,
    "min": MinMultiReducer,
    "min_no_inf": MinNoInfMultiReducer,
    "sum": SumMultiReducer,
    "prod": ProdMultiReducer,
}


def get_registered_reduce_operation_names() -> list[str]:
  """Returns the registered list of supported reduce operation names."""
  return list(_MULTI_REDUCER_CLASSES.keys())
