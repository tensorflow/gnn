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
"""The pool_v2() and broadcast_v2() operations for multiple graph pieces.

TODO(b/265760014): Replace the current implementations of tfgnn.pool_*()
and tfgnn.broadcast() from ./graph_tensor_ops.py with this code.
"""
from __future__ import annotations
import abc
import functools
from typing import Optional, Sequence, Union

import tensorflow as tf

from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import graph_tensor_ops as ops
from tensorflow_gnn.graph import tensor_utils as utils

Field = const.Field
FieldName = const.FieldName
NodeSetName = const.NodeSetName
EdgeSetName = const.EdgeSetName
IncidentNodeTag = const.IncidentNodeTag
IncidentNodeOrContextTag = const.IncidentNodeOrContextTag
GraphTensor = gt.GraphTensor


# TODO(b/269076334): When multi-graph piece support is ready, replace the old
# tfgnn.broadcast() by this, but keep the underlying basic broadcast operations.
def broadcast_v2(
    graph: GraphTensor,
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
    graph: A scalar GraphTensor.
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
  gt.check_scalar_graph_tensor(graph, "broadcast()")
  edge_set_names, node_set_names, got_sequence_args = (
      _get_edge_and_node_set_name_args(
          "broadcast()", graph, from_tag,
          edge_set_name=edge_set_name, node_set_name=node_set_name))
  del edge_set_name, node_set_name  # Replaced by their cleaned-up versions.
  if (feature_value is None) == (feature_name is None):
    raise ValueError(
        "broadcast() requires exactly one of feature_name of feature_value.")
  feature_kwargs = dict(feature_value=feature_value, feature_name=feature_name)

  if from_tag == const.CONTEXT:
    if edge_set_names is not None:
      result = [ops.broadcast_context_to_edges(graph, name, **feature_kwargs)
                for name in edge_set_names]
    else:
      result = [ops.broadcast_context_to_nodes(graph, name, **feature_kwargs)
                for name in node_set_names]
  else:
    result = [
        ops.broadcast_node_to_edges(graph, name, from_tag, **feature_kwargs)
        for name in edge_set_names]

  if got_sequence_args:
    return result
  else:
    assert len(result) == 1
    return result[0]


def _get_edge_and_node_set_name_args(
    function_name: str,
    graph: GraphTensor,
    tag: IncidentNodeOrContextTag,
    *,
    edge_set_name: Union[Sequence[EdgeSetName], EdgeSetName, None] = None,
    node_set_name: Union[Sequence[NodeSetName], NodeSetName, None] = None,
) -> tuple[Optional[Sequence[EdgeSetName]],
           Optional[Sequence[NodeSetName]],
           bool]:
  """Returns canonicalized edge/node args for broadcast() and pool()."""
  if tag == const.CONTEXT:
    num_names = bool(edge_set_name is None) + bool(node_set_name is None)
    if num_names != 1:
      raise ValueError(
          f"{function_name} with tag tfgnn.CONTEXT requires to pass exactly "
          f"1 of edge_set_name, node_set_name but got {num_names}.")
  else:
    if not isinstance(tag, const.IncidentNodeTag):
      raise ValueError(
          f"{function_name} got tag {tag} but requires either "
          "the special value tfgnn.CONTEXT "
          "or a valid IndicentNodeTag, that is, tfgnn.SOURCE, tfgnn.TARGET, "
          "or another valid integer in case of hypergraphs.")
    if edge_set_name is None or node_set_name is not None:
      raise ValueError(
          f"{function_name} requires to pass edge_set_name, "
          "not node_set_name, unless tag is set to tfgnn.CONTEXT.")

  got_sequence_args = not (isinstance(node_set_name, str) or
                           isinstance(edge_set_name, str))
  edge_set_names = _get_nonempty_name_list_or_none(
      function_name, "edge_set_name", edge_set_name)
  node_set_names = _get_nonempty_name_list_or_none(
      function_name, "node_set_name", node_set_name)

  if tag != const.CONTEXT:
    incident_node_set_names = {graph.edge_sets[e].adjacency.node_set_name(tag)
                               for e in edge_set_names}
    if len(incident_node_set_names) > 1:
      raise ValueError(
          f"{function_name} requires the same endpoint for all named edge sets "
          f"but got node sets {incident_node_set_names} from tag={tag} and "
          f"edge_set_name={edge_set_name}")

  return edge_set_names, node_set_names, got_sequence_args


def _get_nonempty_name_list_or_none(
    function_name: str,
    arg_name: str,
    name: Union[Sequence[str], str, None]) -> Optional[Sequence[str]]:
  if name is None:
    return None
  if isinstance(name, str):
    return [name]
  if len(name) > 0:  # Crash here if len() doesn't work.  pylint: disable=g-explicit-length-test
    return name
  raise ValueError(
      f"{function_name} requires {arg_name} to be a non-empty Sequence or None")


# TODO(b/269076334): When multi-graph piece support is ready, replace the old
# tfgnn.pool() by this, and turn the legacy pooling functions into wrappers
# of this function.
def pool_v2(
    graph: GraphTensor,
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

  Moreover, `reduce_type` can be set to a `|`-separated list of reduce types,
  such as `reduce_type="mean|sum"`, which will return the concatenation of
  their individual results.

  TODO(b/265760014): pool() from multiple edge sets (or node sets) does not yet
  support RaggedTensors.

  Args:
    graph: A scalar GraphTensor.
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
  gt.check_scalar_graph_tensor(graph, "pool()")

  edge_set_names, node_set_names, feature_values, _ = (
      get_pool_args_as_sequences(
          "pool()", graph, to_tag,
          edge_set_name=edge_set_name, node_set_name=node_set_name,
          feature_value=feature_value, feature_name=feature_name))
  del edge_set_name, node_set_name, feature_value  # Use canonicalized forms.

  if len(feature_values) > 1 and any(
      utils.is_ragged_tensor(fv) for fv in feature_values):
    raise ValueError(
        "TODO(b/265760014): pool() from multiple edge sets (or node sets) "
        "does not yet support RaggedTensors.")

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
      graph, to_tag,
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
    # TODO(b/265760014): Automate this kind of fallback.
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
    function_name: str,
    graph: GraphTensor,
    tag: IncidentNodeOrContextTag,
    *,
    edge_set_name: Union[Sequence[EdgeSetName], EdgeSetName, None] = None,
    node_set_name: Union[Sequence[NodeSetName], NodeSetName, None] = None,
    feature_value: Union[Sequence[Field], Field, None] = None,
    feature_name: Optional[FieldName] = None,
) -> tuple[Sequence[EdgeSetName], Sequence[NodeSetName], Sequence[Field], bool]:
  """Returns pool()-style args checked and with canonicalized types.

  Args:
    function_name: The user-visible name of the function whose args are
      processed.
    graph: The `GraphTensor`, as for `pool()`.
    tag: Same as for `pool()`.
    edge_set_name: As for `pool()`, can be set to a name or sequence of names.
    node_set_name: As for `pool()`, can be set to a name or sequence of names.
    feature_value: As for `pool()`, can be set to a value or sequence of values.
    feature_name: As for `pool()`, can be set to a feature name.

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
      _get_edge_and_node_set_name_args(
          function_name, graph, tag,
          edge_set_name=edge_set_name, node_set_name=node_set_name))

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
    if isinstance(adjacency, adj.HyperAdjacency):
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
    return tf.where(tf.less_equal(result, result.dtype.min),
                    tf.zeros([], dtype=result.dtype),
                    result)


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
    return tf.where(tf.greater_equal(result, result.dtype.max),
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
