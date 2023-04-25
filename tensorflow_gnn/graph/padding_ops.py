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
"""Defines padding operations over a GraphTensor."""

import functools
from typing import Any, Callable, List, Optional, Tuple, Union, cast

import numpy as np
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_piece as gp
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import preprocessing_common as preprocessing
from tensorflow_gnn.graph import tensor_utils


def pad_to_total_sizes(
    graph_tensor: gt.GraphTensor,
    size_constraints: preprocessing.SizeConstraints,
    *,
    padding_values: Optional[preprocessing.FeatureDefaultValues] = None,
    validate: bool = True) -> Tuple[gt.GraphTensor, tf.Tensor]:
  """Pads graph tensor to the total sizes by inserting fake graph components.

  Padding is done by inserting "fake" graph components at the end of the input
  graph tensor until target total sizes are exactly matched. If that is not
  possible (e.g. input already has more nodes than allowed by the constraints)
  function raises `tf.errors.InvalidArgumentError`.

  If size_constraints.min_nodes_per_component is specified for a node set,
  the inserted graph components satisfy that constraint (e.g., such that there
  is a node for tf.gather_first_node()). Components in the input graph tensor
  must satisfy that constraint already, or tf.errors.InvalidArgumentError will
  be raised. (This function cannot add padding within existing components.)

  Context, node or edge features of the appended fake components are filled with
  user-provided scalar values or with zeros if the latter are not specified.
  Fake edges are created such that each fake node has an approximately uniform
  number of incident edges (this behavior is not guaranteed and may change in
  the future).

  NOTE(b/275338236): This operation is not available in TFLite (last checked
  for TF 2.12).

  Args:
    graph_tensor: scalar graph tensor (rank=0) to pad.
    size_constraints: target total sizes for each graph piece. Must define the
      target number of graph components (`.total_num_components`), target total
      number of items for each node set (`.total_num_nodes[node_set_name]`) and
      likewise for each edge set (`.total_num_edges[edge_set_name]`).
      If `min_nodes_per_component` is set, the inserted graph components satisfy
      that constraint and graph components of the input graph tensor are checked
      against this constraint.
    padding_values: optional mapping from a context, node set or edge set
      feature name to a scalar tensor to use for padding. If no value is
      specified for some feature, its type 'zero' is used (as in `tf.zeros()`).
    validate: If true, then use assertions to check that the input graph tensor
      could be padded. NOTE: while these assertions provide more readable error
      messages, they incur a runtime cost, since assertions must be checked for
      each input value.

  Returns:
    Tuple of padded graph tensor and padding mask. The mask is a rank-1 dense
    boolean tensor wth size equal to the number of graph compoents is the result
    containing `True` for real graph components and `False` - for fake one used
    for padding.

  Raises:
    ValueError: if input parameters are invalid.
    tf.errors.InvalidArgumentError: if input graph tensor could not be padded to
      the `size_constraints` or has less nodes in a component than allowed by
      the `min_nodes_per_component`
  """
  gt.check_scalar_graph_tensor(graph_tensor, 'tfgnn.pad_to_total_sizes()')

  def _ifnone(value, default):
    return value if value is not None else default

  if padding_values is None:
    padding_values = preprocessing.FeatureDefaultValues()

  def get_default_value(
      graph_piece_spec: gt._GraphPieceWithFeaturesSpec,  # pylint: disable=protected-access
      padding_values: gt.Fields,
      feature_name: str,
      debug_context: str) -> tf.Tensor:
    spec = graph_piece_spec.features_spec[feature_name]
    value = padding_values.get(feature_name, None)
    if value is None:
      value = tf.zeros([], dtype=spec.dtype)
    else:
      value = tf.convert_to_tensor(value, spec.dtype)
      if value.shape.rank != 0:
        raise ValueError(f'Default value for {debug_context} must be scalar,'
                         f' got shape={value.shape}')
    return value

  def get_min_max_fake_nodes_indices(
      node_set_name: str) -> Tuple[tf.Tensor, tf.Tensor]:
    min_node_index = graph_tensor.node_sets[node_set_name].total_size
    max_node_index = tf.constant(
        size_constraints.total_num_nodes[node_set_name],
        dtype=min_node_index.dtype) - 1
    return min_node_index, max_node_index

  # Note: we check that graph tensor could potentially fit into the target sizes
  # before running padding. This simplifies padding implementation and removes
  # duplicative validations.
  if validate:
    validation_ops = assert_satisfies_size_constraints(graph_tensor,
                                                       size_constraints)
  else:
    validation_ops = []

  with tf.control_dependencies(validation_ops):
    total_num_components = graph_tensor.total_num_components
    target_total_num_components = size_constraints.total_num_components

    padded_context = _pad_to_total_sizes(
        graph_tensor.context,
        target_total_num_components=target_total_num_components,
        padding_value_fn=functools.partial(
            get_default_value,
            graph_tensor.context.spec,
            _ifnone(padding_values.context, {}),
            debug_context='context'))

    min_nodes_per_component = dict(size_constraints.min_nodes_per_component)
    padded_node_sets = {}
    for name, item in graph_tensor.node_sets.items():
      padded_node_sets[name] = _pad_to_total_sizes(
          item,
          target_total_num_components=target_total_num_components,
          target_total_size=size_constraints.total_num_nodes[name],
          min_nodes_per_component=min_nodes_per_component.get(name, 0),
          padding_value_fn=functools.partial(
              get_default_value,
              item.spec,
              _ifnone(padding_values.node_sets, {}).get(name, {}),
              debug_context=f'{name} nodes'))

    padded_edge_sets = {}
    for name, item in graph_tensor.edge_sets.items():
      padded_edge_sets[name] = _pad_to_total_sizes(
          item,
          target_total_num_components=target_total_num_components,
          target_total_size=size_constraints.total_num_edges[name],
          min_max_node_index_fn=get_min_max_fake_nodes_indices,
          min_edges_per_component=0,
          padding_value_fn=functools.partial(
              get_default_value,
              item.spec,
              _ifnone(padding_values.edge_sets, {}).get(name, {}),
              debug_context=f'{name} edges'))

  padded_graph_tensor = gt.GraphTensor.from_pieces(
      context=padded_context,
      node_sets=padded_node_sets,
      edge_sets=padded_edge_sets)

  num_padded = tf.constant(target_total_num_components,
                           total_num_components.dtype) - total_num_components
  padding_mask = tensor_utils.ensure_static_nrows(
      tf.concat(
          values=[
              tf.ones([total_num_components], dtype=tf.bool),
              tf.zeros([num_padded], dtype=tf.bool)
          ],
          axis=0), target_total_num_components)
  return padded_graph_tensor, cast(tf.Tensor, padding_mask)


def satisfies_size_constraints(
    graph_tensor: gt.GraphTensor,
    total_sizes: preprocessing.SizeConstraints) -> tf.Tensor:
  """Returns whether the input `graph_tensor` satisfies `total_sizes`.

  Args:
    graph_tensor: a graph tensor to check against target total sizes.
    total_sizes: target total sizes for each graph piece.

  Returns:
    A scalar boolean tensor equal to `True` if the `graph_tensor` statisifies
    `total_sizes`, and `False` if not.
  """

  def check_fn(cond: tf.Tensor, message: str):
    del message
    return cond

  conditions = _satisfies_size_constraints_internal(graph_tensor, total_sizes,
                                                    check_fn)
  if not tf.executing_eagerly():
    static_conditions = [tf.get_static_value(cond) for cond in conditions]
    if None not in static_conditions:
      return tf.constant(all(static_conditions))
  return tf.math.reduce_all(tf.stack(conditions, axis=0))


def assert_satisfies_size_constraints(
    graph_tensor: gt.GraphTensor,
    size_constraints: preprocessing.SizeConstraints):
  """Raises InvalidArgumentError if graph_tensor exceeds size_constraints.

  This function can be used as follows:

  ```python
  with tf.control_dependencies([
    assert_satisfies_size_constraints(graph_tensor, size_constraints)]):
    # Use graph_tensor after sizes have been checked.
  ```

  Conceptually, that means this function is like standard tensorflow assertions,
  like `tf.debugging.Assert(satisfies_size_constraints(...))`, but with the
  following important advantages:

  - This functions logs a detailed message which size constraint is violated.
  - This function works around a TensorFlow issue to make sure the assertion is
    executed before the ops it guards, even in the presence of conflicting
    attempts to eliminate constant subexpressions.

  Args:
    graph_tensor: a graph tensor to check against target total sizes.
    size_constraints: target total sizes for each graph piece.

  Returns:
    Validation operations to execute within a `tf.control_dependencies`.

  Raises:
    tf.errors.InvalidArgumentError: if input graph tensor could not be padded to
     the `size_constraints`.
  """

  def check_fn(cond: tf.Tensor, message: str):
    # NOTE: this code assumes that `tf.debugging.assert_equal()` raises
    # immediately if `cond` has static False value.
    return tf.debugging.assert_equal(cond, True, message=message)

  return _satisfies_size_constraints_internal(graph_tensor, size_constraints,
                                              check_fn)


@functools.singledispatch
def _pad_to_total_sizes(piece: gp.GraphPieceBase, **argw) -> gp.GraphPieceBase:
  raise NotImplementedError(type(piece).__name__)


@_pad_to_total_sizes.register
def _(context: gt.Context, *,
      padding_value_fn: Callable[[gt.FieldName], gt.Field],
      target_total_num_components: int) -> gt.Context:
  """Pads graph context to the target number of graph components."""

  diff = tf.ones(
      shape=[target_total_num_components - context.total_num_components],
      dtype=context.spec.sizes_spec.dtype)
  sizes = tf.concat([context.sizes, diff], axis=0)
  sizes = tensor_utils.ensure_static_nrows(
      sizes, nrows=target_total_num_components)
  assert sizes.shape == tf.TensorShape([target_total_num_components])
  return context.from_fields(
      features=_pad_features(
          context.features,
          padding_value_fn=padding_value_fn,
          target_size=target_total_num_components),
      sizes=sizes)


@_pad_to_total_sizes.register
def _(node_set: gt.NodeSet, *,
      padding_value_fn: Callable[[gt.FieldName], gt.Field],
      target_total_num_components: int,
      target_total_size: int,
      min_nodes_per_component: int) -> gt.NodeSet:
  """Pads node set to the target number of nodes."""

  return node_set.from_fields(
      features=_pad_features(
          node_set.features,
          padding_value_fn=padding_value_fn,
          target_size=target_total_size),
      sizes=_pad_sizes(
          node_set.sizes,
          min_entities_per_component=min_nodes_per_component,
          target_num_components=target_total_num_components,
          target_size=target_total_size))


@_pad_to_total_sizes.register
def _(edge_set: gt.EdgeSet, *,
      padding_value_fn: Callable[[gt.FieldName], gt.Field],
      min_max_node_index_fn: Callable[[gt.NodeSetName],
                                      Tuple[tf.Tensor, tf.Tensor]],
      target_total_num_components: int,
      target_total_size: int,
      min_edges_per_component: int) -> gt.EdgeSet:
  """Pads edge set to the target number of edges."""

  return edge_set.from_fields(
      features=_pad_features(
          edge_set.features,
          padding_value_fn=padding_value_fn,
          target_size=target_total_size),
      sizes=_pad_sizes(
          edge_set.sizes,
          min_entities_per_component=min_edges_per_component,
          target_num_components=target_total_num_components,
          target_size=target_total_size),
      adjacency=_pad_to_total_sizes(
          edge_set.adjacency,
          target_total_size=target_total_size,
          min_max_node_index_fn=min_max_node_index_fn))


@_pad_to_total_sizes.register
def _(adjacency: adj.Adjacency, *,
      target_total_size: int,
      min_max_node_index_fn: Callable[[gt.NodeSetName],
                                      Tuple[tf.Tensor, tf.Tensor]]
     ) -> adj.Adjacency:
  """Pads adjacency the target number of edges."""
  return adjacency.from_indices(
      source=(adjacency.source_name,
              _pad_adjacency_index_with_linspace(
                  adjacency.source, target_total_size,
                  *min_max_node_index_fn(adjacency.source_name))),
      target=(adjacency.target_name,
              _pad_adjacency_index_with_linspace(
                  adjacency.target, target_total_size,
                  *min_max_node_index_fn(adjacency.target_name))),
      validate=False)


@_pad_to_total_sizes.register
def _(adjacency: adj.HyperAdjacency, *,
      target_total_size: int,
      min_max_node_index_fn: Callable[[gt.NodeSetName],
                                      Tuple[tf.Tensor, tf.Tensor]]
     ) -> adj.HyperAdjacency:
  """Pads hyper adjacency the target number of edges."""
  padded_indices = {}
  for tag, (name, index) in adjacency.get_indices_dict().items():
    padded_indices[tag] = (name,
                           _pad_adjacency_index_with_linspace(
                               index, target_total_size,
                               *min_max_node_index_fn(name)))

  return adjacency.from_indices(padded_indices, validate=False)


def _pad_features(features: gt.Fields, *,
                  padding_value_fn: Callable[[gt.FieldName], gt.Field],
                  target_size: int) -> gt.Fields:
  """Pads features to the target total number of elemenets."""
  map_fn = functools.partial(_pad_feature, target_size=target_size)
  return {
      fname: map_fn(feature, padding_value=padding_value_fn(fname))
      for fname, feature in features.items()
  }


def _pad_feature(feature: gt.Field, *, padding_value: gt.Field,
                 target_size: int) -> gt.Field:
  """Pads a single feature to the target total number of elemenets."""
  result = tensor_utils.pad_to_nrows(
      feature,
      target_nrows=tf.constant(target_size),
      padding_value=padding_value,
      validate=False)
  return tensor_utils.ensure_static_nrows(result, nrows=target_size)


def _pad_adjacency_index_with_linspace(index: const.Field, target_size: int,
                                       min_index: tf.Tensor,
                                       max_index: tf.Tensor) -> gt.Field:
  """Pads adjacency `index` with linearly increasing indices.

  The function pads `index` tensor to the target size by appending linearly
  non-decreasing sequence of indices starting from the `min_index` (including)
  and ending at the `max_index` (including). This padding schema allows to
  uniformly connect fake nodes with fake edges. Also it preserves memory
  locality as fake edges with the same incident node are consecutive.

  Args:
    index: rank-1 integer tensor with (real) node indices.
    target_size: size of the result index tensor.
    min_index: smallest allowed index for padding.
    max_index: largest allowed index for padding.

  Returns:
    index tensor padded to the `target_size` with non-decreasing linear
    sequence of indices from `min_index` to `max_index`.
  """
  assert index.shape.rank == 1
  assert min_index.shape.rank == 0
  assert max_index.shape.rank == 0
  diff_size = tf.constant(
      target_size, dtype=index.dtype) - tf.size(index, index.dtype)
  diff = tf.linspace(
      start=tf.cast(min_index, tf.float32),
      stop=tf.cast(max_index + 1, tf.float32),
      num=diff_size)
  diff = tf.cast(diff, index.dtype)
  diff = tf.clip_by_value(diff, min_index, max_index)
  result = tf.concat([index, diff], axis=0)
  return tensor_utils.ensure_static_nrows(result, nrows=target_size)


def _pad_sizes(sizes: gt.Field, *, target_num_components: int,
               min_entities_per_component: int,
               target_size: int) -> tf.Tensor:
  """Pads sizes tensor to the target number of elements and graph components."""
  assert sizes.shape.rank == 1
  assert min_entities_per_component >= 0
  # Note that total_num_components could be 0, so we could not simply append
  # total_num_components - 1 zeros at the end of [target_sizes].
  fake1p_size = tf.constant(min_entities_per_component, dtype=sizes.dtype)
  fake0_size = tf.constant(
      target_size, dtype=sizes.dtype) - tf.reduce_sum(sizes)
  if min_entities_per_component > 0:
    num_components = tf.size(sizes, sizes.dtype)
    fake0_size -= (target_num_components - 1 - num_components) * fake1p_size

  diff_in_size_plus_1 = tf.constant(
      target_num_components, dtype=sizes.dtype) - tf.size(sizes, sizes.dtype)
  result = tf.concat(
      values=[
          sizes,
          tf.expand_dims(fake0_size, axis=-1),
          tf.fill([diff_in_size_plus_1], value=fake1p_size)
      ],
      axis=0)
  result = result[:target_num_components]
  result = tensor_utils.ensure_static_nrows(result, target_num_components)
  assert result.shape == tf.TensorShape([target_num_components])
  return cast(tf.Tensor, result)


_BinaryOp = Callable[
    [Union[tf.Tensor, np.ndarray], Union[tf.Tensor, np.ndarray]], tf.Tensor]


def _fold_constants(binary_op: _BinaryOp, x: tf.Tensor,
                    y: tf.Tensor) -> tf.Tensor:
  """Attempts to call binary_op with static values of `x` and `y`.

  Implementes constant folding for TF binary operations (see b/210985575).

  TODO(b/212250026): remove as soon as the root issue is fixed.

  Args:
    binary_op: function of two arguments that could be called with tf.Tensor
      arguments (must return tf.Tensor) and np.ndarray arguments (must return
      np.ndarray).
    x: the first argument.
    y: the second argument.

  Returns:
    tf.constant if both `x` and `y` have static values and tf.Tensor otherwise.
  """
  if not tf.executing_eagerly():
    xs = tf.get_static_value(x)
    ys = tf.get_static_value(y)
    if xs is not None and ys is not None:
      return tf.constant(binary_op(xs, ys))
  return binary_op(x, y)


def _satisfies_size_constraints_internal(
    graph_tensor: gt.GraphTensor, total_sizes: preprocessing.SizeConstraints,
    check_fn: Callable[[tf.Tensor, str], Any]) -> List[Any]:
  """Checks that the graph tensor could fit in the target sizes.

  This operation tests multiple conditions that all must be `True` for the input
  `graph_tensor` to satisfy the `total_sizes`. The evaluated conditions along
  with a description string are passed to the caller using `check_fn` callbacks.

  The function tries to statically evaluate each condition and pass its result
  as tf.constant so that it could be extracted (see `tf.get_static_value()`).
  See `assert_satisfies_size_constraints()` for more information on how this
  might be useful.

  Args:
    graph_tensor: a graph tensor to check against total sizes.
    total_sizes: total sizes constraints for each graph piece.
    check_fn: callable with two arguments. The first argument is an evaluation
      result for one of required conditions. It is a boolean scalar tensor where
      `True` means condition is satisfied. If all conditions result in `True`,
      the `graph_tensor` satisfies `total_sizes`. The second argument is a
      string description of the condition. All values returned by the `check_fn`
      are accumulated and returned.

  Returns:
    List of all results returned by the `check_fn`.
  """
  # NOTE: TF implements for some operations s.c. contant folding when those
  # operations are evaluated statically if all their inputs have static values.
  # Those operations could also raise an exception staticaly if their arguments
  # are invalid. The constant folding is only supported by some operations (e.g.
  # tf.fill) and is not supported by others (e.g. tf.debug.Assert). This could
  # break control flow rules (https://www.tensorflow.org/guide/intro_to_graphs).
  # See b/205974487 for more examples. This function always attempts to evaluate
  # assertions statically by using python logical operators to test conditions
  # in the _fold_constants. Because those operators are overriden both by
  # np.ndarray and tf.Tensor they could be evaluated statically on in the
  # runtime depending on its arguments.
  total_num_components = graph_tensor.total_num_components
  could_add_new_component = _fold_constants(lambda x, y: x < y,
                                            total_num_components,
                                            total_sizes.total_num_components)
  num_fake_components = total_sizes.total_num_components - graph_tensor.total_num_components
  assert_ops = [
      check_fn(
          _fold_constants(
              lambda x, y: x <= y, total_num_components,
              tf.convert_to_tensor(
                  total_sizes.total_num_components,
                  dtype=total_num_components.dtype)),
          ('Could not pad graph as it already has more graph components'
           ' then it is allowed by `total_sizes.total_num_components`'))
  ]

  def _check_sizes(entity_type: str, entity_name: str,
                   sizes: tf.Tensor,
                   total_size: tf.Tensor,
                   target_total_size: Optional[int],
                   min_entities_per_component: int):
    if target_total_size is None:
      raise ValueError(
          f'The target total number of <{entity_name}> {entity_type} must be'
          ' specified as'
          f' `total_sizes.total_num_{entity_type}[<{entity_name}>]`.')
    if min_entities_per_component > 0:
      assert_ops.append(
          check_fn(
              tf.reduce_all(
                  tf.greater_equal(sizes, min_entities_per_component)),
              (f'Some graph components has fewer <{entity_name}>'
               '  than it is allowed by the'
               ' `total_sizes'
               f'.min_{entity_type}_per_component[<{entity_name}>]`.'
              )))

    target_total_size = tf.convert_to_tensor(
        target_total_size, dtype=total_size.dtype)
    padded_size = total_size
    if min_entities_per_component > 0:
      min_fake_entities_count = num_fake_components * tf.convert_to_tensor(
          min_entities_per_component, total_size.dtype)
      padded_size += min_fake_entities_count
    overflow_msg = (f'Could not pad <{entity_name}> as it already has more'
                    f' {entity_type} then it is allowed by the'
                    f' `total_sizes.total_num_{entity_type}[<{entity_name}>]`')
    if min_entities_per_component > 0:
      overflow_msg += (
          ', taking into account that at least'
          f' {min_entities_per_component} {entity_type} should be added'
          ' to each fake component.')
    else:
      overflow_msg += '.'

    # Check independetly a weaker case that total_size <= target_total_size
    # (compared to a stronger condition padded_size <= target_total_size)
    # because the weaker case may support constant folding.
    assert_ops.append(
        check_fn(
            _fold_constants(lambda x, y: x <= y, total_size,
                            target_total_size),
            overflow_msg))

    assert_ops.append(
        check_fn(
            _fold_constants(lambda x, y: x <= y, padded_size,
                            target_total_size),
            overflow_msg))

    assert_ops.append(
        check_fn(
            _fold_constants(
                lambda x, y: x | y, could_add_new_component,
                _fold_constants(lambda x, y: x == y, padded_size,
                                target_total_size)),
            (f'Could not pad <{entity_name}> {entity_type}. To do this, at'
             ' least one graph component must be added to the input graph.'
             ' The latter is not possible as the input graph has already'
             ' `total_sizes.total_num_components` graph components.')))

  min_nodes_per_component = dict(total_sizes.min_nodes_per_component)
  total_num_nodes = {}
  for name, item in graph_tensor.node_sets.items():
    total_size = item.total_size
    target_total_size = total_sizes.total_num_nodes.get(name, None)
    total_num_nodes[name] = total_size
    _check_sizes(
        'nodes',
        name,
        item.sizes,
        total_size,
        target_total_size,
        min_entities_per_component=min_nodes_per_component.get(name, 0))

  for name, item in graph_tensor.edge_sets.items():
    total_size = item.total_size
    target_total_size = total_sizes.total_num_edges.get(name, None)
    _check_sizes('edges', name, item.sizes, total_size, target_total_size,
                 min_entities_per_component=0)

    assert target_total_size is not None
    has_all_edges = _fold_constants(lambda x, y: x == y, total_size,
                                    target_total_size)
    indices = item.adjacency.get_indices_dict()
    for _, (incident_node_set_name, _) in indices.items():
      permits_new_incident_nodes = _fold_constants(
          lambda x, y: x < y, total_num_nodes[incident_node_set_name],
          total_sizes.total_num_nodes[incident_node_set_name])
      assert_ops.append(
          check_fn(
              _fold_constants(lambda x, y: x | y, has_all_edges,
                              permits_new_incident_nodes),
              ('Could not create fake incident edges for the node set'
               f' {incident_node_set_name}. This could happen when the'
               ' total number of real nodes is equal to the target total'
               ' number of nodes, so there are no fake nodes that could be'
               ' connected by inserted fake edges.')))

  return assert_ops
