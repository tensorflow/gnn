"""Defines padding operations over a GraphTensor."""

import functools
from typing import cast, Callable, Optional, Tuple

import tensorflow as tf

from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_piece as gp
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import preprocessing_common as preprocessing
from tensorflow_gnn.graph import tensor_utils


def pad_to_total_sizes(
    graph_tensor: gt.GraphTensor,
    target_total_sizes: preprocessing.SizesConstraints,
    *,
    padding_values: Optional[preprocessing.DefaultValues] = None
) -> Tuple[gt.GraphTensor, tf.Tensor]:
  """Pads graph tensor to the total sizes by inserting fake graph components.

  Padding is done by inserting "fake" graph components at the end of the input
  graph tensor until target total sizes are exactly matched. If that is not
  possible (e.g. input already has more nodes than allowed by the constraints)
  function raises tf.errors.InvalidArgumentError. Context, node or edge features
  of the appended fake components are filled using user-provided scalar values
  or with zeros if the latter are not specified. Fake edges are created such
  that each fake node has an approximately uniform number of incident edges
  (NOTE: this behavior is not guaranteed and may change in the future).

  Args:
    graph_tensor: scalar graph tensor (rank=0) to pad.
    target_total_sizes: target total sizes for each graph piece. Must define the
      target number of graph components (`.total_num_components`), target total
      number of items for each node set (`.total_num_nodes[node_set_name]`) and
      likewise for each edge set (`.total_num_edges[edge_set_name]`).
    padding_values: optional mapping from a context, node set or edge set
      feature name to a scalar tensor to use for padding. If no value is
      specified for some feature, its type 'zero' is used (as in tf.zeros(...)).

  Returns:
    Tuple of padded graph tensor and padding mask. The mask is a rank-1 dense
    boolean tensor wth size equal to the number of graph compoents is the result
    containing True for real graph components and False - for fake one used for
    padding.

  Raises:
    ValueError: if input paramters are invalid.
    tf.errors.InvalidArgumentError: if input graph tensor could not be padded to
      the `target_total_sizes`.
  """
  gt.check_scalar_graph_tensor(graph_tensor, 'tfgnn.pad_to_total_sizes()')

  def _ifnone(value, default):
    return value if value is not None else default

  if padding_values is None:
    padding_values = preprocessing.DefaultValues()

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
        target_total_sizes.total_num_nodes[node_set_name],
        dtype=min_node_index.dtype) - 1
    return min_node_index, max_node_index

  # Note: we check that graph tensor could potentially fit into the target sizes
  # before running padding. This simplifies padding implementation and removes
  # duplicative validations.
  with tf.control_dependencies(
      _check_that_could_fit(graph_tensor, target_total_sizes)):
    total_num_components = graph_tensor.total_num_components
    target_total_num_components = target_total_sizes.total_num_components

    padded_context = _pad_to_total_sizes(
        graph_tensor.context,
        target_total_num_components=target_total_num_components,
        padding_value_fn=functools.partial(
            get_default_value,
            graph_tensor.context.spec,
            _ifnone(padding_values.context, {}),
            debug_context='context'))

    padded_node_sets = {}
    for name, item in graph_tensor.node_sets.items():
      padded_node_sets[name] = _pad_to_total_sizes(
          item,
          target_total_num_components=target_total_num_components,
          target_total_size=target_total_sizes.total_num_nodes[name],
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
          target_total_size=target_total_sizes.total_num_edges[name],
          min_max_node_index_fn=get_min_max_fake_nodes_indices,
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
      target_total_size: int) -> gt.NodeSet:
  """Pads node set to the target number of nodes."""

  return node_set.from_fields(
      features=_pad_features(
          node_set.features,
          padding_value_fn=padding_value_fn,
          target_size=target_total_size),
      sizes=_pad_sizes(
          node_set.sizes,
          target_num_components=target_total_num_components,
          target_size=target_total_size))


@_pad_to_total_sizes.register
def _(edge_set: gt.EdgeSet, *,
      padding_value_fn: Callable[[gt.FieldName], gt.Field],
      min_max_node_index_fn: Callable[[gt.NodeSetName],
                                      Tuple[tf.Tensor, tf.Tensor]],
      target_total_num_components: int,
      target_total_size: int) -> gt.EdgeSet:
  """Pads edge set to the target number of edges."""

  return edge_set.from_fields(
      features=_pad_features(
          edge_set.features,
          padding_value_fn=padding_value_fn,
          target_size=target_total_size),
      sizes=_pad_sizes(
          edge_set.sizes,
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
                  adjacency.source_name, adjacency.source, target_total_size,
                  *min_max_node_index_fn(adjacency.source_name))),
      target=(adjacency.target_name,
              _pad_adjacency_index_with_linspace(
                  adjacency.target_name, adjacency.target, target_total_size,
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
                               name, index, target_total_size,
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


def _pad_adjacency_index_with_linspace(
    incident_node_set_name: const.NodeSetName, index: const.Field,
    target_size: int, min_index: tf.Tensor, max_index: tf.Tensor) -> gt.Field:
  """Pads adjacency `index` with linearly increasing indices.

  The function pads `index` tensor to the target size by appending linearly
  non-decreasing sequence of indices starting from the `min_index` (including)
  and ending at the `max_index` (including). This padding schema allows to
  uniformly connect fake nodes with fake edges. Also it preserves memory
  locality as fake edges with the same incident node are consecutive.

  Args:
    incident_node_set_name: incident node set name.
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
  with tf.control_dependencies([
      tf.debugging.assert_equal(
          tf.logical_or(
              tf.equal(diff_size, tf.constant(0, diff_size.dtype)),
              tf.math.less_equal(min_index, max_index)),
          True,
          message=(
              'Could not create fake incident edges for the node set'
              f' \'{incident_node_set_name}\'. This could happen when the total'
              ' number of real nodes is equal to the target total number of'
              ' nodes, so there are no fake nodes that could be connected by'
              ' inserted fake edges.'))
  ]):

    diff = tf.linspace(
        start=tf.cast(min_index, tf.float32),
        stop=tf.cast(max_index + 1, tf.float32),
        num=diff_size)
    diff = tf.cast(diff, index.dtype)
    diff = tf.clip_by_value(diff, min_index, max_index)
    result = tf.concat([index, diff], axis=0)
    return tensor_utils.ensure_static_nrows(result, nrows=target_size)


def _pad_sizes(sizes: gt.Field, *, target_num_components: int,
               target_size: int) -> tf.Tensor:
  """Pads sizes tensor to the target number of elements and graph components."""
  assert sizes.shape.rank == 1

  # Note that total_num_components could be 0, so we could not simply append
  # total_num_components - 1 zeros at the end of [target_sizes].
  fake0_size = tf.constant(
      target_size, dtype=sizes.dtype) - tf.reduce_sum(sizes)
  diff_in_size_plus_1 = tf.constant(
      target_num_components, dtype=sizes.dtype) - tf.size(sizes, sizes.dtype)
  result = tf.concat(
      values=[
          sizes,
          tf.expand_dims(fake0_size, axis=-1),
          tf.zeros([diff_in_size_plus_1], dtype=sizes.dtype)
      ],
      axis=0)
  result = result[:target_num_components]
  assert result.shape == tf.TensorShape([target_num_components])
  return result


def _check_that_could_fit(graph_tensor: gt.GraphTensor,
                          target_total_sizes: preprocessing.SizesConstraints):
  """Checks at runtime that the graph tensor could fit in the target sizes.

  Args:
    graph_tensor: a graph tensor to check against target total sizes.
    target_total_sizes: target total sizes for each graph piece.

  Returns:
    Validation operations to execute within a tf.control_dependencies.
  """

  total_num_components = graph_tensor.total_num_components
  could_add_new_component = tf.math.less(
      total_num_components, target_total_sizes.total_num_components)
  assert_ops = [
      tf.debugging.assert_less_equal(
          total_num_components,
          tf.constant(
              target_total_sizes.total_num_components,
              dtype=total_num_components.dtype),
          message=(
              'Could not pad graph as it already has more graph components'
              ' then it is allowed by `target_total_sizes.total_num_components`'
          ))
  ]

  def _check(entity_type: str, entity_name: str, total_size: tf.Tensor,
             target_total_size: Optional[int]):
    if target_total_size is None:
      raise ValueError(
          f'The target total number of \'{entity_name}\' {entity_type} must be'
          ' specified as'
          f' `target_total_sizes.total_num_{entity_type}[\'{entity_name}\']`.')
    target_total_size = tf.constant(target_total_size, dtype=total_size.dtype)
    assert_ops.append(
        tf.debugging.assert_less_equal(
            total_size,
            target_total_size,
            message=(
                f'Could not pad \'{entity_name}\' as it already has more'
                f' {entity_type} then it is allowed by the'
                f' `target_total_sizes.total_num_{entity_type}[\'{entity_name}\']`.'
            )))

    assert_ops.append(
        tf.debugging.Assert(
            tf.math.logical_or(could_add_new_component,
                               tf.math.equal(total_size, target_total_size)),
            data=[
                f'Could not pad \'{entity_name}\' {entity_type}. To do this, at'
                ' least one graph component must be added to the input graph.'
                ' The latter is not possible as the input graph has already'
                ' `target_total_sizes.total_num_components` graph components.',
                f' The total number of \'{entity_name}\' {entity_type} is ',
                total_size, ' The total number of graph components is ',
                total_num_components
            ]))

  for name, item in graph_tensor.node_sets.items():
    _check('nodes', name, item.total_size,
           target_total_sizes.total_num_nodes.get(name, None))

  for name, item in graph_tensor.edge_sets.items():
    _check('edges', name, item.total_size,
           target_total_sizes.total_num_edges.get(name, None))

  return assert_ops
