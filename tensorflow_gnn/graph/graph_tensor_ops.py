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
"""Operations on the GraphTensor."""

from __future__ import annotations

import functools
from typing import Any, Callable, Collection, List, Mapping, Optional

import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import broadcast_ops
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import pool_ops
from tensorflow_gnn.graph import tag_utils
from tensorflow_gnn.graph import tensor_utils as utils
from tensorflow_gnn.keras import keras_tensors as kt


Field = const.Field
FieldName = const.FieldName
NodeSetName = const.NodeSetName
EdgeSetName = const.EdgeSetName
IncidentNodeTag = const.IncidentNodeTag
GraphTensor = gt.GraphTensor
GraphKerasTensor = kt.GraphKerasTensor

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


_EdgeFeatureInitializer = Callable[[gt.FieldName, tf.TensorShape], tf.Tensor]


def _zero_edge_feat_init(
    feature_name: gt.FieldName, shape: tf.TensorShape) -> tf.Tensor:
  """Returns zeros with shape `shape`."""
  del feature_name
  return tf.zeros(shape, dtype=tf.float32)


def add_self_loops(
    graph: GraphTensor, edge_set_name: gt.EdgeSetName, *,
    edge_feature_initializer: _EdgeFeatureInitializer = _zero_edge_feat_init,
    ) -> GraphTensor:
  """Adds self-loops for edge with name `edge_set_name` EVEN if already exist.

  Edge `edge_set_name` must connect pair of nodes of the same node set.

  Args:
    graph: GraphTensor without self-loops. NOTE: If it has self-loops, then
      another round if self-loops will be added.
    edge_set_name: Must connect node pairs of the same node set.
    edge_feature_initializer: initializes edge features for the self-loop edges.
      It defaults to initializing features of new edges to tf.zeros.

  Returns:
    GraphTensor with self-loops added.
  """
  gt.check_scalar_graph_tensor(graph, 'tfgnn.add_self_loops()')

  edge_set = graph.edge_sets[edge_set_name]

  if edge_set.adjacency.source_name != edge_set.adjacency.target_name:
    raise ValueError(
        'Edge set "%s" must connect source and target nodes from the same node '
        'set. Got: node set names %s != %s' % (
            edge_set_name, edge_set.adjacency.source_name,
            edge_set.adjacency.target_name))

  node_set_name = edge_set.adjacency.source_name
  node_set = graph.node_sets[node_set_name]

  num_nodes = node_set.total_size
  num_edges = edge_set.total_size

  self_loop_source = tf.range(  # == self_loop_target
      num_nodes, dtype=edge_set.adjacency.target.dtype)

  # shape (components, 2)
  stacked_sizes = tf.stack([edge_set.sizes, node_set.sizes], 1)

  # [ |E1|, |N1|, |E2|, |N2|, ... ]
  alternate_sizes = tf.reshape(stacked_sizes, [-1])
  # where |Ei| and |Ni| are number of edges and nodes in i'th component.

  # [0, 0, ..,    1, 1, ..,   2, 2, ...,   3, 3, 3, ...]
  #  ---------    ---------   ----------   ------------
  #    |E1|          |N1|         |E2|        |N2|
  segment_indicator = utils.repeat(
      tf.range(tf.shape(alternate_sizes)[0], dtype=tf.int32), alternate_sizes,
      repeats_sum_hint=tf.get_static_value(num_nodes + num_edges))

  node_indicator = segment_indicator % 2  # Marks odd (i.e. node positions)
  edge_indicator = 1 - node_indicator     # Marks even (i.e. edge positions)

  # [0, 1, 2,..,  x, x, ...,  |E1|, |E1|+1,..,  x, x, x, ...];  "x" = dont care.
  #  -----------  ---------   ----------------  ------------
  #    |E1|          |N1|           |E2|            |N2|
  edge_positions = tf.cumsum(edge_indicator) - 1
  # Some "x" values can be -1. Remove.
  edge_positions = tf.clip_by_value(
      edge_positions, clip_value_min=0, clip_value_max=num_edges)

  # [x, x, x,..,  0, 1, ...,  x, x,..,  |N1|, |N1|+1, ...];  "x" = dont care.
  #  -----------  ---------   --------  -----------------
  #    |E1|          |N1|        |E2|         |N2|
  node_positions = tf.cumsum(node_indicator) - 1

  # Some "x" values can be -1. Remove.
  node_positions = tf.clip_by_value(
      node_positions, clip_value_min=0, clip_value_max=num_nodes)

  bool_edge_indicator = tf.cast(edge_indicator, tf.bool)
  indices = tf.where(bool_edge_indicator, edge_positions,
                     node_positions + num_edges)

  new_source = tf.gather(
      tf.concat([edge_set.adjacency.source, self_loop_source], 0), indices)
  new_target = tf.gather(
      tf.concat([edge_set.adjacency.target, self_loop_source], 0), indices)

  updated_edge_sets = {}
  for existing_edge_set_name, existing_edge_set in graph.edge_sets.items():
    if edge_set_name != existing_edge_set_name:
      # Unmodified.
      updated_edge_sets[existing_edge_set_name] = existing_edge_set
      continue

    updated_features = {}
    for feat_name, existing_feat_value in existing_edge_set.features.items():
      feat_shape = tf.shape(existing_feat_value)[1:]
      self_loop_edge_feature = edge_feature_initializer(feat_name, feat_shape)
      self_loop_edge_feature = utils.repeat(
          tf.expand_dims(self_loop_edge_feature, axis=0),
          tf.expand_dims(num_nodes, axis=0),
          repeats_sum_hint=tf.get_static_value(num_nodes + 0))
      # Transposing twice so that we get broadcasting for free (instead of
      # reshaping, adding 1's on some axis dimensions).
      new_feature = tf.transpose(tf.where(
          bool_edge_indicator,
          tf.transpose(tf.gather(existing_feat_value, edge_positions)),
          tf.transpose(tf.gather(self_loop_edge_feature, node_positions))))
      updated_features[feat_name] = new_feature

    updated_edge_sets[existing_edge_set_name] = gt.EdgeSet.from_fields(
        sizes=tf.reduce_sum(stacked_sizes, 1),
        features=updated_features,
        adjacency=adj.Adjacency.from_indices(
            source=(node_set_name, new_source),
            target=(node_set_name, new_target),
        )
    )

  return GraphTensor.from_pieces(
      context=graph.context, node_sets=graph.node_sets,
      edge_sets=updated_edge_sets)


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
  node_value = gt.resolve_value(
      node_set, feature_value=feature_value, feature_name=feature_name)

  sizes = node_set.sizes
  assert_positive_sizes = tf.debugging.assert_positive(
      sizes,
      message=f'tfgnn.gather_first_node(..., node_set_name={node_set_name}) '
      'called for a graph in which one or more components contain no nodes.')
  with tf.control_dependencies([assert_positive_sizes]):
    components_starts = tf.math.cumsum(sizes, exclusive=True)
    return tf.gather(node_value, components_starts)


def mask_edges(
    graph: GraphTensor,
    edge_set_name: gt.EdgeSetName,
    boolean_edge_mask: tf.Tensor,
    masked_info_edge_set_name: Optional[gt.EdgeSetName] = None) -> GraphTensor:
  """Creates a GraphTensor after applying edge_mask over the specified edge-set.

  After applying the given boolean mask to the edge-set, removed edges will be
  kept on a different edge-set within the GraphTensor generated by this
  function. Edge masking doesn't change the node sets or the context node
  information.

  Not compatible with XLA.

  Args:
    graph: A scalar GraphTensor.
    edge_set_name: Name of edge-set to apply the boolean mask.
    boolean_edge_mask: A boolean mask with shape `[num_edges]` to mask edges in
      the specified edge-set of the `graph` with rank=1.
    masked_info_edge_set_name: Masked out edge-set information will be kept in a
      new edge-set, with name masked_info_edge_set_name.

  Returns:
    GraphTensor with configured edges masked and removed edges information
      stored in the edget_set with name masked_info_edge_set_name.
  """
  gt.check_scalar_graph_tensor(graph, 'tfgnn.mask_edges()')
  if edge_set_name not in graph.edge_sets:
    raise ValueError(
        f'Please ensure edge_set_name: {edge_set_name} exists as an edge-set '
        f'within the graph: {graph} passed to mask_edges().')
  if (masked_info_edge_set_name and
      masked_info_edge_set_name in graph.edge_sets):
    raise ValueError(
        f'Please ensure edge_set_name:{masked_info_edge_set_name} is not an '
        f'existing edge-set within the graph: {graph} passed to mask_edges().')
  edge_set = graph.edge_sets[edge_set_name]
  adj_indices = edge_set.adjacency.get_indices_dict()

  validation_ops = [
      tf.debugging.assert_equal(
          tf.shape(boolean_edge_mask),
          tf.shape(list(adj_indices.values())[0][1]),
          f'boolean_edge_mask should have the same shape with the adjacency '
          f'node index vectors of the edge-set: {edge_set_name} '
          f'adjacency shape: {tf.shape(edge_set.adjacency[const.SOURCE])}')
  ]
  with tf.control_dependencies(validation_ops):
    negated_boolean_edge_mask = tf.math.logical_not(boolean_edge_mask)
    masked_features = {}
    masked_info_features = {}
    for feature_name, feature_value in edge_set.features.items():
      if utils.is_ragged_tensor(feature_value):
        masked_feature = tf.ragged.boolean_mask(feature_value,
                                                boolean_edge_mask)
        masked_info_feature = tf.ragged.boolean_mask(feature_value,
                                                     negated_boolean_edge_mask)
      else:
        assert utils.is_dense_tensor(feature_value)
        masked_feature = tf.boolean_mask(feature_value, boolean_edge_mask)
        masked_info_feature = tf.boolean_mask(feature_value,
                                              negated_boolean_edge_mask)
      masked_features[feature_name] = masked_feature
      masked_info_features[feature_name] = masked_info_feature

    component_ids = utils.row_lengths_to_row_ids(
        edge_set.sizes, sum_row_lengths_hint=edge_set.spec.total_size)
    num_remaining_edges = tf.math.unsorted_segment_sum(
        tf.cast(boolean_edge_mask, edge_set.sizes.dtype), component_ids,
        edge_set.num_components)
    num_masked_edges = edge_set.sizes - num_remaining_edges

    masked_indices_update = {}
    masked_info_indices_update = {}
    for node_set_tag, (node_set_name, indices) in adj_indices.items():
      masked_indices = tf.boolean_mask(indices, boolean_edge_mask)
      masked_info_indices = tf.boolean_mask(indices, negated_boolean_edge_mask)
      masked_indices_update[node_set_tag] = (node_set_name, masked_indices)
      masked_info_indices_update[node_set_tag] = (node_set_name,
                                                  masked_info_indices)

    new_edge_sets = {}
    for gt_edge_set_name, gt_edge_set in graph.edge_sets.items():
      if gt_edge_set_name != edge_set_name:
        new_edge_sets[gt_edge_set_name] = gt_edge_set
    if isinstance(edge_set.adjacency, adj.Adjacency):
      masked_adj = adj.Adjacency.from_indices(
          source=masked_indices_update[const.SOURCE],
          target=masked_indices_update[const.TARGET],
          validate=const.validate_internal_results,
      )
      masked_info_adj = adj.Adjacency.from_indices(
          source=masked_info_indices_update[const.SOURCE],
          target=masked_info_indices_update[const.TARGET],
          validate=const.validate_internal_results,
      )
    else:
      masked_adj = adj.HyperAdjacency.from_indices(
          masked_indices_update, validate=const.validate_internal_results)
      masked_info_adj = adj.HyperAdjacency.from_indices(
          masked_info_indices_update, validate=const.validate_internal_results)
    new_edge_sets[edge_set_name] = gt.EdgeSet.from_fields(
        sizes=num_remaining_edges,
        features=masked_features,
        adjacency=masked_adj)
    if masked_info_edge_set_name:
      new_edge_sets[masked_info_edge_set_name] = gt.EdgeSet.from_fields(
          sizes=num_masked_edges,
          features=masked_info_features,
          adjacency=masked_info_adj)
    return GraphTensor.from_pieces(
        context=graph.context,
        node_sets=graph.node_sets,
        edge_sets=new_edge_sets)


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


def shuffle_features_globally(graph_tensor: GraphTensor,
                              *,
                              seed: Optional[int] = None) -> GraphTensor:
  """Shuffles context, node set and edge set features of a scalar GraphTensor.

  NOTE(b/277938756): This operation is not available in TFLite (last checked
  for TF 2.12).

  Args:
    graph_tensor: A scalar GraphTensor.
    seed: A seed for random uniform shuffle.

  Returns:
    A scalar GraphTensor `result` with the same graph structure as the input,
    but randomly shuffled feature tensors. More precisely, the result satisfies
    `result.node_sets[ns][ft][i] = graph_tensor.node_sets[ns][ft][sigma(i)]`
    for all node set names `ns` (including auxiliary node sets), all feature
    names `ft` and all indices `i` in `range(n)`, where `n` is the total_size
    of the node set and `sigma` is a permutation of `range(n)`.
    Moreover, the result satisfies the the analogous equations for all features
    of all edge sets (including auxiliary edge sets) and the context.
    The permutation `sigma` is drawn uniformly at random, independently for
    each graph piece and each feature(!). That is, separate features are
    permuted differently, and features on any one item (edge, node, component)
    can form combinations not seen on an input item.
  """
  gt.check_scalar_graph_tensor(graph_tensor,
                               'tfgnn.shuffle_features_globally()')

  context = _shuffle_features(graph_tensor.context.features, seed=seed)
  node_sets, edge_sets = {}, {}

  # NOTE(b/269076334): "_shadow/" node sets (for readout from edge sets)
  # are not exempted here, because they have no features anyways.
  for node_set_name, node_set in graph_tensor.node_sets.items():
    node_sets[node_set_name] = _shuffle_features(node_set.features, seed=seed)

  for edge_set_name, edge_set in graph_tensor.edge_sets.items():
    edge_sets[edge_set_name] = _shuffle_features(edge_set.features, seed=seed)

  return graph_tensor.replace_features(context, node_sets, edge_sets)


def reorder_nodes(graph_tensor: GraphTensor,
                  node_indices: Mapping[gt.NodeSetName, tf.Tensor],
                  *,
                  validate: bool = True) -> GraphTensor:
  """Reorders nodes within node sets according to indices.

  Args:
    graph_tensor: A scalar GraphTensor.
    node_indices: A mapping from node sets name to new nodes indices (positions
      within the node set). Each index is an arbitrary permutation of
      `tf.range(num_nodes)`, where `index[i]` is an index of an original node
      to be placed at position `i`.
    validate: If True, checks that `node_indices` are valid permutations.

  Returns:
    A scalar GraphTensor with randomly shuffled nodes within `node_sets`.

  Raises:
    ValueError: If `node_sets` contains non existing node set names.
    ValueError: If indices are not `rank=1` `tf.int32` or `tf.int64` tensors.
    InvalidArgumentError: if an index shape is not `[num_nodes]`.
    InvalidArgumentError: if an index is not a permutation of
      `tf.range(num_nodes)`. Only if validate is set to True.
  """
  gt.check_scalar_graph_tensor(graph_tensor, 'tfgnn.reorder_nodes()')
  diff = set(node_indices.keys()) - set(graph_tensor.node_sets.keys())
  if diff:
    raise ValueError(f'`node_indices` contains non existing node sets: {diff}.')

  node_sets, edge_sets = {}, {}
  new_nodes_positions = {}
  for node_set_name, node_set in graph_tensor.node_sets.items():
    if node_set_name not in node_indices:
      node_sets[node_set_name] = node_set
    else:
      indices = node_indices[node_set_name]
      num_nodes = node_set.total_size
      validation_ops = []
      validation_ops.append(
          tf.debugging.assert_equal(
              tf.size(indices, out_type=node_set.indices_dtype),
              num_nodes,
              message=(f'Indices for {node_set_name}'
                       ' must have shape `[num_nodes]`.')))

      if validate:
        segment_counts = tf.math.unsorted_segment_sum(
            tf.ones_like(indices), indices, num_nodes)
        validation_ops.append(
            tf.debugging.assert_equal(
                segment_counts, tf.ones_like(segment_counts),
                (f'Indices for {node_set_name}'
                 ' are not valid `tf.range(num_nodes)` permutation.')))

      with tf.control_dependencies(validation_ops):
        # Keep track of new nodes positions after shuffle as a `new_positions`
        # tensor, where `new_positions[i]` returns index of a new node position
        # (after shuffle) for the original node `i`. Note that we could use
        # segment sum operation as indices are unique mapping.
        old_positions = tf.range(num_nodes, dtype=node_set.indices_dtype)
        new_positions = tf.math.unsorted_segment_sum(
            old_positions, indices, num_nodes)

      new_nodes_positions[node_set_name] = new_positions
      features = tf.nest.map_structure(
          functools.partial(tf.gather, indices=indices), node_set.features)
      node_sets[node_set_name] = node_set.replace_features(features)

  for edge_set_name, edge_set in graph_tensor.edge_sets.items():
    if not isinstance(edge_set.adjacency, adj.HyperAdjacency):
      raise ValueError(
          'Expected adjacency type `tfgnn.Adjacency` or `tfgnn.HyperAdjacency`,'
          f' got {type(edge_set.adjacency).__name__},'
          f' edge set {edge_set_name}.')

    adj_indices = edge_set.adjacency.get_indices_dict()
    indices_update = {}
    for node_set_tag, (node_set_name, indices) in adj_indices.items():
      if node_set_name not in new_nodes_positions:
        continue
      indices_update[node_set_tag] = (
          node_set_name,
          tf.gather(new_nodes_positions[node_set_name], indices))
    if not indices_update:
      edge_sets[edge_set_name] = edge_set
    else:
      adj_indices.update(indices_update)
      if isinstance(edge_set.adjacency, adj.Adjacency):
        adjacency = adj.Adjacency.from_indices(
            source=adj_indices[const.SOURCE],
            target=adj_indices[const.TARGET],
            validate=const.validate_internal_results)
      else:
        adjacency = adj.HyperAdjacency.from_indices(
            adj_indices, validate=const.validate_internal_results)
      edge_sets[edge_set_name] = gt.EdgeSet.from_fields(
          features=edge_set.features, sizes=edge_set.sizes, adjacency=adjacency)

  result = GraphTensor.from_pieces(graph_tensor.context, node_sets, edge_sets)

  if const.validate_internal_results:
    result.spec.is_compatible_with(graph_tensor.spec)
  return result


def shuffle_nodes(graph_tensor: GraphTensor,
                  *,
                  node_sets: Optional[Collection[gt.NodeSetName]] = None,
                  seed: Optional[int] = None) -> GraphTensor:
  """Randomly reorders nodes of given node sets, within each graph component.

  The order of edges does not change; only their adjacency is modified to match
  the new order of shuffled nodes. The order of graph components (as created
  by `merge_graph_to_components()`) does not change, nodes are shuffled
  separately within each component.

  Auxiliary node sets are not shuffled, unless they are explicitly included in
  `node_sets`. Not shuffling is the correct behavior for the auxiliary node
  sets used by `tfgnn.structured_readout()`.

  NOTE(b/277938756): This operation is not available in TFLite (last checked
  for TF 2.12).

  Args:
    graph_tensor: A scalar GraphTensor.
    node_sets: An optional collection of node sets names to shuffle. If None,
      all node sets are shuffled.  Should not overlap with `shuffle_indices`.
    seed: Optionally, a fixed seed for random uniform shuffle.

  Returns:
    A scalar GraphTensor with randomly shuffled nodes within `node_sets`.

  Raises:
    ValueError: If `node_sets` containes non existing node set names.
  """
  gt.check_scalar_graph_tensor(graph_tensor, 'tfgnn.shuffle_nodes()')

  if node_sets is None:
    target_node_sets = set(
        node_set_name for node_set_name in graph_tensor.node_sets
        if not gt.get_aux_type_prefix(node_set_name))
  else:
    target_node_sets = set(node_sets)
    diff = target_node_sets - set(graph_tensor.node_sets.keys())
    if diff:
      raise ValueError(f'`node_sets` contains non existing node sets: {diff}.')

  def index_shuffle_singleton(node_set: gt.NodeSet) -> tf.Tensor:
    total_size = node_set.spec.total_size or node_set.total_size
    return tf.random.shuffle(
        tf.range(total_size, dtype=node_set.indices_dtype), seed=seed)

  def index_shuffle_generic(node_set: gt.NodeSet) -> tf.Tensor:
    row_ids = utils.row_lengths_to_row_ids(
        node_set.sizes, sum_row_lengths_hint=node_set.spec.total_size)
    return utils.segment_random_index_shuffle(segment_ids=row_ids, seed=seed)

  if graph_tensor.spec.total_num_components == 1:
    index_fn = index_shuffle_singleton
  else:
    index_fn = index_shuffle_generic

  node_indices = {
      node_set_name: index_fn(graph_tensor.node_sets[node_set_name])
      for node_set_name in target_node_sets
  }

  return reorder_nodes(
      graph_tensor, node_indices, validate=const.validate_internal_results)


def node_degree(graph_tensor: GraphTensor,
                edge_set_name: EdgeSetName,
                node_tag: IncidentNodeTag) -> Field:
  """Returns the degree of each node w.r.t. one side of an edge set.

  Args:
    graph_tensor: A scalar GraphTensor.
    edge_set_name: The name of the edge set for which degrees are calculated.
    node_tag: The side of each edge for which the degrees are calculated,
      specified by its tag in the edge set (e.g., `tfgnn.SOURCE`,
      `tfgnn.TARGET`).

  Returns:
    An integer Tensor of shape `[num_nodes]` and dtype equal to `indices_dtype`
    of the GraphTensor. Element `i` contains the number of edges in the given
    edge set that have node index `i` as their endpoint with the given
    `node_tag`. The dimension `num_nodes` is the number of nodes in the
    respective node set.
  """
  gt.check_scalar_graph_tensor(graph_tensor, 'tfgnn.node_degree()')
  adjacency = graph_tensor.edge_sets[edge_set_name].adjacency
  aggregate_node_count = pool_ops.pool_edges_to_node(
      graph_tensor,
      edge_set_name,
      node_tag,
      reduce_type='sum',
      feature_value=tf.ones_like(adjacency[node_tag]))
  return aggregate_node_count


def _shuffle_features(features: gt.Fields,
                      *,
                      seed: Optional[int] = None) -> gt.Fields:
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


def _check_line_graph_args(
    graph_tensor: gt.GraphTensor,
    *,
    connect_with_original_nodes: bool,
) -> None:
  """Check arguments for `tfgnn.convert_to_line_graph()`."""
  gt.check_scalar_graph_tensor(graph_tensor, 'tfgnn.convert_to_line_graph()')

  # Check that the graph tensor does not use a hyper adjacency
  for edge_set in graph_tensor.edge_sets.values():
    if (
        set(edge_set.adjacency.get_indices_dict())
        != {const.SOURCE, const.TARGET}
    ):
      # TODO(b/276726198): Handle more general HyperAdjacency, perhaps by
      # introducing a special key for line graph nodes.
      raise ValueError(
          'Expected an adjacency with exactly one tfgnn.SOURCE and one '
          'tfgnn.TARGET endpoint in `tfgnn.convert_to_line_graph()`. '
          'Other cases are currently not supported for '
          '`connect_with_original_nodes=True`.'
      )

  # Check for unhandled auxiliary node sets
  for node_set_name in graph_tensor.node_sets:
    if aux_type := gt.get_aux_type_prefix(node_set_name):
      if aux_type == '_readout':
        if not connect_with_original_nodes:
          # TODO(b/276907237): Handle special readout node set names.
          raise ValueError(
              'The original graph contains an auxiliary node set '
              f'\'{node_set_name}\'. Pass `connect_with_original_nodes=True` '
              'to keep any readout node/edge sets, or delete them before '
              'calling tfgnn.convert_to_line_graph().'
          )
      else:
        # TODO(b/276742948): Handle shadow node sets properly, perhaps by
        # converting them to a regular node readout.
        raise ValueError(
            'The original graph contains an auxiliary node set '
            f'\'{node_set_name}\' that `tfgnn.convert_to_line_graph()` '
            'currently cannot handle. Please delete this set before calling '
            '`tfgnn.convert_to_line_graph()`.'
        )

  # Check for unhandled auxiliary edge sets
  for edge_set_name in graph_tensor.edge_sets:
    if aux_type := gt.get_aux_type_prefix(edge_set_name):
      if aux_type == '_readout':
        if not connect_with_original_nodes:
          raise ValueError(
              'The original graph contains an auxiliary edge set '
              f'\'{edge_set_name}\'. Pass `connect_with_original_nodes=True` '
              'to keep any readout node/edge sets, or delete them before '
              'calling tfgnn.convert_to_line_graph().'
          )
      else:
        raise ValueError(
            'The original graph contains an auxiliary edge set '
            f'\'{edge_set_name}\' that `tfgnn.convert_to_line_graph()` '
            'currently cannot handle. Please delete this set before calling '
            '`tfgnn.convert_to_line_graph()`.'
        )


def _connect_line_graph_with_original(
    graph_tensor: gt.GraphTensor,
) -> tuple[dict[str, gt.NodeSet], dict[str, gt.EdgeSet]]:
  """Collect prefixed node sets and auxilary pieces and connect to line graph.

  For nodes, we collect (1) all original node sets and prefix them with
  'original/', and (2) all auxiliary node sets. For edges, we (1) collect and
  adjust all auxiliary edge sets, (2) add edges from line graph node sets
  (original edge sets) to the original   target nodes, and (3) add edges from
  original source nodes to the line graph node sets (original edge sets).

  Args:
    graph_tensor: The original graph tensor.

  Returns:
    Two dictionaries containing (1) the node sets and (2) the edge sets.
  """
  line_node_sets = {}
  line_edge_sets = {}

  for node_set_name, node_set in graph_tensor.node_sets.items():
    if gt.get_aux_type_prefix(node_set_name):
      line_node_sets[node_set_name] = node_set
    else:
      line_node_sets[f'original/{node_set_name}'] = node_set

  for edge_set_name, edge_set in graph_tensor.edge_sets.items():
    if gt.get_aux_type_prefix(edge_set_name):
      source_name = edge_set.adjacency.node_set_name(const.SOURCE)
      target_name = edge_set.adjacency.node_set_name(const.TARGET)
      if not gt.get_aux_type_prefix(source_name):
        source_name = f'original/{source_name}'
      if not gt.get_aux_type_prefix(target_name):
        target_name = f'original/{target_name}'

      line_edge_sets[edge_set_name] = gt.EdgeSet.from_fields(
          sizes=edge_set.sizes,
          adjacency=adj.Adjacency.from_indices(
              source=(source_name, edge_set.adjacency[const.SOURCE]),
              target=(target_name, edge_set.adjacency[const.TARGET]),
          ),
          features=edge_set.features,
      )
      continue

    line_edge_sets[
        f'original/from/{edge_set_name}'
    ] = gt.EdgeSet.from_fields(
        sizes=edge_set.sizes,
        adjacency=adj.Adjacency.from_indices(
            source=(edge_set_name, tf.range(edge_set.total_size)),
            target=(
                f'original/{edge_set.adjacency.node_set_name(const.TARGET)}',
                edge_set.adjacency[const.TARGET],
            ),
        ),
    )
    line_edge_sets[
        f'original/to/{edge_set_name}'
    ] = gt.EdgeSet.from_fields(
        sizes=edge_set.sizes,
        adjacency=adj.Adjacency.from_indices(
            source=(
                f'original/{edge_set.adjacency.node_set_name(const.SOURCE)}',
                edge_set.adjacency[const.SOURCE],
            ),
            target=(edge_set_name, tf.range(edge_set.total_size)),
        ),
    )

  return line_node_sets, line_edge_sets


def _get_line_graph_nodes(
    graph_tensor: gt.GraphTensor,
) -> dict[str, gt.NodeSet]:
  """Get the graph's edges as node sets for the line graph."""
  line_node_sets = {}
  for edge_set_name, edge_set in graph_tensor.edge_sets.items():
    if not gt.get_aux_type_prefix(edge_set_name):
      line_node_sets[edge_set_name] = gt.NodeSet.from_fields(
          features=edge_set.get_features_dict(), sizes=edge_set.sizes
      )
  return line_node_sets


def _get_line_graph_edges(
    graph_tensor: gt.GraphTensor,
    *,
    connect_from: const.IncidentNodeTag = const.TARGET,
    connect_to: const.IncidentNodeTag = const.SOURCE,
    use_node_features_as_line_graph_edge_features: bool = False,
) -> dict[str, gt.EdgeSet]:
  """Construct the edges for the line graph.

  Args:
    graph_tensor: Graph to convert to a line graph.
    connect_from: Specifies which endpoint of the original edges
      will be the source for the line graph edges.
    connect_to: Specifies which endpoint of the original edges
      will be the target for the line graph edges.
    use_node_features_as_line_graph_edge_features: Whether to use the original
      graph's node features as edge features in the line graph.

  Returns:
    A dictionary containing the line graph's edge sets.
  """
  line_edge_sets = {}

  for edge_set_name_source, edge_set_source in graph_tensor.edge_sets.items():
    if gt.get_aux_type_prefix(edge_set_name_source):
      continue

    node_set_name_source = edge_set_source.adjacency.node_set_name(
        connect_from
    )
    node_set = graph_tensor.node_sets[node_set_name_source]
    num_nodes = node_set.total_size

    for edge_set_name_target, edge_set_target in graph_tensor.edge_sets.items():
      if (
          edge_set_target.adjacency.node_set_name(connect_to)
          != node_set_name_source
      ) or gt.get_aux_type_prefix(edge_set_name_target):
        continue

      # Get the number of edges on each side of a node
      num_neighbors_source = pool_ops.pool_edges_to_node(
          graph_tensor,
          edge_set_name_source,
          connect_from,
          feature_value=tf.ones(
              edge_set_source.total_size, dtype=graph_tensor.indices_dtype
          ),
      )
      num_neighbors_target = pool_ops.pool_edges_to_node(
          graph_tensor,
          edge_set_name_target,
          connect_to,
          feature_value=tf.ones(
              edge_set_target.total_size, dtype=graph_tensor.indices_dtype
          ),
      )
      num_neighbors = num_neighbors_source * num_neighbors_target

      # Sort source and target edges by the connecting node index,
      # so they are matched after utils.repeat
      edge_idx_sorted_source = tf.argsort(
          edge_set_source.adjacency[connect_from]
      )
      edge_idx_sorted_target = tf.argsort(
          edge_set_target.adjacency[connect_to]
      )

      # Repeat source line idx according to the number of neighbors per node
      # (outer index loop)
      # e.g. [0 0 1 1 2 2 2 3 3 3]
      idx_line_source = utils.repeat(
          edge_idx_sorted_source,
          utils.repeat(num_neighbors_target, num_neighbors_source),
      )

      # Repeat target line idx according to the number of neighbors per node
      # (inner index loop)
      # via a ragged tensor
      # e.g. [0 1 0 1 2 3 4 2 3 4]
      edge_idx_target_grouped_by_node = tf.RaggedTensor.from_row_lengths(
          edge_idx_sorted_target, num_neighbors_target
      )
      idx_line_target = utils.repeat(
          edge_idx_target_grouped_by_node, num_neighbors_source
      ).flat_values

      # Calculate the number of edges per graph in a batch
      num_neighbors_grouped_by_graph = tf.RaggedTensor.from_row_lengths(
          num_neighbors, node_set.sizes
      )
      line_edge_sizes = tf.reduce_sum(
          num_neighbors_grouped_by_graph, axis=1
      )

      if use_node_features_as_line_graph_edge_features:
        # Create index mapping nodes to line graph edges
        node_idx = utils.repeat(tf.range(num_nodes), num_neighbors)

        # Create line graph feature dictionary
        line_features = dict()
        for feature_name, feature in node_set.get_features_dict().items():
          line_features[feature_name] = tf.gather(feature, node_idx)
      else:
        line_features = None

      line_edge_sets[
          f'{edge_set_name_source}=>{edge_set_name_target}'
      ] = gt.EdgeSet.from_fields(
          sizes=line_edge_sizes,
          adjacency=adj.Adjacency.from_indices(
              source=(edge_set_name_source, idx_line_source),
              target=(edge_set_name_target, idx_line_target),
          ),
          features=line_features,
      )

  return line_edge_sets


def _convert_to_non_backtracking_line_graph(
    graph_tensor: gt.GraphTensor,
    line_graph_tensor: gt.GraphTensor,
    *,
    node_tag_in: const.IncidentNodeTag,
    node_tag_out: const.IncidentNodeTag,
) -> gt.GraphTensor:
  """Convert a line graph to a non-backtracking line graph.

  This removes all backtracking edges from the line graph, i.e. edges where the
  node_tag_in endpoint of the incoming edge is the same as the node_tag_out
  endpoint of the outgoing edge.

  Args:
    graph_tensor: The original graph.
    line_graph_tensor: The line graph, potentially with backtracking edges.
    node_tag_in: receiver_tag of the "outer" node of a sending edge in the line
      graph, i.e. the node that is _not_ used to connect the sending and
      receiving edges.
    node_tag_out: receiver_tag of the "outer" node of a receiving edge in the
      line graph, i.e. the node that is _not_ used to connect the sending
      and receiving edges.

  Returns:
    GraphTensor without backtracking edges.
  """

  for edge_set_name, edge_set in line_graph_tensor.edge_sets.items():
    if (gt.get_aux_type_prefix(edge_set_name)
        or edge_set_name.startswith('original/')):
      continue

    original_edge_set_in = graph_tensor.edge_sets[
        edge_set.adjacency.node_set_name(const.SOURCE)
    ]
    original_edge_set_out = graph_tensor.edge_sets[
        edge_set.adjacency.node_set_name(const.TARGET)
    ]

    # Check if the "outer" node set names are different
    if (
        original_edge_set_in.adjacency.node_set_name(node_tag_in)
        != original_edge_set_out.adjacency.node_set_name(node_tag_out)
    ):
      continue

    node_in = broadcast_ops.broadcast_node_to_edges(
        line_graph_tensor,
        edge_set_name,
        const.SOURCE,
        feature_value=original_edge_set_in.adjacency[node_tag_in],
    )
    node_out = broadcast_ops.broadcast_node_to_edges(
        line_graph_tensor,
        edge_set_name,
        const.TARGET,
        feature_value=original_edge_set_out.adjacency[node_tag_out],
    )

    non_backtracking_edges = tf.math.not_equal(node_in, node_out)
    line_graph_tensor = mask_edges(
        line_graph_tensor, edge_set_name, non_backtracking_edges
    )

  return line_graph_tensor


def convert_to_line_graph(
    graph_tensor: gt.GraphTensor,
    *,
    connect_from: const.IncidentNodeTag = const.TARGET,
    connect_to: const.IncidentNodeTag = const.SOURCE,
    connect_with_original_nodes: bool = False,
    non_backtracking: bool = False,
    use_node_features_as_line_graph_edge_features: bool = False,
) -> gt.GraphTensor:
  """Obtain a graph's line graph.

  In the line graph, every edge in the original graph becomes a node,
  see https://en.wikipedia.org/wiki/Line_graph. Line graph nodes are connected
  whenever the corresponding edges share a specified endpoint.
  The _node_ sets of the resulting graph are the _edge_ sets of the original
  graph, with the same name. The resulting edge sets are named
  `{edge_set_name1}=>{edge_set_name2}`, for every pair of edge sets that
  connects through a common node set (as selected by the args). In particular,
  a pair of edges `u_0->u_1`, `v_0->v_1` will be connected if `u_i == v_j`,
  where the index `i in {0, 1}` is specified by `connect_from` and
  `j in {0, 1}` is specified by `connect_to`.

  If `non_backtracking=True`, edges will only be connected if they also fulfill
  `u_{1-i} != v_{1-j}`.

  This function only supports graphs where all edge set adjacencies contain only
  one SOURCE and one TARGET end point, i.e. non-hypergraphs.
  Note that representing undirected edges {u,v} as a pair of two directed edges
  u->v and v->u will result in a pair of separate line graph nodes.

  Auxiliary node sets are not converted. This will raise an error if (a) the
  graph contains a _readout node set and `preserve_node_sets` is False or
  (b) it contains a _shadow node set.

  Example: Consider a directed triangle represented as a homogeneous graph.
    The node set 'points' contains nodes a, b and c while the edge set 'lines'
    contains edges a->b, b->c, and c->a. The resulting line graph will
    contain a node set(!) 'lines' and an edge set 'lines=>lines'.
    The nodes in node set 'lines' correspond to the original edges;
    let's call them ab, bc, and ca. The edges in edge set 'lines=>lines'
    represent the connections of lines at points: ab->bc, bc->ca, and ca->ab.

    If `connect_with_original_nodes=True`, the resulting graph will retain
    the original nodes and their connection to edges as follows:
    Node set 'original/points' keeps the original nodes a, b and c, and there
    are two edge sets: 'original/to/lines' with edges a->ab, b->bc, c->ca,
    and 'original/from/lines' with edges ab->b, bc->c, ca->a.

  Args:
    graph_tensor: Graph to convert to a line graph.
    connect_from: Specifies which endpoint of the original edges
      will determine the source for the line graph edges.
    connect_to: Specifies which endpoint of the original edges
      will determine the target for the line graph edges.
    connect_with_original_nodes: If true, keep the original node sets (not the
      original edge sets) and connect them to line graph nodes according to
      source and target in the original graph. The node set names will be called
      `original/{node_set}` and the new edges `original/to/{edge_set}` for the
      SOURCE nodes and `original/from/{edge_set}` for the TARGET nodes.
    non_backtracking: Whether to return the non-backtracking line graph. Setting
      this to True will only connect edges where the "outer" nodes are
      different, i.e. `u_{1-i} != v_{1-j}`. For default connection settings,
      for every edge u->v this _removes_ line graph edges uv->vu. If
      connect_to=TARGET, this _removes_ line graph edges uv->uv.
    use_node_features_as_line_graph_edge_features: Whether to use the original
      graph's node features as edge features in the line graph.

  Returns:
    A GraphTensor defining the graph's line graph.
  """
  _check_line_graph_args(
      graph_tensor, connect_with_original_nodes=connect_with_original_nodes
  )

  line_node_sets = {}
  line_edge_sets = {}

  if connect_with_original_nodes:
    original_node_sets, aux_and_to_from_original_edge_sets = (
        _connect_line_graph_with_original(graph_tensor)
    )
    line_node_sets.update(original_node_sets)
    line_edge_sets.update(aux_and_to_from_original_edge_sets)

  line_node_sets.update(
      _get_line_graph_nodes(graph_tensor)
  )
  line_edge_sets.update(
      _get_line_graph_edges(
          graph_tensor,
          connect_from=connect_from,
          connect_to=connect_to,
          use_node_features_as_line_graph_edge_features
          =use_node_features_as_line_graph_edge_features,
      )
  )

  line_graph_tensor = gt.GraphTensor.from_pieces(
      node_sets=line_node_sets,
      edge_sets=line_edge_sets,
      context=graph_tensor.context,
  )

  if non_backtracking:
    line_graph_tensor = _convert_to_non_backtracking_line_graph(
        graph_tensor,
        line_graph_tensor,
        node_tag_in=tag_utils.reverse_tag(connect_from),
        node_tag_out=tag_utils.reverse_tag(connect_to),
    )

  return line_graph_tensor
