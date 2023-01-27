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
"""Contains GPT-GNN related tensor operations."""
from __future__ import annotations

import dataclasses
from typing import Optional, Text

import tensorflow as tf
import tensorflow_gnn as tfgnn


def _assert_rank1_int(t: tf.Tensor, tensor_name: Text) -> None:
  """Checks that the tensor `t` has dtype int32 or int64 with rank 1."""
  if t.shape.rank != 1 or t.dtype not in (tf.int32, tf.int64):
    raise ValueError(
        f'Expected `{tensor_name}` as rank-1 integer tensor,'
        f' got rank={t.shape.rank}, dtype={t.dtype.name}'
    )


def _row_lengths_to_row_ids(
    row_lengths: tf.Tensor, sum_row_lengths_hint: Optional[int] = None
) -> tf.Tensor:
  """Converts rank-1 ragged row lengths to row ids.

  For XLA compatibility `sum_row_lengths_hint` has to be provided to guarantee
  statically (compile-time) known result size.

  Example:

  ```python
  _row_lengths_to_row_ids([2, 1, 0, 2], 5)  # returns [0, 0, 1, 3, 3].
  ```

  Args:
    row_lengths: rank-1 integer tensor with ragged row lengths.
    sum_row_lengths_hint: value optionally provided by the client if the sum of
      `row_lengths` is known statically.

  Returns:
    Rank-1 integer tensor with ragged row ids.
  """
  _assert_rank1_int(row_lengths, 'row_lengths')

  row_starts = tf.math.cumsum(row_lengths)

  sum_row_lengths = (
      tf.reduce_sum(row_lengths)
      if sum_row_lengths_hint is None
      else sum_row_lengths_hint
  )

  cuts = tf.math.unsorted_segment_sum(
      tf.ones_like(row_starts), row_starts, sum_row_lengths + 1
  )
  result = tf.math.cumsum(cuts, exclusive=False)
  return result[:sum_row_lengths]


def _num_samples_per_segment(
    segment_sizes: tf.Tensor, max_samples: int
) -> tf.Tensor:
  """Returns a tensor representing counts to sample within each segment.

  Args:
    segment_sizes: Sizes tensors specifying counts within each segment.
    max_samples: Max number of entities to sample in total. Note that from each
      segment there'll be at most: `max_samples` / <segment_count> entities
      sampled based on the respective segment count or 1 entity per segment if
      the division `max_samples` / <segment_count> is below one. To improve the
      chances of samples returned from each segment, keep this number higher.

  Returns:
    Return a tensor with shape (num_segments,) that has the counts of elements
    to sample from each segment.
  """
  num_segments = segment_sizes.shape[0]
  segment_sample_count = max_samples // num_segments
  if segment_sample_count < 1:
    samples_per_segment = tf.ones_like(segment_sizes)
  else:
    segment_sample_count = tf.convert_to_tensor(
        [segment_sample_count], dtype=segment_sizes.dtype
    )
    samples_per_segment = tf.tile(
        segment_sample_count, tf.convert_to_tensor([num_segments])
    )

  return tf.math.minimum(samples_per_segment, segment_sizes)


def _find_different_indices(
    indices: tf.Tensor,
    another_indices: tf.Tensor,
    max_cols: int,
    default_value: int = -1,
) -> tf.Tensor:
  """Returns values(ids) from `indices` tensor that aren't in `another_indices`.

  Find the set difference between `indices` and `another_indices`
  and returns the difference in a dense matrix where columns are bounded by
  `max_cols` or the minimum number of identified differences in a row.

  Args:
    indices: A 2D matrix of indices.
    another_indices: Another 2D matrix of indices.
    max_cols: Maximum number of different elements to keep per row in the
      returned matrix.
    default_value: Default value to be used for dense tensor generation, it
      needs to be a value outside the vocabulary of either indices, and can be
      used as an oov id. Default is -1.

  Returns:
    Returns a dense tensor with shape [X, MIN(`max_cols`, Y)] where X is the
    first dimension of both `indices` and `another_indices`, while Y is
    the minimum row length of their set difference:
    (`indices` - `another_indices`).
  """
  diffed_indices = tf.sets.difference(indices, another_indices)
  diffed_indices = tf.sparse.to_dense(
      diffed_indices, default_value=default_value
  )
  ragged_diffed_indices = tf.RaggedTensor.from_tensor(
      diffed_indices, padding=default_value
  )
  row_count = indices.shape[0]
  min_index = int(tf.math.reduce_min(ragged_diffed_indices.row_lengths()))
  return ragged_diffed_indices.to_tensor(
      shape=[row_count, min(min_index, max_cols)]
  )


def segment_samples_to_indices(
    samples_per_segment: tf.Tensor,
    segment_sizes: tf.Tensor,
    *,
    seed: Optional[int] = None,
    sum_sample_sizes_hint: Optional[int] = None,
    sum_segment_sizes_hint: Optional[int] = None,
) -> tf.Tensor:
  """Returns sampled ids for each segment using counts in `samples_per_segment`.

  For XLA compatibility `sum_sample_sizes_hint` and `sum_segment_sizes_hint` has
  to be provided to guarantee statically (compile-time) known output size.

  Args:
    samples_per_segment: Number of samples to collect from each respective
      segment. This tensor's elements should be smaller than or equal to the
      `segment_sizes`.
    segment_sizes: Actual segment counts.
    seed: A random seed for node index shuffling.
    sum_sample_sizes_hint: Optional sizes value provided if the sum of the
      `samples_per_segment` known statically.
    sum_segment_sizes_hint: Optional sizes vlaue provided if the sum of the
      `segment_sizes` known statically.

  Returns:
    Tensor that lists ids sampled from a segment with sizes tensors passed
    in as `segment_sizes`. Returned tensor will have a certain number of ids
    sampled from each segment which is specified in `samples_per_segment`.
  """
  validation_ops = [
      tf.debugging.assert_shapes(
          [(segment_sizes, ('N',)), (samples_per_segment, ('N',))],
          message=(
              f'Tensor shapes for {segment_sizes} and '
              f'{samples_per_segment} has to match.'
          ),
      ),
      tf.debugging.assert_equal(
          tf.math.greater_equal(segment_sizes, samples_per_segment),
          tf.ones_like(segment_sizes, tf.bool),
          message=(
              f'Segment sizes in {segment_sizes} must be greater than or '
              f'equal to samples counts in {samples_per_segment}.'
          ),
      ),
  ]
  with tf.control_dependencies(validation_ops):
    samples_row_ids = _row_lengths_to_row_ids(
        samples_per_segment, sum_sample_sizes_hint
    )
    samples_segment_offsets = tf.cumsum(samples_per_segment, exclusive=True)
    samples_segment_offsets_per_index = tf.gather(
        samples_segment_offsets, samples_row_ids
    )
    # Find zero based indices within each segment.
    samples_segment_indices = tf.range(tf.math.reduce_sum(samples_per_segment))
    samples_segment_indices = tf.math.subtract(
        samples_segment_indices, samples_segment_offsets_per_index
    )

    full_segment_offsets = tf.cumsum(segment_sizes, exclusive=True)
    samples_full_segment_offsets_per_indice = tf.gather(
        full_segment_offsets, samples_row_ids
    )
    # Find actual node indices from full node ids, by adding the respective
    # segment row offsets to segment based indices
    samples_node_indices = tf.math.add(
        samples_segment_indices, samples_full_segment_offsets_per_indice
    )

    full_segment_row_ids = _row_lengths_to_row_ids(
        segment_sizes, sum_segment_sizes_hint
    )
    segment_shuffled_full_node_ids = (
        tfgnn.experimental.segment_random_index_shuffle(
            segment_ids=full_segment_row_ids, seed=seed
        )
    )
    subsampled_node_ids = tf.gather(
        segment_shuffled_full_node_ids, samples_node_indices
    )
    return subsampled_node_ids


@dataclasses.dataclass
class ConnectedNodes:
  """Keeps connected source and target node ids and node-set names.

  Attributes:
    source_node_name: Node-set name of the source endpoint.
    source_node_ids: Node ids from the source node-set.
    target_node_name: Node-set name of the target endpoint.
    target_node_ids: Node ids from the target node-set.
  """

  source_node_name: str
  source_node_ids: tf.Tensor
  target_node_name: str
  target_node_ids: tf.Tensor


def _get_connected_node_ids(
    graph: tfgnn.GraphTensor,
    *,
    edge_set_names: list[str],
    target_node_tag: tfgnn.IncidentNodeTag,
) -> ConnectedNodes:
  """Returns source and target node ids connected with edges in edge_set_names.

  Args:
    graph: A scalar GraphTensor.
    edge_set_names: Edge-set names for existing(positive) edge-set information.
      Each edge-set should have the same source and target node-set name.
    target_node_tag: Incident side of the edge-set adjacency representing target
      node set.

  Returns:
    A ConnectedNodes class which keeps source node ids, source node name and
    target node ids and target node name, that are connected with edges in
    `edge_set_names`. Target node-set is identified via the target_node_tag.

  Raises:
    ValueError: if graph is not scalar (rank > 0) or edge_set_names contains
    edge_sets with different source and target names.
  """
  tfgnn.check_scalar_graph_tensor(graph, '_get_connected_node_ids()')
  source_node_id_list = []
  target_node_id_list = []
  source_node_name = None
  target_node_name = None
  for edge_set_name in edge_set_names:
    edge_set = graph.edge_sets[edge_set_name]
    positive_adjacency = edge_set.adjacency
    if not source_node_name:
      source_node_name = positive_adjacency.source_name
    if source_node_name != positive_adjacency.source_name:
      raise ValueError(
          'source_name and target_name doesnt match among the edge_sets in '
          'the positive_edge_set_names, source_node_name: '
          f'{positive_adjacency.source_name} vs {source_node_name}'
      )
    if not target_node_name:
      target_node_name = positive_adjacency.target_name
    if target_node_name != positive_adjacency.target_name:
      raise ValueError(
          'source_name and target_name does not match among the'
          ' edge_sets in the positive_edge_set_names, '
          f'target_node_name: {positive_adjacency.target_name} vs '
          f'{target_node_name}'
      )
    source_node_id_list.append(positive_adjacency.source)
    target_node_id_list.append(positive_adjacency.target)
  if target_node_tag == tfgnn.TARGET:
    sampling_pos_source_node_ids = tf.concat(source_node_id_list, axis=0)
    sampling_pos_target_node_ids = tf.concat(target_node_id_list, axis=0)
  else:
    sampling_pos_source_node_ids = tf.concat(target_node_id_list, axis=0)
    sampling_pos_target_node_ids = tf.concat(source_node_id_list, axis=0)
  sampling_source_node_tag = tfgnn.reverse_tag(target_node_tag)
  sampling_source_node_name = positive_adjacency.node_set_name(
      sampling_source_node_tag
  )
  sampling_target_node_name = positive_adjacency.node_set_name(target_node_tag)
  return ConnectedNodes(
      sampling_source_node_name,
      sampling_pos_source_node_ids,
      sampling_target_node_name,
      sampling_pos_target_node_ids,
  )


def sample_unconnected_nodes(
    graph: tfgnn.GraphTensor,
    *,
    edge_set_names: list[str],
    max_negative_samples: int,
    negative_samples_node_tag: tfgnn.IncidentNodeTag,
    sample_buffer_scale: int = 2,
    seed: Optional[int] = None,
) -> tfgnn.Field:
  """Generates unconnected node-ids for the node-set of the specified tag.

  This operation returns up to `max_negative_samples` node-ids(sampling target
  node) from the node-set with the node-tag `negative_samples_node_tag` for each
  of the node-ids(sampling source) at the other end-point of the edge_sets
  specified by the `edge_set_names`. Returned node-ids indicate that there isn't
  an edge between the sampling target node-id and the respective sampling source
  node-id, also node-ids will be ordered by index starting with the smallest
  possible node-id. In case you prefer to shuffle the nodes within the graph
  tensor then please use  `shuffle_nodes()` before `sample_unconnected_nodes()`.

  Args:
    graph: A scalar GraphTensor.
    edge_set_names: Edge-set names for existing(positive) edge-set information.
      Each edge-set should have the same source and target node-set name.
    max_negative_samples: Maximum number of unconnected(negative) node-ids to
      return per node.
    negative_samples_node_tag: Incident side of the edge-set adjacency, among
      which unconnected(negative) node-ids will be generated.
    sample_buffer_scale: This scaling parameter will increase the total number
      of elements sampled per segment. Essentially we sample
      `max_negative_samples`*`sample_buffer_scale` elements in total.
    seed: Random seed parameter for shuffling node indices per segment before
      subsampling node-ids from each.

  Returns:
    Tensor specifying unconnected(negative) node-ids(sampling target) generated
    for the node-set identified with tag `negative_samples_node_tag` for each
    node(sampling source) at the other endpoint of the edge-set among the given
    `edge_set_names`. Returned tensor shape will be
    [X, MIN(`max_negative_samples`, Y)]; X equal to the total_size of the
    sampling source node-set, Y is the minimum of `max_negative_samples` and the
    minimum number of unconnected nodes across the sampling source node-set.
    Returned node-ids will be ordered at each row, starting at the smallest
    possible negative node-id.

  Raises:
    ValueError: if graph is not scalar (rank > 0) or positive_edge_set_names is
    empty.
  """
  tfgnn.check_scalar_graph_tensor(graph, 'sample_unconnected_nodes()')
  if not edge_set_names:
    raise ValueError('edge_set_names cant be empty')

  # Collect the list of source and target node ids which are connected by the
  # given edge_sets.
  sampling_connected_nodes = _get_connected_node_ids(
      graph,
      edge_set_names=edge_set_names,
      target_node_tag=negative_samples_node_tag,
  )

  total_sampling_sources = graph.node_sets[
      sampling_connected_nodes.source_node_name
  ].total_size
  sampling_target_node_sizes = graph.node_sets[
      sampling_connected_nodes.target_node_name
  ].sizes
  sampling_target_node_sizes_hint = graph.node_sets[
      sampling_connected_nodes.target_node_name
  ].spec.total_size

  # Generate sample count to collect from each component.
  segment_samples_count = _num_samples_per_segment(
      sampling_target_node_sizes, max_negative_samples * sample_buffer_scale
  )
  subsampled_negative_node_ids = segment_samples_to_indices(
      segment_samples_count,
      sampling_target_node_sizes,
      seed=seed,
      sum_sample_sizes_hint=tf.reduce_sum(segment_samples_count),
      sum_segment_sizes_hint=sampling_target_node_sizes_hint,
  )
  # Tile negative_node_ids to represent the potential negative nodes for each
  # sampling source node (SSN).
  # [0 1 ... N, 0 1 ... N, ... 0 1 ... N]
  # < SSN_1   > <  SSN_2 > ... <  SSN_X >
  # Tile to represent each sampling source node in their respective row.
  subsampled_negative_node_ids = tf.tile(
      tf.expand_dims(subsampled_negative_node_ids, axis=0),
      [total_sampling_sources, 1],
  )
  # Get positive(existing edge-sets) sampling target node-ids for each
  # sampling source node.
  sampling_pos_source_node_sorted_indices = tf.argsort(
      sampling_connected_nodes.source_node_ids
  )
  sampling_pos_source_node_ids = tf.gather(
      sampling_connected_nodes.source_node_ids,
      sampling_pos_source_node_sorted_indices,
  )
  sampling_pos_target_node_ids = tf.gather(
      sampling_connected_nodes.target_node_ids,
      sampling_pos_source_node_sorted_indices,
  )
  positive_target_node_ids = tf.RaggedTensor.from_value_rowids(
      sampling_pos_target_node_ids,
      sampling_pos_source_node_ids,
      nrows=total_sampling_sources,
  )
  positive_target_node_ids = positive_target_node_ids.to_tensor(
      default_value=-1
  )
  unconnected_node_ids = _find_different_indices(
      subsampled_negative_node_ids,
      positive_target_node_ids,
      max_negative_samples,
  )
  return unconnected_node_ids
