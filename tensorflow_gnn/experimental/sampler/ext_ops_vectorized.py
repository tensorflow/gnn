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
"""Implements `ext_ops.py` using TF vectorized operations."""
from typing import Optional, Tuple

import tensorflow as tf
import tensorflow_gnn as tfgnn


def ragged_choice(
    num_samples: tf.Tensor,
    row_splits: tf.Tensor,
    *,
    global_indices: bool,
    seed: Optional[int],
) -> tf.RaggedTensor:
  """Implements `ext_ops.py:ragged_choice()`."""
  row_starts, row_limits = row_splits[:-1], row_splits[1:]
  row_lengths = row_limits - row_starts
  num_samples = tf.math.minimum(num_samples, row_lengths)

  row_ids = tf.ragged.row_splits_to_segment_ids(row_splits)
  # Shuffles indices of target nodes.
  # TODO(aferludin): row-based indices for `segment_random_index_shuffle`.
  shuffle_indices = tfgnn.experimental.segment_random_index_shuffle(
      segment_ids=row_ids, seed=seed
  )
  row_limits = row_starts + tf.cast(num_samples, row_starts.dtype)
  subranges = tf.ragged.range(row_starts, row_limits)
  result = tf.gather(shuffle_indices, subranges)
  if not global_indices:
    result -= tf.expand_dims(row_starts, axis=-1)
  return result


def ragged_unique(ragged: tf.RaggedTensor) -> tf.RaggedTensor:
  """Implements `ext_ops.py:ragged_unique()`."""
  global_vocabulary, row_ids, row_ids_base = _index_rows(ragged)
  unique_row_ids = tf.unique(row_ids).y
  unique_idx = unique_row_ids % row_ids_base
  result_values = tf.gather(global_vocabulary, unique_idx)
  result_rowids = tf.cast(
      unique_row_ids // row_ids_base, ragged.row_splits.dtype
  )

  return tf.RaggedTensor.from_value_rowids(
      result_values, result_rowids, ragged.nrows(), validate=False
  )


def ragged_lookup(
    values: tf.RaggedTensor,
    vocabulary: tf.RaggedTensor,
    *,
    global_indices: bool,
    validate: bool = True,
) -> tf.RaggedTensor:
  """Implements `ext_ops.py:ragged_lookup()`."""
  # Replace values and vocabulary items with their unique indices `idx`.
  joined = tf.concat([vocabulary, values], axis=-1)
  _, row_ids, _ = _index_rows(joined)

  # Because `joined` was created by joining vocabulary with values column-wise,
  # so vocabulary columns precede values columns. Having that `row_ids` do not
  # overlap between rows, the `tf.unique` operation returns indices of values
  # within their vocabularies from the same rows.
  row_vocabulary, joined_global_idx = tf.unique(row_ids, tf.int64)
  joined_global_idx = joined.with_values(joined_global_idx)

  # `joined_global_idx` are global in the sense that by construciton the indices
  # in each row have an offset so that they do not overlap with previous rows.
  # To restore row-based indices, we need to substract from all indices
  # within each row `i` the total number of unique ids for rows `j < i`.
  # If we could fully trust the input `vocabulary`, this could be as simple as
  # `tf.size(vocabulary[:i, :])`, but some values could be OOV and code should
  # detect this and raise an exception. So we compute and substract row offsets
  # explicitly.
  joined_idx = joined_global_idx - tf.reduce_min(
      joined_global_idx, axis=-1, keepdims=True
  )
  joined_idx = tf.cast(joined_idx, values.row_splits.dtype)

  # Selects values columns from `joined`.
  values_selector = tf.ragged.range(
      vocabulary.row_lengths(),
      joined.row_lengths(),
      row_splits_dtype=values.row_splits.dtype,
  )
  result = tf.gather(joined_idx, values_selector, batch_dims=1)
  validation_ops = []
  if validate:
    has_oov = tf.debugging.assert_less(
        tf.math.reduce_max(result, axis=-1),
        vocabulary.row_lengths(),
        'Out of vocabulary values',
    )
    has_vocabulary_dups = tf.debugging.assert_equal(
        tf.size(row_vocabulary, out_type=tf.int64),
        tf.size(vocabulary, out_type=tf.int64),
        'Vocabulary has repeated values in rows',
    )
    validation_ops.extend([has_oov, has_vocabulary_dups])

  if global_indices:
    result += tf.expand_dims(vocabulary.row_starts(), axis=-1)

  with tf.control_dependencies(validation_ops):
    return tf.identity(result)


def _index_rows(
    ragged: tf.RaggedTensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Helper function for computing row unique values of a ragged tensor.

  Args:
    ragged: The ragged tensor of rank 2 (ragged matrix).

  Returns:
    Three tensors as global unique values across ragged dimensions, value row
    ids (`row_ids`) with a shape `[tf.size(ragged)]` and row ids base (`base`)
    as a scalar. By the contruction: if any two values from the same row of
    `ragged` are equal, their `row_ids` are also equal and vice-versa; if two
    values are from different rows their `row_ids` are always different. The
    value index in the returned vocabulary could be found as `row_ids % base`
    and its row index as `row_ids // base`.
  """
  # Create vocabulary of global (ignoring row splits) unique values.
  # `idx` are indices of input values in the global vocabulary.
  vocabulary, idx = tf.unique(ragged.values, out_idx=tf.int64)

  # For any i, idx[i] < idx_upper_bound.
  idx_upper_bound = tf.size(vocabulary, out_type=tf.int64) + 1
  # For all values offset their indices `idx` by their row numbers in `values`
  # multiplied by the `idx_upper_bound`. By the contruction: if any two values
  # from the same row are equal, their `row_ids` are also equal and vice-versa;
  # if two values are from different rows their `row_ids` are always different.
  # This allows to find unique elements indepentently for each row.
  row_ids = idx + idx_upper_bound * tf.cast(ragged.value_rowids(), tf.int64)
  return vocabulary, row_ids, idx_upper_bound
