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
"""Implements `ext_ops.py` using parallel map (as `tf.map_fn`)."""

from typing import Optional, Tuple

import tensorflow as tf


def ragged_choice(
    num_samples: tf.Tensor,
    row_splits: tf.Tensor,
    *,
    global_indices: bool,
    seed: Optional[int],
) -> tf.RaggedTensor:
  """Implements `ext_ops.py:ragged_choice()`."""

  def fn(inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    num_samples, range_size = inputs
    values = tf.range(range_size, dtype=range_size.dtype)
    return tf.random.shuffle(values, seed)[:num_samples]

  row_starts, row_limits = row_splits[:-1], row_splits[1:]
  row_lengths = row_limits - row_starts
  num_samples = tf.math.minimum(num_samples, row_lengths)
  result = tf.map_fn(
      fn,
      (num_samples, row_lengths),
      fn_output_signature=tf.RaggedTensorSpec(
          [None],
          dtype=row_splits.dtype,
          ragged_rank=0,
          row_splits_dtype=row_splits.dtype,
      ),
  )
  if global_indices:
    result += tf.expand_dims(row_starts, axis=-1)
  return result


def ragged_unique(ragged: tf.RaggedTensor) -> tf.RaggedTensor:
  """Implements `ext_ops.py:ragged_unique()`."""

  def fn(values: tf.Tensor) -> tf.Tensor:
    return tf.unique(values, out_idx=ragged.row_splits.dtype).y

  return tf.map_fn(
      fn,
      ragged,
      fn_output_signature=tf.RaggedTensorSpec(
          [None],
          dtype=ragged.dtype,
          ragged_rank=0,
          row_splits_dtype=ragged.row_splits.dtype,
      ),
  )


def ragged_lookup(
    values: tf.RaggedTensor,
    vocabulary: tf.RaggedTensor,
    *,
    global_indices: bool,
    validate: bool = True,
) -> tf.RaggedTensor:
  """Implements `ext_ops.py:ragged_lookup()`."""

  dtype = values.row_splits.dtype

  def fn(inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    row_values, row_vocabulary = inputs
    joined = tf.concat([row_vocabulary, row_values], axis=-1)
    unique_values, idx = tf.unique(joined, out_idx=dtype)
    vocab_size = tf.size(row_vocabulary, out_type=dtype)
    result = idx[vocab_size:]
    validation_ops = []
    if validate:
      has_oov = tf.debugging.assert_less(
          tf.math.reduce_max(result, axis=-1),
          vocab_size,
          'Out of vocabulary values',
      )
      has_vocabulary_dups = tf.debugging.assert_equal(
          vocab_size,
          tf.size(unique_values, out_type=dtype),
          'Vocabulary has repeated values in rows',
      )
      validation_ops.extend([has_oov, has_vocabulary_dups])
    with tf.control_dependencies(validation_ops):
      return tf.identity(result)

  result = tf.map_fn(
      fn,
      (values, vocabulary),
      fn_output_signature=tf.RaggedTensorSpec(
          [None],
          dtype=dtype,
          ragged_rank=0,
          row_splits_dtype=dtype,
      ),
  )
  if global_indices:
    result += tf.expand_dims(vocabulary.row_starts(), axis=-1)
  return result

