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
"""Extended set of Tensorflow operations."""
from typing import Optional

import tensorflow as tf
# copybara:uncomment_begin (internal implementation of ext_ops)
# from tensorflow_gnn.experimental.sampler import ext_ops_custom
# copybara:uncomment_end
from tensorflow_gnn.experimental.sampler import ext_ops_parallel
from tensorflow_gnn.experimental.sampler import ext_ops_vectorized


_IMPLEMENTATIONS = {
    'parallel': ext_ops_parallel,
    'vectorized': ext_ops_vectorized,
    # copybara:uncomment_begin (internal implementation of ext_ops)
    # 'custom': ext_ops_custom,
    # copybara:uncomment_end
}
_OPS_LIB = _IMPLEMENTATIONS['vectorized']


def set_ops_implementation(implementation_name: str):
  """Sets ops implementation to one of the supported libs."""
  global _OPS_LIB
  lib = _IMPLEMENTATIONS.get(implementation_name, None)
  if lib is None:
    raise ValueError(
        f'The {implementation_name} must be one of'
        f' {list(_IMPLEMENTATIONS.keys())}.'
    )
  _OPS_LIB = lib


def ragged_choice(
    num_samples: tf.Tensor,
    row_splits: tf.Tensor,
    *,
    global_indices: bool = False,
    seed: Optional[int] = None,
) -> tf.RaggedTensor:
  """Draws given number of elements without replacement from each ragged row.

  Example:

    ```python
    ragged_choice([2, 2], [0, 2, 5], global_indices=True)
    # [[1, 0], [2, 4]]
    ragged_choice([2, 2], [0, 2, 5], global_indices=False)
    # [[1, 0], [0, 2]]
    ```

  Args:
    num_samples: The maximum number of samples to draw from each row without
      replacement. An integer tensor broadcastable to `[nrows]` (so a scalar or
      a 1D `[nrows]` tensor),
    row_splits: Ragged dimensions splits. A 1-D integer tensor with shape
      `[nrows+1]`. Must not be empty, and must be sorted in ascending order.
      `row_splits[0]` must be zero and `row_splits[-1]` must be `nvals`.
    global_indices: If True, the returned indices are defined for flat values
      ignoring the ragged row splits. If False, the returned indices are defined
      independently for each ragged row.
    seed: A Python integer. Used to create a random seed for sampling.

  Returns:
    A ragged tensor of the same type as `row_splits` containing indices of
    sampled elements in each row (row-based or global, depending on the
    `global_indices` argument).
  """
  num_samples = tf.convert_to_tensor(num_samples)
  row_splits = tf.convert_to_tensor(row_splits)
  return _OPS_LIB.ragged_choice(
      num_samples, row_splits, global_indices=global_indices, seed=seed
  )


def ragged_unique(ragged: tf.RaggedTensor) -> tf.RaggedTensor:
  """Returns unique values for each ragged row preserving order.

  Example:

  ```python
    ragged_unique([
      ['a', 'a', 'b'],
      ['a', 'b', 'a', 'b', 'c']
    ])
    # [
    #   ['a', 'b'],
    #   ['a', 'b', 'c']
    # ]
  ```

  Args:
    ragged: The ragged tensor of rank 2 (ragged matrix).

  Returns:
    Ragged tensor of rank 2, containing unique values for each row of the input.
  """
  ragged = _convert_to_ragged_tensor(ragged)
  if ragged.shape.rank != 2:
    raise ValueError(
        f'Expected rank 2 ragged tensor, got {tf.type_spec_from_value(ragged)}'
    )

  return _OPS_LIB.ragged_unique(ragged)


def ragged_lookup(
    values: tf.RaggedTensor,
    vocabulary: tf.RaggedTensor,
    *,
    global_indices: bool = False,
    validate: bool = True,
) -> tf.RaggedTensor:
  """Indices of values in vocabulary found independently for each row.

  Example:

  ```python
    values = [['a', 'a', 'b'], ['a', 'b', 'a', 'b', 'c']]
    vocabulary = [['b', 'a'], ['a', 'b', 'c']]
    ragged_lookup(values, vocabulary)
    # [ [1, 1, 0], [0, 1, 0, 1, 2] ]
    ragged_lookup(values, vocabulary, global_indices=True)
    # [ [1, 1, 0], [0, 1, 0, 1, 2] ]
  ```

  Args:
    values: The ragged tensor of rank 2 (ragged matrix) with values to index.
    vocabulary: The ragged tensor of rank 2 (ragged matrix) with vocabulary
      values to use for indexing. Vocabulary must contain unique values in its
      rows.
    global_indices: If True, the returned indices are defined for flat values
      ignoring the ragged row splits. If False, the returned indices are defined
      independently for each ragged row.
    validate: If True, runs potentially expensive checks for OOV values and
      repeated values in vocabulary rows.

  Returns:
    Ragged tensor of rank 2 and the same type as values row split, with indices
    of values in vocabulary (row-based or global, depending on the
    `global_indices` argument).
  """
  values = _convert_to_ragged_tensor(values)
  vocabulary = _convert_to_ragged_tensor(vocabulary)

  if values.shape.rank != 2:
    raise ValueError(
        'Expected values to be rank 2 ragged tensor, got'
        f' {tf.type_spec_from_value(values)}'
    )

  if vocabulary.shape.rank != 2:
    raise ValueError(
        'Expected vocabulary to be rank 2 ragged tensor, got'
        f' {tf.type_spec_from_value(values)}'
    )

  if values.dtype != vocabulary.dtype:
    raise ValueError(
        'Values and vocabulary must have the same dtype, got'
        f' {values.dtype} != {vocabulary.dtype}'
    )
  if values.row_splits.dtype != vocabulary.row_splits.dtype:
    raise ValueError(
        'Values and vocabulary must have the same row splits dtype, got'
        f' {values.row_splits.dtype} != {vocabulary.row_splits.dtype}'
    )

  return _OPS_LIB.ragged_lookup(
      values, vocabulary, global_indices=global_indices, validate=validate
  )


def _convert_to_ragged_tensor(values) -> tf.RaggedTensor:
  if isinstance(values, (list, tuple)):
    return tf.ragged.constant(values)
  return values
