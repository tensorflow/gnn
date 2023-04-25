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
"""Utils for tensors and ragged tensors."""
from typing import List, Optional, Text, Union, Mapping

import tensorflow as tf

from tensorflow_gnn.graph import tf_internal

Value = Union[tf.Tensor, tf.RaggedTensor]
ValueSpec = Union[tf.TensorSpec, tf.RaggedTensorSpec]


def dims_list(tensor: tf.Tensor) -> List[Union[int, tf.Tensor]]:
  """Lists tensor dimensions with a preference for statically known sizes."""
  static = tensor.shape.as_list()
  if None not in static:
    return static
  dynamic = tf.unstack(tf.shape(tensor))
  return [(d if s is None else s) for s, d in zip(static, dynamic)]


def outer_dimension_size(value: Value) -> Union[int, tf.Tensor]:
  """Size of the outer-most dense or ragged tensor dimension."""
  if is_ragged_tensor(value):
    return outer_dimension_size(value.row_lengths())

  if is_dense_tensor(value):
    return dims_list(value)[0]

  raise ValueError(f'Unsupported type {type(value).__name__}')


def row_lengths_to_row_ids(
    row_lengths: tf.Tensor,
    sum_row_lengths_hint: Optional[int] = None) -> tf.Tensor:
  """Converts rank-1 ragged row lengths to row ids.

  For XLA compatibility `sum_row_lengths_hint` has to be provided to guarantee
  statically (compile-time) known result size.

  Example:

  ```python
  row_lengths_to_row_ids([2, 1, 0, 2], 5)  # returns [0, 0, 1, 3, 3].
  ```

  Args:
    row_lengths: rank-1 integer tensor with ragged row lengths.
    sum_row_lengths_hint: value optionally provided by the client if the sum of
      `row_lengths` known statically.

  Returns:
    Rank-1 integer tensor with ragged row ids.
  """
  _assert_rank1_int(row_lengths, 'row_lengths')

  row_starts = tf.math.cumsum(row_lengths, exclusive=False)

  sum_row_lengths = tf.reduce_sum(
      row_lengths) if sum_row_lengths_hint is None else sum_row_lengths_hint

  cuts = tf.math.unsorted_segment_sum(
      tf.ones_like(row_starts), row_starts, sum_row_lengths + 1)
  result = tf.math.cumsum(cuts, exclusive=False)
  return result[:sum_row_lengths]


def segment_random_index_shuffle(
    *, segment_ids: tf.Tensor, seed: Optional[int] = None) -> tf.Tensor:
  """Returns a global permutation that shuffles randomly within each segment.

  XLA compatible.

  NOTE: This function is **experimental** and may change or disappear in future
  releases of the TF-GNN library.

  NOTE: This implementation is based on `tf.argsort()`. Because TF argsort
  returns `tf.int32` indices, it is required that the total number of values in
  all segments, `N = tf.size(segment_ids)`, is `N <= tf.dtypes.int32.max`.
  Although this sorting-based implementation results in O(N log N) complexity,
  in practice it can sometimes be significantly faster than `tf.map_fn()`.

  NOTE: Unlike `tf.random_index_shuffle`, this function returns all indices
  implied by `segment_ids`.

  Args:
    segment_ids: rank-1 tensor of dtype int32 or int64 with sorted segment ids.
    seed: A Python integer. Used to create a random seed for shuffle.

  Returns:
    A tensor `result` with the same shape and dtype as `segment_ids` such that
    `lambda i: result[i]` is a permutation of [0, N) that was drawn uniformly at
    random from the set of all permutations that satisfy
    `segment_ids[result[i]] == segment_ids[i]` for all `i`.

  Raises:
    ValueError: if `segment_ids` has `rank != 1` or has not integer type.
    InvalidArgumentError: if `segment_ids` are not sorted in ascending order.
    InvalidArgumentError: if `segment_ids` size is greater than
      `tf.dtypes.int32.max`.
  """

  segment_ids = tf.convert_to_tensor(segment_ids)
  if segment_ids.shape.rank != 1:
    raise ValueError('`segment_ids` must have rank=1,'
                     f' got {segment_ids.shape.rank}.')
  if segment_ids.dtype not in (tf.int32, tf.int64):
    raise ValueError('`segment_ids` must have dtype tf.int32 or tf.int64,'
                     f' got {segment_ids.dtype}.')

  num_values = outer_dimension_size(segment_ids)

  validation_ops = [
      tf.debugging.assert_less_equal(
          num_values, tf.dtypes.int32.max, '`values` with dimension 0 size > '
          f' {tf.dtypes.int32.max} is currently not suported.'),
      tf.debugging.assert_greater_equal(
          segment_ids[1:], segment_ids[:-1],
          '`segment_ids` must be sorted in ascending order.')
  ]
  with tf.control_dependencies(validation_ops):
    shuffled_indices = tf.random.shuffle(
        tf.range(num_values, dtype=segment_ids.dtype), seed=seed)
    shuffled_segment_ids = tf.gather(segment_ids, shuffled_indices)
    restores_segments_positions = tf.argsort(shuffled_segment_ids, stable=True)
    return tf.gather(shuffled_indices, restores_segments_positions)


def flatten_indices(indices: tf.Tensor, indices_row_lengths: tf.Tensor,
                    values_row_lengths: tf.Tensor) -> tf.Tensor:
  """Changes ragged values indices from row-local to global.

  Example:

  ```python
  flatten_indices([1, 0, 1, 0], [3, 1], [2, 1])  # returns [1, 0, 1, 2 + 1].
  # Here there are 2 rows with 2 values in the first row and 1 in the second.
  # [1, 0, 1] are values indices in the first row; [0] - in the second.
  # [1, 0, 1, 2 + 1] are global value indices.
  ```

  Args:
    indices: rank-1 integer tensor with row-local values' indices.
    indices_row_lengths: rank-1 integer tensor with number of values' indices in
      each subgraph.
    values_row_lengths: rank-1 integer tensor with number of values in each
      subgraph. Must have the shape as `indices_row_lengths`.

  Returns:
    Rank-1 integer tensor with values' global indices.
  """
  _assert_rank1_int(indices, 'indices')
  _assert_rank1_int(indices_row_lengths, 'indices_row_lengths')
  _assert_rank1_int(values_row_lengths, 'values_row_lengths')

  indices_total = outer_dimension_size(indices)
  indices_row_ids = row_lengths_to_row_ids(indices_row_lengths, indices_total)

  values_row_starts = tf.math.cumsum(values_row_lengths, axis=0, exclusive=True)
  offsets = tf.gather(values_row_starts, indices_row_ids)
  offsets = tf.cast(offsets, indices.dtype)
  return tf.math.add(indices, offsets)


def static_size(shape: tf.TensorShape) -> Optional[int]:
  """Returns the total number of elements for static `shape`. None otherwise."""
  shape = shape.as_list()
  if None in shape:
    return None
  result = 1
  for dim in shape:
    assert dim > 0, shape
    result *= dim
  return result


def repeat(value: Value,
           repeats: tf.Tensor,
           repeats_sum_hint: Optional[int] = None) -> Value:
  """Repeats value (ragged or dense) along its first dimension.

  Each `value[i, ...]` is repeated `repeats[i]` times.

  For XLA compatibility `repeats_sum_hint` has to be provided to guarantee
  statically (compile-time) known result size.

  Example:

  ```python
  repeat(['a', 'b'], [2, 1], 3)  # returns ['a', 'a', 'b'].
  ```

  Args:
    value: ragged or dense tensor with `rank > 0`.
    repeats: rank-1 integer tensor, where `repeats[i]` is the number of times
      `value[i, ...]` must be repeated. Must have the same first dimension as
      the `value`.
    repeats_sum_hint: value optionally provided by the client if the sum of
      `repeats` known statically.

  Returns:
    value tensor repeated along its first dimension.
  """
  if value.shape.rank == 0:
    raise ValueError(f'`value` must have rank>1, got rank={value.shape.rank}')

  _assert_rank1_int(repeats, 'repeats')
  return tf.gather(value, row_lengths_to_row_ids(repeats, repeats_sum_hint))


def static_repeat(value: Value, multiplier: int) -> Value:
  """Repeats value (ragged or dense) along its first dimension.

  Each value[i, ...] is repeated `multiplier` times.
  The function is XLA compatible, if the `value` is a dense tensor.

  Example:

  ```python
    repeat(['a', 'b'], 3)  # returns ['a', 'a', 'a', 'b', 'b', 'b'].
  ```

  Args:
    value: ragged or dense tensor with `rank > 0`.
    multiplier: number of times each `value[i, ...]` must be repeated.

  Returns:
    value tensor repeated along its first dimension.
  """
  if value.shape.rank == 0:
    raise ValueError(f'`value` must have rank>1, got rank={value.shape.rank}')

  value = tf.expand_dims(value, 1)
  dims = [1, int(multiplier)] + [1] * (value.shape.rank - 2)
  result = tf.tile(value, dims)
  if is_ragged_tensor(result):
    return result.values
  return tf.reshape(result, [-1] + dims_list(result)[2:])


def ones_like_leading_dims(value: Value, rank: int,
                           dtype: tf.dtypes.DType) -> Value:
  """Creates a tensor of all ones for first `rank` dimensions."""
  if rank == 0:
    raise ValueError(f'Expected `rank > 0`, got {rank}')
  if rank > value.shape.rank:
    raise ValueError('`rank` is greater then `value` rank,'
                     f' got rank={rank},'
                     f' value.shape.rank={value.shape.rank}')

  if is_dense_tensor(value):
    size_shape = tf.shape(value)[:rank]
    return tf.ones(size_shape, dtype=dtype)

  if not is_ragged_tensor(value):
    raise ValueError(f'Unsupported type {type(value).__name__}')

  def iterate(value: Value, rank: int) -> Value:
    if rank == 0:
      if is_ragged_tensor(value):
        nrows = value.nrows()
      else:
        nrows = tf.shape(value)[0]
      return tf.ones(tf.expand_dims(nrows, -1), dtype=dtype)
    if value.uniform_row_length:
      return tf.RaggedTensor.from_uniform_row_length(
          iterate(value.values, rank - 1), value.uniform_row_lengths)
    return tf.RaggedTensor.from_row_splits(
        iterate(value.values, rank - 1), value.row_splits)

  return iterate(value, rank - 1)


def ensure_static_nrows(value: Value, nrows: int) -> Value:
  """Updates value type spec to have static number of rows (see below).

  This function allows to restore static dimension sizes without relying on the
  tensorflow shape inference. If `value` is a dense tensor, the result tensor
  has `result.shape[0] == nrows`. If `value` is a ragged tensor, the
  `result.nrows() == nrows`. Function checks at runtime that `value` allows
  that update.

  NOTE(b/275338236): This operation is not available in TFLite (last checked
  for TF 2.12).

  Args:
    value: dense tensor (`rank > 0`) or ragged tensor that allows static
      `nrows`.
    nrows: static number of rows.

  Returns:
    Tensor that is equal to the input tensor but with static number of rows.
  """
  if value.shape.rank == 0:
    raise ValueError(f'Expected `rank > 0` tensor, got {value.shape.rank}')

  if is_dense_tensor(value):
    return tf.ensure_shape(value, tf.TensorShape([nrows, *value.shape[1:]]))

  if is_ragged_tensor(value):
    return tf.RaggedTensor.from_row_splits(
        value.values,
        tf.ensure_shape(value.row_splits, tf.TensorShape([nrows + 1])),
        validate=False)

  raise ValueError(f'Unsupported type {type(value).__name__}')


def fill(spec: ValueSpec, nrows: tf.Tensor, value: tf.Tensor) -> Value:
  """Creates tensor filled with a scalar `value` according to the constraints.

  This function returns a Tensor or RaggedTensor compatible with `spec`.
  Its outermost dimension is `nrows`. Its further dimensions must be dense
  dimensions of a size defined in `spec`, or ragged dimensions for which
  the value contains 0 items. The elements of the tensor (if any) are set to
  `value`.

  Args:
    spec: type spec the result should be compatible with.
    nrows: number of rows in the result tensor. For a dense tensor, this is the
      outermost dimension size. For a ragged tensor, this is the number of rows
      in the outermost split (`tf.RaggedTensor.nrows`).
    value: scalar value to use for filling.

  Returns:
    Tensor filled with `value` that is compatible with `spec` and has `nrows`
    number of rows.
  """
  value = tf.convert_to_tensor(value, dtype=spec.dtype)
  if value.shape.rank != 0:
    raise ValueError('The `value` must be scalar tensor,'
                     f' got rank={value.shape.rank}')

  nrows = tf.convert_to_tensor(nrows)
  if nrows.shape.rank != 0:
    raise ValueError('The `nrows` must be scalar tensor,'
                     f' got rank={nrows.shape.rank}')

  if isinstance(spec, tf.TensorSpec) or spec.ragged_rank == 0:
    inner_dims = spec.shape[1:]
    outer_dim = spec.shape[0]
    if outer_dim is not None and outer_dim != nrows:
      raise ValueError(f'The leading dimension in `spec` is {outer_dim} and'
                       f' it is not compatible with nrows={nrows}.')
    if not inner_dims.is_fully_defined():
      raise ValueError('All except the leading shape dimensions in `spec`'
                       ' must be fully defined,'
                       f' got shape={spec.shape}')
    result_dims = [nrows, *inner_dims.as_list()]
    result = tf.fill(result_dims, value)
    assert result.shape[1:].as_list() == inner_dims.as_list()

  elif isinstance(spec, tf.RaggedTensorSpec):

    # By convension: scalar entries represent uniform row length, vector entries
    # represent ragged row lenghts.
    row_partitions = []
    # The `cum_dim` tracks the minimum positive number of entities that could be
    # partitioned by the continuous sequence of higher-up uniform dimensions.
    cum_dim = nrows
    for dim in spec.shape[1:(spec.ragged_rank + 1)]:
      if dim is None:
        # Ragged dimension: add row lengths ([0, 0.., 0]) for empty values that
        # are compatible with outer dimensions.
        row_partitions.append(
            tf.fill([cum_dim], tf.constant(0, dtype=spec.row_splits_dtype)))
        cum_dim = 0
      else:
        row_partitions.append(tf.constant(dim, dtype=spec.row_splits_dtype))
        cum_dim = cum_dim * dim

    assert spec.shape[spec.ragged_rank] is None, spec
    features_shape = spec.shape[(spec.ragged_rank + 1):]
    flat_values_shape = tf.TensorShape([0]).concatenate(features_shape)
    flat_values = tf.fill(flat_values_shape, value)
    result = flat_values
    for row_partition in reversed(row_partitions):
      if row_partition.shape.rank == 0:
        result = tf.RaggedTensor.from_uniform_row_length(result, row_partition)
      else:
        assert row_partition.shape.rank == 1, row_partition.rank
        result = tf.RaggedTensor.from_row_lengths(result, row_partition)
  else:
    raise ValueError(f'Unsupported type spec {type(spec).__name__}')

  assert spec.is_compatible_with(
      result), f'{spec}, {tf.type_spec_from_value(result)}'
  return result


def pad_to_nrows(value: Value,
                 target_nrows: tf.Tensor,
                 padding_value: tf.Tensor,
                 validate: bool = True) -> Value:
  """Pads `value` to the target number of rows with scalar `padding_value`.

  Args:
    value: tensor of `rank > 0` or ragged tensor to pad.
    target_nrows: number of rows in the result tensor. For a dense tensor, this
      is the outermost dimension size. For a ragged tensor, this is the number
      of rows in the outermost split (`tf.RaggedTensor.nrows`).
    padding_value: scalar value to use for padding.
    validate: if true, adds runtime checks that value could be padded.

  Returns:
    Input `value` padded to the target number of rows.
  """
  if value.shape.rank == 0:
    raise ValueError('The `value` must have rank>0, got scalar (rank=0)')

  if is_dense_tensor(value):
    diff_size = tf.cast(target_nrows, tf.int64) - tf.shape(value, tf.int64)[0]
  elif is_ragged_tensor(value):
    diff_size = tf.cast(target_nrows, tf.int64) - tf.cast(
        value.nrows(), tf.int64)
  else:
    raise ValueError(f'Unsupported type {type(value).__name__}')

  spec = tf.type_spec_from_value(value)
  relaxed_shape = tf.TensorShape([None, *spec.shape[1:]])
  if isinstance(spec, tf.RaggedTensorSpec):
    spec = tf.RaggedTensorSpec(
        shape=relaxed_shape,
        dtype=spec.dtype,
        ragged_rank=spec.ragged_rank,
        row_splits_dtype=spec.row_splits_dtype)
  else:
    assert isinstance(spec, tf.TensorSpec)
    spec = tf.TensorSpec(shape=relaxed_shape, dtype=spec.dtype)

  if validate:
    validation_ops = [
        tf.debugging.assert_non_negative(
            diff_size,
            f'The `value` has more rows then the target_nrows={target_nrows}.')
    ]
  else:
    validation_ops = []

  with tf.control_dependencies(validation_ops):
    diff = fill(spec, nrows=diff_size, value=padding_value)
    return tf.concat([value, diff], axis=0)


def _assert_rank1_int(t: tf.Tensor, tensor_name: Text) -> None:
  if t.shape.rank != 1 or t.dtype not in (tf.int32, tf.int64):
    raise ValueError(f'Expected `{tensor_name}` as rank-1 integer tensor,'
                     f' got rank={t.shape.rank}, dtype={t.dtype.name}')


def with_undefined_outer_dimension(spec: ValueSpec) -> ValueSpec:
  """Sets outer most shape dimension to None (undefined)."""
  if spec.shape.rank == 0:
    raise ValueError('The `spec` must have rank>0, got scalar (rank=0)')

  if spec.shape[0] is None:
    return spec
  relaxed_shape = tf.TensorShape([None, *spec.shape[1:]])

  if isinstance(spec, tf.TensorSpec):
    return tf.TensorSpec(shape=relaxed_shape, dtype=spec.dtype)

  if isinstance(spec, tf.RaggedTensorSpec):
    return tf.RaggedTensorSpec(
        shape=relaxed_shape,
        dtype=spec.dtype,
        ragged_rank=spec.ragged_rank,
        row_splits_dtype=spec.row_splits_dtype)

  raise ValueError(f'Unsupported type {type(spec).__name__}')


def is_ragged_tensor(value: Value) -> bool:
  """Returns whether a tensor (TF or Keras) is a RaggedTensor."""
  return isinstance(value, (tf.RaggedTensor, tf_internal.RaggedKerasTensor))


def is_dense_tensor(value: Value) -> bool:
  """Returns whether a tensor (TF or Keras) is a Tensor."""
  if isinstance(value, tf.Tensor):
    return True

  if isinstance(value, tf_internal.KerasTensor):
    # KerasTensor is the base class for all Keras tensors, including the
    # RaggedKerasTensor. Below we rely on the type spec to resolve actual type.
    return isinstance(value.type_spec, tf.TensorSpec)

  return False


def short_repr(value: Value) -> str:
  """A string for a dense or ragged tensor without the contained values.

  This is a helper function to print metadata (dtype and shape) of a feature
  tensor without all the values, to make GraphTensor and GraphPiece reprs
  more readable.

  Args:
    value: tensor or ragged tensor.

  Returns:
    Shortened string representation of the input.
  """
  if is_dense_tensor(value):
    return f'<tf.Tensor: shape={value.shape}, dtype={value.dtype!r}>'
  elif is_ragged_tensor(value):
    return f'<tf.RaggedTensor: dtype={value.dtype!r}>'
  else:
    return repr(value)


def short_features_repr(features: Mapping[str, Value]) -> str:
  return ('{' + ', '.join(f"'{key}': {short_repr(value)}"
                          for key, value in features.items()) + '}')
