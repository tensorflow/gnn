"""Utils for tensors and ragged tensors."""
from typing import List, Optional, Text, Union

from keras.engine import keras_tensor as kt
import tensorflow as tf

Value = Union[tf.Tensor, tf.RaggedTensor]


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
    row_lengths_to_row_ids([2, 1, 0, 2], 5) -> [0, 0, 1, 3, 3]

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


def flatten_indices(indices: tf.Tensor, indices_row_lengths: tf.Tensor,
                    values_row_lengths: tf.Tensor) -> tf.Tensor:
  """Changes ragged values indices from row-local to global.

  Example:
    flatten_indices([1, 0, 1, 0], [3, 1], [2, 1]) -> [1, 0, 1, 2 + 1]
    Here there are 2 rows with 2 values in the first row and 1 in the second.
    [1, 0, 1] are values indices in the first row; [0] - in the second.
    [1, 0, 1, 2 + 1] are global value indices.

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

  Each value[i, ...] is repeated repeats[i] times.

  For XLA compatibility `repeats_sum_hint` has to be provided to guarantee
  statically (compile-time) known result size.

  Example:
    repeat(['a', 'b'], [2, 1], 3) -> ['a', 'a', 'b']

  Args:
    value: ragged or dense tensor with rank > 0.
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
    repeat(['a', 'b'], 3) -> ['a', 'a', 'a', 'b', 'b', 'b']

  Args:
    value: ragged or dense tensor with rank > 0.
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


def _assert_rank1_int(t: tf.Tensor, tensor_name: Text) -> None:
  if t.shape.rank != 1 or t.dtype not in (tf.int32, tf.int64):
    raise ValueError(f'Expected `{tensor_name}` as rank-1 integer tensor,'
                     f' got rank={t.shape.rank}, dtype={t.dtype.name}')


def is_ragged_tensor(value: Value) -> bool:
  return isinstance(value, (tf.RaggedTensor, kt.RaggedKerasTensor))


def is_dense_tensor(value: Value) -> bool:
  return isinstance(value, (tf.Tensor, kt.KerasTensor))
