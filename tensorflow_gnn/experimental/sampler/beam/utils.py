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
"""Set of Beam utils."""
import functools
from typing import Iterable, Iterator, List, Optional, Tuple, TypeVar, Union, cast
import apache_beam as beam
from apache_beam import typehints
from apache_beam.typehints import trivial_inference
import numpy as np
import tensorflow as tf
from tensorflow_gnn.experimental.sampler import proto as pb

PCollection = beam.pvalue.PCollection

K = TypeVar('K')
Q = TypeVar('Q')
V = TypeVar('V')

# Flattened components of tensor, ragged tensor or generic composite tensor. For
# regular tensor `t` this is `[t.numpy()]`. For the ragged tensor `rt` this is
# `[c.numpy() for c in [*rt.flat_values, *rt.nested_row_lengths()]]`. In generic
# case of not-specialized composite tensor type `c`, this is
# `[c.numpy() for c in tf.nest.flatten(c, expand_composites=True)]`.
Value = List[np.ndarray]
ShapeLike = Tuple[int, ...]


class SafeLeftLookupJoin(beam.PTransform):
  """Matches keys from queries with unique values.

  NOTE: the implementation does not rely on `CoGroupByKey`, because the latter,
  depending on the implementation of Beam runner used, may materialize join
  results in memory (as Python `list`), which is prohibitively for large
  datasets. This comes at a cost that join result streams may be re-iterated
  twice.
  """

  def expand(
      self,
      inputs: Tuple[PCollection[Tuple[K, Q]], PCollection[Tuple[K, V]]],
  ) -> PCollection[Tuple[K, Tuple[Q, Optional[V]]]]:
    def add_tag(
        key: K, value: Union[Q, V], *, tag: bytes
    ) -> Tuple[K, Tuple[bytes, Union[Q, V]]]:
      return (key, (tag, value))

    def extract_results(
        inputs: Tuple[K, Iterable[Tuple[bytes, Union[Q, V]]]]
    ) -> Iterator[Tuple[K, Tuple[Q, Optional[V]]]]:
      key, join_results = inputs
      # Because `join_results` stream could be very large for hot keys, we
      # iterate over it two times. The first time to find result value (if any)
      # and the second time to output lookup results for each query.
      value = None
      for tag, element in join_results:
        if tag == b'V':
          value = cast(V, element)
          break

      values_ctr = 0
      for tag, element in join_results:
        if tag == b'Q':
          query = cast(Q, element)
          yield (key, (query, value))
        else:
          values_ctr += 1

      if values_ctr > 1:
        raise ValueError(
            'Left side of the lookup join must contain unique keys. There'
            f' {values_ctr} entries of `{key}` key'
        )

    queries, values = inputs
    key_type, query_type = trivial_inference.key_value_types(
        queries.element_type
    )
    _, value_type = trivial_inference.key_value_types(values.element_type)

    tagged_queries = queries | 'AddQueryTag' >> beam.MapTuple(
        functools.partial(add_tag, tag=b'Q')
    ).with_output_types(
        typehints.Tuple[key_type, typehints.Tuple[bytes, query_type]]
    )
    tagged_values = values | 'AddValueTag' >> beam.MapTuple(
        functools.partial(add_tag, tag=b'V')
    ).with_output_types(
        typehints.Tuple[key_type, typehints.Tuple[bytes, value_type]]
    )

    group_by_type = typehints.Tuple[
        key_type,
        typehints.Tuple[bytes, typehints.Union[query_type, value_type]],
    ]
    result_type = typehints.Tuple[
        key_type, typehints.Tuple[query_type, typehints.Optional[value_type]]
    ]
    return (
        (tagged_queries, tagged_values)
        | 'Flatten' >> beam.Flatten().with_output_types(group_by_type)
        | 'GroupByKey' >> beam.GroupByKey()
        | 'Extract'
        >> beam.ParDo(extract_results).with_output_types(result_type)
    )


LeftLookupJoin = SafeLeftLookupJoin


def as_pytype(scalar: Union[bytes, np.ndarray]) -> Union[bytes, int, float]:
  """Converts scalar value to the Python type."""
  return scalar if isinstance(scalar, bytes) else scalar.item()


def get_np_dtype(type_pb) -> np.dtype:
  """Converts input TF proto enum type value into numpy dtype."""
  result = tf.dtypes.as_dtype(type_pb)
  return np.dtype(result.as_numpy_dtype)


def get_ragged_np_types(spec: pb.RaggedTensorSpec) -> Tuple[np.dtype, ...]:
  """Returns numpy types for ragged flat value and each ragged partition."""
  splits_types = (get_np_dtype(spec.row_splits_dtype),) * spec.ragged_rank
  return (get_np_dtype(spec.dtype), *splits_types)


def get_outer_dim_size(value: List[np.ndarray]) -> int:
  """Returns outermost dimension size for potentially ragged value."""
  if len(value) == 1:
    return value[0].shape[0]
  else:
    # (flat_values, outermost_dim,..., innermost_dim)
    return value[1].shape[0]


def get_batch_size(values: List[List[np.ndarray]]) -> int:
  """Returns batch size for the batch of tensors."""
  assert values
  dims = [get_outer_dim_size(value) for value in values]
  if any(dims[0] != d for d in dims):
    raise ValueError(f'Values have different outer dimensions: {dims}')
  return dims[0]


def ragged_slice(
    value: List[np.ndarray], start: int, limit: int
) -> List[np.ndarray]:
  """Extracts `value[index:(index+1), :]` for potentially ragged values."""
  assert value
  b, e = start, limit
  partition_slices = []
  for dim in range(1, len(value)):
    partition = value[dim]
    partition_slices.append(partition[b:e])
    b, e = np.sum(partition[:b]), np.sum(partition[:e])
  flat_value = value[0][b:e]
  return [flat_value, *partition_slices]


def stack(values: List[Value]) -> Value:
  """Stacks dense values along axis 0.

  Args:
    values: dense tensors with the same `rank` and compatible shapes.

  Returns:
    Stacking result as a dense tensor with the `rank + 1` rank.
  """
  if not values:
    raise ValueError('values must not be empty')

  batch = []
  for value in values:
    if len(value) != 1:
      raise ValueError('Expected single value for dense tensor')

    batch.append(value[0])

  try:
    return [np.stack(batch, axis=0)]
  except ValueError as e:
    raise ValueError(f'values are not compatible: {e}') from e


def stack_ragged(values: List[Value], row_splits_dtype: np.dtype) -> Value:
  """Stacks potentially ragged values along axis 0.

  Args:
    values: list of potentionally ragged tensors with the same `rank` and
      compatible row partitions.
    row_splits_dtype: dtype of row splits.

  Returns:
    Stacking result as a ragged tensor with rank + 1.
  """
  if not values:
    raise ValueError('values must not be empty')

  batches = []
  sizes = []
  for value in values:
    sizes.append(get_outer_dim_size(value))
    if not batches:
      batches = [[p] for p in value]
    else:
      if len(batches) != len(value):
        raise ValueError('values must have the same number of components')
      for batch, component in zip(batches, value):
        batch.append(component)

  try:
    flat_values = np.concatenate(batches[0], axis=0)
    outer_partition = np.array(sizes, dtype=row_splits_dtype)
    inner_partitions = [np.concatenate(v, axis=0) for v in batches[1:]]
    return [flat_values, outer_partition, *inner_partitions]

  except ValueError as e:
    raise ValueError(f'values are not compatible: {e}') from e


def parse_tf_example(
    example: tf.train.Example, name: str, spec: pb.ValueSpec
) -> Value:
  """Parses values from TF example message by its name and spec.

  For `tensor` the name is the name of feature with flat tensor values.

  For `ragged_tensor` the name is the name of feature with ragged tensor flat
  values. The nested row partitions are stored as `int64` features with
  `{name}.d{index}` names, where `index` is a 1-based ragged partition index
  counting from the outermost to the innermost dimensions.


  For other cases, e.g. `flattened`, the `NotImplementedError` is raised.


  Args:
    example: tensorflow Example proto instance.
    name: value name.
    spec: value type spec.

  Returns:
    Value instance as a list of numpy array. For regular tensor `t` this is
    `[t.numpy()]`. For the ragged tensor `rt` this is
    `[c.numpy() for c in [*rt.flat_values, *rt.nested_row_lengths()]]`.
  """
  if spec.HasField('tensor'):
    spec = spec.tensor
    return [
        _parse_tf_feature(
            example,
            name,
            get_np_dtype(spec.dtype),
            tuple(dim.size for dim in spec.shape.dim),
        )
    ]
  elif spec.HasField('ragged_tensor'):
    spec = spec.ragged_tensor
    dtype = get_np_dtype(spec.dtype)
    ragged_rank = spec.ragged_rank
    row_splits_dtype = get_np_dtype(spec.row_splits_dtype)
    shape = tuple(dim.size for dim in spec.shape.dim)

    flat_value = _parse_tf_feature(example, name, dtype, shape[ragged_rank:])

    nested_row_lengths = [
        _parse_tf_feature(example, f'{name}.d{d}', row_splits_dtype, (-1,))
        for d in range(1, ragged_rank + 1)
        if shape[d] == -1
    ]
    return [flat_value, *nested_row_lengths]

  else:
    raise NotImplementedError(
        f'Example parsing is not supported for {spec} of {name}'
    )


def _parse_tf_feature(
    example: tf.train.Example,
    fname: str,
    dtype: np.dtype,
    shape: ShapeLike,
) -> np.ndarray:
  """Extract single feature from TF Example."""
  features = example.features.feature
  if fname not in features:
    raise ValueError(f'Expected feature "{fname}" is missing')

  feature = features[fname]
  if np.issubdtype(dtype, np.integer):
    if not feature.HasField('int64_list'):
      raise ValueError(f'Expected int64 feature for "{fname}"')
    values = feature.int64_list.value
  elif np.issubdtype(dtype, np.floating):
    if not feature.HasField('float_list'):
      raise ValueError(f'Expected float feature for "{fname}"')
    values = feature.float_list.value
  elif np.issubdtype(dtype, np.object_):
    if not feature.HasField('bytes_list'):
      raise ValueError(f'Expected bytes feature for "{fname}"')
    values = feature.bytes_list.value
  else:
    raise ValueError(f'Unsupported dtype: {dtype}')

  if -1 not in shape:
    expected_num_elements = np.prod(shape, dtype=np.int32)
    if len(values) != expected_num_elements:
      raise ValueError(
          f'shape={shape} requires {expected_num_elements} elements for'
          f' "{fname}", actual {len(values)}'
      )
  return np.array(values, dtype=dtype).reshape(shape)
