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

from typing import cast, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union
import apache_beam as beam
from apache_beam import typehints
from apache_beam.typehints import trivial_inference

import numpy as np
import tensorflow as tf

from tensorflow_gnn.experimental.sampler import eval_dag_pb2 as pb

PCollection = beam.pvalue.PCollection

K = TypeVar('K')
Q = TypeVar('Q')
V = TypeVar('V')


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


def get_outer_dim_size(values: List[List[np.ndarray]]) -> int:
  """Returns outermost dimension for the batch of tensors."""
  assert values
  dims = [value[-1].shape[0] for value in values]
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
