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
"""Stage executors for features accessors."""

from typing import Dict, Iterable, Iterator, Optional, Tuple, cast

import apache_beam as beam
from apache_beam import typehints as beam_typehints
import numpy as np
from tensorflow_gnn.experimental.sampler import eval_dag_pb2 as pb
from tensorflow_gnn.experimental.sampler.beam import executor_lib
from tensorflow_gnn.experimental.sampler.beam import utils

PCollection = beam.pvalue.PCollection
SourceId = executor_lib.SourceId
ExampleId = executor_lib.ExampleId
Value = executor_lib.Value
Values = executor_lib.Values
PValues = executor_lib.PValues

ExampleId = executor_lib.ExampleId
PKeyToBytes = executor_lib.PKeyToBytes


_Query = Tuple[ExampleId, int]
_Value = bytes


@beam_typehints.with_output_types(Tuple[ExampleId, Values])
class KeyToBytesAccessor(beam.PTransform):
  """Extracts serialized values from a table using lookup keys.

  Implements `KeyToBytesAccessor` interface. Takes as an input two collections
  containing lookup queries and lookup values. The lookup values are tuples of
  unique keys (integer or bytes) and serialized lookup values (bytes). The
  lookup queries are tuples of unique `ExampleId`s and ragged rank 1 lookup
  keys. The lookup results are returned for each input key, either as a value
  if one is found, or as a configured default value if no value is found.
  """

  @beam_typehints.with_input_types(Tuple[ExampleId, Values])
  @beam_typehints.with_output_types(ExampleId)
  class FilterEmptyInputs(beam.DoFn):
    """Filters example ids for inputs with empty sets of lookup keys."""

    def __init__(self):
      self._count = beam.metrics.Metrics.counter('FilterEmptyInputs', 'Count')

    def process(self, inputs: Tuple[ExampleId, Values]) -> Iterator[ExampleId]:
      example_id, keys = inputs
      assert len(keys) == 1 and len(keys[0]) == 2
      if keys[0][0].size == 0:
        yield example_id
        self._count.inc()

  @beam_typehints.with_input_types(Tuple[_Query, Optional[_Value]])
  @beam_typehints.with_output_types(Tuple[ExampleId, Tuple[int, bytes]])
  class ProcessLookupResults(beam.DoFn):
    """Rekeys by example id and replaces missing values with default value."""

    def __init__(self, default_value: bytes):
      self._default_value = default_value
      self._missing_values_counter = beam.metrics.Metrics.counter(
          'KeyToBytesAccessor', 'MissingValues'
      )

    def process(
        self, lookup_result: Tuple[_Query, Optional[_Value]]
    ) -> Iterator[Tuple[ExampleId, Tuple[int, _Value]]]:
      (example_id, index), value = lookup_result
      if value is None:
        value = self._default_value
        self._missing_values_counter.inc()
      yield (example_id, (index, value))

  @beam_typehints.with_input_types(
      Tuple[ExampleId, Iterable[Tuple[int, bytes]]]
  )
  @beam_typehints.with_output_types(Tuple[ExampleId, Values])
  class AggregateResults(beam.DoFn):
    """Aggregates final results from pieces grouped by example ids.

    The inputs are tuples of output positions and lookup values grouped by their
    example ids.
    """

    def __init__(self, layer: pb.Layer):
      self._layer = layer

    def setup(self):
      self._value_dtype, self._splits_dtype = utils.get_ragged_np_types(
          self._layer.outputs[0].ragged_tensor
      )

    def process(
        self,
        inputs: Tuple[ExampleId, Iterable[Tuple[int, bytes]]],
    ) -> Iterator[Tuple[ExampleId, Values]]:
      example_id, values_iter = inputs
      values = sorted(values_iter, key=lambda kv: kv[0])
      values = [
          np.array([v for _, v in values], dtype=self._value_dtype),
          np.array([len(values)], dtype=self._splits_dtype),
      ]
      yield (example_id, [values])

  @beam_typehints.with_input_types(ExampleId)
  @beam_typehints.with_output_types(Tuple[ExampleId, Values])
  class CreateEmptyResults(beam.DoFn):
    """Creates empty lookup results for input example ids."""

    def __init__(self, layer: pb.Layer):
      self._layer = layer

    def setup(self):
      value_dtype, splits_dtype = utils.get_ragged_np_types(
          self._layer.outputs[0].ragged_tensor
      )
      self._empty_value = [[
          np.array([], dtype=value_dtype),
          np.array([0], dtype=splits_dtype),
      ]]

    def process(
        self, example_id: ExampleId
    ) -> Iterator[Tuple[ExampleId, Values]]:
      yield (example_id, self._empty_value)

  def __init__(self, layer: pb.Layer):
    _check_signature(layer.inputs, 'input')
    _check_signature(layer.outputs, 'output')

    # TODO(aferludin): default value must be configured by the layer.
    self._layer = layer
    self._default_value = b''

  def expand(self, inputs: Tuple[PValues, PKeyToBytes]) -> PValues:
    keys, values = inputs
    queries = keys | 'RekeyBySourceIds' >> beam.ParDo(_rekey_by_source_ids)
    empty_inputs = keys | 'FilterEmptyInputs' >> beam.ParDo(
        self.FilterEmptyInputs()
    )

    lookup_results = (
        (queries, values)
        | 'LeftJoin' >> utils.LeftLookupJoin()
        | 'DropKeys' >> beam.Values()
        | 'ProcessLookupResults'
        >> beam.ParDo(self.ProcessLookupResults(self._default_value))
        | 'GroupByExampleId' >> beam.GroupByKey()
        | 'AggregateResults' >> beam.ParDo(self.AggregateResults(self._layer))
    )

    empty_results = empty_inputs | 'CreateEmptyResults' >> beam.ParDo(
        self.CreateEmptyResults(self._layer)
    )

    return [lookup_results, empty_results] | 'Flatten' >> beam.Flatten()


def _rekey_by_source_ids(
    inputs: Tuple[ExampleId, Values]
) -> Iterator[Tuple[SourceId, _Query]]:
  """Prepares lookup queries."""
  example_id, keys = inputs
  assert len(keys) == 1 and len(keys[0]) == 2
  keys = keys[0][0]
  for index, key in enumerate(keys):
    key = utils.as_pytype(key)
    assert isinstance(key, (int, bytes))
    yield key, (example_id, index)


def _key_to_bytes_executor(
    label: str,
    layer: pb.Layer,
    inputs: PValues,
    feeds: Dict[str, executor_lib.PFeed],
    unused_artifacts_path: str,
) -> PValues:
  """Returns KeyToBytesAccessor stage executor."""
  del unused_artifacts_path
  values_table = feeds.get(layer.id, None)
  if values_table is None:
    raise ValueError(
        f'Missing values table for KeyToBytesAccessor layer {layer.id}'
    )

  values_table = cast(PKeyToBytes, values_table)
  return (inputs, values_table) | label >> KeyToBytesAccessor(layer)


def _check_signature(args: Iterable[pb.ValueSpec], signature_type: str) -> None:
  args = list(args)
  if len(args) != 1 or not args[0].HasField('ragged_tensor'):
    raise ValueError(
        f'Invalid {signature_type} signature for `KeyToBytesAccessor` layer:'
        f' expected single ragged tensor value, got {args}.'
    )


executor_lib.register_stage_executor(
    'KeyToBytesAccessor', _key_to_bytes_executor
)
