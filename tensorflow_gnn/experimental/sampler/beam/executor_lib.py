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
"""Beam executor for bulk sampling pipelines.

Allows to run sampling model using Apache Beam. Each sampling model operates on
batches of independent examples. The typical sampling model takes batch of seed
nodes as its input and step-by-step adds new neighbors by calling sampling
layers. While executing on Beam, we take this to the extreme: there is a single
batch (as PCollection) containing all examples. Because `PCollection` is not
ordered set of entites, we key all entities by unique "example id". Those
strings are used instead of an batch dimension indices to unqiquely identify
individual examples. The sampling layers are translated into Beam stages (as
`PTransform`) and tensors are replaced with `PCollection`s containing fixed size
lists of "tensor values", as `[value_1, value_2, .., value_n]`, where `n` is
fixed within each `PCollection`. Each tensor value is  fixed-size list of
`numpy.ndarray`s containing flattened tensor or composite tensor components.
The following flattening rule is applied to any tensor value `t`:
  * dense: `[t]`;
  * ragged: `[t.flat_values, *t.nested_row_lengths()]`;
  * other:  `tf.nest.flatten(t)`;
with results tensor components being converted to the `numpy.ndarray`s.
"""

import collections
import os

from typing import Callable, Dict, List, Iterable, Iterator, Mapping, Optional, Set, Tuple, TypeVar, Union
import apache_beam as beam
from apache_beam import typehints as beam_typehints
from apache_beam.coders import typecoders
from apache_beam.typehints import typehints
from apache_beam.utils import windowed_value

import numpy as np
import tensorflow as tf

from tensorflow_gnn.experimental.sampler import eval_dag_pb2 as pb
from tensorflow_gnn.experimental.sampler.beam import utils

PCollection = beam.pvalue.PCollection
# Global unique identifier of a particular example.
ExampleId = bytes
# Tensor or flattened composite tensor.
Value = List[np.ndarray]
# Collection of values belonging to particular example.
Values = List[Value]
# Stage input/output as a batch of example values keyed by unique example ids.
PValues = PCollection[Tuple[ExampleId, Values]]


SourceId = TypeVar('SourceId', int, bytes)
TargetId = TypeVar('TargetId', int, bytes)

# Collection of unique keys to serialized values.
PKeyToBytes = PCollection[Tuple[SourceId, bytes]]
# Collection of edges as source/target node pairs.
PEdges = PCollection[Tuple[SourceId, TargetId]]
# Supported external data sources types.
PFeed = Union[PKeyToBytes, PEdges]

# Executor for primitive stages. Input arguments are label, layer, collection
# with input values, all feeds (not prefiltered) and path to serialized
# artifacts (e.g. saved TF Model for `TFModel` stages).
Executor = Callable[[str, pb.Layer, PValues, Dict[str, PFeed], str], PValues]


class NDArrayCoder(beam.coders.Coder):
  """Beam coder for Numpy N-dimensional array of TF-compatible data types.

  Supports all numeric data types and bytes (represented as `np.object_`).
  The numpy array is serialized as a tuple of `(dtype, shape, flat values)`.
  For numeric values serialization we rely on `tobytes()` and `frombuffer` from
  the numpy library. It, seems, has the best speed/space tradeoffs. Tensorflow
  represents `tf.string` as `np.object_` (as `np.string_` is for arrays
  containing fixed-width byte strings, which can lead to lots of wasted
  memory). Because `np.object_` is an array of references to arbitrary
  objects, we could not rely on numpy native serialization and using
  `IterableCoder` from the Beam library instead.

  NOTE: for some simple stages the execution time may be dominated by data
  serialization/deserialization, so any imporvement here translates directly to
  the total execution costs.
  """

  def __init__(self):
    encoded_struct = typehints.Tuple[str, typehints.Tuple[int, ...], bytes]
    self._coder = typecoders.registry.get_coder(encoded_struct)
    self._bytes_coder = typecoders.registry.get_coder(typehints.Iterable[bytes])

  def encode(self, value: np.ndarray) -> bytes:
    if value.dtype == np.object_:
      flat_values = self._bytes_coder.encode(value.flat)
    else:
      flat_values = value.tobytes()
    return self._coder.encode((value.dtype.str, value.shape, flat_values))

  def decode(self, encoded: bytes) -> np.ndarray:
    dtype_str, shape, serialized_values = self._coder.decode(encoded)
    dtype = np.dtype(dtype_str)
    if dtype == np.object_:
      flat_values = np.array(
          self._bytes_coder.decode(serialized_values), dtype=np.object_
      )
    else:
      flat_values = np.frombuffer(serialized_values, dtype=dtype)
    return np.reshape(flat_values, shape)

  def is_deterministic(self):
    return True

  def to_type_hint(self):
    return np.ndarray


beam.coders.registry.register_coder(np.ndarray, NDArrayCoder)


def execute(
    program: pb.Program,
    inputs: Mapping[str, PValues],
    *,
    feeds: Optional[Mapping[str, PFeed]] = None,
    artifacts_path: str = '',
) -> PCollection[Tuple[ExampleId, tf.train.Example]]:
  """Executes sampling program for the given inputs and external data feeds.

  Args:
    program: The sampling program, e.g. sampling Keras model converted by the
      `sampler.create_program()` function.
    inputs: A mapping from input layer name to input values.
    feeds: A mapping feed name to feed values, e.g. serialized features keyed by
      unique node ids.
    artifacts_path: The path to file system directory containing subdirectories
      named after layers with artifacts (e.g. saved TF model).

  Returns:
    A collection of unique example ids to execution results as TF Example
    messages.
  """
  sink = program.layers.get('sink', None)
  if sink is None:
    raise ValueError('Sampling program must define `sink` layer.')

  output = _execute(
      program.eval_dag,
      dict(program.layers),
      dict(inputs),
      dict(feeds or {}),
      artifacts_path,
  )

  return output | 'CreateTfExample' >> beam.ParDo(TFExampleSink(sink))


def _execute(
    eval_dag: pb.EvalDAG,
    layers: Dict[str, pb.Layer],
    inputs: Dict[str, PValues],
    feeds: Dict[str, PFeed],
    artifacts_path: str,
) -> PValues:
  """Runs Eval DAG stages and recursively executes composite stages."""
  results = []
  outputs = {}
  for stage in eval_dag.stages:
    layer = layers[stage.layer_id]
    if layer.type == 'Sink':
      results.append(_get_primitive_stage_inputs(stage, layer, outputs))
      continue

    stage_name = _create_stage_name(stage.id, layer.id, layer.type)
    if layer.type == 'InputLayer':
      output = inputs[layer.id]
    elif _is_primitive_stage(layer):
      stage_inputs = _get_primitive_stage_inputs(stage, layer, outputs)
      executor = _get_primitive_stage_executor(layer)
      output = executor(stage_name, layer, stage_inputs, feeds, artifacts_path)
    elif _is_composite_stage(layer):
      substage_inputs = _get_composite_stage_inputs(stage, layer, outputs)
      output = (
          substage_inputs,
          feeds,
      ) | stage_name >> CompositeStage(layer.eval_dag, layers, artifacts_path)
    else:
      raise ValueError(f'Unsupported layer type {layer.type}')
    outputs[stage.id] = output

  if len(results) != 1:
    raise ValueError('Eval DAG must contain exactly one `Sink` stage')

  return results[0]


def _is_composite_stage(layer: pb.Layer) -> bool:
  return layer.HasField('eval_dag')


def _is_primitive_stage(layer: pb.Layer) -> bool:
  return layer.type in _REGISTERED_EXECUTORS


def _get_primitive_stage_executor(layer: pb.Layer) -> beam.PTransform:
  return _REGISTERED_EXECUTORS[layer.type]


def _get_primitive_stage_inputs(
    stage: pb.Stage, layer: pb.Layer, outputs: Dict[str, PValues]
) -> PValues:
  inputs = _filter_matching_inputs(stage, outputs)
  return inputs | _create_stage_name(
      stage.id, layer.id, 'Inputs'
  ) >> CombineInputs(stage)


def _get_composite_stage_inputs(
    stage: pb.Stage, layer: pb.Layer, outputs: Dict[str, PValues]
) -> Dict[str, PValues]:
  inputs = _filter_matching_inputs(stage, outputs)
  return inputs | _create_stage_name(
      stage.id, layer.id, 'Inputs'
  ) >> FilterCompositeStageInputs(stage, layer)


@beam_typehints.with_output_types(Tuple[ExampleId, Values])
class CombineInputs(beam.PTransform):
  """Collects matching inputs for `stage`."""

  @beam_typehints.with_input_types(Tuple[ExampleId, Values])
  @beam_typehints.with_output_types(Tuple[ExampleId, Tuple[str, Values]])
  class ClearUnusedOutputs(beam.DoFn):
    """Clears unused stage outputs by setting their values to empty lists."""

    def __init__(self, indices: Set[int], stage_id: str):
      self._indices = indices
      self._stage_id = stage_id

    def process(
        self, inputs: Tuple[ExampleId, Values]
    ) -> Iterator[Tuple[ExampleId, Tuple[str, Values]]]:
      example_id, values = inputs
      filtered_values = []
      for index, value in enumerate(values):
        filtered_values.append(value if index in self._indices else [])
      yield (example_id, (self._stage_id, filtered_values))

  @beam_typehints.with_input_types(Tuple[ExampleId, Tuple[str, Values]])
  @beam_typehints.with_output_types(Tuple[ExampleId, Values])
  class MatchOutput(beam.DoFn):
    """Collects input values from a single stage output."""

    def __init__(self, stage: pb.Stage):
      self._stage = stage

    def process(
        self, inputs: Tuple[bytes, Tuple[str, Values]]
    ) -> Iterator[Tuple[bytes, Values]]:
      example_id, (stage_id, values) = inputs
      inputs = []
      for matcher in self._stage.input_matchers:
        assert matcher.stage_id == stage_id, example_id
        inputs.append(values[matcher.output_index])
      yield (example_id, inputs)

  @beam_typehints.with_input_types(
      Tuple[ExampleId, Iterable[Tuple[str, Values]]]
  )
  @beam_typehints.with_output_types(Tuple[ExampleId, Values])
  class MatchCombinedOutputs(beam.DoFn):
    """Collects input values from combined outputs of multiple stages."""

    def __init__(self, stage: pb.Stage):
      self._stage = stage

    def process(
        self, inputs: Tuple[bytes, Iterable[Tuple[str, Values]]]
    ) -> Iterator[Tuple[bytes, Values]]:
      example_id, outputs = inputs
      outputs = dict(outputs)
      inputs = []
      for matcher in self._stage.input_matchers:
        values = outputs[matcher.stage_id]
        inputs.append(values[matcher.output_index])
      yield (example_id, inputs)

  def __init__(self, stage: pb.Stage):
    self._stage = stage

  def expand(self, inputs: Dict[str, PValues]) -> PValues:
    outputs_map = collections.defaultdict(set)
    for matcher in self._stage.input_matchers:
      outputs_map[matcher.stage_id].add(matcher.output_index)

    stage_outputs = []
    for stage_id, output_indices in outputs_map.items():
      stage_outputs.append(
          inputs[stage_id]
          | f'ClearUnusedOutputs/{stage_id}'
          >> beam.ParDo(self.ClearUnusedOutputs(output_indices, stage_id))
      )

    if len(stage_outputs) == 1:
      return stage_outputs[0] | 'Match' >> beam.ParDo(
          self.MatchOutput(self._stage)
      )

    return (
        stage_outputs
        | 'Flatten' >> beam.Flatten()
        | 'Group' >> beam.GroupByKey()
        | 'Match' >> beam.ParDo(self.MatchCombinedOutputs(self._stage))
    )


class FilterCompositeStageInputs(beam.PTransform):
  """Collects matching inputs for `stage`."""

  def __init__(self, stage: pb.Stage, layer: pb.Layer):
    if not layer.HasField('input_names'):
      raise ValueError('Composite layer must define `input_names`')

    self._stage = stage
    self._layer = layer

  def expand(self, inputs: Dict[str, PValues]) -> Dict[str, PValues]:
    substage_inputs = {}
    for matcher, name in zip(
        self._stage.input_matchers, self._layer.input_names.feature_names
    ):
      substage_inputs[name] = inputs[
          matcher.stage_id
      ] | f'{name}/{matcher.stage_id}' >> beam.Map(
          _extract_input, index=matcher.output_index
      )
    return substage_inputs


@beam_typehints.with_output_types(Tuple[ExampleId, Values])
class CompositeStage(beam.PTransform):
  """Wraps EvalDAG of a composite stage as a PTransform."""

  def __init__(
      self,
      eval_dag: pb.EvalDAG,
      layers: Dict[str, pb.Layer],
      artifacts_path: str,
  ):
    self._eval_dag = eval_dag
    self._layers = layers
    self._artifacts_path = artifacts_path

  def expand(
      self, inputs: Tuple[Dict[str, PValues], Dict[str, PFeed]]
  ) -> PValues:
    inputs, feeds = inputs
    return _execute(
        self._eval_dag,
        self._layers,
        inputs,
        feeds=feeds,
        artifacts_path=self._artifacts_path,
    )


@beam_typehints.with_input_types(Tuple[ExampleId, Values])
@beam_typehints.with_output_types(Tuple[ExampleId, tf.train.Example])
class TFExampleSink(beam.DoFn):
  """Converts dense or ragged values to TFExample.

  Feature names for each value must be specified in the `config` field of the
  `Sink` layer as `IOFeatures` message. Currently only ragged tensors or dense
  tensors are supported. Dense values are flattened and output as a TF example
  `Feature` of the matching type. Ragged tensors are written as collection of
  their flat values and ragged row lengths from each ragged dimension. The flat
  values have feature name "{feature_name}". The row lengths have feature names
  "{feature_name}.d{index}", where `index` enumerate ragged dimensions from
  outermost to innermost starting from 1 .
  """

  _SUPPORTED_VALUES = ['tensor', 'ragged_tensor']

  def __init__(self, sink: pb.Layer):
    for input_spec in sink.inputs:
      if not any(input_spec.HasField(t) for t in self._SUPPORTED_VALUES):
        raise ValueError(
            'Conversion to TF Example is only supported for'
            f' {self._SUPPORTED_VALUES}, got {sink}'
        )
    if not sink.HasField('config'):
      raise ValueError('Sink layer must define config as `IOFeatures` message.')

    io_config = pb.IOFeatures()
    sink.config.Unpack(io_config)
    self._io_config = io_config
    self._examples_count = beam.metrics.Metrics.counter(
        'TFExampleSink', 'ExamplesCount'
    )

  def process(
      self, inputs: Tuple[ExampleId, Values]
  ) -> Iterator[Tuple[ExampleId, tf.train.Example]]:
    example_id, values = inputs
    batch_size = utils.get_outer_dim_size(values)
    if batch_size != 1:
      raise ValueError(
          f'Expected values of {example_id} to have batch size 1,'
          f' got {batch_size}'
      )
    features = tf.train.Features()
    for feature_name, value in zip(self._io_config.feature_names, values):
      value_dtype = value[0].dtype
      flat_values = np.reshape(value[0], [-1]).tolist()
      if np.issubdtype(value_dtype, np.floating):
        feature = tf.train.Feature(
            float_list=tf.train.FloatList(value=flat_values)
        )
      elif np.issubdtype(value_dtype, np.integer):
        feature = tf.train.Feature(
            int64_list=tf.train.Int64List(value=flat_values)
        )
      else:
        assert value_dtype == np.object_, value_dtype
        feature = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=flat_values)
        )
      features.feature[feature_name].CopyFrom(feature)

      for index, row_lengths in enumerate(value[2:], 1):
        assert row_lengths.dtype in (np.int32, np.int64), row_lengths.dtype
        features.feature[f'{feature_name}.d{index}'].CopyFrom(
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=row_lengths.tolist())
            )
        )

    yield example_id, tf.train.Example(features=features)
    self._examples_count.inc()


def _extract_input(
    inputs: Tuple[ExampleId, Values], index: int
) -> Tuple[ExampleId, Values]:
  example_id, values = inputs
  assert 0 <= index < len(values), example_id
  return example_id, [values[index]]


def _create_stage_name(*argv):
  return '/'.join(argv)


def _filter_matching_inputs(
    stage: pb.Stage, outputs: Dict[str, PValues]
) -> Dict[str, PValues]:
  """Filters stage inputs from upstream stages outputs."""
  result = {}
  for matcher in stage.input_matchers:
    stage_input = outputs.get(matcher.stage_id, None)
    if not stage_input:
      raise ValueError(
          f'Stage {stage.id} is disconnected: could not find matching input'
          f' upstream stage {matcher.stage_id}.'
      )
    result[matcher.stage_id] = stage_input
  return result


class TFModelBase(beam.DoFn):
  """Base abstract class for TFModel layer implementation."""

  def __init__(
      self,
      model_path: str,
      layer: pb.Layer,
  ):
    # Checks that TF model exists before running pipeline.
    if not tf.io.gfile.exists(model_path):
      raise ValueError(
          f'Layer "{layer.id}" refers to a non-existent TF model path:'
          f' {model_path}'
      )

    self._model_path = model_path
    self._output_struct = []
    for output in layer.outputs:
      if output.HasField('tensor'):
        self._output_struct.append([None])
      elif output.HasField('ragged_tensor'):
        self._output_struct.append(
            [None] * (output.ragged_tensor.ragged_rank + 1)
        )
      elif output.HasField('flattened'):
        self._output_struct.append([None] * len(output.flattened.components))
      else:
        raise NotImplementedError(f'Tensor type {output} is not supported')

  def setup(self):
    self._model = tf.saved_model.load(self._model_path)
    self._serving_fn = self._model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    ]

  def teardown(self):
    del self._serving_fn
    del self._model

  def call_model(self, values: Values) -> Values:
    kwargs = {
        ('argw' + (f'_{i}' if i > 0 else '')): tf.convert_to_tensor(v)
        for i, v in enumerate(tf.nest.flatten(values))
    }
    outputs = self._serving_fn(**kwargs)
    flat_outputs = [None] * len(outputs)
    def output_index(name: str) -> int:
      s = len('output_')
      return int(name[s:])

    for k, v in outputs.items():
      flat_outputs[output_index(k)] = v.numpy()

    return tf.nest.pack_sequence_as(self._output_struct, flat_outputs)


@beam_typehints.with_input_types(Tuple[ExampleId, Values])
@beam_typehints.with_output_types(Tuple[ExampleId, Values])
class TFModelBasic(TFModelBase):
  """Executes saved TF Model for given inputs."""

  def process(
      self, inputs: Tuple[ExampleId, Values]
  ) -> Iterator[Tuple[ExampleId, Values]]:
    example_id, values = inputs
    yield example_id, self.call_model(values)


@beam_typehints.with_input_types(Tuple[ExampleId, Values])
@beam_typehints.with_output_types(Tuple[ExampleId, Values])
class TFModelWithAutoBatch(TFModelBase):
  """Calls TF Model on aggregated maximum possible batches.

  Batches are created by concatenating input values by their outermost
  dimension untill either batch memory or size constrains are met. After model
  is called, the result values are unstacked according to the input values
  splits and returned from the function during  `process()` or `finish_bundle()`
  calls. This implementation may be orders fo magnitude faster compared to the
  `TFModelBasic`, but it requires that the underlying model supports batching,
  as `concat(model(input1), model(input2)) == model(concat(input1, input2))` for
  any possible inputs 1 and 2.
  """

  def __init__(
      self,
      model_path: str,
      layer: pb.Layer,
      *,
      max_examples_per_batch: int,
      max_batch_size_distr_in_bytes: int,
  ):
    # Checks that TF model exists before running pipeline.
    super().__init__(model_path, layer)
    self._max_examples_per_batch = max_examples_per_batch
    self._max_batch_size_distr_in_bytes = max_batch_size_distr_in_bytes

    self._batch_size_distr = beam.metrics.Metrics.distribution(
        layer.type, 'BatchSize'
    )

  def start_bundle(self):
    self._reset()

  def finish_bundle(self):
    if self._batch_splits:
      yield from self._flush()

  def process(
      self, inputs: Tuple[ExampleId, Values]
  ) -> Iterator[Tuple[ExampleId, Values]]:
    assert len(self._batch_splits) <= self._max_examples_per_batch

    example_id, values = inputs

    if not self._batch_splits:
      self._stackable_components = []
      for value in values:
        self._stackable_components.append([[vi] for vi in value])
    else:
      assert len(self._stackable_components) == len(values)
      for components, value in zip(self._stackable_components, values):
        for pieces, v in zip(components, value):
          pieces.append(v)

    try:
      outer_dim_size = utils.get_outer_dim_size(values)
    except ValueError as e:
      raise ValueError(
          f'Values for {example_id} has inconsistent outer dimension sizes.'
          ' This could be result if some layers of sampling models does not'
          ' support batch dimension.'
      ) from e

    self._batch_splits.append((example_id, outer_dim_size))
    self._estimated_memsize += sum(
        tf.nest.flatten(tf.nest.map_structure(_estimate_memsize, values))
    )
    if (
        self._estimated_memsize >= self._max_batch_size_distr_in_bytes
        or len(self._batch_splits) >= self._max_examples_per_batch
    ):
      yield from self._flush()

  def _flush(self):
    batch_size = len(self._batch_splits)
    assert batch_size <= self._max_examples_per_batch, batch_size

    self._batch_size_distr.update(batch_size)

    inputs = []
    for value_components in self._stackable_components:
      inputs.append(
          [np.concatenate(pieces, axis=0) for pieces in value_components]
      )
    result_batch = self.call_model(inputs)

    window = beam.transforms.window.GlobalWindow()
    start = 0
    for example_id, outer_dim_size in self._batch_splits:
      limit = start + outer_dim_size
      slices = [
          utils.ragged_slice(value, start, limit) for value in result_batch
      ]
      yield windowed_value.WindowedValue(
          (example_id, slices),
          beam.utils.timestamp.MAX_TIMESTAMP,
          [window],
      )
      start = limit

    self._reset()

  def _reset(self):
    self._stackable_components = []
    self._batch_splits = []
    self._estimated_memsize = 0


def _estimate_memsize(value: np.ndarray) -> int:
  result = 8 + value.size * value.itemsize
  if value.dtype == np.object_:
    result += sum(len(v) for v in value)
  return result


def _tf_model_executor(
    label: str,
    layer: pb.Layer,
    inputs: PValues,
    unused_feeds: Dict[str, PFeed],
    artifacts_path: str,
) -> PValues:
  """Returns TFModel stage executor."""
  model_path = os.path.join(artifacts_path, layer.id)
  if _supports_batching(layer):
    model_fn = TFModelWithAutoBatch(
        model_path,
        layer,
        max_examples_per_batch=10_000,
        max_batch_size_distr_in_bytes=10_000_000,
    )

  else:
    model_fn = TFModelBasic(model_path, layer)

  return inputs | label >> beam.ParDo(model_fn)


def _supports_batching(layer: pb.Layer) -> bool:
  """Checks if layer supports batching by concatenating inputs and splitting outputs.

  The check guarantees that
  `concat(layer(input1), layer(input2)) == layer(concat(input1, input2))` for
  any inputs 1 and 2.

  TODO(aferludin): this must be controlled by the `Layer` property and set
  during sampling model export, only if Keras layers has this guarantee.

  Args:
    layer: The layer to check.

  Returns:
    `True` if layer supports batching.
  """

  def _is_stackable(spec) -> bool:
    return spec.HasField('ragged_tensor') or spec.HasField('tensor')

  return all(_is_stackable(t) for t in [*layer.inputs, *layer.outputs])


_REGISTERED_EXECUTORS: Dict[str, Executor] = {
    'TFModel': _tf_model_executor,
}


def register_stage_executor(layer_type: str, executor: Executor) -> None:
  """Allows to assing stage executor to the stage layer type."""
  if layer_type in _REGISTERED_EXECUTORS:
    raise ValueError(
        f'Executor for layer type {layer_type} is already registered'
    )
  _REGISTERED_EXECUTORS[layer_type] = executor
