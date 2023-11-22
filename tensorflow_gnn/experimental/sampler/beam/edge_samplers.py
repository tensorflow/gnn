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
"""Executors for edge sampling stages."""

from typing import Dict, Iterable, Iterator, List, Optional, Tuple, cast

import apache_beam as beam
from apache_beam import typehints as beam_typehints
from apache_beam.coders import typecoders
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.experimental.sampler import proto as pb
from tensorflow_gnn.experimental.sampler.beam import executor_lib
from tensorflow_gnn.experimental.sampler.beam import utils


PCollection = beam.pvalue.PCollection
SourceId = executor_lib.SourceId
TargetId = executor_lib.TargetId
ExampleId = executor_lib.ExampleId
Value = executor_lib.Value
Values = executor_lib.Values
PValues = executor_lib.PValues

ExampleId = executor_lib.ExampleId
PEdges = executor_lib.PEdges


class _UniformEdgesSamplerBase(beam.PTransform):
  """Base class for all edge samplers."""

  def __init__(self, layer: pb.Layer):
    config = pb.EdgeSamplingConfig()
    layer.config.Unpack(config)

    self._debug_context = _get_error_message_details(layer, config)
    if config.sample_size <= 0:
      raise ValueError(
          f'Expected sampling size > 0, got {config.sample_size} for'
          f' {self._debug_context}'
      )

    if len(config.edge_feature_names.feature_names) != len(layer.outputs):
      raise ValueError(
          'The numbers of edge features'
          f' {len(config.edge_feature_names.feature_names)} and layer outputs '
          f' {len(layer.outputs)} do not match for'
          f' {self._debug_context}'
      )
    # Checks that layer config is valid.
    for fname in (
        tfgnn.SOURCE_NAME,
        config.edge_target_feature_name,
    ):
      if fname not in config.edge_feature_names.feature_names:
        raise ValueError(
            f'The {fname} index feature is missing for {self._debug_context}'
        )

    # Output values are sorted by their names.
    output_feature_names = sorted(
        [
            (f if f != config.edge_target_feature_name else tfgnn.TARGET_NAME)
            for f in config.edge_feature_names.feature_names
        ]
    )

    self._input_features_spec = []
    self._output_features_spec = []
    for output_spec, output_name in zip(layer.outputs, output_feature_names):
      input_name = (
          output_name
          if output_name != tfgnn.TARGET_NAME
          else config.edge_target_feature_name
      )
      if not output_spec.HasField('ragged_tensor'):
        raise ValueError(
            f'Expected ragged feature {input_name} for {self._debug_context}'
        )

      self._input_features_spec.append((
          input_name,
          _get_input_spec(output_spec.ragged_tensor),
      ))
      self._output_features_spec.append((output_name, output_spec))

    self._layer = layer
    self._config = config


@beam_typehints.with_output_types(Tuple[ExampleId, Values])
class UniformEdgesSampler(_UniformEdgesSamplerBase):
  """Samples outgoing edges uniformly at random without replacement.

  Implements `UniformEdgesSampler` interface. It takes a collection of source
  node ids and all existing edges as input, and outputs a sample of outgoing
  edges. The source node ids are tuples of unique `ExampleId` and a ragged
  tensor of source node ids from which outgoing edges are selected. The edges
  collection contains tuples of edge source and target node ids (integer or
  bytes). The sampled edges are returned for each input `ExampleId` as two
  ragged tensors containing the source and and target node ids of the sampled
  edges. The maximum number of sampled edges per source node is defined in the
  layer configuration (`EdgeSamplingConfig`).

  The class implements uniform edge sampling by first grouping edges for each
  source node and then bucketing them into buckets of `EDGE_BUCKET_SIZE` so that
  there is at most one incomplete bucket. Edges are sampled in two stages.
  First source node ids are joined with their out degrees. This allows to select
  edge buckets and number of edges to sample from each bucket. On the second
  stage edges are sampled from each buckets.  The `EDGE_BUCKET_SIZE` controls
  data chunk sizes of distributed shuffle operation (as `beam.GroupByKey`).
  Setting `EDGE_BUCKET_SIZE >> 1` reduces the total number of shuffle pairs
  and serialization/deserialization overhead. On the other hand, it should be
  "not too big" for better load-balancing and small memory overhead.
  """

  EDGE_BUCKET_SIZE = 100

  @beam_typehints.with_input_types(Tuple[ExampleId, Values])
  @beam_typehints.with_output_types(Tuple[SourceId, Tuple[ExampleId, int]])
  class RekeyBySourceIds(beam.DoFn):
    """Extracts source node ids from the input values.

    The input are tuples of (example id, source node ids). The source node ids
    from the input are unpacked, indexed for each example id and output as
    (source node id, example id, source node id index). The "source node id
    index" is used on the last stage of the PTransform to sort sampled edges
    for each example id in the order of the input source node ids.
    """

    def process(
        self, inputs: Tuple[ExampleId, Values]
    ) -> Iterator[Tuple[SourceId, Tuple[ExampleId, int]]]:
      example_id, values = inputs
      assert len(values) == 1 and len(values[0]) == 2
      values = values[0][0]
      for index, source_id in enumerate(values):
        source_id = utils.as_pytype(source_id)
        assert isinstance(source_id, (int, bytes))
        yield source_id, (example_id, index)

  @beam_typehints.with_input_types(Tuple[ExampleId, Values])
  @beam_typehints.with_output_types(ExampleId)
  class FilterEmptyInputs(beam.DoFn):
    """Filters example ids of empty source node ids inputs."""

    def process(self, inputs: Tuple[ExampleId, Values]) -> Iterator[ExampleId]:
      example_id, values = inputs
      assert len(values) == 1 and len(values[0]) == 2
      if values[0][0].size == 0:
        yield example_id

  @beam_typehints.with_input_types(
      Tuple[
          SourceId,
          Tuple[Tuple[ExampleId, int], Optional[int]],
      ]
  )
  @beam_typehints.with_output_types(
      Tuple[Tuple[SourceId, int], Tuple[int, ExampleId, int]]
  )
  class CreateQueries(beam.DoFn):
    """Creates sampling queries from source node ids and their node degrees.

    The "sampling query" specifies the exact number of edges to sample without
    replacement from each edge bucket. Because edges are grouped into equally
    sized groups and sampled uniformly at random, the node out degree contains
    enough information to generate sampling queries. The output values are key-
    value pairs with (source node id, edge bucket index) keys. The output values
    are (number of edges to sample from bucket, example id, source node id
    index).
    """

    def __init__(self, sample_size: int):
      self._sample_size = sample_size

    def setup(self):
      self._rng = np.random.Generator(np.random.Philox())

    def process(
        self,
        inputs: Tuple[
            SourceId,
            Tuple[Tuple[ExampleId, int], Optional[int]],
        ],
    ) -> Iterator[Tuple[Tuple[SourceId, int], Tuple[int, ExampleId, int]]]:
      source_id, ((example_id, index), out_degree) = inputs
      assert isinstance(source_id, (bytes, int)), type(source_id)
      if out_degree is None:
        edge_indices = np.array([-1])
      elif out_degree <= self._sample_size:
        edge_indices = np.arange(out_degree)
      else:
        edge_indices = self._rng.choice(
            out_degree, size=self._sample_size, replace=False
        )
      buckets, num_samples_per_bucket = np.unique(
          edge_indices // UniformEdgesSampler.EDGE_BUCKET_SIZE,
          return_counts=True,
      )
      for bucket_id, sample_size in zip(buckets, num_samples_per_bucket):
        yield (source_id, bucket_id.item()), (
            sample_size.item(),
            example_id,
            index,
        )

  @beam_typehints.with_input_types(tf.train.Example)
  @beam_typehints.with_output_types(Tuple[SourceId, bytes])
  class ExtractEdges(beam.DoFn):
    """Extracts edge source and target ids, plus edge features, if any.

    To simplify upstream processing, the parsed edge target node ids and its
    features are converted to `Values` and serialized to `bytes` using Beam
    `PCoder`.
    """

    def __init__(
        self,
        features_spec: List[Tuple[str, pb.ValueSpec]],
        *,
        debug_context: str,
    ):
      self._debug_context = debug_context
      self._features_spec = features_spec

    def setup(self):
      # NOTE: we use generic `Values` type to represent edge features for
      # intermediate computation. Although for the typical case of scalar edge
      # features it may be an overkill (NDArrayCoder serializes `np.array` shape
      # and dtype for each instance). In the feature, we may optimize this by
      # specializing `PCoder`s directly for `Values` type.
      self._coder = typecoders.registry.get_coder(executor_lib.Values)
      self._values_spec = []
      for fname, spec in self._features_spec:
        if fname == tfgnn.SOURCE_NAME:
          self._source_spec = (fname, spec)
        else:
          self._values_spec.append((fname, spec))

    def process(
        self, example: tf.train.Example
    ) -> Iterator[Tuple[SourceId, bytes]]:
      features = []
      try:
        source_id = utils.parse_tf_example(example, *self._source_spec)[0]
        source_id = source_id.item(0)
        assert isinstance(source_id, (int, bytes)), type(source_id)
        for value_spec in self._values_spec:
          features.append(utils.parse_tf_example(example, *value_spec))

        yield (source_id, self._coder.encode(features))
      except ValueError as e:
        raise ValueError(f'{e} for {self._debug_context}') from e

  @beam_typehints.with_input_types(Tuple[SourceId, Iterable[bytes]])
  @beam_typehints.with_output_types(Tuple[Tuple[SourceId, int], np.ndarray])
  class CreateValues(beam.DoFn):
    """Splits edges grouped by their source node ids into equally sized buckets.

    The input are streams of edge target node ids grouped by source node ids.
    Those streams are batched into `EDGE_BUCKET_SIZE` buckets and indexed
    starting from 0. The output is a collection of those buckets keyed by
    (source node id, bucket index) tuples.
    """

    def process(
        self, inputs: Tuple[SourceId, Iterable[bytes]]
    ) -> Iterator[Tuple[Tuple[SourceId, int], np.ndarray]]:
      source_id, edge_data = inputs
      buffer = []
      bucket_id = 0
      num_edges = 0
      for serialized_feature in edge_data:
        buffer.append(serialized_feature)
        num_edges += 1
        while len(buffer) >= UniformEdgesSampler.EDGE_BUCKET_SIZE:
          yield (
              (source_id, bucket_id),
              self._make_bucket(buffer[: UniformEdgesSampler.EDGE_BUCKET_SIZE]),
          )
          buffer = buffer[UniformEdgesSampler.EDGE_BUCKET_SIZE :]
          bucket_id += 1

      if buffer:
        yield (source_id, bucket_id), self._make_bucket(buffer)
        bucket_id += 1

    def _make_bucket(self, edges: List[TargetId]) -> np.ndarray:
      assert len(edges) <= UniformEdgesSampler.EDGE_BUCKET_SIZE
      return np.array(edges, dtype=np.object_)

  @beam_typehints.with_input_types(
      Tuple[
          Tuple[SourceId, int],
          Tuple[
              Tuple[int, ExampleId, int],
              Optional[np.ndarray],
          ],
      ]
  )
  @beam_typehints.with_output_types(
      Tuple[ExampleId, Tuple[int, SourceId, Optional[bytes]]]
  )
  class SampleFromBuckets(beam.DoFn):
    """Samples edges from the edge buckets.

    The input is a join result of "sampling queries" with matching edge buckets.
    It is keyed by edges' source node ids and bucket indices. Each sampling
    query is a 3-tuple if (number of edges to sample, example ids, source id
    index). The output is a key-value pairs. The keys are example ids. The
    values are (source id index, source id, optional sampled target node id)
    tuples.


    NOTE: we use `None` as sampled target node id to identify queries for which
    there are no edges. This is done to generate sampling results for all input
    example ids, even if there are no edges for all its source node ids.
    """

    def setup(self):
      self._rng = np.random.Generator(np.random.Philox())

    def process(
        self,
        inputs: Tuple[
            Tuple[SourceId, int],
            Tuple[
                Tuple[int, ExampleId, int],
                Optional[np.ndarray],
            ],
        ],
    ) -> Iterator[Tuple[ExampleId, Tuple[int, SourceId, Optional[bytes]]]]:
      (source_id, _), values = inputs
      (num_samples, example_id, index), bucket = values

      if bucket is None:
        yield (example_id, (index, source_id, None))
        return

      if num_samples == len(bucket):
        sampled = bucket
      else:
        assert num_samples < len(bucket), source_id
        indices = np.arange(len(bucket))
        sampled_indices = self._rng.choice(
            indices, size=num_samples, replace=False
        )
        sampled = bucket[sampled_indices]
      for value in sampled:
        yield (
            example_id,
            (index, source_id, value),
        )

  @beam_typehints.with_input_types(
      Tuple[ExampleId, Iterable[Tuple[int, SourceId, Optional[bytes]]]]
  )
  @beam_typehints.with_output_types(Tuple[ExampleId, Values])
  class AggregateResults(beam.DoFn):
    """Aggregates final sampling results from pieces grouped by example ids.

    The inputs are tuples of (source node id index, source node id, serialized
    edge features including edge target ids) grouped by example ids. The edge
    features are sorted in their output order except the edge source node ids
    which are omitted.

    The source node id index stores order of the source node ids in the sampling
    query, so the sampled edges in the `PTransform` output are sorted in the
    same order as their queries. The output contains ragged for tensors source
    node ids, target node ids of and edge features (if any) for sampled edges in
    the `features_spec` order.
    """

    def __init__(
        self,
        features_spec: List[Tuple[str, pb.ValueSpec]],
        *,
        debug_context: str,
    ):
      self._features_spec = features_spec[:]

      self._edge_source_index = -1
      self._features_index = []
      for index, (name, _) in enumerate(features_spec):
        if name == tfgnn.SOURCE_NAME:
          self._edge_source_index = index
        else:
          self._features_index.append(index)

      assert self._edge_source_index >= 0
      assert self._features_index

      self._debug_context = debug_context

    def setup(self):
      self._coder = typecoders.registry.get_coder(executor_lib.Values)

      self._empty_value = _get_empty_value(self._features_spec)

      def restore_batch_dimension(
          value: executor_lib.Value, num_edges: int, dtype: np.dtype
      ) -> executor_lib.Value:
        """Restores batch dimension of size 1."""
        return [value[0], np.array([num_edges], dtype=dtype), *value[1:]]

      def get_dense_stacking_fn(row_splits_dtype: np.dtype):
        def fn(batch: List[executor_lib.Value]) -> executor_lib.Value:
          return restore_batch_dimension(
              utils.stack(batch), len(batch), row_splits_dtype
          )

        return fn

      def get_ragged_stacking_fn(row_splits_dtype: np.dtype):
        def fn(batch: List[executor_lib.Value]) -> executor_lib.Value:
          return restore_batch_dimension(
              utils.stack_ragged(batch, row_splits_dtype),
              len(batch),
              row_splits_dtype,
          )

        return fn

      self._stacking_fns = []
      for name, output in self._features_spec:
        dtypes = utils.get_ragged_np_types(output.ragged_tensor)
        if name == tfgnn.SOURCE_NAME:
          self._edge_source_dtype = dtypes[0]

        if output.ragged_tensor.ragged_rank > 1:
          stacking_fn = get_ragged_stacking_fn(dtypes[1])
        else:
          stacking_fn = get_dense_stacking_fn(dtypes[1])

        self._stacking_fns.append(stacking_fn)

    def process(
        self,
        inputs: Tuple[ExampleId, Iterable[Tuple[int, SourceId, bytes]]],
    ) -> Iterator[Tuple[ExampleId, Values]]:
      example_id, values_iter = inputs
      sampled_values = [(i, s, v) for i, s, v in values_iter if v is not None]
      if not sampled_values:
        yield (example_id, self._empty_value)
        return

      sampled_values.sort(key=lambda x: x[0])

      batches = [[] for _ in range(1 + len(self._features_index))]
      try:
        for _, source_id, serialized_features in sampled_values:
          batches[self._edge_source_index].append(
              [np.array(source_id, dtype=self._edge_source_dtype)]
          )
          features = self._coder.decode(serialized_features)
          assert isinstance(features, list)
          assert len(self._features_index) == len(features)

          for index, feature in zip(self._features_index, features):
            batches[index].append(feature)

        values = [
            stacking_fn(batch)
            for stacking_fn, batch in zip(self._stacking_fns, batches)
        ]

      except Exception as e:
        raise ValueError(
            f'Failed to aggregate results for {example_id}: {e} for'
            f' {self._debug_context}'
        ) from e

      yield (example_id, values)

  @beam_typehints.with_input_types(ExampleId)
  @beam_typehints.with_output_types(Tuple[ExampleId, Values])
  class CreateEmptyResults(beam.DoFn):
    """Creates empty sampling results for input example ids."""

    def __init__(self, features_spec: List[Tuple[str, pb.ValueSpec]]):
      self._features_spec = features_spec

    def setup(self):
      self._empty_value = _get_empty_value(self._features_spec)

    def process(
        self,
        example_id: ExampleId,
    ) -> Iterator[Tuple[ExampleId, Values]]:
      yield (example_id, self._empty_value)

  def expand(self, inputs) -> PValues:
    source_ids, raw_edges = inputs

    edges = raw_edges | 'ExtractEdges' >> beam.ParDo(
        self.ExtractEdges(
            self._input_features_spec, debug_context=self._debug_context
        )
    )

    out_degrees = edges | 'OutDegree' >> beam.combiners.Count.PerKey()
    queries = source_ids | 'RekeyBySourceIds' >> beam.ParDo(
        self.RekeyBySourceIds()
    )
    empty_inputs = source_ids | 'FilterEmptyInputs' >> beam.ParDo(
        self.FilterEmptyInputs()
    )

    query_buckets = (
        (queries, out_degrees)
        | 'JoinSourceIdsWithOutDegrees' >> utils.LeftLookupJoin()
        | 'Queries' >> beam.ParDo(self.CreateQueries(self._config.sample_size))
    )

    value_buckets = (
        edges
        | 'GroupEdges' >> beam.GroupByKey()
        | 'Values' >> beam.ParDo(self.CreateValues())
    )

    sampling_results = (
        (
            query_buckets,
            value_buckets,
        )
        | 'LookupEdgeBuckets' >> utils.LeftLookupJoin()
        | 'SampleFromBuckets' >> beam.ParDo(self.SampleFromBuckets())
        | 'GroupByExampleId' >> beam.GroupByKey()
        | 'AggregateResults'
        >> beam.ParDo(
            self.AggregateResults(
                self._output_features_spec,
                debug_context=self._debug_context,
            )
        )
    )
    empty_results = empty_inputs | 'CreateEmptyResults' >> beam.ParDo(
        self.CreateEmptyResults(self._output_features_spec)
    )
    return [sampling_results, empty_results] | 'Flatten' >> beam.Flatten()


def _uniform_edge_sampler(
    label: str,
    layer: pb.Layer,
    inputs: PValues,
    feeds: Dict[str, executor_lib.PFeed],
    unused_artifacts_path: str,
) -> PValues:
  """Returns UniformEdgesSampler stage executor."""
  del unused_artifacts_path
  edges_table = feeds.get(layer.id, None)
  if edges_table is None:
    raise ValueError(
        f'Missing edges table for UniformEdgesSampler layer {layer.id}', feeds
    )

  edges_table = cast(PEdges, edges_table)
  return (inputs, edges_table) | label >> UniformEdgesSampler(layer)


def _get_error_message_details(
    layer: pb.Layer, config: pb.EdgeSamplingConfig
) -> str:
  return (
      f'edge set "{config.edge_set_name}" sampling layer {layer.type} with'
      f' id={layer.id}.'
  )


def _get_input_spec(output_spec: pb.RaggedTensorSpec) -> pb.ValueSpec:
  shape = tf.TensorShape(output_spec.shape)[2:].as_proto()
  if output_spec.ragged_rank <= 2:
    return pb.ValueSpec(
        tensor=pb.TensorSpec(dtype=output_spec.dtype, shape=shape)
    )

  return pb.ValueSpec(
      ragged_tensor=pb.RaggedTensorSpec(
          dtype=output_spec.dtype,
          shape=shape,
          ragged_rank=output_spec.ragged_rank - 1,
          row_splits_dtype=output_spec.row_splits_dtype,
      )
  )


def _get_empty_value(features_spec: List[Tuple[str, pb.ValueSpec]]) -> Values:
  features_dtypes = [
      utils.get_ragged_np_types(spec.ragged_tensor) for _, spec in features_spec
  ]
  features = []
  for dtypes in features_dtypes:
    features.append([
        np.array([], dtype=dtypes[0]),
        np.array([0], dtype=dtypes[1]),
    ])
  return features


executor_lib.register_stage_executor(
    'UniformEdgesSampler', _uniform_edge_sampler
)
