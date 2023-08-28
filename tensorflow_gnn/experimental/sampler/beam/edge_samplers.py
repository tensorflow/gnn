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
from tensorflow_gnn.experimental.sampler import eval_dag_pb2 as pb
from tensorflow_gnn.experimental.sampler.beam import executor_lib
from tensorflow_gnn.experimental.sampler.beam import utils


PCollection = beam.pvalue.PCollection
SourceId = executor_lib.SourceId
TargetId = executor_lib.TargetId
EdgeData = executor_lib.EdgeData
ExampleId = executor_lib.ExampleId
Value = executor_lib.Value
Values = executor_lib.Values
PValues = executor_lib.PValues

ExampleId = executor_lib.ExampleId
PEdges = executor_lib.PEdges


class _UniformEdgesSamplerBase(beam.PTransform):
  """Base class for all edge samplers."""

  def __init__(self, layer: pb.Layer):
    if len(layer.inputs) != 1 or not layer.inputs[0].HasField('ragged_tensor'):
      raise ValueError(
          'Invalid input signature for `UniformEdgesSampler` layer:'
          f' expected single ragged tensor value, got {layer.inputs}.'
      )
    if len(layer.outputs) < 2 or not all(
        output.HasField('ragged_tensor') for output in layer.outputs
    ):
      raise ValueError(
          'Invalid output signature for `UniformEdgesSampler` layer: expected'
          ' at least a pair of source and target ragged tensors,'
          f' got {layer.outputs}.'
      )
    self._layer = layer
    for index, output in enumerate(self._layer.outputs):
      if len(output.ragged_tensor.shape.dim) != 2:
        raise ValueError(
            f'Shape {output.ragged_tensor.shape.dim} for output {index} of'
            ' UniformEdgesSampler is currently unsupported (b/297411286).'
            f' Expected feature {index} to be a scalar with the same dimensions'
            ' as source and target'
        )
    config = pb.EdgeSamplingConfig()
    layer.config.Unpack(config)
    if config.sample_size <= 0:
      raise ValueError(f'Expected sampling size > 0, got {config.sample_size}')
    self._sample_size = int(config.sample_size)


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

  @beam_typehints.with_input_types(Tuple[SourceId, EdgeData])
  @beam_typehints.with_output_types(Tuple[SourceId, bytes])
  class MakeFeatureTuples(beam.DoFn):
    """Makes and serializes feature tuples from serialized features.
    
    When the input contains a serialized tf.train.Example, this stage will
    deserialize it, build a tuple out of its feature values and discard the
    feature names, and serialize that tuple with a beam Coder. This should save
    a significant amount of intermediate disk space and memory.
    """

    def __init__(self, layer: pb.Layer):
      self._layer = layer

    def setup(self):
      self._coder = _get_feature_coder(self._layer)
      self._np_types = [
          utils.get_np_dtype(output.ragged_tensor.dtype)
          for output in self._layer.outputs
      ]
      if np.issubdtype(self._np_types[1], np.integer):
        self._featureless_target_typehint = int
      else:
        self._featureless_target_typehint = bytes

    def process(
        self, inputs: Tuple[SourceId, EdgeData]
    ) -> Iterator[Tuple[SourceId, bytes]]:
      source_id, edge_data = inputs
      target_id, serialized_feature = edge_data
      if serialized_feature is not None:
        example = tf.train.Example.FromString(serialized_feature)
        # TODO: b/297185857 - Add the expected feature names to pb.Layer
        # We should also add the edge_target_feature_name.
        feature_names = list(sorted(example.features.feature.keys()))
        feature_values = []
        # TODO: b/297411286 - Enable non-scalar edge features to be encoded as
        # np.ndarrays.
        for idx, feature_name in enumerate(feature_names):
          if self._np_types[idx] == np.dtype(np.object_):
            if not (
                feature_name in example.features.feature
                and isinstance(
                    example.features.feature[feature_name].bytes_list.value[0],
                    bytes,
                )
            ):
              raise ValueError(
                  f'Feature {feature_name} does not exist or is not a bytes'
                  ' feature.'
              )
            feature_values.append(
                example.features.feature[feature_name].bytes_list.value[0]
            )
          elif np.issubdtype(self._np_types[idx], np.floating):
            if not (
                feature_name in example.features.feature
                and isinstance(
                    example.features.feature[feature_name].float_list.value[0],
                    float,
                )
            ):
              raise ValueError(
                  f'Feature {feature_name} does not exist or is not a float'
                  ' feature.'
              )
            feature_values.append(
                example.features.feature[feature_name].float_list.value[0]
            )
          elif np.issubdtype(self._np_types[idx], np.integer):
            if not (
                feature_name in example.features.feature
                and isinstance(
                    example.features.feature[feature_name].int64_list.value[0],
                    int,
                )
            ):
              raise ValueError(
                  f'Feature {feature_name} does not exist or is not an int'
                  ' feature.'
              )
            feature_values.append(
                example.features.feature[feature_name].int64_list.value[0]
            )
        feature_tuple = tuple(feature_values)
        yield (source_id, self._coder.encode(feature_tuple))
      else:
        featureless_target_encoder = typecoders.registry.get_coder(
            self._featureless_target_typehint
        )
        yield (source_id, featureless_target_encoder.encode(target_id))

  @beam_typehints.with_input_types(Tuple[SourceId, Iterable[bytes]])
  @beam_typehints.with_output_types(
      Tuple[Tuple[SourceId, int], np.ndarray]
  )
  class CreateValues(beam.DoFn):
    """Splits edges grouped by their source node ids into equally sized buckets.

    The input are streams of edge target node ids grouped by source node ids.
    Those streams are batched into `EDGE_BUCKET_SIZE` buckets and indexed
    starting from 0. The output is a collection of those buckets keyed by
    (source node id, bucket index) tuples.
    """

    def __init__(self, layer: pb.Layer):
      self._layer = layer

    def setup(self):
      self._dtype = np.dtype(np.object_)

    def process(
        self, inputs: Tuple[SourceId, Iterable[bytes]]
    ) -> Iterator[
        Tuple[Tuple[SourceId, int], np.ndarray]
    ]:
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
              self._make_bucket(
                  buffer[: UniformEdgesSampler.EDGE_BUCKET_SIZE], self._dtype
              ),
          )
          buffer = buffer[UniformEdgesSampler.EDGE_BUCKET_SIZE :]

      if buffer:
        yield (source_id, bucket_id), self._make_bucket(buffer, self._dtype)
        bucket_id += 1

    def _make_bucket(
        self, edges: List[TargetId], dtype: np.dtype
    ) -> np.ndarray:
      assert len(edges) <= UniformEdgesSampler.EDGE_BUCKET_SIZE
      return np.array(edges, dtype=dtype)

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
      (num_samples, example_id, index), edge_data = values

      if edge_data is None:
        yield (example_id, (index, source_id, None))
        return

      serialized_features = edge_data

      if num_samples == len(serialized_features):
        sampled_serialized_features = edge_data
      else:
        assert num_samples < len(serialized_features), source_id
        indices = np.arange(len(serialized_features))
        sampled_indices = self._rng.choice(
            indices, size=num_samples, replace=False
        )
        sampled_serialized_features = (
            serialized_features[sampled_indices]
            if serialized_features is not None
            else None
        )
      for serialized_features in sampled_serialized_features:
        yield (
            example_id,
            (index, source_id, serialized_features),
        )

  @beam_typehints.with_input_types(
      Tuple[ExampleId, Iterable[Tuple[int, SourceId, Optional[bytes]]]]
  )
  @beam_typehints.with_output_types(Tuple[ExampleId, Values])
  class AggregateResults(beam.DoFn):
    """Aggregates final sampling results from pieces grouped by example ids.

    The inputs are tuples of (source node id index, source node id, sampled
    target node id) grouped by example ids. The source node id index is used
    to sort sampled edges in the order of source node ids in the `PTransform`s
    input. The output contains two ragged tensors with source and target node
    ids of sampled edges.
    """

    def __init__(self, layer: pb.Layer):
      self._layer = layer

    def setup(self):
      self._source_dtypes = utils.get_ragged_np_types(
          self._layer.outputs[0].ragged_tensor
      )
      self._target_dtypes = utils.get_ragged_np_types(
          self._layer.outputs[1].ragged_tensor
      )
      self._features_dtypes = [
          utils.get_ragged_np_types(output.ragged_tensor)
          for output in self._layer.outputs
      ]
      self._coder = _get_feature_coder(self._layer)
      if np.issubdtype(self._features_dtypes[1][0], np.integer):
        self._featureless_target_typehint = int
      else:
        self._featureless_target_typehint = bytes

    def _make_features(self, features: List[bytes]) -> Values:
      examples = [self._coder.decode(f) for f in features]
      extra_features = []
      for idx, feature_dtype in enumerate(self._features_dtypes):
        ragged_row_length = np.array([len(examples)], dtype=feature_dtype[1])
        if feature_dtype[0] == np.dtype(np.object_):
          extra_features.append([
              np.array([ex[idx] for ex in examples], dtype=feature_dtype[0]),
              ragged_row_length,
          ])
        elif np.issubdtype(feature_dtype[0], np.floating):
          extra_features.append([
              np.array([ex[idx] for ex in examples], dtype=feature_dtype[0]),
              ragged_row_length,
          ])
        elif np.issubdtype(feature_dtype[0], np.integer):
          extra_features.append([
              np.array([ex[idx] for ex in examples], dtype=feature_dtype[0]),
              ragged_row_length,
          ])
      return extra_features

    def process(
        self,
        inputs: Tuple[
            ExampleId, Iterable[Tuple[int, SourceId, bytes]]
        ],
    ) -> Iterator[Tuple[ExampleId, Values]]:
      example_id, values_iter = inputs
      values = [(i, s, e) for i, s, e in values_iter if e is not None]
      values = sorted(values, key=lambda kv: kv[0])
      serialized_features = [t for _, _, t in values if t is not None]
      if len(self._layer.outputs) > 2:
        features = self._make_features(serialized_features)
        yield (example_id, features)
      else:
        sources = [
            np.array([s for _, s, _ in values], dtype=self._source_dtypes[0]),
            np.array([len(values)], dtype=self._source_dtypes[1]),
        ]
        coder = typecoders.registry.get_coder(self._featureless_target_typehint)
        targets = [
            np.array(
                [coder.decode(t) for _, _, t in values],
                dtype=self._target_dtypes[0],
            ),
            np.array([len(values)], dtype=self._target_dtypes[1]),
        ]
        yield (example_id, [sources, targets])

  @beam_typehints.with_input_types(ExampleId)
  @beam_typehints.with_output_types(Tuple[ExampleId, Values])
  class CreateEmptyResults(beam.DoFn):
    """Creates empty sampling results for input example ids."""

    def __init__(self, layer: pb.Layer):
      self._layer = layer

    def setup(self):
      features_dtypes = [
          utils.get_ragged_np_types(output.ragged_tensor)
          for output in self._layer.outputs
      ]
      features = []
      for dtypes in features_dtypes:
        features.append([
            np.array([], dtype=dtypes[0]),
            np.array([0], dtype=dtypes[1]),
        ])
      self._empty_value = features

    def process(
        self,
        example_id: ExampleId,
    ) -> Iterator[Tuple[ExampleId, Values]]:
      yield (example_id, self._empty_value)

  def expand(self, inputs) -> PValues:
    source_ids, edges = inputs

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
        | 'Queries' >> beam.ParDo(self.CreateQueries(self._sample_size))
    )
    value_buckets = (
        edges
        | 'MakeFeatureTuples' >> beam.ParDo(self.MakeFeatureTuples(self._layer))
        | 'GroupEdges' >> beam.GroupByKey()
        | 'Values' >> beam.ParDo(self.CreateValues(self._layer))
    )

    sampling_results = (
        (
            query_buckets,
            value_buckets,
        )
        | 'LookupEdgeBuckets' >> utils.LeftLookupJoin()
        | 'SampleFromBuckets' >> beam.ParDo(self.SampleFromBuckets())
        | 'GroupByExampleId' >> beam.GroupByKey()
        | 'AggregateResults' >> beam.ParDo(self.AggregateResults(self._layer))
    )
    empty_results = empty_inputs | 'CreateEmptyResults' >> beam.ParDo(
        self.CreateEmptyResults(self._layer)
    )
    return [sampling_results, empty_results] | 'Flatten' >> beam.Flatten()


def _get_feature_coder(layer: pb.Layer) -> beam.coders.Coder:
  """Returns beam feature coder for layer."""
  output_types = []
  np_types = [
      utils.get_np_dtype(output.ragged_tensor.dtype)
      for output in layer.outputs
  ]
  for i, np_type in enumerate(np_types):
    if np.issubdtype(np_type, np.floating):
      output_types.append(float)
    elif np.issubdtype(np_type, np.integer):
      output_types.append(int)
    elif np_type == np.dtype(np.object_):
      output_types.append(bytes)
    else:
      raise ValueError(f'Unsupported dtype: {np_type} for output {i}')
  output_typehint = beam.typehints.Tuple[tuple(output_types)]
  return typecoders.registry.get_coder(output_typehint)


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


executor_lib.register_stage_executor(
    'UniformEdgesSampler', _uniform_edge_sampler
)
