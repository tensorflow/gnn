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
"""Creates `DatasetProvider` and `Tasks` instances to invoke TF-GNN runner."""

from __future__ import annotations
import abc
import dataclasses
import functools
from typing import Any, Callable

import tensorflow as tf
from tensorflow_gnn import runner
from tensorflow_gnn.experimental.in_memory import datasets
from tensorflow_gnn.experimental.in_memory import int_arithmetic_sampler as ia_sampler
from tensorflow_gnn.experimental.sampler import link_samplers
from tensorflow_gnn.experimental.sampler import subgraph_pipeline
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.proto import graph_schema_pb2
from tensorflow_gnn.sampler import sampling_spec_builder
from tensorflow_gnn.sampler import sampling_spec_pb2


RANDOM_UNIFORM = sampling_spec_builder.SamplingStrategy.RANDOM_UNIFORM


class SizedDatasetProvider(runner.DatasetProvider):

  @property
  @abc.abstractmethod
  def cardinality(self) -> int:
    raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class TaskedData:
  """Bundles instances required to train by `tfgnn.runner`: dataset and task."""
  task: runner.Task
  provider: SizedDatasetProvider
  validation_provider: None|SizedDatasetProvider = None


class Sampling:
  """Union (one-of) sampling options."""

  @property
  def spec(self) -> None | sampling_spec_pb2.SamplingSpec:
    return self._spec

  @property
  def sample_sizes(self) -> None | tuple[int, ...]:
    return self._sample_sizes

  def __init__(self, spec: None | sampling_spec_pb2.SamplingSpec,
               sample_sizes: None | tuple[int, ...]):
    if int(spec is None) + int(sample_sizes is None) != 1:
      raise ValueError('Expecting exactly one of `spec` and `sample_sizes`')
    self._spec = spec
    self._sample_sizes = sample_sizes

  @staticmethod
  def from_spec(spec: sampling_spec_pb2.SamplingSpec) -> Sampling:
    return Sampling(spec=spec, sample_sizes=None)

  @staticmethod
  def from_sample_sizes(sample_sizes: list[int]) -> Sampling:
    return Sampling(spec=None, sample_sizes=tuple(sample_sizes))


Sampling.SMALL_SAMPLING = Sampling.from_sample_sizes([5, 5])


def provide_link_prediction_data(
    *,
    dataset_name: None|str = None,
    graph_data: None|datasets.LinkPredictionGraphData = None,
    source_sampling: None|Sampling = None,
    target_sampling: None|Sampling = None,
    task: None|runner.Task = None,
    ) -> TaskedData:
  """Provides `TaskedData` for link prediction tasks.

  Given a graph (indicated by `dataset_name` or `graph_data`), a dataset
  iterating all edges and producing subgraph `GraphTensor` around each edge will
  created. The size of all `GraphTensor`s (depth and fanout) is determined by
  `sampling`. Positive edges are sampled uniformly. For each positive edge
  connecting `source->target`, negative edges `source->n_target` and
  `n_source->target` will be sampled where `n_source` and `n_target` are sampled
  uniformly at random.

  Args:
    dataset_name: Mutually-exclusive with `graph_data`. If given, it must be
      name of node link prediction dataset recognized by in_memory/datasets.py,
      e.g., "ogbl-collab". This is used to create `graph_data` without any
      modifications (edges will not be undirected, self-connections will not be
      added, etc).
    graph_data: Mutually-exclusive with `dataset_name`. It provides the list of
      (positive) training edges and validation (positive and negative) edges.
    source_sampling: Specifies how each `GraphTensor` instance will be sampled
      around an (positive or negative) endpoint `source` around edge
      `source->target`. If None, defaults to `SMALL_SAMPLING` (samples 5 nodes
      from each edge-set connected to `source`, and for each sampled node,
      samples 5 of its neighbors per edgeset).
    target_sampling: Similar to `source_sampling`, but for `target` endpoint.
    task: Runner task to apply. If not given, defaults to
      `DotProductLinkPrediction`. If given, user must provide a task that works
      well with the produced subgraphs.

  Returns:
    `TaskedData` (`td`) with `td.task` set to a link-prediction task and
    `provider` set to `DatasetProvider` that yields a dataset of `GraphTensor`
    instances. Each instance is a subgraph sample around source-node of an edge
    unioned with a subgraph sample around target-node of the same edge. The
    positions of the edge endpoints are specified in edge-sets `_readout/source`
    and `_readout/target` and the label of the edge (positive or negative) can
    accessed as `graph.node_sets['_readout']['label']`.
  """
  graph_data = _resolve_graph_data(
      datasets.LinkPredictionGraphData, dataset_name, graph_data)
  assert isinstance(graph_data, datasets.LinkPredictionGraphData)
  schema = graph_data.graph_schema()

  providers = _provide_link_prediction_data_with_int_arithmetic_sampling(
      graph_data,
      _resolve_sampling_spec(
          schema, graph_data.source_node_set_name, source_sampling),
      _resolve_sampling_spec(
          schema, graph_data.target_node_set_name, target_sampling))
  train_provider, validation_provider = providers

  if task is None:
    task = runner.DotProductLinkPrediction()

  return TaskedData(task=task, provider=train_provider,
                    validation_provider=validation_provider)


def provide_node_classification_data(
    *,
    dataset_name: None|str = None,
    graph_data: None|datasets.NodeClassificationGraphData = None,
    sampling: None|Sampling = None) -> TaskedData:
  """Provides `TaskedData` for node classification tasks."""
  graph_data = _resolve_graph_data(
      datasets.NodeClassificationGraphData, dataset_name, graph_data)
  assert isinstance(graph_data, datasets.NodeClassificationGraphData)

  sampler = ia_sampler.GraphSampler(
      graph_data, sampling_mode=ia_sampler.EdgeSampling.WITH_REPLACEMENT)

  node_split = graph_data.node_split()
  schema = graph_data.graph_schema()
  provider_kwargs = dict(
      sampling_pipeline=subgraph_pipeline.SamplingPipeline(
          schema,
          _resolve_sampling_spec(schema, graph_data.labeled_nodeset, sampling),
          sampler.make_edge_sampler,
          create_node_features_lookup_factory(graph_data, True)),
      labeled_node_set_name=graph_data.labeled_nodeset)
  train_provider = _NodeClassificationSamplingProvider(
      node_split.train.shape[0],
      dataset_fn=functools.partial(
          _create_nodes_and_labels_dataset_from_eager_tensors,
          node_split.train, graph_data.labels(), shuffle=True),
      **provider_kwargs)
  validation_provider = _NodeClassificationSamplingProvider(
      node_split.validation.shape[0],
      dataset_fn=functools.partial(
          _create_nodes_and_labels_dataset_from_eager_tensors,
          node_split.validation, graph_data.labels()),
      **provider_kwargs)
  task = _ManyNodesClassification(
      graph_data.labeled_nodeset, label_fn=_readout_seed_node_labels,
      num_classes=graph_data.num_classes())
  return TaskedData(
      task=task, provider=train_provider,
      validation_provider=validation_provider)


## <Helper functions for Node Classification>


def _create_nodes_and_labels_dataset_from_eager_tensors(
    seed_nodes: tf.Tensor,
    labels: tf.Tensor,
    context: None|tf.distribute.InputContext = None,
    *,
    shuffle=False) -> tf.data.Dataset:
  """Creates dataset of `(node, label)` from `seed_nodes` and `labels`."""
  context = context or tf.distribute.InputContext()
  dataset = (tf.data.Dataset.from_tensor_slices(seed_nodes)
             .shard(context.num_input_pipelines, context.input_pipeline_id))
  if shuffle:
    dataset = dataset.shuffle(
        seed_nodes.shape[0], reshuffle_each_iteration=True)
  dataset = dataset.map(lambda node: (node, tf.gather(labels, node)))
  return dataset


def _readout_seed_node_labels(graph: gt.GraphTensor) -> tuple[
    gt.GraphTensor, graph_constants.Field]:
  labels = tf.gather(
      graph.node_sets['_readout']['label'],
      graph.edge_sets['_readout/seed'].adjacency.target)
  return graph, labels


class _ManyNodesClassification(runner.RootNodeMulticlassClassification):
  """Classification setup with possibly multiple labeled nodes."""

  def gather_activations(self, graph: gt.GraphTensor) -> graph_constants.Field:
    x = graph.node_sets[self._node_set_name][graph_constants.HIDDEN_STATE]
    return tf.gather(x, graph.edge_sets['_readout/seed'].adjacency.source)


class _NodeClassificationSamplingProvider(SizedDatasetProvider):
  """DatasetProvider for Node Classification."""

  def __init__(
      self,
      cardinality: int,
      dataset_fn: Callable[[tf.distribute.InputContext], tf.data.Dataset],
      sampling_pipeline: subgraph_pipeline.SamplingPipeline,
      labeled_node_set_name: graph_constants.NodeSetName,
      sampling_batch_size: int = 100):
    """Constructs DatasetProvider for Node Classification.

    Args:
      cardinality: Total number of examples in (one iteration of) dataset.
      dataset_fn: Callable returns dataset that with each element containing
        `(seed ID, label)`. The first will be batched to invoke
        `sampling_pipeline`.
      sampling_pipeline: Maps seed IDs from dataset into `GraphTensor`s.
      labeled_node_set_name: The node set that the labels correspond to.
      sampling_batch_size: Batch size to send to `sampling_model`.
    """
    super().__init__()
    self._cardinality = cardinality
    self._dataset_fn = dataset_fn
    self._sampling_pipeline = sampling_pipeline
    self._node_set_name = labeled_node_set_name
    self._batch_size = sampling_batch_size

  @property
  def cardinality(self) -> int:
    return self._cardinality

  def get_dataset(self, context: tf.distribute.InputContext) -> tf.data.Dataset:
    """Returns dataset with `GraphTensor` elements for node classification.

    Each `GraphTensor` instance is to classify one or more nodes. Specifically,
    edge-set `'_readout/seed'` will contain `K` edges, with source indices
    pointing to labeled nodes. Further, node-set `'_readout'` will contain `K`
    nodes, with feature `'label'` containing the corresponding node labels, 
    where `K` is the number of labeled nodes in `GraphTensor`.

    Args:
      context: To select the data shard. If constructed with default constructor
        (`tf.distribute.InputContext()`), then all data will used by worker.
    """
    sampler = self._sampling_pipeline  # for short.
    return (
        self._dataset_fn(context)
        .batch(self._batch_size)  # For invoking sampler (unbatch to follow).
        # Sample subgraph.
        .map(lambda nodes, labels: (sampler(nodes), nodes, labels),
             num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        .unbatch()
        .map(self._add_structured_readout,
             num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    )

  def _add_structured_readout(self, graph, node_id, label):
    node_sets = dict(graph.node_sets)
    edge_sets = dict(graph.edge_sets)
    ones = tf.ones([1], dtype=graph.indices_dtype)
    node_sets['_readout'] = gt.NodeSet.from_fields(
        sizes=ones, features={'label': tf.expand_dims(label, 0)})
    node_ids = tf.cast(
        graph.node_sets[self._node_set_name]['#id'], node_id.dtype)
    node_id_position = tf.argmax(node_id == node_ids,
                                 output_type=graph.indices_dtype)
    edge_sets['_readout/seed'] = gt.EdgeSet.from_fields(
        sizes=ones, adjacency=adj.Adjacency.from_indices(
            source=(self._node_set_name, tf.expand_dims(node_id_position, 0)),
            target=('_readout', tf.zeros([1], dtype=graph.indices_dtype)),
        ))
    return gt.GraphTensor.from_pieces(
        context=graph.context, edge_sets=edge_sets, node_sets=node_sets)


## <Helper functions for Link Prediction.>


def _create_labeled_edges_dataset_from_eager_tensors(
    sources: tf.Tensor,
    targets: tf.Tensor,
    context: None|tf.distribute.InputContext = None,
    *,
    shuffle=False,
    negative_links_sampling: NegativeLinksSampling,
    ) -> tf.data.Dataset:
  """Creates link-prediction dataset from `sources`, `targets`, and negatives.

  It must be that `source_nodes` and `target_nodes` are both vectors with equal
  shape, with each `(source_nodes[i], target_nodes[i])` indicating a positive
  edge.

  For each positive edge, a total of
  ```num_negatives_per_source + num_negatives_per_target```
  negative examples will be sampled (uniformly at random).

  Args:
    sources: Eager vector of size M. It is (currently) assumed that these
      nodes are integral, and come from range `[0, max_source_node]`. Each
      `(sources[i], targets[i])` indicate an edge.
    targets: Eager vector of size M. It is (currently) assumed that these
      nodes are integral, and come from range `[0, max_target_node]`.
    context: If given, a subset of the data will be used that is designated for
      to the worker by the `context`.
    shuffle: If set, input edges (`sources, targets`) will be shuffled.
    negative_links_sampling: Contains functions that generate negative source
      and target node IDs.

  Returns:
    tf.data.Dataset with total examples:
    `pe_matrix.shape[0] *(1+num_negatives_per_source+num_negatives_per_target)`.
    Each positive edge will be followed by its negatives. The returned dataset
    will yield pair: `(edge, label)`, where `label` is binary tf.float32, and
    `edge` is a 2-size vector containing `[src, tgt]`: the vector can be a row
    of `positive_edge_matrix` (in which case, `label == 1.0`), or one endpoint
    is from `[src, tgt]` and the other is (uniformly) randomly sampled.
  """
  if (len(sources.shape) != 1 or len(targets.shape) != 1
      or int(sources.shape[0]) != targets.shape[0]):
    raise ValueError(
        'Expecting vectors `sources` and `targets` of same size')

  context = context or tf.distribute.InputContext()
  ds = (
      tf.data.Dataset.range(targets.shape[0])
      .shard(context.num_input_pipelines, context.input_pipeline_id))

  if shuffle:
    ds = ds.shuffle(targets.shape[0], reshuffle_each_iteration=True)

  return amend_negative_edges(
      ds.map(lambda i: (tf.gather(sources, i), tf.gather(targets, i))),
      negative_links_sampling=negative_links_sampling)


@dataclasses.dataclass(frozen=True)
class NegativeLinksSampling:
  """Instructs how to sample negative edges for link-prediction task."""

  # Function that can generate negative source node IDs. `int` argument
  # indicates the number of desired negative IDs.
  negative_sources_fn: Callable[[int], tf.Tensor]
  # Function that can generate negative target node IDs. `int` argument
  # indicates the number of desired negative IDs.
  negative_targets_fn: Callable[[int], tf.Tensor]

  # For each positive edge `(src, tgt)`, the `src` endpoint will be paired with
  # this many negative targets.
  num_negatives_per_source: int
  # ... and the `tgt` endpoint will be paired with this many negative sources.
  num_negatives_per_target: int

  def generate_sources(self) -> tf.Tensor:
    """Returns negative-sampled sources.

    Equivalent to `negative_sources_fn(num_negatives_per_target)`.
    """
    return self.negative_sources_fn(self.num_negatives_per_target)

  def generate_targets(self) -> tf.Tensor:
    """Returns negative-sampled targets.

    Equivalent to `negative_targets_fn(num_negatives_per_source)`.
    """
    return self.negative_targets_fn(self.num_negatives_per_source)

  @property
  def total_negatives(self) -> int:
    return self.num_negatives_per_source + self.num_negatives_per_target


def _provide_link_prediction_data_with_int_arithmetic_sampling(
    graph_data: datasets.LinkPredictionGraphData,
    source_sampling_spec: sampling_spec_pb2.SamplingSpec,
    target_sampling_spec: sampling_spec_pb2.SamplingSpec
    ) -> tuple[_LinkPredictionSamplingProvider,
               _LinkPredictionSamplingProvider]:
  """Dataset of subgraph samples for link prediction task."""
  adjacency = graph_data.edge_sets()[graph_data.target_edgeset].adjacency
  validation_edges = graph_data.edge_split().validation_edges
  negative_links_sampling = NegativeLinksSampling(
      negative_sources_fn=functools.partial(
          _uniform_random_int_negatives, output_dtype=adjacency.source.dtype,
          max_node_id=graph_data.num_source_nodes - 1),
      negative_targets_fn=functools.partial(
          _uniform_random_int_negatives, output_dtype=adjacency.target.dtype,
          max_node_id=graph_data.num_target_nodes - 1),
      num_negatives_per_source=2, num_negatives_per_target=2)

  train_edges_dataset_fn = functools.partial(
      _create_labeled_edges_dataset_from_eager_tensors,
      adjacency.source, adjacency.target, shuffle=True,
      negative_links_sampling=negative_links_sampling)

  validation_edges_dataset_fn = functools.partial(
      _create_labeled_edges_dataset_from_eager_tensors,
      validation_edges[0], validation_edges[1],
      negative_links_sampling=negative_links_sampling)

  # Sampling model.
  sampler = ia_sampler.GraphSampler(
      graph_data, sampling_mode=ia_sampler.EdgeSampling.WITH_REPLACEMENT)

  sampling_model = link_samplers.create_link_prediction_sampling_model(
      graph_data.graph_schema(),
      source_sampling_spec=source_sampling_spec,
      target_sampling_spec=target_sampling_spec,
      source_edge_sampler_factory=sampler.make_edge_sampler,
      target_edge_sampler_factory=sampler.make_edge_sampler,
      node_features_accessor_factory=(
          graph_data.create_node_features_lookup_factory()))

  return (
      _LinkPredictionSamplingProvider(
          adjacency.source.shape[0], train_edges_dataset_fn, sampling_model),
      _LinkPredictionSamplingProvider(
          validation_edges.shape[-1], validation_edges_dataset_fn,
          sampling_model))


class _LinkPredictionSamplingProvider(SizedDatasetProvider):
  """DatasetProvider for Link Prediction."""

  def __init__(
      self,
      cardinality: int,
      dataset_fn: Callable[[tf.distribute.InputContext], tf.data.Dataset],
      sampling_model: tf.keras.Model,
      sampling_batch_size: int = 100):
    """Constructs DatasetProvider for Link Prediction.

    Args:
      cardinality: Total number of examples in (one iteration of) dataset.
      dataset_fn: Callable returns dataset that with each element containing
        `(source ID, target ID, label)`. The first two will be batched to invoke
        `sampling_model`.
      sampling_model: tf.keras.Model to map items in `dataset` to `GraphTensor`.
      sampling_batch_size: Batch size to send to `sampling_model`.
    """
    super().__init__()
    self._cardinality = cardinality
    self._dataset_fn = dataset_fn
    self._sampling_model = sampling_model
    self._batch_size = sampling_batch_size

  @property
  def cardinality(self) -> int:
    return self._cardinality

  def sample_subgraphs(self, source_ids, target_ids, labels):
    return self._sampling_model((source_ids, target_ids)), labels

  def attach_label(self, graph, label):
    return graph.replace_features(
        node_sets={'_readout': {'label': tf.expand_dims(label, 0)}})

  def get_dataset(self, context: tf.distribute.InputContext) -> tf.data.Dataset:
    return (
        self._dataset_fn(context)
        .batch(self._batch_size)
        .map(self.sample_subgraphs)
        .unbatch()
        .map(self.attach_label))


def _uniform_random_int_negatives(
    num_negatives: int, *,
    output_dtype: tf.DType, max_node_id: int) -> tf.Tensor:
  return tf.random.uniform(
      [num_negatives], minval=0, maxval=max_node_id, dtype=output_dtype)


def _sample_negative_edges(
    source_node: tf.Tensor, target_node: tf.Tensor, *,
    negative_links_sampling: NegativeLinksSampling) -> tuple[
        tf.Tensor, tf.Tensor, tf.Tensor]:
  """Returns positive edge `{source,target}_node`, sampled negatives and labels.

  Args:
    source_node: scalar `tf.Tensor` containing ID of source node.
    target_node: scalar `tf.Tensor` containing ID of target node.
    negative_links_sampling: Contains functions that generate negative source
      and target node IDs.

  Returns:
    Three tensors `(sources, targets, labels)`, all are vectors of length
    `1 + N.num_negatives_per_source + N.num_negatives_per_target`, with
    `N == negative_links_sampling`. First entry corresponds to positive edge,
    with `sources[0] == source_node`, `targets[0] == target_node`, and
    `labels[0] == 1`. Remainder entries correspond to negative edges, with
    `labels[1:] == 0`, and `targets[i]` contains either `target_node` or a
    random negative returned from `N.negative_targets_fn`. In case, it was
    `target_node`, then `sources[i]` must contain random negative from
    `N.negative_sources_fn` else it must be `sources[i] == source_id`.
  """
  source_node = tf.ensure_shape(source_node, [])
  target_node = tf.ensure_shape(target_node, [])
  negative_sources = negative_links_sampling.generate_sources()
  negative_targets = negative_links_sampling.generate_targets()

  all_sources = tf.concat([
      tf.expand_dims(source_node, 0),
      tf.fill([negative_links_sampling.num_negatives_per_source], source_node),
      negative_sources,
  ], axis=0)

  all_targets = tf.concat([
      tf.expand_dims(target_node, 0),
      negative_targets,
      tf.fill([negative_links_sampling.num_negatives_per_target], target_node),
  ], axis=0)

  all_labels = tf.concat([
      tf.ones([1], dtype=tf.float32),
      tf.zeros([negative_links_sampling.total_negatives], dtype=tf.float32),
  ], 0)
  return all_sources, all_targets, all_labels


def amend_negative_edges(
    positive_edges_dataset: tf.data.Dataset,
    *,
    negative_links_sampling: NegativeLinksSampling,
    ) -> tf.data.Dataset:
  """Creates link-prediction dataset from `sources`, `targets`, and negatives.

  It must be that `source_nodes` and `target_nodes` are both vectors with equal
  shape, with each `(source_nodes[i], target_nodes[i])` indicating a positive
  edge.

  For each positive edge, a total of
  ```num_negatives_per_source + num_negatives_per_target```
  negative examples will be sampled (uniformly at random).

  Args:
    positive_edges_dataset: Dataset containing pairs of positive edge endpoints.
      Each element must be `(source_id, target_id)`.
    negative_links_sampling: Contains functions that generate negative source
      and target node IDs.

  Returns:
    Dataset with each element being 3-scalars: `(source_id, target_id, label)`.
    The `label` will be `1` if `source_id, target_id` are from
    `positive_edges_dataset`. Otherwise, `label == 0`, in which case, either
    `source_id` or `target_id` (but not both) are, respectively, determined by
    `negative_sources_fn` or `negative_targets_fn` (and the other comes from
    `positive_edges_dataset`).
  """
  return (
      positive_edges_dataset
      .map(functools.partial(
          _sample_negative_edges,
          negative_links_sampling=negative_links_sampling))
      .unbatch())


## <Helper functions for both tasks>


def _resolve_graph_data(
    graph_data_class: type[datasets.InMemoryGraphData],
    dataset_name: None|str = None,
    graph_data: None|datasets.InMemoryGraphData = None,
    ) -> datasets.InMemoryGraphData:
  """Returns `graph_data` or `get_in_memory_graph_data(dataset_name)`.

  Exactly one of `dataset_name` or `graph_data` must be given. If `graph_data`
  is given, then it must be an instance of `graph_data_class`. If `dataset_name`
  is given, then it must correspond to an `InMemoryGraphData` with type
  `graph_data_class`. If any if these conditions are violated, `ValueError` will
  be raised.

  Args:
    graph_data_class: Type of instance that will be returned.
    dataset_name: Mutually-exclusive with `graph_data`. If given, it must be
      name of `InMemoryGraphData` with type `graph_data_class`.
    graph_data: Mutually-exclusive with `dataset_name`. If given, it must be
      an instance of `graph_data_class`.

  Returns:
    `InMemoryGraphData` with type `graph_data_class`.
  """
  if int(dataset_name is None) + int(graph_data is None) != 1:
    raise ValueError('Expecting exactly one of `dataset_name` or `graph_data`')
  if dataset_name is not None:
    graph_data = datasets.get_in_memory_graph_data(dataset_name)
    if not isinstance(graph_data, graph_data_class):
      raise ValueError(
          f'dataset "{dataset_name}" corresponds to object of type '
          f'{type(graph_data)}. Expecting {graph_data_class}.')
  else:
    if not isinstance(graph_data, graph_data_class):
      raise ValueError(
          f'`graph_data` {graph_data} corresponds to object of type '
          f'{type(graph_data)}. Expecting {graph_data_class}.')
  assert graph_data is not None
  return graph_data


def _resolve_sampling_spec(
    graph_schema: graph_schema_pb2.GraphSchema,
    seed_node_set_name: graph_constants.NodeSetName,
    sampling: Sampling
    ) -> sampling_spec_pb2.SamplingSpec:
  """Returns `SamplingSpec` from `sampling.spec` or `make_sampling_spec_tree`.

  If `sampling.spec` is given, then it will be returned. Else, int list
  `steps = sampling.sample_sizes` will be read and returned `SamplingSpec`
  will sample `steps[0]` from every node set that `seed_node_set_name`
  connects to, and from every sampled node, `steps[1]` neighbors will be sampled
  for every node-set, etc, until depth `len(steps)`.

  Args:
    graph_schema: Only accessed if `sampling.spec is None`.
    seed_node_set_name: Only accessed if `sampling.spec is None`. If accessed,
      the result `SamplingSpec` will have its `seed_op.node_set_name` set to
      this.
    sampling: Must have either attribute `spec` or `sample_sizes` set.

  Returns:
    `sampling.spec` if not None. Else,
    `make_sampling_spec_tree(graph_schema, seed_node_set_name,
    sampling.sample_sizes)`. In which case, if `sampling.sample_sizes` is not
    set, it will default to `Sampling.SMALL_SAMPLING`.
  """
  if sampling is None:
    sampling = Sampling.SMALL_SAMPLING

  sampling_spec = sampling.spec
  if sampling_spec is None:  # if `sampling`` is constructed `from_sizes`.
    sampling_spec = sampling_spec_builder.make_sampling_spec_tree(
        graph_schema, seed_node_set_name,
        sample_sizes=list(sampling.sample_sizes))
  return sampling_spec


## <Temporary HACK to be migrated to datasets.py or subgraph_pipeline.py>


# Helper functions to obtain FeatureAccessor that keeps '#id' feature.
# NOTE: These can be copied to update `datasets.py`.
def create_node_features_lookup_factory(
    self_gd: datasets.InMemoryGraphData,
    keep_id_feature: bool = False) -> datasets.NodeFeaturesLookupFactory:
  return functools.partial(
      _node_features_lookup, node_sets=dict(self_gd.node_sets()), cache={},
      resource_prefix=self_gd.name, keep_id_feature=keep_id_feature)


class _Accessor(
    tf.keras.layers.Layer, datasets.interfaces.KeyToFeaturesAccessor):
  """Wraps `NodeSet` with `call` that can select features for node subsets."""

  def __init__(
      self, node_set: Any,
      resource_name: str, keep_id_feature: bool):
    super().__init__()
    self._node_set = node_set
    self._resource_name = resource_name
    self._keep_id_feature = keep_id_feature

  def call(self, keys: tf.RaggedTensor) -> datasets.interfaces.Features:
    """Gathers features corresponding to (tf.int) node keys."""
    return {feature_name: tf.gather(feature_value, keys)
            for feature_name, feature_value in self._node_set.features.items()
            if self._keep_id_feature or feature_name != '#id'}

  @property
  def resource_name(self) -> str:
    return self._resource_name


def _node_features_lookup(
    node_set_name: graph_constants.NodeSetName,
    *,
    node_sets: dict[graph_constants.NodeSetName, Any],
    cache: dict[graph_constants.NodeSetName, _Accessor],
    resource_prefix: str,
    keep_id_feature: bool) -> datasets.interfaces.KeyToFeaturesAccessor:
  """Returns `KeyToFeaturesAccessor` for a given node set."""
  if node_set_name in cache:
    return cache[node_set_name]
  cache[node_set_name] = _Accessor(
      node_sets[node_set_name], f'{resource_prefix}/nodes/{node_set_name}',
      keep_id_feature)
  return cache[node_set_name]
