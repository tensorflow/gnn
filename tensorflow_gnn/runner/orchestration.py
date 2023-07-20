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
"""The runner entry point."""
from __future__ import annotations

import collections
import dataclasses
import functools
import operator
import os
from typing import Callable, Mapping, Optional, Sequence, Tuple, TypeVar, Union

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.runner import interfaces
from tensorflow_gnn.runner.utils import model_export
from tensorflow_gnn.runner.utils import parsing as parsing_utils

T = TypeVar("T")
OneOrMappingOf = Union[T, Mapping[str, T]]

Field = tfgnn.Field
GraphTensor = tfgnn.GraphTensor
# TODO(b/274672364): make this tuple[...] in Python 3.9 style
# when we drop py38 support.
GraphTensorAndField = Tuple[GraphTensor, Field]
GraphTensorSpec = tfgnn.GraphTensorSpec
TaskPreprocessFn = Callable[
    [GraphTensor],
    Tuple[Union[GraphTensor, Sequence[GraphTensor]], Field]
]
SizeConstraints = tfgnn.SizeConstraints

DatasetProvider = interfaces.DatasetProvider
GraphTensorPadding = interfaces.GraphTensorPadding
GraphTensorProcessorFn = interfaces.GraphTensorProcessorFn
ModelExporter = interfaces.ModelExporter
Task = interfaces.Task
Trainer = interfaces.Trainer
RunResult = interfaces.RunResult

_BASE_MODEL_TAG = "UNDERSCORE_TFGNN_RUNNER_BASE_MODEL"
_TPU_DEFAULT_STEPS_PER_EXECUTION = 100


@dataclasses.dataclass
class TFDataServiceConfig:
  """Provides tf.data service related configuration options.

  tf.data service has data flexible visitation guarantees, its impact over your
  training pipelines will be empirical. Check out the tf.data service internals
  and operation details from
  https://www.tensorflow.org/api_docs/python/tf/data/experimental/service.
  """
  tf_data_service_address: str
  tf_data_service_job_name: str
  tf_data_service_mode: Union[str, tf.data.experimental.service.ShardingPolicy]


class _WrappedDatasetProvider(DatasetProvider):
  """Wraps a `DatasetProvider` with batching and processing."""

  def __init__(self,
               apply_fn: Callable[[tf.data.Dataset], tf.data.Dataset],
               delegate: DatasetProvider,
               drop_remainder: bool,
               global_batch_size: int,
               tf_data_service_config: Optional[TFDataServiceConfig] = None):
    self._apply_fn = apply_fn
    self._delegate = delegate
    self._drop_remainder = drop_remainder
    self._global_batch_size = global_batch_size
    self._tf_data_service_config = tf_data_service_config

  def get_dataset(self, context: tf.distribute.InputContext) -> tf.data.Dataset:
    """Gets a batched dataset with `apply_fn` applied."""
    if self._tf_data_service_config:
      # Prevent any sharding for tf.data service.
      context = tf.distribute.InputContext(
          num_input_pipelines=1,
          input_pipeline_id=0,
          num_replicas_in_sync=context.num_replicas_in_sync)
    ds = self._delegate.get_dataset(context)
    ds = ds.batch(
        context.get_per_replica_batch_size(self._global_batch_size),
        drop_remainder=self._drop_remainder)
    if self._tf_data_service_config:
      if (self._tf_data_service_config.tf_data_service_address is None or
          self._tf_data_service_config.tf_data_service_mode is None or
          self._tf_data_service_config.tf_data_service_job_name is None):
        raise ValueError(
            "Specify tf_data_service_address, tf_data_service_mode and"
            "tf_data_service_job_name.")
      ds = ds.apply(
          tf.data.experimental.service.distribute(
              processing_mode=self._tf_data_service_config.tf_data_service_mode,
              service=self._tf_data_service_config.tf_data_service_address,
              job_name=self._tf_data_service_config.tf_data_service_job_name))
    return ds.apply(self._apply_fn)


def _make_padding_preprocessing_model(
    gtspec: GraphTensorSpec,
    preprocessing_model: tf.keras.Model,
    size_constraints: SizeConstraints) -> tf.keras.Model:
  """Builds a `tf.keras.Model` that applies padding and preprocessing.

  Args:
    gtspec: The `GraphTensorSpec` for input.
    preprocessing_model: The preprocessing model.
    size_constraints: Size constraints for padding.

  Returns:
    A `tf.keras.Model` with three outputs where outputs are (`GraphTensor`,
    `Field`, `Field`) as model input, label and mask, respectively.
  """
  inputs = tf.keras.Input(type_spec=gtspec)
  gt, mask = tfgnn.keras.layers.PadToTotalSizes(size_constraints)(inputs)
  gt, labels = preprocessing_model(gt)
  return tf.keras.Model(inputs, (gt, labels, mask))


def _make_parsing_preprocessing_model(
    gtspec: GraphTensorSpec,
    preprocessing_model: tf.keras.Model) -> tf.keras.Model:
  """Builds a `tf.keras.Model` that parses GraphTensors."""
  examples = tf.keras.Input(
      shape=(),
      dtype=tf.string,
      name="examples")  # Name seen in SignatureDef.
  parsed = tfgnn.keras.layers.ParseExample(gtspec)(examples)
  parsed = parsed.merge_batch_to_components()
  return tf.keras.Model(examples, preprocessing_model(parsed))


def _make_preprocessing_model(
    gtspec: GraphTensorSpec,
    processors: Sequence[GraphTensorProcessorFn],
    task_processor_fn: OneOrMappingOf[TaskPreprocessFn]
) -> tuple[tf.keras.Model, OneOrMappingOf[Union[int, Sequence[int]]]]:
  """Builds a `tf.keras.Model` that applies preprocessing.

  This function returns a `tf.keras.Model` for preprocessing and a
  `OneOrMappingOf` integer indices into that model's output.

  The preprocessing model accepts a scalar `GraphTensor` batch (i.e.: it has
  been merged to components) that satisfies `gtspec`. `processors` are applied
  in order before any `task_processor_fn` are applied in parallel, producing,
  with the same structure as `task_processor_fn`: training labels and one or
  more base GNN inputs. (In the simplest case: a single base GNN input and a
  single training label.) The base GNN inputs are deduplicated by Tensor object
  identity (many tasks may use the same imput: an unchanged one) and arranged
  in a flat `Sequence`.

  The preprocessing model returns a `Tuple[Sequence[GraphTensor],
  OneOrMappingOf[Field]]` with the sequence of deduplicated inputs for the base
  GNN and the `OneOrMappingOf` training labels.

  The `OneOrMappingOf` integer indices returned by the function contains indices
  into the preprocessing model `Sequence[GraphTensor]` output and takes the same
  structure as `task_processor_fn`: For each `task_processor_fn`, the
  output/input map contains the single index or sequence of indices at which
  the output(s) of the `Task.preprocess(...)` method are found in preprocessing
  model `Sequence[GraphTensor]` output.

  Example for a single task with a single model input:

    ```python
    task = RootNodeBinaryClassification(...)
    preprocessing_model, oimap = _make_preprocessing_model(...)
    # `task.preprocess(...)` returns (along with labels) a single unmutated
    # `GraphTensor`, s.t. here, `oimap` is a scalar `0`.
    .
    .
    .
    # Later, prediction heads are created using preprocessing model output and
    # the base GNN...
    xs, *_ = preprocessing_model(...)
    gnn = model_fn(...)
    _ = task.predict(*[gnn(xs[i]) for i in tf.nest.flatten(oimap)])
    ```

  Example for multiple tasks with perturbed model inputs:

    ```python
    tasks = {
        "classification": RootNodeBinaryClassification(...),
        "dgi": tfgnn.models.contrastive_losses.DeepGraphInfomaxTask(...)
    }
    preprocessing_model, oimap = _make_preprocessing_model(...)
    # `tasks["classification"].preprocess(...)` returns (along with labels) a
    # single unmutated `GraphTensor`. But, `tasks["dgi"].preprocess(...)`
    # returns (along with labels) two `GraphTensor`s: one unmutated and one
    # corrupted. The unmutated `GraphTensor`s are deduplicated, s.t. here,
    # `oimap` is a mapping of {"classification": 0, "dgi": (0, 1)}--the
    # unmutated `GraphTensor` is shared between both `tasks`.
    .
    .
    .
    # Later, prediction heads are created using preprocessing model output and
    # the base GNN...
    xs, *_ = preprocessing_model(...)
    gnn = model_fn(...)
    inputs = tuple(gnn(x) for x in xs)
    for name, task in tasks.items():
      _ = task.predict(*[inputs(i) for i in tf.nest.flatten(oimap[name])])
    ```

  Args:
    gtspec: The `GraphTensorSpec` for input.
    processors: The sequence of `GraphTensorProcessorFn`s to apply.
    task_processor_fn: A `Task` preprocessor, used to apply any final objective
      specific processing and label generation. May be a single
      `TaskPreprocessFn` or, for multi-Task, a `Mapping[str, TaskPreprocessFn]`.

  Returns:
    A preprocessing `tf.keras.Model` and an output/input mapping.
  """
  inputs = tf.keras.Input(type_spec=gtspec)

  gt = functools.reduce(lambda acc, fn: fn(acc), processors, inputs)
  is_multi_task = isinstance(task_processor_fn, collections.abc.Mapping)

  if is_multi_task:
    gt_processed_and_labels = {k: p(gt) for k, p in task_processor_fn.items()}
    gt_processed = {k: gt for k, (gt, lbl) in gt_processed_and_labels.items()}
    labels = {k: lbl for k, (gt, lbl) in gt_processed_and_labels.items()}
  else:
    gt_processed, labels = task_processor_fn(gt)

  specs = [x.spec for x in tf.nest.flatten(gt_processed)]

  if is_multi_task and any(gt.spec != s for s in specs):
    raise ValueError(
        "All `GraphTensorSpec` expected to match shared preprocessing (got"
        f" gt.spec={gt.spec} and specs={specs}). For multi-Task, `Task`s should"
        " read out labels from the auxiliary '_readout' node set and not mutate"
        " the `GraphTensorSpec` (see, e.g.:"
        " `RootNodeBinaryClassification(label_feature_name=...)`)."
    )
  elif any(specs[0] != s for s in specs[1:]):
    raise ValueError(
        f"All `GraphTensorSpec` expected to match (got specs={specs})"
    )

  # Deduplicate preprocessing outputs by identity of Tensor objects. This way,
  # `Task`s can share preprocessing results (e.g., the unmodified input).

  # Tensors are not hashable, instead of making a `set` of Tensors: rely on
  # `dict` key uniqueness.
  ref_to_output = dict((x.ref(), x) for x in tf.nest.flatten(gt_processed))
  # Index the unique outputs as 0, 1, 2, ...
  outputs = tuple(ref_to_output.values())
  ref_to_index = dict((x.ref(), i) for i, x in enumerate(outputs))
  #  Strip features on aux graph pieces (notably: labels).
  woauxfeatures = _without_aux_graph_piece_features()
  outputs = tuple(woauxfeatures(x) for x in outputs)
  # For each task, report the output indices of its `Task.preprocess(...)`
  # results.
  oimap = tf.nest.map_structure(lambda x: ref_to_index[x.ref()], gt_processed)

  return tf.keras.Model(inputs, (outputs, labels)), oimap


_map_over_dataset = functools.partial(
    tf.data.Dataset.map,
    deterministic=False,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)


def _per_replica_batch_size(global_batch_size: int, num_replicas: int) -> int:
  if global_batch_size % num_replicas != 0:
    raise ValueError(f"The `global_batch_size` {global_batch_size} is not "
                     f"divisible by `num_replicas_in_sync` {num_replicas}")
  return global_batch_size // num_replicas


def _without_aux_graph_piece_features() -> tf.keras.layers.Layer:
  def fn(inputs, *, node_set_name=None, edge_set_name=None):
    del inputs
    if node_set_name is not None and tfgnn.get_aux_type_prefix(node_set_name):
      return dict()  # Drop features.
    if edge_set_name is not None and tfgnn.get_aux_type_prefix(edge_set_name):
      return dict()  # Drop features.
    return None  # Passthru.
  return tfgnn.keras.layers.MapFeatures(
      node_sets_fn=fn,
      edge_sets_fn=fn,
      name="without_aux_graph_piece_features")


def run(*,
        train_ds_provider: DatasetProvider,
        model_fn: Callable[[GraphTensorSpec], tf.keras.Model],
        optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
        trainer: Trainer,
        task: OneOrMappingOf[Task],
        loss_weights: Optional[Mapping[str, float]] = None,
        gtspec: GraphTensorSpec,
        global_batch_size: int,
        epochs: int = 1,
        drop_remainder: bool = False,
        export_dirs: Optional[Sequence[str]] = None,
        model_exporters: Optional[Sequence[ModelExporter]] = None,
        feature_processors: Optional[Sequence[GraphTensorProcessorFn]] = None,
        valid_ds_provider: Optional[DatasetProvider] = None,
        train_padding: Optional[GraphTensorPadding] = None,
        valid_padding: Optional[GraphTensorPadding] = None,
        tf_data_service_config: Optional[TFDataServiceConfig] = None,
        steps_per_execution: Optional[int] = None):
  """Runs training (and validation) of a model on task(s) with the given data.

  This includes preprocessing the input data, appending any suitable head(s),
  and running training (and validation) with the requested distribution
  strategy.

  The input data is processed in multiple stages, starting from the contents
  of the datasets provided by `train_ds_provider` and `valid_ds_provider`:

   1. Input examples are batched.
   2. If necessary, input batches are parsed as `GraphTensor` values and merged
      into components (see: `GraphTensor.merge_batch_to_components`).
   3. If set, `train_padding` and `valid_padding`, resp., are applied.
   4. The given `feature_processors` are applied in order for all non-trainable
      feature transformations on CPU (as part of `tf.data.Dataset.map(...)`).
   5. The `Task.preprocess(...)` method is applied to extract training targets
      (for supervised learning, that means: labels) and optionally transform
      the value of the preprocessed `GraphTensor` into a model input (or
      multiple model inputs for tasks like self-supervised contrastive losses).
   6. If the resulting `GraphTensor`s have any auxillary pieces (as indicated by
      `tfgnn.get_aux_type_prefix(...)`): all features (typically: labels) are
      removed from those graph pieces.

  The base GNN (as built by `model_fn`) is run on all results from step (6).
  `Task.predict(...)` is called on the model outputs that correspond to the
  one or more graphs requested in step (5) by `Task.preprocess(...)`.

  Trainable transformations of inputs (notably lookups in trainable embedding
  tables) are required to happen inside `model_fn`.

  For supervised learning, training labels enter the pipeline as features
  on the `GraphTensor` that undergo the `feature_processors` (shared by all
  `Task`s) and are read out of the `GraphTensor` by `Task.preprocess(...)`.

  Users are strongly encouraged to take one of the following two approaches
  to prevent the leakage of label information into the training:

    * Store labels on the auxiliary `"_readout"` node set and let
      `Task.preprocess(...)` read them from there. (For library-supplied
      `Task`s, that means initializing with `label_feature_name="..."`.) If
      that is not already true for the input datasets, the label feature can
      be moved there by one of the `feature_processors`, using
      `tfgnn.structured_readout_into_feature(...)` or a similar helper function.
    * For single-Task training only: Let `Task.preprocess()` return modified
      `GraphTensor`s that no longer contain the separately returned labels.
      (Library-supplied Tasks delegate this to the `label_fn="..."` passed
      in initialization.)

  Args:
    train_ds_provider: A `DatasetProvider` for training. The `tf.data.Dataset`
      is not batched and contains scalar `GraphTensor` values conforming to
      `gtspec`, possibly serialized as a `tf.train.Example` proto.
    model_fn: Returns the base GNN `tf.keras.Model` for use in training and
      validation.
    optimizer_fn: Returns a `tf.keras.optimizers.Optimizer` for use in training.
    trainer: A `Trainer`.
    task: A `Task` for single-Task training or a `Mapping[str, Task]` for
      multi-Task training. In multi-Task training, `Task.preprocess(...)`
      must return `GraphTensors` with the same spec as its inputs, only the
      values may change (so that there remains a single spec for `model_fn`).
    loss_weights: An optional `Mapping[str, float]` for multi-Task training. If
      given, this structure must match (with `tf.nest.assert_same_structure`)
      the structure of `task`. The mapping contains, for each `task`, a scalar
      coefficient to weight the loss contributions of that `task`.
    gtspec: A `GraphTensorSpec` matching the elements of `train` and `valid`
      datasets. If `train` or `valid` contain `tf.string` elements, this
      `GraphTensorSpec` is used for parsing; otherwise, `train` or `valid` are
      expected to contain `GraphTensor` elements whose relaxed spec matches
      `gtspec`.
    global_batch_size: The `tf.data.Dataset` global batch size for both training
      and validation.
    epochs: The epochs to train.
    drop_remainder: Whether to drop a `tf.data.Dataset` remainder at batching.
    export_dirs: Optional directories for exports (SavedModels); if unset,
      default behavior is `os.path.join(model_dir, "export")`.
    model_exporters: Zero or more `ModelExporter` for exporting (SavedModels) to
      `export_dirs`. If unset, default behavior is `[KerasModelExporter()]`.
    feature_processors: A sequence of callables for feature processing with the
      Keras functional API. Each callable must accept and return a symbolic
      scalar `GraphTensor`. The callables are composed in order and may change
      the `GraphTensorSpec` (e.g., add/remove features). The resulting Keras
      model is executed on CPU as part of a `tf.data.Dataset.map` operation.
    valid_ds_provider: A `DatasetProvider` for validation. The `tf.data.Dataset`
      is not batched and contains scalar `GraphTensor` values conforming to
      `gtspec`, possibly serialized as a `tf.train.Example` proto.
    train_padding: `GraphTensor` padding for training. Required if training on
      TPU.
    valid_padding: `GraphTensor` padding for validation. Required if training on
      TPU.
    tf_data_service_config: tf.data service speeds-up tf.data input pipeline
      runtime reducing input bottlenecks for model training. Particularly for
      training on accelerators consider enabling it. For more info please see:
      https://www.tensorflow.org/api_docs/python/tf/data/experimental/service.
    steps_per_execution: The number of batches to run during each training
      iteration. If not set, for TPU strategy default to 100 and to `None`
      otherwise.

  Returns:
    A `RunResult` object containing models and information about this run.
  """
  validate = valid_ds_provider is not None

  if isinstance(trainer.strategy, tf.distribute.TPUStrategy):
    if train_padding is None:
      raise ValueError("`TPUStrategy` requires a `train_padding` (got None)")
    elif validate and valid_padding is None:
      raise ValueError("`TPUStrategy` requires a `valid_padding` (got None)")

    steps_per_execution = (
        steps_per_execution or _TPU_DEFAULT_STEPS_PER_EXECUTION
    )

  if not validate and valid_padding is not None:
    raise ValueError("`valid_padding` specified without a validation dataset")

  if loss_weights is not None:
    tf.nest.assert_same_structure(task, loss_weights)

  preprocess_model, oimap = _make_preprocessing_model(
      gtspec,
      feature_processors or tuple(),
      tf.nest.map_structure(operator.attrgetter("preprocess"), task))

  def apply_fn(
      ds,
      *,
      filter_fn: Optional[Callable[..., bool]] = None,
      size_constraints: Optional[SizeConstraints] = None):
    ds = parsing_utils.maybe_parse_graph_tensor_dataset(ds, gtspec)
    ds = _map_over_dataset(ds, tfgnn.GraphTensor.merge_batch_to_components)
    if filter_fn is not None:
      ds = ds.filter(filter_fn)
    if size_constraints is not None:
      padding_preprocess_model = _make_padding_preprocessing_model(
          gtspec,
          preprocess_model,
          size_constraints)
      ds = _map_over_dataset(ds, padding_preprocess_model)
    else:
      ds = _map_over_dataset(ds, preprocess_model)
    return ds

  target_batch_size = _per_replica_batch_size(
      global_batch_size,
      trainer.strategy.num_replicas_in_sync)

  # The following code computes the size_constraints for padding (on the host
  # that runs this Python code) before the actual training or validation
  # datasets are created (possibly replicated, possibly distributed to
  # one or more worker jobs).
  if train_padding is not None:
    size_constraints = train_padding.get_size_constraints(target_batch_size)
    train_apply_fn = functools.partial(
        apply_fn,
        filter_fn=train_padding.get_filter_fn(size_constraints),
        size_constraints=size_constraints)
  else:
    train_apply_fn = apply_fn

  if validate and valid_padding is not None:
    size_constraints = valid_padding.get_size_constraints(target_batch_size)
    valid_apply_fn = functools.partial(
        apply_fn,
        filter_fn=valid_padding.get_filter_fn(size_constraints),
        size_constraints=size_constraints)
  elif validate:
    valid_apply_fn = apply_fn

  train_ds_provider = _WrappedDatasetProvider(
      train_apply_fn,
      train_ds_provider,
      drop_remainder,
      global_batch_size,
      tf_data_service_config)

  if validate:
    valid_ds_provider = _WrappedDatasetProvider(
        valid_apply_fn,
        valid_ds_provider,
        drop_remainder,
        global_batch_size)

  def adapted_model_fn():
    xs, *_ = preprocess_model.output
    specs = [x.spec for x in xs]
    inputs = [tf.keras.Input(type_spec=spec) for spec in specs]
    # All specs are the same (asserted in `_make_preprocessing_model`).
    gnn = tf.keras.Sequential((model_fn(specs[0]),), name=_BASE_MODEL_TAG)
    outputs = [gnn(i) for i in inputs]

    if isinstance(task, collections.abc.Mapping):
      outputs = {
          k: v.predict(*[outputs[i] for i in tf.nest.flatten(oimap[k])])
          for k, v in task.items()
      }
    else:
      outputs = task.predict(*[outputs[i] for i in tf.nest.flatten(oimap)])

    model = tf.keras.Model(inputs, outputs)

    losses = tf.nest.map_structure(operator.methodcaller("losses"), task)
    metrics = tf.nest.map_structure(operator.methodcaller("metrics"), task)

    if train_padding is None:
      model.compile(
          optimizer_fn(),
          loss=losses,
          loss_weights=loss_weights,
          metrics=metrics,
          steps_per_execution=steps_per_execution)
    else:
      model.compile(
          optimizer_fn(),
          loss=losses,
          loss_weights=loss_weights,
          weighted_metrics=metrics,
          steps_per_execution=steps_per_execution)

    return model

  model = trainer.train(
      adapted_model_fn,
      train_ds_provider,
      epochs=epochs,
      valid_ds_provider=valid_ds_provider)

  if model_exporters is None:
    model_exporters = (model_export.KerasModelExporter(),)

  parsing_and_preprocess_model = _make_parsing_preprocessing_model(
      gtspec,
      preprocess_model)

  run_result = RunResult(
      preprocess_model=parsing_and_preprocess_model,
      base_model=model.get_layer(_BASE_MODEL_TAG),
      trained_model=model)

  for export_dir in export_dirs or (os.path.join(trainer.model_dir, "export"),):
    for exporter in model_exporters:
      exporter.save(run_result, export_dir)

  return run_result
