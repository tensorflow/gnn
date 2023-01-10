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
import collections
import dataclasses
import functools
import itertools
import os
from typing import Callable, Optional, Sequence, Tuple, Union

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.runner import interfaces
from tensorflow_gnn.runner.utils import model as model_utils
from tensorflow_gnn.runner.utils import model_export
from tensorflow_gnn.runner.utils import parsing as parsing_utils

GraphTensor = tfgnn.GraphTensor
GraphTensorAndField = Tuple[GraphTensor, tfgnn.Field]
GraphTensorSpec = tfgnn.GraphTensorSpec
SizeConstraints = tfgnn.SizeConstraints

DatasetProvider = interfaces.DatasetProvider
GraphTensorPadding = interfaces.GraphTensorPadding
GraphTensorProcessorFn = interfaces.GraphTensorProcessorFn
ModelExporter = interfaces.ModelExporter
Task = interfaces.Task
Trainer = interfaces.Trainer


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
    super().__init__()
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


def _make_parsing_model(gtspec: GraphTensorSpec) -> tf.keras.Model:
  """Builds a `tf.keras.Model` that parses GraphTensors."""
  examples = tf.keras.Input(
      shape=(),
      dtype=tf.string,
      name="examples")  # Name seen in SignatureDef.
  parsed = tfgnn.keras.layers.ParseExample(gtspec)(examples)
  return tf.keras.Model(examples, parsed)


def _make_preprocessing_model(
    gtspec: GraphTensorSpec,
    preprocessors: Sequence[GraphTensorProcessorFn],
    task_preprocessor: GraphTensorProcessorFn,
    size_constraints: Optional[SizeConstraints] = None) -> tf.keras.Model:
  """Builds a `tf.keras.Model` that applies preprocessing.

  Args:
    gtspec: The `GraphTensorSpec` for input.
    preprocessors: The `GraphTensorProcessorFn` to apply.
    task_preprocessor: A `Task` preprocessor, used to apply any final objective
      specific processing.
    size_constraints: Any size constraints for padding.

  Returns:
    A `tf.keras.Model` with one, two or three outputs depending on the presence
    of `size_constraints` and the return values of `preprocessors.` Where
    outputs are (`GraphTensor`, `tfgnn.Field`, `tfgnn.Field`) as model
    input, label and mask (respectively).
  """
  gt = tf.keras.Input(type_spec=gtspec)
  x = gt.merge_batch_to_components()

  if size_constraints is not None:
    x, mask = tfgnn.keras.layers.PadToTotalSizes(size_constraints)(x)
  else:
    mask = None

  # Apply preprocessors to GraphTensor x: exactly one may split out a label y.
  y = None

  for fn in itertools.chain(preprocessors, (task_preprocessor,)):
    output = fn(x)

    if isinstance(output, collections.abc.Sequence):
      x, *ys = output
      if len(ys) == 1:
        yy = ys[0]
      else:
        raise ValueError(f"Expected (`GraphTensor`, `Field`) (got {output})")
      if y is not None and yy is not None:
        raise ValueError(f"Expected one label (got {y} and {yy})")
      else:
        y = yy
    else:
      x = output

    if not tfgnn.is_graph_tensor(x):
      raise ValueError(f"Expected `GraphTensor` (got {x})")

  if y is None and mask is None:
    return tf.keras.Model(gt, x)
  elif y is not None and mask is None:
    return tf.keras.Model(gt, (x, y))
  elif y is not None and mask is not None:
    return tf.keras.Model(gt, (x, y, mask))

  raise ValueError(f"Expected labels with a mask (got None and {mask})")


_map_over_dataset = functools.partial(
    tf.data.Dataset.map,
    deterministic=False,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)


def _per_replica_batch_size(global_batch_size: int, num_replicas: int) -> int:
  if global_batch_size % num_replicas != 0:
    raise ValueError(f"The `global_batch_size` {global_batch_size} is not "
                     f"divisible by `num_replicas_in_sync` {num_replicas}")
  return global_batch_size // num_replicas


def run(*,
        train_ds_provider: DatasetProvider,
        model_fn: Callable[[GraphTensorSpec], tf.keras.Model],
        optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
        trainer: Trainer,
        task: Task,
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
        tf_data_service_config: Optional[TFDataServiceConfig] = None):
  """Runs training (and validation) of a model on a task with the given data.

  This includes preprocessing the input data, adapting the model by appending
  the suitable head(s), and running training (and validation) with the
  requested distribution strategy.

  Args:
    train_ds_provider: A `DatasetProvider` for training. The `tf.data.Dataset`
      is not batched and contains scalar `GraphTensor` values.
    model_fn: Returns a `tf.keras.Model` for use in training and validation.
    optimizer_fn: Returns a `tf.keras.optimizers.Optimizer` for use in training.
    trainer: A `Trainer.`
    task: A `Task.`
    gtspec: A `GraphTensorSpec` matching the elements of `train` and `valid`
      datasets. If `train` or `valid` contain `tf.string` elements, this
      `GraphTensorSpec` is used for parsing; otherwise, `train` or `valid` are
      expected to contain `GraphTensor` elements whose relaxed spec matches
      `gtspec.`
    global_batch_size: The `tf.data.Dataset` global batch size for both training
      and validation.
    epochs: The epochs to train.
    drop_remainder: Whether to drop a `tf.data.Dataset` remainder at batching.
    export_dirs: Optional directories for exports (SavedModels); if unset,
      default behavior is `os.path.join(model_dir, "export").`
    model_exporters: Zero or more `ModelExporter` for exporting (SavedModels) to
      `export_dirs.` If unset, default behavior is `[KerasModelExporter()].`
    feature_processors: `GraphTensor` functions for feature processing:
      These may change some `GraphTensorSpec.` Functions are composed in
      order using `functools.reduce`; each function should accept a scalar
      `GraphTensor.`. All except one function should return a scalar
      `GraphTensor` with a single component; a single function, anywhere
      within `feature_processors` may return a tuple (`GraphTensor`,
      `tfgnn.Field`) where the `tfgnn.Field` is used for training labels.
    valid_ds_provider: A `DatasetProvider` for validation. The `tf.data.Dataset`
      is not batched and contains scalar `GraphTensor` values.
    train_padding: `GraphTensor` padding for training. Required if training on
      TPU.
    valid_padding: `GraphTensor` padding for validation. Required if training on
      TPU.
    tf_data_service_config: tf.data service speeds-up tf.data input pipeline
      runtime reducing input bottlenecks for model training. Particularly for
      training on accelerators consider enabling it. For more info please see:
      https://www.tensorflow.org/api_docs/python/tf/data/experimental/service.
  """
  validate = valid_ds_provider is not None

  if isinstance(trainer.strategy, tf.distribute.TPUStrategy):
    if train_padding is None:
      raise ValueError("`TPUStrategy` requires a `train_padding` (got None)")
    elif validate and valid_padding is None:
      raise ValueError("`TPUStrategy` requires a `valid_padding` (got None)")

  if not validate and valid_padding is not None:
    raise ValueError("`valid_padding` specified without a validation dataset")

  preprocess_model = _make_preprocessing_model(
      gtspec,
      feature_processors or (),
      task.preprocess)

  def apply_fn(ds,
               *,
               filter_fn: Optional[Callable[..., bool]] = None,
               size_constraints: Optional[SizeConstraints] = None):
    ds = parsing_utils.maybe_parse_graph_tensor_dataset(ds, gtspec)
    if filter_fn is not None:
      ds = ds.filter(filter_fn)
    if size_constraints is not None:
      padding_preprocess_model = _make_preprocessing_model(
          gtspec,
          feature_processors or (),
          task.preprocess,
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
    if isinstance(preprocess_model.output, collections.abc.Sequence):
      x, *_ = preprocess_model.output
    else:
      x = preprocess_model.output
    m = task.adapt(model_fn(x.spec))
    optimizer = optimizer_fn()
    if train_padding is None:
      m.compile(optimizer, loss=task.losses(), metrics=task.metrics())
    else:
      m.compile(optimizer, loss=task.losses(), weighted_metrics=task.metrics())
    return m

  model = trainer.train(
      adapted_model_fn,
      train_ds_provider,
      epochs=epochs,
      valid_ds_provider=valid_ds_provider)

  if model_exporters is None:
    model_exporters = [model_export.KerasModelExporter()]

  parsing_and_preprocess_model = model_utils.chain_first_output(
      _make_parsing_model(gtspec),
      preprocess_model, first_output_only=False)

  for export_dir in export_dirs or [os.path.join(trainer.model_dir, "export")]:
    for exporter in model_exporters:
      exporter.save(parsing_and_preprocess_model, model, export_dir)
