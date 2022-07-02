"""The runner entry point."""
import functools
import itertools
import os
import sys
from typing import Callable, Optional, Sequence, Tuple, Union
import uuid

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.runner.utils import model_export

# pylint:disable=g-import-not-at-top
if sys.version_info >= (3, 8):
  from typing import Protocol
  from typing import runtime_checkable
else:
  from typing_extensions import Protocol
  from typing_extensions import runtime_checkable
# pylint:enable=g-import-not-at-top


@runtime_checkable
class DatasetProvider(Protocol):

  def get_dataset(self, context: tf.distribute.InputContext) -> tf.data.Dataset:
    """Get a `tf.data.Dataset` by `context` per replica."""
    raise NotImplementedError()


class _WrappedDatasetProvider:
  """Wraps a `DatasetProvider` with batching and processing."""

  def __init__(self,
               apply_fn: Callable[[tf.data.Dataset], tf.data.Dataset],
               delegate: DatasetProvider,
               drop_remainder: bool,
               global_batch_size: int,
               tf_data_service_address: Optional[str] = None):
    self._apply_fn = apply_fn
    self._delegate = delegate
    self._drop_remainder = drop_remainder
    self._global_batch_size = global_batch_size
    self._tf_data_service_address = tf_data_service_address
    # TODO(b/196880966): The tf.data service name should be unique per dataset
    # (and not unique per TensorFlow worker).
    self._tf_data_service_job_name = uuid.uuid4().hex

  def get_dataset(self, context: tf.distribute.InputContext) -> tf.data.Dataset:
    """Get a `tf.data.Dataset`."""
    if self._tf_data_service_address:
      # Prevent any sharding for tf.data service.
      context = tf.distribute.InputContext(
          num_input_pipelines=1,
          input_pipeline_id=0,
          num_replicas_in_sync=context.num_replicas_in_sync)
    ds = self._delegate.get_dataset(context)
    ds = ds.batch(
        context.get_per_replica_batch_size(self._global_batch_size),
        drop_remainder=self._drop_remainder)
    if self._tf_data_service_address:
      ds = ds.apply(
          tf.data.experimental.service.distribute(
              processing_mode="parallel_epochs",
              service=self._tf_data_service_address,
              job_name=self._tf_data_service_job_name))
    return ds.apply(self._apply_fn)


@runtime_checkable
class Task(Protocol):
  """Collects the ancillary, supporting pieces to train a Keras model.

  `Task`s are applied and used to compile a `tf.keras.Model` in the scope
  of a training invocation: they are subject to the executing context
  of the `Trainer` and should, when needed, override it (e.g., a global
  policy, like `tf.keras.mixed_precision.global_policy()` and its implications
  over logit and activation layers).
  """

  def adapt(self, model: tf.keras.Model) -> tf.keras.Model:
    """Adapt a model to a task by appending arbitrary head(s)."""
    raise NotImplementedError()

  def preprocessors(self) -> Sequence[Callable[..., tf.data.Dataset]]:
    """Preprocess a `tf.data.Dataset`: e.g., extract labels."""
    raise NotImplementedError()

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    """Arbitrary losses matching any head(s)."""
    raise NotImplementedError()

  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    """Arbitrary task specific metrics."""
    raise NotImplementedError()


@runtime_checkable
class Trainer(Protocol):
  """A class for training and validation."""

  @property
  def model_dir(self) -> str:
    raise NotImplementedError()

  @property
  def strategy(self) -> tf.distribute.Strategy:
    raise NotImplementedError()

  def train(
      self,
      model_fn: Callable[[], tf.keras.Model],
      train_ds_provider: DatasetProvider,
      *,
      epochs: int = 1,
      valid_ds_provider: Optional[DatasetProvider] = None) -> tf.keras.Model:
    """Trains a `tf.keras.Model` with optional validation.

    Args:
      model_fn: Returns a `tf.keras.Model` for use in training and validation.
      train_ds_provider: A `DatasetProvider` for training. The items of the
        `tf.data.Dataset` are pairs `(graph_tensor, label)` that represent one
        batch of per-replica training inputs after
        `GraphTensor.merge_batch_to_components()` has been applied.
      epochs: The epochs to train.
      valid_ds_provider: A `DatasetProvider` for validation. The items of the
        `tf.data.Dataset` are pairs `(graph_tensor, label)` that represent one
        batch of per-replica training inputs after
        `GraphTensor.merge_batch_to_components()` has been applied.

    Returns:
      A trained `tf.keras.Model.`
    """
    raise NotImplementedError()


GraphTensorAndField = Tuple[tfgnn.GraphTensor, tfgnn.Field]


@runtime_checkable
class GraphTensorProcessorFn(Protocol):
  """A class for `GraphTensor` processing."""

  def __call__(
      self,
      gt: tfgnn.GraphTensor) -> Union[tfgnn.GraphTensor, GraphTensorAndField]:
    """Processes a `GraphTensor` with optional `Field` extraction.

    Args:
      gt: A `GraphTensor` for processing.

    Returns:
      A processed `GraphTensor` or a processed `GraphTensor` and `Field.`
    """
    raise NotImplementedError()


@runtime_checkable
class ModelExporter(Protocol):
  """Saves a Keras model."""

  def save(self, model: tf.keras.Model, export_dir: str):
    """Saves a Keras model.

    All persistence decisions are left to the implementation: e.g., a Keras
    model with full API or a simple `tf.train.Checkpoint` may be saved.

    Args:
      model: A `tf.keras.Model` to save.
      export_dir: A destination directory for the model.
    """
    raise NotImplementedError()


def make_preprocess_model(
    gtspec: tfgnn.GraphTensorSpec,
    preprocessors: Sequence[GraphTensorProcessorFn]) -> tf.keras.Model:
  """Builds a `tf.keras.Model` that applies preprocessing.

  Args:
    gtspec: The `tfgnn.GraphTensorSpec` for input.
    preprocessors: The `GraphTensorProcessorFn` to apply.

  Returns:
    A `tf.keras.Model.`
  """
  examples = tf.keras.layers.Input(
      shape=(),
      dtype=tf.string,
      name="examples")  # Name seen in SignatureDef.
  x = tfgnn.keras.layers.ParseExample(gtspec)(examples)

  x = x.merge_batch_to_components()
  y = None

  for fn in preprocessors:
    output = fn(x)
    if isinstance(output, tuple):
      if len(output) == 2 and tfgnn.is_graph_tensor(output[0]) and y is None:
        x, y = output
      elif len(output) == 2 and y is not None:
        raise ValueError(f"Received more than one label: {y} and {output[1]}")
      else:
        msg = f"Expected (`tfgnn.GraphTensor`, `tf.Tensor`), received: {output}"
        raise ValueError(msg)
    elif tfgnn.is_graph_tensor(output):
      x = output
    else:
      raise ValueError(f"Expected `tfgnn.GraphTensor`, received: {output}")

  return tf.keras.Model(examples, x if y is None else (x, y))


def run(*,
        train_ds_provider: DatasetProvider,
        model_fn: Callable[[tfgnn.GraphTensorSpec], tf.keras.Model],
        optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
        trainer: Trainer,
        task: Task,
        gtspec: tfgnn.GraphTensorSpec,
        global_batch_size: int,
        epochs: int = 1,
        drop_remainder: bool = True,
        export_dirs: Optional[Sequence[str]] = None,
        model_exporters: Optional[Sequence[ModelExporter]] = None,
        feature_processors: Optional[Sequence[GraphTensorProcessorFn]] = None,
        valid_ds_provider: Optional[DatasetProvider] = None,
        tf_data_service_address: Optional[str] = None):
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
    gtspec: A `tfgnn.GraphTensorSpec` for parsing the GraphTensors in the
      `train` and `valid` datasets.
    global_batch_size: The `tf.data.Dataset` global batch size for both training
      and validation.
    epochs: The epochs to train.
    drop_remainder: Whether to drop a `tf.data.Dataset` remainder at batching.
    export_dirs: Optional directories for exports (SavedModels); if unset,
      default behavior is `os.path.join(model_dir, "export").`
    model_exporters: Zero or more `ModelExporter` for exporting (SavedModels) to
      `export_dirs.` If unset, default behavior is `[KerasModelExporter(...)].`
    feature_processors: `tfgnn.GraphTensor` functions for feature processing:
      These may change some `tfgnn.GraphTensorSpec.` Functions are composed in
      order using `functools.reduce`; each function should accept a scalar
      `tfgnn.GraphTensor.`. All except one function should return a scalar
      `tfgnn.GraphTensor` with a single component; a single function, anywhere
      within `feature_processors` may return a tuple (`tfgnn.GraphTensor`,
      `tfgnn.Field`) where the `tfgnn.Field` is used for training labels.
    valid_ds_provider: A `DatasetProvider` for validation. The `tf.data.Dataset`
      is not batched and contains scalar `GraphTensor` values.
    tf_data_service_address: Address for tf.data service. tf.data service is
      run with `processing_mode="parallel_epochs".` Parallel epochs corresponds
      to the `ShardingPolicy.OFF` mode and it is important for users to shuffle
      any filenames in this case, see: go/tf-data-service#processing-modes.
      Note, `parallel_epochs` differs greatly in its data visitation promises
      and notion of epochs from `SimpleDatasetProvider` and
      `SimpleSampleDatasetsProvider.` These providers, without tf.data service,
      shard by filenames (corresponding to the tf.data service processing mode
      of `ShardingPolicy.FILE`).
  """
  preprocess_model = make_preprocess_model(gtspec, feature_processors or ())

  def feature_preprocess_fn(ds):
    return ds.map(
        preprocess_model,
        deterministic=False,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def apply_fn(ds):
    # TODO(b/196880966): Revisit if any `task.preprocessors` should be included
    # in the `preprocess_model.`
    fns = itertools.chain((feature_preprocess_fn,), task.preprocessors())
    return functools.reduce(lambda acc, fn: fn(acc), fns, ds)

  def adapted_model_fn():
    x = preprocess_model.outputs[0]  # Ignore other ouputs.
    m = task.adapt(model_fn(x.spec))
    m.compile(optimizer_fn(), loss=task.losses(), metrics=task.metrics())
    return m

  train_ds_provider = _WrappedDatasetProvider(apply_fn, train_ds_provider,
                                              drop_remainder, global_batch_size,
                                              tf_data_service_address)

  if valid_ds_provider is not None:
    valid_ds_provider = _WrappedDatasetProvider(
        apply_fn,
        valid_ds_provider,
        drop_remainder,
        global_batch_size)

  model = trainer.train(
      adapted_model_fn,
      train_ds_provider,
      epochs=epochs,
      valid_ds_provider=valid_ds_provider)

  inputs = preprocess_model.inputs
  outputs = preprocess_model(inputs)
  # Ignore other ouputs.
  x = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
  model_for_export = tf.keras.Model(inputs, model(x))

  if model_exporters is None:
    model_exporters = [model_export.KerasModelExporter()]

  for export_dir in export_dirs or [os.path.join(trainer.model_dir, "export")]:
    for exporter in model_exporters:
      exporter.save(model_for_export, export_dir)
