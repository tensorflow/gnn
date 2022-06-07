"""The runner entry point."""
import dataclasses
import functools
import operator
import os
import sys
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, TypeVar, Union
from absl import logging

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.runner.utils import model as model_utils
from tensorflow_gnn.runner.utils import model_export

# pylint:disable=g-import-not-at-top
if sys.version_info >= (3, 8):
  from typing import Protocol
  from typing import runtime_checkable
else:
  from typing_extensions import Protocol
  from typing_extensions import runtime_checkable
# pylint:enable=g-import-not-at-top

GraphTensorAndField = Tuple[tfgnn.GraphTensor, tfgnn.Field]
SizeConstraints = tfgnn.SizeConstraints

V = TypeVar("V")
NestedStructure = Union[V, Sequence[V], Mapping[str, V]]


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


@runtime_checkable
class GraphTensorPadding(Protocol):
  """Collects `GraphtTensor` padding helpers."""

  def get_filter_fn(self, target_batch_size: int) -> Callable[..., bool]:
    """"""
    raise NotImplementedError()

  def get_size_constraints(self, target_batch_size: int) -> SizeConstraints:
    """"""
    raise NotImplementedError()


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


class _LabeledGraphTensorProcessorFn:
  """A `GraphTensorProcessorFn` with optional bound label."""

  def __init__(self, fn: GraphTensorProcessorFn, label: Optional[tfgnn.Field]):
    self._fn = fn
    self._label = label

  def __call__(
      self,
      gt: tfgnn.GraphTensor) -> Tuple[tfgnn.GraphTensor, tfgnn.Field]:
    output = self._fn(gt)
    try:
      x, y = output
      if self._label is not None:
        raise ValueError(f"Expected one label (got {y} and {self._label})")
    except ValueError:
      raise ValueError(f"Expected (`GraphTensor`, `Field`) (got {output})")  # pylint:disable=raise-missing-from
    except TypeError:
      x, y = output, self._label
      if not tfgnn.is_graph_tensor(x):
        raise ValueError(f"Expected `GraphTensor` (got {x})")  # pylint:disable=raise-missing-from
    return x, y  # pytype:disable=bad-return-type


@runtime_checkable
class ModelExporter(Protocol):
  """Saves a Keras model."""

  def save(
      self,
      preprocess_model: Optional[tf.keras.Model],
      model: tf.keras.Model,
      export_dir: str):
    """Saves a Keras model.

    All persistence decisions are left to the implementation: e.g., a Keras
    model with full API or a simple `tf.train.Checkpoint` may be saved.

    Args:
      preprocess_model: An optional `tf.keras.Model` for preprocessing.
      model: A `tf.keras.Model` to save.
      export_dir: A destination directory for the model.
    """
    raise NotImplementedError()


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

  def preprocessor(
      self,
      gt: tfgnn.GraphTensor) -> Union[tfgnn.GraphTensor, GraphTensorAndField]:
    """Preprocess a `GraphTensor`: e.g., extract labels."""
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


def _make_parsing_model(gtspec: tfgnn.GraphTensorSpec) -> tf.keras.Model:
  """Builds a `tf.keras.Model` that parses GraphTensors."""
  examples = tf.keras.layers.Input(
      shape=(),
      dtype=tf.string,
      name="examples")  # Name seen in SignatureDef.
  parsed = tfgnn.keras.layers.ParseExample(gtspec)(examples)
  return tf.keras.Model(examples, parsed)


def _make_preprocessing_model(
    gtspec: tfgnn.GraphTensorSpec,
    preprocessors: Sequence[GraphTensorProcessorFn],
    task: Union[Task, Sequence[Task], Mapping[str, Task]],
    size_constraints: Optional[SizeConstraints] = None) -> tf.keras.Model:
  """Builds a `tf.keras.Model` that applies preprocessing.

  Args:
    gtspec: The `GraphTensorSpec` for input.
    preprocessors: The `GraphTensorProcessorFn` to apply.
    task: A possibly nested structure (see: `tf.nest`) of `Task` atoms, used
      to apply any final `GraphTensorProcessorFn.`
    size_constraints: Any size constraints for padding.

  Returns:
    A `tf.keras.Model` with one, two or three outputs depending on the presence
    of `size_constraints` and the return values of `preprocessors.`
  """
  gt = tf.keras.layers.Input(type_spec=gtspec)

  x = gt.merge_batch_to_components()
  y = mask = None

  if size_constraints is not None:
    x, mask = tfgnn.keras.layers.PadToTotalSizes(size_constraints)(x)

  for fn in preprocessors:
    x, y = _LabeledGraphTensorProcessorFn(fn, y)(x)

  xs, ys = [], []

  for atom in tf.nest.flatten(task):
    xx, yy = _LabeledGraphTensorProcessorFn(atom.preprocessor, None)(x)
    if yy is None and y is None:
      raise ValueError(f"`Task` ({type(atom).__name__}) is missing labels")
    if yy is None:
      yy = y
    else:
      logging.info("`Task` (%s) has multiple labels (%s vs. %s)",
                   type(atom).__name__, f"{y}", f"{yy}")
    xs.append(xx)
    ys.append(yy)

  xs = tf.nest.pack_sequence_as(task, xs)
  ys = tf.nest.pack_sequence_as(task, ys)

  # `tf.nest.flatten` on `tf.data.Dataset` does not recurse into lists: convert
  # any lists to tuples.
  if isinstance(xs, list): xs = tuple(xs)
  if isinstance(ys, list): ys = tuple(ys)

  if mask is None:
    return tf.keras.Model(gt, (xs, ys))
  else:
    return tf.keras.Model(gt, (xs, ys, mask))


def _map_structure_to_attr(
    structure: NestedStructure[Any],
    attr: str) -> NestedStructure[Any]:
  return tf.nest.map_structure(operator.attrgetter(attr), structure)


def _map_structure_to_invocation(
    structure: NestedStructure[Any],
    method: str) -> NestedStructure[Any]:
  return tf.nest.map_structure(operator.methodcaller(method), structure)


def _map_structure_to_item(
    structure: NestedStructure[Any],
    item: int) -> NestedStructure[Any]:
  return tf.nest.map_structure(operator.itemgetter(item), structure)


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
        model_fn: Callable[[tfgnn.GraphTensorSpec], tf.keras.Model],
        optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
        trainer: Trainer,
        task: NestedStructure[Task],
        task_weights: Optional[NestedStructure[float]] = None,
        gtspec: tfgnn.GraphTensorSpec,
        global_batch_size: int,
        epochs: int = 1,
        # TODO(b/196880966): Flip this default to `False.`
        drop_remainder: bool = True,
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
    task: A possibly nested structure (see: `tf.nest`) of `Task` atoms. If a
      nested structure (vs. a single `Task` atom), multitask orchestration is
      perform. (Multiple labels, models, losses and metrics are coordinated.)
    task_weights: Optional weights for multitask orchestration. A possibly
      nested structure (see: `tf.nest`) of float atoms. If provided, the
      structure must match `task` (see: `tf.nest.assert_same_structure`).
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
    train_padding: `GraphTensor` padding for training. Required if training on
      TPU.
    valid_padding: `GraphTensor` padding for validation. Required if training on
      TPU.
    tf_data_service_config: tf.data service speeds up tf.data input pipeline
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

  if task_weights is not None:
    tf.nest.assert_same_structure(task, task_weights)

  parsing_model = _make_parsing_model(gtspec)
  preprocess_model = _make_preprocessing_model(
      gtspec,
      feature_processors or (),
      task)

  def apply_fn(ds, *, padding: Optional[GraphTensorPadding] = None):
    ds = _map_over_dataset(ds, parsing_model)
    if padding is not None:
      target_batch_size = _per_replica_batch_size(
          global_batch_size,
          trainer.strategy.num_replicas_in_sync)
      padding_preprocess_model = _make_preprocessing_model(
          gtspec,
          feature_processors or (),
          task,
          padding.get_size_constraints(target_batch_size))
      ds = ds.filter(padding.get_filter_fn(target_batch_size))
      ds = _map_over_dataset(ds, padding_preprocess_model)
    else:
      ds = _map_over_dataset(ds, preprocess_model)
    return ds

  if train_padding is not None:
    train_apply_fn = functools.partial(apply_fn, padding=train_padding)
  else:
    train_apply_fn = apply_fn

  if validate and valid_padding is not None:
    valid_apply_fn = functools.partial(apply_fn, padding=valid_padding)
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
    # Ignore other ouputs (i.e.: preprocess_model(preprocess_model.input)[1:]).
    gtspecs = _map_structure_to_attr(
        preprocess_model(preprocess_model.input)[0],
        "spec")
    models = tf.nest.pack_sequence_as(task, [
        atom1.adapt(model_fn(atom2))
        for atom1, atom2 in zip(tf.nest.flatten(task), tf.nest.flatten(gtspecs))
    ])
    model = tf.keras.Model(
        _map_structure_to_attr(models, "input"),
        _map_structure_to_attr(models, "output"))
    kwargs = {
        "loss": _map_structure_to_invocation(task, "losses"),
        "loss_weights": task_weights
    }
    if train_padding is None:
      kwargs["metrics"] = _map_structure_to_invocation(task, "metrics")
    else:
      kwargs["weighted_metrics"] = _map_structure_to_invocation(task, "metrics")
    model.compile(optimizer_fn(), **kwargs)
    return model

  model = trainer.train(
      adapted_model_fn,
      train_ds_provider,
      epochs=epochs,
      valid_ds_provider=valid_ds_provider)

  if model_exporters is None:
    model_exporters = [model_export.KerasModelExporter()]

  parsing_and_preprocess_model = model_utils.chain(
      parsing_model,
      preprocess_model)

  for export_dir in export_dirs or [os.path.join(trainer.model_dir, "export")]:
    for exporter in model_exporters:
      exporter.save(parsing_and_preprocess_model, model, export_dir)
