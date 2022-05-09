"""`tf.keras.Model.fit` training loop."""
import dataclasses
import functools
import os
from typing import Callable, Optional, Sequence, Union

import tensorflow as tf

from tensorflow_gnn.runner import orchestration

BackupAndRestore = tf.keras.callbacks.experimental.BackupAndRestore
DatasetProvider = orchestration.DatasetProvider


@dataclasses.dataclass
class KerasOptions:
  """Provides Keras Modeling related options."""
  policy: Optional[Union[str, tf.keras.mixed_precision.Policy]] = None


class KerasTrainer:
  """Trains using the `tf.keras.Model.fit` training loop."""

  def __init__(
      self,
      strategy: tf.distribute.Strategy,
      *,
      model_dir: str,
      ckpts_dir: Optional[str] = None,
      backup_dir: Optional[str] = None,
      steps_per_epoch: Optional[int] = None,
      validation_steps: Optional[int] = None,
      validation_per_epoch: Optional[int] = None,
      summarize_every_n_steps: Union[int, str] = 500,
      checkpoint_every_n_steps: Union[int, str] = "epoch",
      callbacks: Optional[Sequence[tf.keras.callbacks.Callback]] = None,
      restore_best_weights: bool = True,
      keras_options: Optional[KerasOptions] = None):
    """Sets training parameters.

    Args:
      strategy: A `tf.distribute.Strategy.`
      model_dir: A model directory for summaries.
      ckpts_dir: An optional directory for checkpoints, if unset;
        `os.path.join(model_dir, "ckpts")` is used.
      backup_dir: An optional directory for backup, if unset;
        `(os.path.join(model_dir, "backup"),)` is used.
      steps_per_epoch: An optional steps per epoch, if unspecified: epochs are
        at `tf.data.Dataset` end.
      validation_steps: An optional numer of validation steps, if unspecified:
        the entire validation `tf.data.Dataset` is evaluated.
      validation_per_epoch: An optional validations per epoch, if unspecified:
        one evaluation per training epoch is performed.
      summarize_every_n_steps: The frequency for writing TensorBoard summaries,
        as an integer number of steps, or "epoch" for once per epoch, or
        "never".
      checkpoint_every_n_steps: The frequency for writing latest models, as an
        integer number of steps, or "epoch" for once per epoch, or "never".
        The best model will always be saved after each validation epoch except
        when this parameter is set to "never", because the validation metric is
        available only after validation epoch.
      callbacks: Optional additional `tf.keras.callbacks.Callback` for
        `tf.keras.Model.fit.`
      restore_best_weights: Requires a `checkpoint_every_n_steps` other than
        "never." Whether to restore the best model weights as determined by
        `tf.keras.callbacks.ModelCheckpoint` after training.
      keras_options: A `KerasOptions` for mixed precision, see:
        https://www.tensorflow.org/guide/mixed_precision.
    """
    if restore_best_weights and checkpoint_every_n_steps == "never":
      raise ValueError("`restore_best_weights` requires a "
                       "`checkpoint_every_n_steps` other than \"never\"")

    if ckpts_dir is None:
      ckpts_dir = os.path.join(model_dir, "ckpts")

    if backup_dir is None:
      backup_dir = os.path.join(model_dir, "backup")

    self._strategy = strategy
    self._model_dir = model_dir
    self._ckpts_dir = ckpts_dir
    self._backup_dir = backup_dir
    self._steps_per_epoch = steps_per_epoch
    self._validation_steps = validation_steps
    self._validation_per_epoch = validation_per_epoch
    self._summarize_every_n_steps = summarize_every_n_steps
    self._checkpoint_every_n_steps = checkpoint_every_n_steps
    self._callbacks = callbacks
    self._restore_best_weights = restore_best_weights
    self._keras_options = keras_options

  @property
  def model_dir(self) -> str:
    return self._model_dir

  @property
  def strategy(self) -> tf.distribute.Strategy:
    return self._strategy

  def train(
      self,
      model_fn: Callable[[], tf.keras.Model],
      train_ds_provider: DatasetProvider,
      *,
      epochs: int = 1,
      valid_ds_provider: Optional[DatasetProvider] = None) -> tf.keras.Model:
    """Runs `tf.keras.Model.fit` with the`tf.distribute.Strategy` provided.

    Args:
      model_fn: A `ModelFn`, to be invoked in the `tf.distribute.Strategty`
        scope.
      train_ds_provider: A function that returns a `tf.data.Dataset` for
        training.The items of the `tf.data.Dataset` are pairs
        `(graph_tensor, label)` that represent one batch of per-replica training
        inputs after `GraphTensor.merge_batch_to_components()` has been applied.
      epochs: The epochs to train: adjusted for `validation_per_epoch.`
      valid_ds_provider: An optional function that returns a `tf.data.Dataset`
        for validation. The items of the `tf.data.Dataset` are pairs
        `(graph_tensor, label)` that represent one batch of per-replica training
        inputs after `GraphTensor.merge_batch_to_components()` has been applied.

    Returns:
      A trained `tf.keras.Model.`
    """

    # Adjust the following given `epochs`:
    # - `summarize_every_n_steps`
    # - `checkpoint_every_n_steps`
    # - `steps_per_epoch`
    # - `validation_steps`
    if self._validation_per_epoch is not None:
      if self._steps_per_epoch is None:
        raise ValueError("`validation_per_epoch` requires a `steps_per_epoch`")
      # Preserve the user-visible notion of "epoch"...
      if self._summarize_every_n_steps == "epoch":
        summarize_every_n_steps = self._steps_per_epoch
      else:
        summarize_every_n_steps = self._summarize_every_n_steps
      if self._checkpoint_every_n_steps == "epoch":
        checkpoint_every_n_steps = self._steps_per_epoch
      else:
        checkpoint_every_n_steps = self._checkpoint_every_n_steps
      # ...before we fudge it for Keras to validate more often.
      epochs = epochs * self._validation_per_epoch
      steps_per_epoch = self._steps_per_epoch // self._validation_per_epoch
      validation_steps = self._validation_steps
    else:
      summarize_every_n_steps = self._summarize_every_n_steps
      checkpoint_every_n_steps = self._checkpoint_every_n_steps
      steps_per_epoch = self._steps_per_epoch
      validation_steps = self._validation_steps

    if validation_steps is not None and valid_ds_provider is None:
      raise ValueError("`validation_steps` requires a `valid_ds_fn`")

    if self._restore_best_weights and valid_ds_provider is None:
      raise ValueError("`restore_best_weights` requires a validation dataset")

    if self._keras_options and self._keras_options.policy:
      tf.keras.mixed_precision.set_global_policy(self._keras_options.policy)

    def per_replica_ds_fn(input_context, *, delegate, repeat):
      ds = delegate.get_dataset(input_context)
      # The dataset could be repeated by the preprocessing, e.g. by
      # augmentations. We repeat it again here if needed.
      if repeat:
        ds = ds.repeat()
      return ds

    train_ds = self._strategy.distribute_datasets_from_function(
        functools.partial(
            per_replica_ds_fn,
            delegate=train_ds_provider,
            # Training is by epochs and steps_per_epoch: not dataset end.
            repeat=steps_per_epoch is not None))

    if valid_ds_provider is not None:
      valid_ds = self._strategy.distribute_datasets_from_function(
          functools.partial(
              per_replica_ds_fn,
              delegate=valid_ds_provider,
              # Validation is by validation_steps: not dataset end.
              repeat=validation_steps is not None))
    else:
      valid_ds = None

    callbacks = [
        *(self._callbacks or []),
        BackupAndRestore(backup_dir=self._backup_dir)
    ]

    if checkpoint_every_n_steps != "never":
      callbacks += [
          tf.keras.callbacks.ModelCheckpoint(
              filepath=os.path.join(self._ckpts_dir, "latest"),
              save_best_only=False,
              save_weights_only=True,
              save_freq=checkpoint_every_n_steps),
          tf.keras.callbacks.ModelCheckpoint(
              filepath=os.path.join(self._ckpts_dir, "best"),
              save_best_only=True,
              save_weights_only=True,
              save_freq="epoch")
      ]

    if summarize_every_n_steps != "never":
      callbacks += [
          tf.keras.callbacks.TensorBoard(
              log_dir=self._model_dir,
              histogram_freq=1,
              write_steps_per_second=True,
              update_freq=summarize_every_n_steps,
              embeddings_freq=1)
      ]

    with self._strategy.scope():
      model = model_fn()

    if self._keras_options and self._keras_options.policy:
      # Cast logits to `tf.keras.backend.floatx()` for mixed_precision.
      # For more details, see:
      # https://www.tensorflow.org/guide/mixed_precision#building_the_model.
      floatx = tf.keras.backend.floatx()
      outputs = [tf.cast(o, dtype=floatx) for o in model.outputs]
      model = tf.keras.Model(model.inputs, outputs)

    model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_ds,
        validation_steps=validation_steps,
        callbacks=callbacks)

    if self._restore_best_weights:
      model.load_weights(os.path.join(self._ckpts_dir, "best"))

    return model

