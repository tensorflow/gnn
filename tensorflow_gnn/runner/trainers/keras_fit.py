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
"""`tf.keras.Model.fit` training loop."""
import dataclasses
import functools
import os
from typing import Callable, Optional, Sequence, Union

import tensorflow as tf
from tensorflow_gnn.runner import interfaces

BackupAndRestore = tf.keras.callbacks.experimental.BackupAndRestore
DatasetProvider = interfaces.DatasetProvider


@dataclasses.dataclass
class KerasTrainerOptions:
  """Provides Keras training related options."""
  policy: Optional[Union[str, tf.keras.mixed_precision.Policy]] = None
  soft_device_placement: bool = False
  # `enable_check_numerics` requires `soft_device_placement` if running on TPU.
  enable_check_numerics: bool = False


@dataclasses.dataclass
class KerasTrainerCheckpointOptions:
  """Provides Keras Checkpointing related configuration options.

  Attributes:
    checkpoint_dir: Directory path to save checkpoint files.
    best_checkpoint: Filename for the best checkpoint.
    latest_checkpoint: Filename for the latest checkpoint.
  """
  checkpoint_dir: Optional[str] = None
  best_checkpoint: str = "best"
  latest_checkpoint: str = "latest"

  def best_checkpoint_filepath(self) -> str:
    return os.path.join(self.checkpoint_dir, self.best_checkpoint)

  def latest_checkpoint_filepath(self) -> str:
    return os.path.join(self.checkpoint_dir, self.latest_checkpoint)


class KerasTrainer(interfaces.Trainer):
  """Trains using the `tf.keras.Model.fit` training loop."""

  def __init__(
      self,
      strategy: tf.distribute.Strategy,
      *,
      model_dir: str,
      checkpoint_options: Optional[KerasTrainerCheckpointOptions] = None,
      backup_dir: Optional[str] = None,
      steps_per_epoch: Optional[int] = None,
      verbose: Union[int, str] = "auto",
      validation_steps: Optional[int] = None,
      validation_per_epoch: Optional[int] = None,
      validation_freq: Optional[int] = None,
      summarize_every_n_steps: Union[int, str] = 500,
      checkpoint_every_n_steps: Union[int, str] = "epoch",
      backup_and_restore: bool = True,
      callbacks: Optional[Sequence[tf.keras.callbacks.Callback]] = None,
      restore_best_weights: Optional[bool] = None,
      options: Optional[KerasTrainerOptions] = None):
    """Sets training parameters.

    Args:
      strategy: A `tf.distribute.Strategy.`
      model_dir: A model directory for summaries.
      checkpoint_options: An optional configuration for checkpointing related
        configs. If checkpoint_options.checkpoint_dir is unset;
        `os.path.join(model_dir, "ckpnt")` is used.
      backup_dir: An optional directory for backup, if unset;
        `(os.path.join(model_dir, "backup"),)` is used.
      steps_per_epoch: The number of training steps per epoch. Optional,
        if unspecified: epochs are at `tf.data.Dataset` end.
      verbose: Forwarded to `tf.keras.Model.fit()`. Possible values are
        0 (silent), 1 (print progress bar), 2 (one line per epoch), and
        "auto" (default) defers to keras to select verbosity.
      validation_steps: The number of steps used during validation. Optional,
        if unspecified: the entire validation `tf.data.Dataset` is evaluated.
      validation_per_epoch: The number of validations done per training epoch.
        Optional, if unspecified: Perform one validation per training epoch.
        Only one of `validation_per_epoch` and `validation_freq` can be
        specified.
      validation_freq: Specifies how many training epochs to run before a new
        validation run is performed. Optional, if unspecified: Performs
        validation after every training epoch. Only one of
        `validation_per_epoch` and `validation_freq` can be specified.
      summarize_every_n_steps: The frequency for writing TensorBoard summaries,
        as an integer number of steps, or "epoch" for once per epoch, or
        "never".
      checkpoint_every_n_steps: The frequency for writing latest models, as an
        integer number of steps, or "epoch" for once per epoch, or "never".
        The best model will always be saved after each validation epoch except
        when this parameter is set to "never", because the validation metric is
        available only after validation epoch.
      backup_and_restore: Whether to backup and restore (According to
        `tf.keras.callbacks.BackupAndRestore`). The backup
        directory is determined by `backup_dir`.
      callbacks: Optional additional `tf.keras.callbacks.Callback` for
        `tf.keras.Model.fit.`
      restore_best_weights: Requires a `checkpoint_every_n_steps` other than
        "never." Whether to restore the best model weights as determined by
        `tf.keras.callbacks.ModelCheckpoint` after training. If unspecified,
        its value is determined at `train(...)` invocation: `True if
        valid_ds_provider is not None else False`.
      options: A `KerasTrainerOptions.`
    """
    if restore_best_weights and checkpoint_every_n_steps == "never":
      raise ValueError("`restore_best_weights` requires a "
                       "`checkpoint_every_n_steps` other than \"never\"")

    if checkpoint_options is None:
      checkpoint_options = KerasTrainerCheckpointOptions()

    if checkpoint_options.checkpoint_dir is None:
      checkpoint_options.checkpoint_dir = os.path.join(model_dir, "ckpnt")

    if backup_dir is None:
      backup_dir = os.path.join(model_dir, "backup")

    if (validation_freq is not None and validation_per_epoch is not None):
      raise ValueError(
          "`validation_freq` and `validation_per_epoch` are mutually exclusive."
      )

    self._strategy = strategy
    self._model_dir = model_dir
    self._checkpoint_options = checkpoint_options
    self._backup_dir = backup_dir
    self._steps_per_epoch = steps_per_epoch
    self._verbose = verbose
    self._validation_steps = validation_steps
    self._validation_per_epoch = validation_per_epoch
    self._validation_freq = validation_freq
    self._summarize_every_n_steps = summarize_every_n_steps
    self._checkpoint_every_n_steps = checkpoint_every_n_steps
    self._backup_and_restore = backup_and_restore
    self._callbacks = callbacks
    self._restore_best_weights = restore_best_weights
    self._options = options

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

    if self._validation_freq is not None and valid_ds_provider is None:
      raise ValueError("`validation_freq` requires a `valid_ds_fn`")

    validation_freq = (
        self._validation_freq if self._validation_freq is not None else 1
    )

    # Adjust `restore_best_weights` given `valid_ds_provider`:
    restore_best_weights = self._restore_best_weights
    if restore_best_weights and valid_ds_provider is None:
      raise ValueError("`restore_best_weights` requires a validation dataset")
    elif restore_best_weights is None:
      restore_best_weights = valid_ds_provider is not None

    if self._options and self._options.soft_device_placement:
      tf.config.set_soft_device_placement(True)

    if self._options and self._options.enable_check_numerics:
      tf.debugging.enable_check_numerics()

    if self._options and self._options.policy:
      tf.keras.mixed_precision.set_global_policy(self._options.policy)

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

    callbacks = list(self._callbacks or [])

    if self._backup_and_restore:
      callbacks += [
          BackupAndRestore(backup_dir=self._backup_dir)
      ]

    if checkpoint_every_n_steps != "never":
      callbacks += [
          tf.keras.callbacks.ModelCheckpoint(
              filepath=self._checkpoint_options.latest_checkpoint_filepath(),
              save_best_only=False,
              save_weights_only=True,
              save_freq=checkpoint_every_n_steps),
          tf.keras.callbacks.ModelCheckpoint(
              filepath=self._checkpoint_options.best_checkpoint_filepath(),
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

    if self._options and self._options.policy:
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
        validation_freq=validation_freq,
        verbose=self._verbose,
        callbacks=callbacks)

    if restore_best_weights:
      model.load_weights(self._checkpoint_options.best_checkpoint_filepath())

    return model
