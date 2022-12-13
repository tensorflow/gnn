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
"""Interfaces for the runner entry point."""
import abc
import sys
from typing import Callable, Optional, Sequence, Tuple, Union

import tensorflow as tf
import tensorflow_gnn as tfgnn

# pylint:disable=g-import-not-at-top
if sys.version_info >= (3, 8):
  from typing import Protocol
  from typing import runtime_checkable
else:
  from typing_extensions import Protocol
  from typing_extensions import runtime_checkable
# pylint:enable=g-import-not-at-top

GraphTensor = tfgnn.GraphTensor
GraphTensorAndField = Tuple[GraphTensor, tfgnn.Field]
SizeConstraints = tfgnn.SizeConstraints


class DatasetProvider(abc.ABC):

  @abc.abstractmethod
  def get_dataset(self, context: tf.distribute.InputContext) -> tf.data.Dataset:
    """Get a `tf.data.Dataset` by `context` per replica."""
    raise NotImplementedError()


class GraphTensorPadding(abc.ABC):
  """Collects `GraphtTensor` padding helpers."""

  @abc.abstractmethod
  def get_filter_fn(
      self,
      size_constraints: SizeConstraints) -> Callable[..., bool]:
    raise NotImplementedError()

  @abc.abstractmethod
  def get_size_constraints(self, target_batch_size: int) -> SizeConstraints:
    raise NotImplementedError()


@runtime_checkable
class GraphTensorProcessorFn(Protocol):
  """A class for `GraphTensor` processing."""

  def __call__(
      self,
      gt: GraphTensor) -> Union[GraphTensor, GraphTensorAndField]:
    """Processes a `GraphTensor` with optional `Field` extraction.

    Args:
      gt: A `GraphTensor` for processing.

    Returns:
      A processed `GraphTensor` or a processed `GraphTensor` and `Field.`
    """
    raise NotImplementedError()


class ModelExporter(abc.ABC):
  """Saves a Keras model."""

  @abc.abstractmethod
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


class Task(abc.ABC):
  """Collects the ancillary, supporting pieces to train a Keras model.

  `Task`s are applied and used to compile a `tf.keras.Model` in the scope
  of a training invocation: they are subject to the executing context
  of the `Trainer` and should, when needed, override it (e.g., a global
  policy, like `tf.keras.mixed_precision.global_policy()` and its implications
  over logit and activation layers).

  A `Task` is expected to coordinate all of its methods and their return values
  to define a graph learning objective. Precisely:

  1) `preprocess` is expected to return a `GraphTensor` or
     (`GraphTensor`, `Field`) where the `GraphTensor` matches the input of the
     model returned by `adapt` and the `Field` is a training label
  2) `adapt` is expected to return a `tf.keras.Model` that accepts a
     `GraphTensor` matching the output of `preprocess`
  3) `losses` is expected to return callables (`tf.Tensor`, `tf.Tensor`) ->
     `tf.Tensor` that accept (`y_true`, `y_pred`) where `y_true` is produced
     by some dataset and `y_pred` is output of the adapted model (see (2))
  4) `metrics` is expected to return callables (`tf.Tensor`, `tf.Tensor`) ->
     `tf.Tensor` that accept (`y_true`, `y_pred`) where `y_true` is produced
     by some dataset and `y_pred` is output of the adapted model (see (2)).

  No constraints are made on the `adapt` method; e.g.: it may adapt its input by
  appending a head, it may add losses to its input, it may add metrics to its
  input or it may do any combination of the aforementioned modifications. The
  `adapt` method is expected to adapt an arbitrary `tf.keras.Model` to the graph
  learning objective. (The entire `Tasks` coordinates what that means with
  respect to input—via `preprocess`—, modeling—via `adapt`— and optimization—via
  `losses.`)
  """

  @abc.abstractmethod
  def adapt(self, model: tf.keras.Model) -> tf.keras.Model:
    """Adapt a model to a task by appending arbitrary head(s)."""
    raise NotImplementedError()

  @abc.abstractmethod
  def preprocess(
      self,
      gt: GraphTensor) -> Union[GraphTensor, GraphTensorAndField]:
    """Preprocess a scalar (after `merge_batch_to_components`) `GraphTensor`."""
    raise NotImplementedError()

  @abc.abstractmethod
  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    """Arbitrary losses matching any head(s)."""
    raise NotImplementedError()

  @abc.abstractmethod
  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    """Arbitrary task specific metrics."""
    raise NotImplementedError()


class Trainer(abc.ABC):
  """A class for training and validation."""

  @property
  @abc.abstractmethod
  def model_dir(self) -> str:
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def strategy(self) -> tf.distribute.Strategy:
    raise NotImplementedError()

  @abc.abstractmethod
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
