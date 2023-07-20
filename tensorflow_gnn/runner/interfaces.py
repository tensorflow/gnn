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
from __future__ import annotations

import abc
import dataclasses
import sys
from typing import Callable, Optional, Sequence, TypeVar, Union

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

T = TypeVar("T")
OneOrSequenceOf = Union[T, Sequence[T]]

Field = tfgnn.Field
GraphTensor = tfgnn.GraphTensor
SizeConstraints = tfgnn.SizeConstraints


@dataclasses.dataclass
class RunResult:
  """Holds the return values of `run(...)`.

  Attributes:
    preprocess_model: Keras model containing only the computation for
      preprocessing inputs. It is not trained. The model takes serialized
      `GraphTensor`s as its inputs and returns preprocessed `GraphTensor`s.
      `None` when no preprocess model exists.
    base_model: Keras base GNN (as returned by the user provided `model_fn`).
      The model both takes and returns `GraphTensor`s. The model contains
      any--but not all--trained weights. The `trained_model` contains all
      `base_model` trained weights in addition to any prediction trained
      weights.
    trained_model: Keras model for the e2e GNN. (Base GNN plus any prediction
      head(s).) The model takes `preprocess_model` output as its inputs and
      returns `Task` predictions as its output. Output matches the structure of
      the `Task`: an atom for single- or a mapping for multi- `Task` training.
      The model contains all trained weights.
  """
  preprocess_model: Optional[tf.keras.Model]
  base_model: tf.keras.Model
  trained_model: tf.keras.Model


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

  def __call__(self, inputs: GraphTensor) -> GraphTensor:
    """Processes a `GraphTensor`."""
    raise NotImplementedError()


class ModelExporter(abc.ABC):
  """Saves a Keras model."""

  @abc.abstractmethod
  def save(self, run_result: RunResult, export_dir: str):
    """Saves a Keras model.

    All persistence decisions are left to the implementation: e.g., a Keras
    model with full API or a simple `tf.train.Checkpoint` may be saved.

    Args:
      run_result: A `RunResult` from training.
      export_dir: A destination directory.
    """
    raise NotImplementedError()


class Task(abc.ABC):
  """Defines a learning objective for a GNN.

  A `Task` represents a learning objective for a GNN model and defines all the
  non-GNN pieces around the base GNN. Specifically:

  1) `preprocess` is expected to return a `GraphTensor` (or `GraphTensor`s) and
     a `Field` where (a) the base GNN's output for each `GraphTensor` is passed
     to `predict` and (b) the `Field` is used as the training label (for
     supervised tasks);
  2) `predict` is expected to (a) take the base GNN's output for each
     `GraphTensor` returned by `preprocess` and (b) return a tensor with the
     model's prediction for this task;
  3) `losses` is expected to return callables (`tf.Tensor`, `tf.Tensor`) ->
     `tf.Tensor` that accept (`y_true`, `y_pred`) where `y_true` is produced
     by some dataset and `y_pred` is the model's prediction from (2);
  4) `metrics` is expected to return callables (`tf.Tensor`, `tf.Tensor`) ->
     `tf.Tensor` that accept (`y_true`, `y_pred`) where `y_true` is produced
     by some dataset and `y_pred` is the model's prediction from (2).

  No constraints are made on the `predict` method; e.g.: it may append a head
  with learnable weights or it may perform tensor computations only. (The entire
  `Task` coordinates what that means with respect to dataset—via `preprocess`—,
  modeling—via `predict`— and optimization—via `losses`.)

  `Task`s are applied in the scope of a training invocation: they are subject to
  the executing context of the `Trainer` and should, when needed, override it
  (e.g., a global policy, like `tf.keras.mixed_precision.global_policy()` and
  its implications over logit and activation layers).
  """

  @abc.abstractmethod
  def preprocess(
      self,
      inputs: GraphTensor) -> tuple[OneOrSequenceOf[GraphTensor], Field]:
    """Preprocesses a scalar (after `merge_batch_to_components`) `GraphTensor`.

    This function uses the Keras functional API to define non-trainable
    transformations of the symbolic input `GraphTensor`, which get executed
    during dataset preprocessing in a `tf.data.Dataset.map(...)` operation.
    It has two responsibilities:

     1. Splitting the training label out of the input for training. It must be
        returned as a separate tensor.
     2. Optionally, transforming input features. Some advanced modeling
        techniques require running the same base GNN on multiple different
        transformations, so this function may return a single `GraphTensor`
        or a non-empty sequence of `GraphTensors`. The corresponding base GNN
        output for each `GraphTensor` is provided to the `predict(...)` method.

    Args:
      inputs: A symbolic Keras `GraphTensor` for processing.

    Returns:
      A tuple of processed `GraphTensor`(s) and a `Field` to be used as labels.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def predict(self, *args: GraphTensor) -> Field:
    """Produces prediction outputs for the learning objective.

    Overall model composition* makes use of the Keras Functional API
    (https://www.tensorflow.org/guide/keras/functional) to map symbolic Keras
    `GraphTensor` inputs to symbolic Keras `Field` outputs.

    *) `outputs = predict(GNN(inputs))` where `inputs` are those `GraphTensor`
       returned by `preprocess(...)`, `GNN` is the base GNN, `predict` is this
       method and `outputs` are the prediction outputs for the learning
       objective.

    Args:
      *args: The symbolic Keras `GraphTensor` inputs(s). These inputs correspond
        (in sequence) to the base GNN output of each `GraphTensor` returned by
        `preprocess(...)`.

    Returns:
      The model's prediction output for this task.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    """Returns arbitrary task specific losses."""
    raise NotImplementedError()

  @abc.abstractmethod
  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    """Returns arbitrary task specific metrics."""
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
      A trained `tf.keras.Model`.
    """
    raise NotImplementedError()
