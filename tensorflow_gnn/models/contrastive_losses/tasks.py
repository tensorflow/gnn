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
"""Contrastive loss tasks."""
from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
import functools
from typing import Optional, Union

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn import runner
from tensorflow_gnn.models.contrastive_losses import layers
from tensorflow_gnn.models.contrastive_losses import losses
from tensorflow_gnn.models.contrastive_losses import metrics

Field = tfgnn.Field
GraphTensor = tfgnn.GraphTensor


class _ConstrastiveLossTask(runner.Task, abc.ABC):
  """Base class for unsupervised contrastive representation learning tasks.

  The default `predict` method implementation shuffles feature across batch
  examples to create positive and negative activations. There are multiple ways
  proposed in the literature to learn representations based on the activations.

  Any subclass must implement `make_contrastive_layer` method, which produces
  the final prediction outputs.

  If the loss involves labels for each example, subclasses should leverage
  `losses` and `metrics` methods to specify task's losses. When the loss only
  involves model outputs, `make_contrastive_layer` should output both positive
  and perturb examples, and the `losses` should use pseudolabels.

  Any model-specific preprocessing should be implemented in the `preprocess`.
  """

  def __init__(
      self,
      node_set_name: str,
      *,
      feature_name: str = tfgnn.HIDDEN_STATE,
      representations_layer_name: Optional[str] = None,
      corruptor: Optional[layers._Corruptor] = None,
      projector_units: Optional[Sequence[int]] = None,
      seed: Optional[int] = None,
  ):
    """Constructs the `runner.Task`.

    Args:
      node_set_name: Name of the node set for readout.
      feature_name: Feature name for readout.
      representations_layer_name: Layer name for uncorrupted representations.
      corruptor: `Corruptor` instance for creating negative samples. If not
        specified, we use `ShuffleFeaturesGlobally` by default.
      projector_units: `Sequence` of layer sizes for projector network.
        Projectors prevent dimensional collapse, but can hinder training for
        easy corruptions. For more details, see
        https://arxiv.org/abs/2304.12210.
      seed: Random seed for the default corruptor (`ShuffleFeaturesGlobally`).
    """
    self._representations_layer_name = (
        representations_layer_name or "clean_representations"
    )
    self._readout = tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name=node_set_name,
        feature_name=feature_name)
    if corruptor is None:
      self._corruptor = layers.ShuffleFeaturesGlobally(seed=seed)
    else:
      self._corruptor = corruptor
    if projector_units:
      self._projector = tf.keras.Sequential(
          [tf.keras.layers.Dense(units, "relu") for units in projector_units]
      )
    else:
      self._projector = None

  def predict(self, *args: tfgnn.GraphTensor) -> tfgnn.Field:
    """Apply a readout head for use with various contrastive losses.

    Args:
      *args: A tuple of (clean, corrupted) `tfgnn.GraphTensor`s.

    Returns:
      The logits for some contrastive loss as produced by the implementing
      subclass.
    """
    gt_clean, gt_corrupted = args

    if not tfgnn.is_graph_tensor(gt_clean):
      raise ValueError(f"Expected a `GraphTensor` input (got {gt_clean})")
    if not tfgnn.is_graph_tensor(gt_corrupted):
      raise ValueError(f"Expected a `GraphTensor` input (got {gt_corrupted})")
    if isinstance(tf.distribute.get_strategy(), tf.distribute.TPUStrategy):
      raise AssertionError(
          "Contrastive learning tasks do not support TPU (see b/269648832)."
      )

    # Clean representations.
    x_clean = tf.keras.layers.Layer(name=self._representations_layer_name)(
        self._readout(gt_clean)
    )
    # Corrupted representations.
    x_corrupted = self._readout(gt_corrupted)
    if self._projector:
      x_clean = self._projector(x_clean)
      x_corrupted = self._projector(x_corrupted)
    outputs = tf.stack((x_clean, x_corrupted), axis=1)
    return self.make_contrastive_layer()(outputs)

  @abc.abstractmethod
  def make_contrastive_layer(self) -> tf.keras.layers.Layer:
    """Returns the layer contrasting clean outputs with the correupted ones."""
    raise NotImplementedError()

  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return tuple()


class DeepGraphInfomaxTask(_ConstrastiveLossTask):
  """A Deep Graph Infomax (DGI) Task."""

  def make_contrastive_layer(self) -> tf.keras.layers.Layer:
    return layers.DeepGraphInfomaxLogits()

  def preprocess(
      self,
      inputs: GraphTensor) -> tuple[Sequence[GraphTensor], Field]:
    """Creates labels--i.e., (positive, negative)--for Deep Graph Infomax."""
    x = (inputs, self._corruptor(inputs))
    y = tf.tile(((1, 0),), (inputs.num_components, 1))
    return x, y

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (tf.keras.losses.BinaryCrossentropy(from_logits=True),)

  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (
        tf.keras.metrics.BinaryCrossentropy(from_logits=True),
        tf.keras.metrics.BinaryAccuracy(),
    )


def _unstack_y_pred(
    metric_fn: Callable[[tf.Tensor], tf.Tensor]
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
  """Wraps a `metric_fn` to operate on clean representations only."""

  @functools.wraps(metric_fn)
  def wrapper_fn(_, y_pred):
    representations_clean, _ = tf.unstack(y_pred, axis=1)
    return metric_fn(representations_clean)

  return wrapper_fn


class BarlowTwinsTask(_ConstrastiveLossTask):
  """A Barlow Twins (BT) Task."""

  def __init__(
      self,
      *args,
      lambda_: Optional[Union[tf.Tensor, float]] = None,
      normalize_batch: bool = True,
      **kwargs):
    super().__init__(*args, **kwargs)
    self._lambda = lambda_
    self._normalize_batch = normalize_batch

  def make_contrastive_layer(self) -> tf.keras.layers.Layer:
    return tf.keras.layers.Layer()

  def preprocess(
      self,
      inputs: GraphTensor) -> tuple[Sequence[GraphTensor], Field]:
    """Creates unused pseudo-labels."""
    x = (inputs, self._corruptor(inputs))
    y = tf.zeros((inputs.num_components, 0), dtype=tf.int32)
    return x, y

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    def loss_fn(_, x):
      return losses.barlow_twins_loss(
          *tf.unstack(x, axis=1),
          lambda_=self._lambda,
          normalize_batch=self._normalize_batch,
      )

    return (loss_fn,)

  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (
        _unstack_y_pred(metrics.self_clustering),
        _unstack_y_pred(metrics.numerical_rank),
        _unstack_y_pred(metrics.pseudo_condition_number),
    )


class VicRegTask(_ConstrastiveLossTask):
  """A VICReg Task."""

  def __init__(
      self,
      *args,
      sim_weight: Union[tf.Tensor, float] = 25.,
      var_weight: Union[tf.Tensor, float] = 25.,
      cov_weight: Union[tf.Tensor, float] = 1.,
      **kwargs):
    super().__init__(*args, **kwargs)
    self._sim_weight = sim_weight
    self._var_weight = var_weight
    self._cov_weight = cov_weight

  def make_contrastive_layer(self) -> tf.keras.layers.Layer:
    return tf.keras.layers.Layer()

  def preprocess(
      self,
      inputs: GraphTensor) -> tuple[Sequence[GraphTensor], Field]:
    """Creates unused pseudo-labels."""
    x = (inputs, self._corruptor(inputs))
    y = tf.zeros((inputs.num_components, 0), dtype=tf.int32)
    return x, y

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    def loss_fn(_, x):
      return losses.vicreg_loss(
          *tf.unstack(x, axis=1),
          sim_weight=self._sim_weight,
          var_weight=self._var_weight,
          cov_weight=self._cov_weight,
      )

    return (loss_fn,)

  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (
        _unstack_y_pred(metrics.self_clustering),
        _unstack_y_pred(metrics.numerical_rank),
        _unstack_y_pred(metrics.pseudo_condition_number),
    )
