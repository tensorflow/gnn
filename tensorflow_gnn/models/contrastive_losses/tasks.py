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
from collections.abc import Callable, Mapping, Sequence
import functools
from typing import Optional, Union

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn import runner
from tensorflow_gnn.models.contrastive_losses import layers
from tensorflow_gnn.models.contrastive_losses import losses
from tensorflow_gnn.models.contrastive_losses import metrics
from tensorflow_gnn.models.contrastive_losses import utils

Field = tfgnn.Field
GraphTensor = tfgnn.GraphTensor


# In our implementation of contrastive learning, `y_pred`` is a tensor with
# clean and corrupted representations stacked in the first dimension.
_UNSTACK_FN = lambda y_pred: tf.unstack(y_pred, axis=1)[0]


class ContrastiveLossTask(runner.Task):
  """Base class for unsupervised contrastive representation learning tasks.

  The process is separated into preprocessing and contrastive parts, with the
  focus on reusability of individual components. The `preprocess` produces
  input GraphTensors to be used with the `predict` as well as labels for the
  task. The default `predict` method implementation expects a pair of positive
  and negative GraphTensors. There are multiple ways proposed in the literature
  to learn representations based on the activations - we achieve that by using
  custom losses.

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
      corruptor: Optional[layers.Corruptor] = None,
      projector_units: Optional[Sequence[int]] = None,
      seed: Optional[int] = None,
  ):
    """Constructs the `runner.Task`.

    NOTE: This class uses `ShuffleFeaturesGlobally` as the default corruptor.
    Per b/269249455, it does not support TPU execution. However, this is most
    robust corruption function in practice, which is why we default to it.

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
        node_set_name=node_set_name, feature_name=feature_name
    )
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

  def preprocess(
      self, inputs: GraphTensor
  ) -> tuple[Sequence[GraphTensor], runner.Predictions]:
    """Applies a `Corruptor` and returns empty pseudo-labels."""
    x = (inputs, self._corruptor(inputs))
    y = tf.zeros((inputs.num_components, 0), dtype=tf.int32)
    return x, y

  def predict(self, *args: tfgnn.GraphTensor) -> runner.Predictions:
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

  def metrics(self) -> runner.Metrics:
    return tuple()


class _DgiPassthrough(tf.keras.layers.Layer):
  """Applies logits layer and returns both predictions and representations."""

  def __init__(
      self,
      logits_layer: tf.keras.layers.Layer,
      *args,
      **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self._logits_layer = logits_layer

  def call(self, inputs: tf.Tensor) -> runner.Predictions:
    return {
        "predictions": self._logits_layer(inputs),
        "representations": inputs,
    }


# Gradients of this loss will be `None`, but that's not a problem as long as
# other loss terms create gradients for all trainable variables.
class _ZeroLoss(tf.keras.losses.Loss):

  def call(
      self,
      y_true: tf.Tensor,
      y_pred: tf.Tensor,
      sample_weight: Optional[tf.Tensor] = None,
  ) -> tf.Tensor:
    # Keras Loss interface expects a tensor of shape `tf.shape(y_pred)[:-1]`
    # with a correct data type.
    return tf.zeros_like(y_pred[..., 0])


class DeepGraphInfomaxTask(ContrastiveLossTask):
  """A Deep Graph Infomax (DGI) Task."""

  def __init__(
      self,
      *args,
      **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self._logits_layer = layers.DeepGraphInfomaxLogits()

  def make_contrastive_layer(self) -> tf.keras.layers.Layer:
    return _DgiPassthrough(self._logits_layer)

  def predict(self, *args: tfgnn.GraphTensor) -> runner.Predictions:
    # Wrap the predictions to have dictionary names as Keras names for loss
    # and metric reporting (e.g., in TensorBoard).
    return {
        k: tf.keras.layers.Layer(name=k)(v)
        for k, v in super().predict(*args).items()
    }

  def preprocess(
      self, inputs: GraphTensor
  ) -> tuple[Sequence[GraphTensor], Mapping[str, Field]]:
    """Creates labels--i.e., (positive, negative)--for Deep Graph Infomax."""
    x = (inputs, self._corruptor(inputs))
    y_dgi = tf.tile(tf.constant([[0, 1]]), (inputs.num_components, 1))
    y_empty = tf.zeros((inputs.num_components, 0), dtype=tf.int32)
    return x, {"predictions": y_dgi, "representations": y_empty}

  def losses(self) -> runner.Losses:
    return {
        "predictions": tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "representations": _ZeroLoss(),
    }

  def metrics(self) -> runner.Metrics:
    return {
        "predictions": (
            tf.keras.metrics.BinaryCrossentropy(from_logits=True),
            tf.keras.metrics.BinaryAccuracy(),
        ),
        "representations": (
            metrics.AllSvdMetrics(y_pred_transform_fn=_UNSTACK_FN),
        ),
    }


def _unstack_y_pred(
    metric_fn: Callable[[tf.Tensor], tf.Tensor]
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
  """Wraps a `metric_fn` to operate on clean representations only."""

  @functools.wraps(metric_fn)
  def wrapper_fn(_, y_pred):
    representations_clean, _ = tf.unstack(y_pred, axis=1)
    return metric_fn(representations_clean)

  return wrapper_fn


class BarlowTwinsTask(ContrastiveLossTask):
  """A Barlow Twins (BT) Task."""

  def __init__(
      self,
      *args,
      lambda_: Optional[Union[tf.Tensor, float]] = None,
      normalize_batch: bool = True,
      **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self._lambda = lambda_
    self._normalize_batch = normalize_batch

  def make_contrastive_layer(self) -> tf.keras.layers.Layer:
    return tf.keras.layers.Layer()

  def losses(self) -> runner.Losses:
    def loss_fn(_, x):
      return losses.barlow_twins_loss(
          *tf.unstack(x, axis=1),
          lambda_=self._lambda,
          normalize_batch=self._normalize_batch,
      )

    return loss_fn

  def metrics(self) -> runner.Metrics:
    return (metrics.AllSvdMetrics(y_pred_transform_fn=_UNSTACK_FN),)


class VicRegTask(ContrastiveLossTask):
  """A VICReg Task."""

  def __init__(
      self,
      *args,
      sim_weight: Union[tf.Tensor, float] = 25.0,
      var_weight: Union[tf.Tensor, float] = 25.0,
      cov_weight: Union[tf.Tensor, float] = 1.0,
      **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self._sim_weight = sim_weight
    self._var_weight = var_weight
    self._cov_weight = cov_weight

  def make_contrastive_layer(self) -> tf.keras.layers.Layer:
    return tf.keras.layers.Layer()

  def losses(self) -> runner.Losses:
    def loss_fn(_, x):
      return losses.vicreg_loss(
          *tf.unstack(x, axis=1),
          sim_weight=self._sim_weight,
          var_weight=self._var_weight,
          cov_weight=self._cov_weight,
      )

    return loss_fn

  def metrics(self) -> runner.Metrics:
    return (metrics.AllSvdMetrics(y_pred_transform_fn=_UNSTACK_FN),)


class TripletLossTask(ContrastiveLossTask):
  """The triplet loss task."""

  def __init__(self, *args, margin: float = 1.0, **kwargs):
    super().__init__(*args, **kwargs)
    self._margin = margin
    self._unstack_graph_tensor_at_index = utils.SliceNodeSetFeatures()

  def make_contrastive_layer(self) -> tf.keras.layers.Layer:
    return layers.TripletEmbeddingSquaredDistances()

  def _unstack_graph_tensor(
      self, inputs: GraphTensor
  ) -> tuple[GraphTensor, GraphTensor]:
    anchor = self._unstack_graph_tensor_at_index(inputs, 0)
    positive_sample = self._unstack_graph_tensor_at_index(inputs, 1)
    return (anchor, positive_sample)

  def preprocess(
      self, inputs: GraphTensor
  ) -> tuple[Sequence[GraphTensor], tfgnn.Field]:
    """Creates unused pseudo-labels.

    The input tensor should have the anchor and positive sample stacked along
    the first dimension for each feature within each node set. The corruptor is
    applied on the positive sample.

    Args:
      inputs: The anchor and positive sample stack along the first axis.

    Returns:
      Sequence of three graph tensors (anchor, positive_sample,
      corrupted_sample) and unused pseudo-labels.
    """
    anchor, positive_sample = self._unstack_graph_tensor(inputs)
    x = (anchor, positive_sample, self._corruptor(positive_sample))
    y = tf.zeros((inputs.num_components, 0), dtype=tf.int32)
    return x, y

  def predict(self, *args: tfgnn.GraphTensor) -> runner.Predictions:
    """Apply a readout head for use with triplet contrastive loss.

    Args:
      *args: A tuple of (anchor, positive_sample, negative_sample)
        `tfgnn.GraphTensor`s.

    Returns:
      The positive and negative distance embeddings for triplet loss as produced
      by the implementing subclass.
    """
    anchor, positive_sample, negative_sample = args

    if not tfgnn.is_graph_tensor(anchor):
      raise ValueError(f"Expected a `GraphTensor` input (got {anchor})")
    if not tfgnn.is_graph_tensor(positive_sample):
      raise ValueError(
          f"Expected a `GraphTensor` input (got {positive_sample})"
      )
    if not tfgnn.is_graph_tensor(negative_sample):
      raise ValueError(
          f"Expected a `GraphTensor` input (got {negative_sample})"
      )
    if isinstance(tf.distribute.get_strategy(), tf.distribute.TPUStrategy):
      raise AssertionError(
          "Contrastive learning tasks do not support TPU (see b/269648832)."
      )

    # Clean representations.
    x_positive = tf.keras.layers.Layer(name=self._representations_layer_name)(
        self._readout(positive_sample)
    )
    x_anchor = self._readout(anchor)
    # Corrupted representations.
    x_corrupted = self._readout(negative_sample)
    if self._projector:
      x_positive = self._projector(x_positive)
      x_anchor = self._projector(x_anchor)
      x_corrupted = self._projector(x_corrupted)
    outputs = tf.stack((x_anchor, x_positive, x_corrupted), axis=1)
    return self.make_contrastive_layer()(outputs)

  def losses(self) -> runner.Losses:
    def loss_fn(_, x):
      positive_distance, negative_distance = tf.unstack(x, axis=1)
      return losses.triplet_loss(
          positive_distance,
          negative_distance,
          margin=self._margin,
      )

    return loss_fn

  def metrics(self) -> runner.Metrics:
    return (metrics.TripletLossMetrics(),)
