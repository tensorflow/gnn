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
"""Deep Graph Infomax tasks."""
from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import Optional, Tuple, Union

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn import runner
from tensorflow_gnn.models.contrastive_losses import layers as perturbation_layers
from tensorflow_gnn.models.contrastive_losses import losses
from tensorflow_gnn.models.contrastive_losses.deep_graph_infomax import layers as dgi_layers

Field = tfgnn.Field
GraphTensor = tfgnn.GraphTensor


class _ConstrastiveLossTask(runner.Task, abc.ABC):
  """Base class for unsupervised contrastive representation learning tasks.

  The default `adapt` method implementation shuffles feature across batch
  examples to create positive and negative activations. There are multiple ways
  proposed in the literature to learn representations based on the activations.

  Any subclass must implement `make_contrastive_layer` method, which re-adapts
  the input `tf.keras.Model` to prepare task-specific outputs.

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
      seed: Optional[int] = None):
    self._representations_layer_name = (
        representations_layer_name or "clean_representations"
    )
    self._feature_name = feature_name
    self._node_set_name = node_set_name
    self._seed = seed
    self._perturber = perturbation_layers.ShuffleFeaturesGlobally(seed=seed)

  def adapt(self, model: tf.keras.Model) -> tf.keras.Model:
    """Adapt a `tf.keras.Model` for use with various contrastive losses.

    The input `tf.keras.Model` must have a single `GraphTensor` input and a
    single `GraphTensor` output.

    Args:
      model: A `tf.keras.Model` to be adapted.

    Returns:
      A `tf.keras.Model` with output logits for contrastive loss as produced by
      the implementing subclass.
    """
    if not tfgnn.is_graph_tensor(model.input):
      raise ValueError(f"Expected a `GraphTensor` input (got {model.input})")
    if not tfgnn.is_graph_tensor(model.output):
      raise ValueError(f"Expected a `GraphTensor` output (got {model.output})")

    if isinstance(tf.distribute.get_strategy(), tf.distribute.TPUStrategy):
      raise AssertionError(
          "Contrastive learning tasks do not support TPU (see b/269648832)."
      )

    # Clean representations: readout
    readout = tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name=self._node_set_name,
        feature_name=self._feature_name)
    # Clean representations.
    submodule_clean = tf.keras.Model(
        model.input,
        readout(model.output),
        name=self._representations_layer_name)
    x_clean = submodule_clean(model.input)

    # Corrupted representations: shuffling, model application and readout
    shuffled = self._perturber(model.input)
    x_corrupted = readout(model(shuffled))

    return tf.keras.Model(
        model.input,
        self.make_contrastive_layer()(tf.stack((x_clean, x_corrupted), axis=1)),
    )

  @abc.abstractmethod
  def make_contrastive_layer(self) -> tf.keras.layers.Layer:
    """Returns the layer contrasting clean outputs with the correupted ones."""
    raise NotImplementedError()

  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return tuple()


class DeepGraphInfomaxTask(_ConstrastiveLossTask):
  """A Deep Graph Infomax (DGI) Task."""

  def make_contrastive_layer(self) -> tf.keras.layers.Layer:
    return dgi_layers.DeepGraphInfomaxLogits()

  def preprocess(
      self,
      inputs: GraphTensor) -> Tuple[Optional[GraphTensor], Field]:
    """Creates labels--i.e., (positive, negative)--for Deep Graph Infomax."""
    return None, tf.tile(((1, 0),), (inputs.num_components, 1))

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (tf.keras.losses.BinaryCrossentropy(from_logits=True),)

  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (
        tf.keras.metrics.BinaryCrossentropy(from_logits=True),
        tf.keras.metrics.BinaryAccuracy(),
    )


class BarlowTwinsTask(_ConstrastiveLossTask):
  """A Barlow Twins (BT) Task."""

  def __init__(
      self,
      node_set_name: str,
      *,
      feature_name: str = tfgnn.HIDDEN_STATE,
      representations_layer_name: Optional[str] = None,
      seed: Optional[int] = None,
      lambda_: Optional[Union[tf.Tensor, float]] = None,
      normalize_batch: bool = True):
    super().__init__(
        node_set_name,
        feature_name=feature_name,
        representations_layer_name=representations_layer_name,
        seed=seed)
    self._lambda = lambda_
    self._normalize_batch = normalize_batch

  def make_contrastive_layer(self) -> tf.keras.layers.Layer:
    return tf.keras.layers.Layer()

  def preprocess(
      self,
      inputs: GraphTensor) -> Tuple[Optional[GraphTensor], Field]:
    """Creates unused pseudo-labels."""
    return None, tf.zeros((inputs.num_components, 0), dtype=tf.int32)

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    def loss_fn(_, x):
      return losses.barlow_twins_loss(
          *tf.unstack(x, axis=1),
          lambda_=self._lambda,
          normalize_batch=self._normalize_batch,
      )

    return (loss_fn,)


class VicRegTask(_ConstrastiveLossTask):
  """A VICReg Task."""

  def __init__(
      self,
      node_set_name: str,
      *,
      feature_name: str = tfgnn.HIDDEN_STATE,
      representations_layer_name: Optional[str] = None,
      seed: Optional[int] = None,
      sim_weight: Union[tf.Tensor, float] = 25.,
      var_weight: Union[tf.Tensor, float] = 25.,
      cov_weight: Union[tf.Tensor, float] = 1.):
    super().__init__(
        node_set_name,
        feature_name=feature_name,
        representations_layer_name=representations_layer_name,
        seed=seed)
    self._sim_weight = sim_weight
    self._var_weight = var_weight
    self._cov_weight = cov_weight

  def make_contrastive_layer(self) -> tf.keras.layers.Layer:
    return tf.keras.layers.Layer()

  def preprocess(
      self,
      inputs: GraphTensor) -> Tuple[Optional[GraphTensor], Field]:
    """Creates unused pseudo-labels."""
    return None, tf.zeros((inputs.num_components, 0), dtype=tf.int32)

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    def loss_fn(_, x):
      return losses.vicreg_loss(
          *tf.unstack(x, axis=1),
          sim_weight=self._sim_weight,
          var_weight=self._var_weight,
          cov_weight=self._cov_weight,
      )

    return (loss_fn,)
