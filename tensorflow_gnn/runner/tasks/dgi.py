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
"""An implementation of Deep Graph Infomax: https://arxiv.org/abs/1809.10341."""
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import tensorflow as tf
import tensorflow_gnn as tfgnn


@tf.keras.utils.register_keras_serializable(package="GNN")
class AddLossDeepGraphInfomax(tf.keras.layers.Layer):
  """"A bilinear layer with losses and metrics for Deep Graph Infomax."""

  def __init__(self, units: int):
    """Builds the bilinear layer weights.

    Args:
      units: Units for the bilinear layer.
    """
    super().__init__()
    self._bilinear = tf.keras.layers.Dense(units, use_bias=False)

  def get_config(self) -> Mapping[Any, Any]:
    """Returns the config of the layer.

    A layer config is a Python dictionary (serializable) containing the
    configuration of a layer. The same layer can be reinstantiated later
    (without its trained weights) from this configuration.
    """
    return dict(
        units=self._bilinear.units,
        **super().get_config())

  def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    """Returns clean representations after adding a Deep Graph Infomax loss.

    Clean representations are the unmanipulated, original model output.

    Args:
      inputs: A tuple of (clean, corrupted) representations for Deep Graph
        Infomax.

    Returns:
      The clean representations: the first item of the `inputs` tuple.
    """
    y_clean, y_corrupted = inputs
    # Summary
    summary = tf.math.reduce_mean(y_clean, axis=0, keepdims=True)
    # Clean losses and metrics
    logits_clean = tf.matmul(y_clean, self._bilinear(summary), transpose_b=True)
    self.add_loss(tf.keras.losses.BinaryCrossentropy(
        from_logits=True,
        name="binary_crossentropy_clean")(
            tf.ones_like(logits_clean),
            logits_clean))
    self.add_metric(
        tf.keras.metrics.binary_crossentropy(
            tf.ones_like(logits_clean),
            logits_clean,
            from_logits=True),
        name="binary_crossentropy_clean")
    self.add_metric(
        tf.keras.metrics.binary_accuracy(
            tf.ones_like(logits_clean),
            logits_clean),
        name="binary_accuracy_clean")
    # Corrupted losses and metrics
    logits_corrupted = tf.matmul(
        y_corrupted,
        self._bilinear(summary),
        transpose_b=True)
    self.add_loss(tf.keras.losses.BinaryCrossentropy(
        from_logits=True,
        name="binary_crossentropy_corrupted")(
            tf.zeros_like(logits_corrupted),
            logits_corrupted))
    self.add_metric(
        tf.keras.metrics.binary_crossentropy(
            tf.zeros_like(logits_corrupted),
            logits_corrupted,
            from_logits=True),
        name="binary_crossentropy_corrupted")
    self.add_metric(
        tf.keras.metrics.binary_accuracy(
            tf.zeros_like(logits_corrupted),
            logits_corrupted),
        name="binary_accuracy_corrupted")
    return y_clean


class DeepGraphInfomax:
  """A Task for training with the Deep Graph Infomax loss.

  Deep Graph Infomax is an unsupervised loss that attempts to learn a bilinear
  layer capable of discriminating between positive examples (any input
  `GraphTensor`) and negative examples (an input `GraphTensor` but with shuffled
  features: this implementation shuffles features across the components of a
  scalar `GraphTensor`).

  Deep Graph Infomax is particularly useful in unsupervised tasks that wish to
  learn latent representations informed primarily by a nodes neighborhood
  attributes (vs. its structure).

  This task can adapt a `tf.keras.Model` with a single, scalar `GraphTensor`
  input and a single, scalar `GraphTensor`  output. The adapted `tf.keras.Model`
  head has--as its output--any latent, root node (according to `node_set_name`
  and `state_name`) represenations. The unsupervised loss is added to the model
  by adding a Layer that calls `tf.keras.Layer.add_loss().`

  For more information, see: https://arxiv.org/abs/1809.10341.
  """

  def __init__(self,
               node_set_name: str,
               *,
               state_name: str = tfgnn.HIDDEN_STATE,
               seed: Optional[int] = None):
    """Captures arguments for the task.

    Args:
      node_set_name: The node set for activations.
      state_name: The state name of any activations.
      seed: A seed for corrupted representations.
    """
    self._state_name = state_name
    self._node_set_name = node_set_name
    self._seed = seed

  def adapt(self, model: tf.keras.Model) -> tf.keras.Model:
    """Adapt a `tf.keras.Model` for Deep Graph Infomax.

    The input `tf.keras.Model` must have a single `GraphTensor` input and a
    single `GraphTensor` output.

    Args:
      model: A `tf.keras.Model` to be adapted.

    Returns:
      A `tf.keras.Model` with clean representation output from Deep Graph
      Infomax. The unsupervised loss is added to the model
      via `tf.keras.Model.add_loss.`
    """
    if not tfgnn.is_graph_tensor(model.input):
      raise ValueError(f"Expected a GraphTensor, received {model.input}")

    if not tfgnn.is_graph_tensor(model.output):
      raise ValueError(f"Expected a GraphTensor, received {model.output}")

    # Clean representations: readout
    y_clean = tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name=self._node_set_name,
        feature_name=self._state_name)(model.output)

    # Corrupted representations: shuffling, model application and readout
    shuffled = tfgnn.shuffle_features_globally(model.input)
    y_corrupted = tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name=self._node_set_name,
        feature_name=self._state_name)(model(shuffled))

    return tf.keras.Model(
        model.input,
        AddLossDeepGraphInfomax(
            y_clean.get_shape()[-1])((y_clean, y_corrupted)))

  def preprocess(self, gt: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
    """Returns the input GraphTensor."""
    return gt

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    """Returns an empty losses tuple.

    Loss signatures are according to `tf.keras.losses.Loss,` here: no losses
    are returned because they have been added to the model via
    `tf.keras.Layer.add_loss.`
    """
    return tuple()

  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    """Returns an empty metrics tuple.

    Metric signatures are according to `tf.keras.metrics.Metric,` here: no
    metrics are returned because they have been added to the model via
    `tf.keras.Layer.add_metric.`
    """
    return tuple()
