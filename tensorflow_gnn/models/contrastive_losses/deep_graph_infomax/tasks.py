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

from collections.abc import Callable, Sequence
from typing import Optional

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn import runner
from tensorflow_gnn.models.contrastive_losses import layers as perturbation_layers
from tensorflow_gnn.models.contrastive_losses.deep_graph_infomax import layers as dgi_layers


class DeepGraphInfomaxTask(runner.Task):
  """A Deep Graph Infomax (DGI) Task."""

  def __init__(
      self,
      node_set_name: str,
      *,
      feature_name: str = tfgnn.HIDDEN_STATE,
      representations_layer_name: Optional[str] = "clean_representations",
      seed: Optional[int] = None):
    self._representations_layer_name = representations_layer_name
    self._node_set_name = node_set_name
    self._feature_name = feature_name
    self._perturber = perturbation_layers.ShuffleFeaturesGlobally(seed=seed)

  def adapt(self, model: tf.keras.Model) -> tf.keras.Model:
    """Adapts an arbitrary Graph Neural Network (GNN) to DGI.

    Args:
      model: A Keras GNN to adapt. The model is expected to have `GraphTensor`
        input and output.

    Returns:
      An adapted Keras GNN whose outputs are DGI (clean, corrupted) logits,
      respectively. Clean representations (those output from `model`) are
      accessible via a submodule (named as the `representations_layer_name` set
      at initialization).

    Raises:
      ValueError: If `model.input` or `model.output` are not `GraphTensor`.
    """
    if not tfgnn.is_graph_tensor(model.input):
      raise ValueError(f"Expected a `GraphTensor` input (got {model.input})")
    if not tfgnn.is_graph_tensor(model.output):
      raise ValueError(f"Expected a `GraphTensor` output (got {model.output})")
    # Readout operation.
    readout = tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name=self._node_set_name,
        feature_name=self._feature_name)
    # Clean representations.
    submodule_clean = tf.keras.Model(
        model.input,
        readout(model.output),
        name=self._representations_layer_name)
    x_clean = submodule_clean(model.input)
    # Corrupted representations.
    x_corrupted = readout(model(self._perturber(model.input)))
    # Clean and Corrupted logits.
    logits = dgi_layers.DeepGraphInfomaxLogits()((x_clean, x_corrupted))
    return tf.keras.Model(model.input, logits)

  def preprocess(
      self,
      gt: tfgnn.GraphTensor) -> tuple[tfgnn.GraphTensor, tfgnn.Field]:
    """Creates labels--i.e., (positive, negative)--for Deep Graph Infomax."""
    y = tf.tile(tf.constant(((1, 0),), dtype=tf.int32), (gt.num_components, 1))
    return gt, y

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (tf.keras.losses.BinaryCrossentropy(from_logits=True),)

  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (tf.keras.metrics.BinaryCrossentropy(from_logits=True),
            tf.keras.metrics.BinaryAccuracy())
