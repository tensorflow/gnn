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
from typing import Callable, Optional, Sequence, Tuple

import tensorflow as tf
import tensorflow_gnn as tfgnn

Field = tfgnn.Field
GraphTensor = tfgnn.GraphTensor


class DeepGraphInfomax:
  """Deep Graph Infomax.

  Deep Graph Infomax is an unsupervised loss that attempts to learn a bilinear
  layer capable of discriminating between positive examples (any input
  `GraphTensor`) and negative examples (an input `GraphTensor` but with shuffled
  features: this implementation shuffles features across the components of a
  scalar `GraphTensor`).

  Deep Graph Infomax is particularly useful in unsupervised tasks that wish to
  learn latent representations informed primarily by a nodes neighborhood
  attributes (vs. its structure).

  This task can adapt a `tf.keras.Model` with single `GraphTensor` input and a
  single `GraphTensor`  output. The adapted `tf.keras.Model` head is for binary
  classification on the pseudo-labels (positive or negative) for Deep Graph
  Infomax.

  For more information, see: https://arxiv.org/abs/1809.10341.
  """

  def __init__(self,
               node_set_name: str,
               *,
               state_name: str = tfgnn.HIDDEN_STATE,
               seed: Optional[int] = None):
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
      A `tf.keras.Model` with output logits for Deep Graph Infomax, i.e.: a
      positive logit and a negative logit for each example in a batch.
    """
    if not tfgnn.is_graph_tensor(model.input):
      raise ValueError(f"Expected a GraphTensor, received {model.input}")

    if not tfgnn.is_graph_tensor(model.output):
      raise ValueError(f"Expected a GraphTensor, received {model.output}")

    # Positive activations: readout
    readout = tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name=self._node_set_name,
        feature_name=self._state_name,
        name="embeddings")(model.output)

    # A submodel with DeepGraphInfomax embeddings only as output
    submodel = tf.keras.Model(
        model.input,
        readout,
        name="DeepGraphInfomaxEmbeddings")
    pactivations = submodel(submodel.input)

    # Negative activations: shuffling, model application and readout
    shuffled = tfgnn.shuffle_features_globally(model.input)
    nactivations = tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name=self._node_set_name,
        feature_name=self._state_name)(model(shuffled))

    # Summary and bilinear layer
    summary = tf.math.reduce_mean(pactivations, axis=0, keepdims=True)
    bilinear = tf.keras.layers.Dense(summary.get_shape()[-1], use_bias=False)

    # Positive and negative logits
    plogits = tf.matmul(pactivations, bilinear(summary), transpose_b=True)
    nlogits = tf.matmul(nactivations, bilinear(summary), transpose_b=True)

    # Combined logits
    logits = tf.keras.layers.Concatenate(name="logits")((plogits, nlogits))

    return tf.keras.Model(model.input, logits)

  def preprocess(self, gt: GraphTensor) -> Tuple[GraphTensor, Field]:
    """Create labels--i.e., (positive, negative)--for Deep Graph Infomax."""
    num_components = gt.num_components
    y = tf.tile(tf.constant([[1, 0]], dtype=tf.int32), (num_components, 1))
    return gt, y

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    """Sparse categorical crossentropy loss."""
    return (tf.keras.losses.BinaryCrossentropy(from_logits=True),)

  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    """Sparse categorical metrics."""
    return (tf.keras.metrics.BinaryCrossentropy(from_logits=True),
            tf.keras.metrics.BinaryAccuracy())
