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
"""Deep Graph Infomax layers (see: https://arxiv.org/abs/1809.10341)."""
from __future__ import annotations

import tensorflow as tf

_PACKAGE = "GNN>models>contrastive_losses"


@tf.keras.utils.register_keras_serializable(package=_PACKAGE)
class DeepGraphInfomaxLogits(tf.keras.layers.Layer):
  """Computes clean and corrupted logits for Deep Graph Infomax (DGI)."""

  def build(self, input_shape: tf.TensorShape) -> None:
    """Buils a bilinear layer."""
    if not isinstance(input_shape, tf.TensorShape):
      raise ValueError(f"Expected `TensorShape` (got {type(input_shape)})")
    units = input_shape.as_list()[-1]
    if units is None:
      raise ValueError(f"Expected a defined inner dimension (got {units})")
    # Bilinear layer.
    self._bilinear = tf.keras.layers.Dense(units, use_bias=False)

  def call(self, inputs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Computes clean and corrupted logits for DGI.

    Args:
      inputs: A stacked tensor with (clean, corrupted) representations.

    Returns:
      A concatenated (clean, corrupted) logits, respectively.
    """
    x_clean, x_corrupted = tf.unstack(inputs, axis=1)
    # Summary.
    summary = tf.math.reduce_mean(x_clean, axis=0, keepdims=True)
    # Clean logits.
    logits_clean = tf.matmul(x_clean, self._bilinear(summary), transpose_b=True)
    # Corrupted logits.
    logits_corrupted = tf.matmul(
        x_corrupted,
        self._bilinear(summary),
        transpose_b=True)
    return tf.keras.layers.Concatenate()((logits_clean, logits_corrupted))
