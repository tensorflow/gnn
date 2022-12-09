# Copyright 2022 The TensorFlow GNN Authors. All Rights Reserved.
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
"""The ItemDropout class."""

from typing import Optional

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="GNN")
class ItemDropout(tf.keras.layers.Layer):
  """Dropout of feature values for entire edges, nodes or components.

  This Layer class wraps `tf.keras.layers.Dropout` to perform edge dropout
  or node dropout (or "component dropout", which is rarely useful) on
  Tensors shaped like features of a **scalar** GraphTensor.

  Init args:
    rate: The dropout rate, forwarded to `tf.keras.layers.Dropout`.
    seed: The random seed, forwarded `tf.keras.layers.Dropout`.

  Call args:
    x: A float Tensor of shape `[num_items, *feature_dims]`. This is the shape
      of node features or edge features (or context features) in a *scalar**
      GraphTensor. Across calls, all inputs must have the same known rank.

  Call returns:
    A Tensor `y` with the same shape and dtype as the input `x`.
    In non-training mode, the output is the same as the input: `y == x`.
    In training mode, each row `y[i]` is either zeros (with probability `rate`)
    or a scaled-up copy of the input row: `y[i] = x[i] * 1./(1-rate)`.
    This is similar to ordinary dropout, except all or none of the feature
    values for each item are dropped out.
  """

  def __init__(self,
               rate: float,
               seed: Optional[int] = None,
               **kwargs):
    super().__init__(**kwargs)
    self._rate = rate
    self._seed = seed

  def get_config(self):
    return dict(
        rate=self._rate,
        seed=self._seed,
        **super().get_config())

  def build(self, shape):
    if shape.rank is None or shape.rank < 1:
      raise ValueError(
          "ItemDropout requires inputs of known fixed non-zero rank")
    noise_shape = tf.TensorShape([None] + (shape.rank - 1)*[1])
    self._dropout = tf.keras.layers.Dropout(
        rate=self._rate, noise_shape=noise_shape, seed=self._seed)

  def call(self, inputs):
    if self._dropout.noise_shape.rank != inputs.shape.rank:
      raise ValueError(f"Built for rank {self._dropout.noise_shape.rank}, "
                       f"called with input of rank {inputs.shape.rank}")
    return self._dropout(inputs)
