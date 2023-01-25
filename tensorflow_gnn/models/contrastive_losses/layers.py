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
"""Perturbation layers."""
from typing import Optional

import tensorflow as tf
import tensorflow_gnn as tfgnn

_PACKAGE = "GNN>models>contrastive_losses"


@tf.keras.utils.register_keras_serializable(package=_PACKAGE)
class ShuffleFeaturesGlobally(tf.keras.layers.Layer):
  """Shuffles input `GraphTensor` features globally.

  Please see `tfgnn.shuffle_features_globally(...)` for more details.
  """

  def __init__(self, *, seed: Optional[int] = None, **kwargs):
    """Captures arguments for `call`.

    Args:
      seed: An optional random seed for `shuffle_features_globally`, used for
        deterministic perturbations.
      **kwargs: Additional keyword arguments.
    """
    super().__init__(**kwargs)
    self._seed = seed

  def get_config(self):
    return dict(seed=self._seed, **super().get_config())

  def call(self, inputs: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
    return tfgnn.shuffle_features_globally(inputs, seed=self._seed)
