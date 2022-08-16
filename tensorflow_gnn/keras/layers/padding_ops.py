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
"""Keras layer types for padding ops."""

import tensorflow as tf

from tensorflow_gnn.graph import padding_ops
from tensorflow_gnn.graph import preprocessing_common


@tf.keras.utils.register_keras_serializable(package="GNN")
class PadToTotalSizes(tf.keras.layers.Layer):
  """Applies tfgnn.pad_to_total_sizes() to a GraphTensor.

  This Keras layer maps a GraphTensor to a GraphTensor by calling
  `tfgnn.pad_to_total_sizes()` with the additional arguments, notably
  `sizes_constraints`, passed at initialization time. See that function
  for detailed documentation.

  Serialization to a Keras model config requires the `sizes_constraints` to
  contain Python integers or eager Tensors, not symbolic Tensors.
  """

  def __init__(self,
               sizes_constraints: preprocessing_common.SizeConstraints,
               *,
               validate: bool = True,
               **kwargs):
    super().__init__(**kwargs)
    self._sizes_constraints = sizes_constraints
    self._validate = validate

  def get_config(self):
    try:
      int_constraints = tf.nest.map_structure(int, self._sizes_constraints)
    except TypeError as e:
      raise NotImplementedError(  # Let SavedModel export skip this gracefully.
          "get_config() requires sizes_constraints convertible to int.") from e
    return dict(
        sizes_constraints=int_constraints._asdict(),
        validate=self._validate,
        **super().get_config())

  @classmethod
  def from_config(cls, config):
    sizes_constraints = preprocessing_common.SizeConstraints(
        **config.pop("sizes_constraints"))
    return cls(sizes_constraints=sizes_constraints, **config)

  def call(self, graph):
    return padding_ops.pad_to_total_sizes(
        graph, self._sizes_constraints, validate=self._validate)
