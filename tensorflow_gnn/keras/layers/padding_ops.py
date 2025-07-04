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

from typing import Optional

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

  This layer can be restored from config by `tf.keras.models.load_model()`
  when saved as part of a Keras model using `save_format="tf"`.
  Serialization to a Keras model config requires the `sizes_constraints` to
  contain Python integers or eager Tensors, not symbolic Tensors.

  Init args:
    sizes_constraints: Total sizes for each graph piece, forwarded to
      `tfgnn.pad_to_total_sizes()`.
    return_mask: Controls whether the padding mask is returned as a
      second output (see below); true by default.
    mask_output_feature_name: Optionally, the name of a context feature that
      will be set in the output graph to the padding mask.
    validate: Forwarded to `tfgnn.pad_to_total_sizes()`.

  Call args:
    graph: A scalar `tfgnn.GraphTensor` to be padded

  Call returns:
    Either just the `padded_graph` or a tuple `(padded_graph, padding_mask)`,
    depending on `return_mask`. The padding mask is a boolean tensor of shape
    `[num_components]` on which the i-th element is `True` for the components
    present in the input graph and `False` for the components added for padding.
    If `mask_output_feature_name` is set, the padding mask is stored under that
    name in the context features of the padded graph.
  """

  def __init__(self,
               sizes_constraints: preprocessing_common.SizeConstraints,
               *,
               return_mask: bool = True,
               mask_output_feature_name: Optional[str] = None,
               validate: bool = True,
               **kwargs):
    super().__init__(**kwargs)
    self._sizes_constraints = sizes_constraints
    self._return_mask = return_mask
    self._mask_output_feature_name = mask_output_feature_name
    self._validate = validate

  def get_config(self):
    try:
      int_constraints = tf.nest.map_structure(int, self._sizes_constraints)
    except TypeError as e:
      raise NotImplementedError(  # Let SavedModel export skip this gracefully.
          "get_config() requires sizes_constraints convertible to int.") from e
    return dict(
        sizes_constraints=int_constraints._asdict(),
        return_mask=self._return_mask,
        mask_output_feature_name=self._mask_output_feature_name,
        validate=self._validate,
        **super().get_config())

  @classmethod
  def from_config(cls, config):
    sizes_constraints = preprocessing_common.SizeConstraints(
        **config.pop("sizes_constraints"))
    return cls(sizes_constraints=sizes_constraints, **config)

  def call(self, graph):
    padded_graph, component_mask = padding_ops.pad_to_total_sizes(
        graph, self._sizes_constraints, validate=self._validate)
    if self._mask_output_feature_name:
      context_features = padded_graph.context.get_features_dict()
      context_features[self._mask_output_feature_name] = component_mask
      padded_graph = padded_graph.replace_features(context=context_features)
    if self._return_mask:
      return padded_graph, component_mask
    else:
      return padded_graph
