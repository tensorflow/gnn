# Copyright 2023 The TensorFlow GNN Authors. All Rights Reserved.
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
"""Label readout helpers."""
from __future__ import annotations

import tensorflow as tf
import tensorflow_gnn as tfgnn


@tf.keras.utils.register_keras_serializable(package="GNN")
class ContextLabelFn(tf.keras.layers.Layer):
  """Reads out a context `tfgnn.Field`."""

  def __init__(self, feature_name: str, **kwargs):
    super().__init__(**kwargs)
    self._feature_name = feature_name

  def call(
      self,
      inputs: tfgnn.GraphTensor) -> tuple[tfgnn.GraphTensor, tfgnn.Field]:
    if not tfgnn.is_graph_tensor(inputs):
      raise ValueError(f"Expected `GraphTensor` inputs (got {inputs})")
    y = inputs.context[self._feature_name]
    x = inputs.remove_features(context=(self._feature_name,))
    return x, y

  def  get_config(self):
    return dict(feature_name=self._feature_name, **super().get_config())


@tf.keras.utils.register_keras_serializable(package="GNN")
class RootNodeLabelFn(tf.keras.layers.Layer):
  """Reads out a root node `tfgnn.Field`."""

  def __init__(
      self,
      node_set_name: tfgnn.NodeSetName,
      *,
      feature_name: tfgnn.FieldName = tfgnn.HIDDEN_STATE,
      **kwargs):
    super().__init__(**kwargs)
    self._node_set_name = node_set_name
    self._feature_name = feature_name

  def call(
      self,
      inputs: tfgnn.GraphTensor) -> tuple[tfgnn.GraphTensor, tfgnn.Field]:
    y = tfgnn.gather_first_node(
        inputs,
        self._node_set_name,
        feature_name=self._feature_name)
    node_sets = {self._node_set_name: (self._feature_name,)}
    x = inputs.remove_features(node_sets=node_sets)
    return x, y

  def  get_config(self):
    return dict(
        node_set_name=self._node_set_name,
        feature_name=self._feature_name,
        **super().get_config())
