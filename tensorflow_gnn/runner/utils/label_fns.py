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
import tensorflow as tf
import tensorflow_gnn as tfgnn


@tf.keras.utils.register_keras_serializable(package="GNN")
class ContextLabelFn(tf.keras.layers.Layer):
  """Reads out a context `tfgnn.Field`."""

  def __init__(self, feature_name: str, **kwargs):
    super().__init__(**kwargs)
    self._feature_name = feature_name

  def call(self, inputs: tfgnn.GraphTensor) -> tfgnn.Field:
    if not tfgnn.is_graph_tensor(inputs):
      raise ValueError(f"Expected `GraphTensor` inputs (got {inputs})")
    return inputs.context[self._feature_name]

  def  get_config(self):
    return dict(feature_name=self._feature_name, **super().get_config())


@tf.keras.utils.register_keras_serializable(package="GNN")
class RootNodeLabelFn(tfgnn.keras.layers.ReadoutFirstNode):
  pass

