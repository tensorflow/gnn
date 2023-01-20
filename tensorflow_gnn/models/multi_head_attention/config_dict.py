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
"""ConfigDict for Multi-Head Attention."""

from ml_collections import config_dict
import tensorflow as tf
from tensorflow_gnn.models.multi_head_attention import layers


def graph_update_get_config_dict() -> config_dict.ConfigDict:
  """Returns ConfigDict for graph_update_from_config_dict() with defaults."""
  # Keep in sync with default args of
  # MultiHeadAttentionMPNNGraphUpdate.__init__.
  cfg = config_dict.ConfigDict()
  cfg.units = config_dict.placeholder(int)  # Sets type to Optional[int].
  cfg.message_dim = config_dict.placeholder(int)
  cfg.num_heads = config_dict.placeholder(int)
  cfg.receiver_tag = config_dict.placeholder(int)
  cfg.l2_regularization = 0.0
  cfg.edge_dropout_rate = 0.0
  cfg.state_dropout_rate = 0.0
  cfg.conv_activation = "relu"
  cfg.activation = "relu"
  cfg.lock()
  return cfg


def graph_update_from_config_dict(
    cfg: config_dict.ConfigDict) -> tf.keras.layers.Layer:
  """Returns a MultiHeadAttentionMPNNGraphUpdate initialized from `cfg`.

  Args:
    cfg: A `ConfigDict` with the fields defined by
      `graph_update_get_config_dict()`. All fields with non-`None` values are
      used as keyword arguments for initializing and returning a
      `MultiHeadAttentionMPNNGraphUpdate` object. For the required arguments of
      `MultiHeadAttentionMPNNGraphUpdate.__init__`, users must set a value in
      `cfg` before passing it here.

  Returns:
    A new `MultiHeadAttentionMPNNGraphUpdate` object.

  Raises:
    TypeError: if `cfg` fails to supply a required argument for
    `MultiHeadAttentionMPNNGraphUpdate.__init__`.
  """
  kwargs = {k: v for k, v in cfg.items() if v is not None}
  return layers.MultiHeadAttentionMPNNGraphUpdate(**kwargs)
