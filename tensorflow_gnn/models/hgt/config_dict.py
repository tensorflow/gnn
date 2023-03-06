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
from tensorflow_gnn.models.hgt import layers


def graph_update_get_config_dict() -> config_dict.ConfigDict:
  """Returns ConfigDict for graph_update_from_config_dict() with defaults."""
  cfg = config_dict.ConfigDict()
  # LINT.IfChange(graph_update_get_config_dict)
  cfg.num_heads = config_dict.placeholder(int)  # Sets type to Optional[int].
  cfg.per_head_channels = config_dict.placeholder(int)
  cfg.receiver_tag = config_dict.placeholder(int)
  cfg.use_weighted_skip = config_dict.placeholder(bool)
  cfg.dropout_rate = config_dict.placeholder(float)
  cfg.use_layer_norm = config_dict.placeholder(bool)
  cfg.use_bias = config_dict.placeholder(bool)
  cfg.activation = config_dict.placeholder(str)
  # LINT.ThenChange(./layers.py:HGTGraphUpdate_args)
  cfg.lock()
  return cfg


def graph_update_from_config_dict(
    cfg: config_dict.ConfigDict) -> tf.keras.layers.Layer:
  """Returns a HGTGraphUpdate initialized from `cfg`.

  Args:
    cfg: A `ConfigDict` with the fields defined by
      `graph_update_get_config_dict()`. All fields with non-`None` values are
      used as keyword arguments for initializing and returning a
      `HGTGraphUpdate` object. For the required arguments of
      `HGTGraphUpdate.__init__`, users must set a value in
      `cfg` before passing it here.

  Returns:
    A new `HGTGraphUpdate` object.

  Raises:
    TypeError: if `cfg` fails to supply a required argument for
    `HGTGraphUpdate.__init__`.
  """
  kwargs = {k: v for k, v in cfg.items() if v is not None}
  return layers.HGTGraphUpdate(**kwargs)
