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
"""ConfigDict for MtAlbis."""

from ml_collections import config_dict
import tensorflow as tf
from tensorflow_gnn.models.mt_albis import layers


def graph_update_get_config_dict() -> config_dict.ConfigDict:
  """Returns ConfigDict for graph_update_from_config_dict() with defaults."""
  # LINT.IfChange(graph_update_get_config_dict)
  # TODO(b/261835577): What about node_set_names, edge_feature_name,
  # attention_edge_set_names?
  cfg = config_dict.ConfigDict()
  cfg.units = config_dict.placeholder(int)  # Sets type to Optional[int].
  cfg.message_dim = config_dict.placeholder(int)
  cfg.receiver_tag = config_dict.placeholder(int)
  cfg.attention_type = "none"
  cfg.attention_num_heads = 4
  cfg.simple_conv_reduce_type = "mean"
  cfg.simple_conv_use_receiver_state = True
  cfg.state_dropout_rate = 0.0
  cfg.edge_dropout_rate = 0.0
  cfg.l2_regularization = 0.0
  cfg.normalization_type = "layer"
  cfg.batch_normalization_momentum = 0.99
  cfg.next_state_type = "dense"
  cfg.edge_set_combine_type = "concat"
  cfg.lock()
  # LINT.ThenChange(./layers.py:MtAlbisGraphUpdate_args)
  return cfg


def graph_update_from_config_dict(
    cfg: config_dict.ConfigDict) -> tf.keras.layers.Layer:
  """Constructs a MtAlbisGraphUpdate from a ConfigDict.

  Args:
    cfg: A `ConfigDict` with the fields defined by
      `graph_update_get_config_dict()`. All fields with non-`None` values are
      used as keyword arguments for initializing and returning a
      `MtAlbisGraphUpdate` object. For the required arguments of
      `MtAlbisGraphUpdate.__init__`, users must set a value in `cfg` before
      passing it here.

  Returns:
    A new Layer object as returned by `MtAlbisGraphUpdate()`.

  Raises:
    TypeError: if `cfg` fails to supply a required argument for
    `MtAlbisGraphUpdate()`.
  """
  # Turn the ConfigDict into kwargs. Unfilled placeholders show up as `None`,
  # so we omit them here, and leave it to the called function to catch
  # missing required arguments.
  kwargs = {k: v for k, v in cfg.items() if v is not None}
  return layers.MtAlbisGraphUpdate(**kwargs)
