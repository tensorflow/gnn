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
"""ConfigDict for VanillaMPNN."""

from ml_collections import config_dict
import tensorflow as tf
from tensorflow_gnn.models.vanilla_mpnn import layers


def graph_update_get_config_dict() -> config_dict.ConfigDict:
  """Returns ConfigDict for graph_update_from_config_dict() with defaults."""
  # LINT.IfChange(graph_update_get_config_dict)
  cfg = config_dict.ConfigDict()
  cfg.units = config_dict.placeholder(int)  # Sets type to Optional[int].
  cfg.message_dim = config_dict.placeholder(int)
  cfg.receiver_tag = config_dict.placeholder(int)
  cfg.reduce_type = "sum"
  cfg.l2_regularization = 0.0
  cfg.dropout_rate = 0.0
  cfg.use_layer_normalization = False
  cfg.lock()
  # LINT.ThenChange(./layers.py:VanillaMPNNGraphUpdate_args)
  return cfg


def graph_update_from_config_dict(
    cfg: config_dict.ConfigDict) -> tf.keras.layers.Layer:
  """Returns a VanillaMPNNGraphUpdate initialized from `cfg`.

  Args:
    cfg: A `ConfigDict` with the fields defined by
      `graph_update_get_config_dict()`. All fields with non-`None` values are
      used as keyword arguments for initializing and returning a
      `VanillaMPNNGraphUpdate` object. For the required arguments of
      `VanillaMPNNGraphUpdate.__init__`, users must set a value in `cfg` before
      passing it here.

  Returns:
    A new `VanillaMPNNGraphUpdate` object.

  Raises:
    TypeError: if `cfg` fails to supply a required argument for
    `VanillaMPNNGraphUpdate.__init__`.
  """
  kwargs = {k: v for k, v in cfg.items() if v is not None}
  return layers.VanillaMPNNGraphUpdate(**kwargs)
