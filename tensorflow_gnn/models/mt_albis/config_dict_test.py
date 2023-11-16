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
"""Tests for config_dict."""

import json
from typing import Mapping

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models.mt_albis import config_dict as mt_albis_config_dict
from tensorflow_gnn.models.mt_albis import layers


class ConfigDictTest(tf.test.TestCase):

  def test_graph_update_defaults(self):
    units = 32
    message_dim = 16
    receiver_tag = tfgnn.SOURCE

    cfg = mt_albis_config_dict.graph_update_get_config_dict()
    cfg.units = units
    cfg.message_dim = message_dim
    cfg.receiver_tag = receiver_tag
    actual = mt_albis_config_dict.graph_update_from_config_dict(cfg)

    expected = layers.MtAlbisGraphUpdate(
        units=units, message_dim=message_dim, receiver_tag=receiver_tag)

    self.assertEqual(to_model_config(actual), to_model_config(expected))

  def test_graph_update_custom(self):
    units = 64
    message_dim = 32
    receiver_tag = tfgnn.TARGET
    attention_type = "multi_head"
    attention_edge_set_names = ("cites",)
    node_set_names = ("paper",)

    cfg = mt_albis_config_dict.graph_update_get_config_dict()
    cfg.units = units
    cfg.message_dim = message_dim
    cfg.receiver_tag = receiver_tag
    cfg.attention_type = attention_type
    cfg.attention_edge_set_names = attention_edge_set_names
    actual = mt_albis_config_dict.graph_update_from_config_dict(
        cfg, node_set_names=node_set_names)

    expected = layers.MtAlbisGraphUpdate(
        units=units,
        message_dim=message_dim,
        receiver_tag=receiver_tag,
        attention_type=attention_type,
        attention_edge_set_names=attention_edge_set_names,
        node_set_names=node_set_names,
    )
    self.assertEqual(to_model_config(actual), to_model_config(expected))


# TODO(b/265776928): De-duplicate the multiple copies of this test helper.
def to_model_config(layer: tf.keras.layers.Layer):
  """Returns a parsed model config for `layer`, without `"name"` fields."""
  # Need a full model to serialize *recursively*.
  model = tf.keras.Sequential([layer])
  # Subobjects are only built in the first call.
  _ = model(_make_test_graph_loop())
  model_config = json.loads(model.to_json())
  # The names of layers are uniquified and impede the hparam comparison.
  return _remove_names(model_config)


def _remove_names(obj):
  """Returns parsed JSON `obj` without dict entries called "name"."""
  if isinstance(obj, Mapping):
    return {k: _remove_names(v) for k, v in obj.items() if k != "name"}
  elif isinstance(obj, (list, tuple)):
    return type(obj)([_remove_names(v) for v in obj])
  else:
    return obj


def _make_test_graph_loop():
  """Returns a scalar GraphTensor with one node and one egde."""
  return tfgnn.GraphTensor.from_pieces(
      node_sets={
          "paper": tfgnn.NodeSet.from_fields(
              sizes=tf.constant([1]),
              features={tfgnn.HIDDEN_STATE: tf.constant([[1.]])}),
          "author": tfgnn.NodeSet.from_fields(
              sizes=tf.constant([1]),
              features={tfgnn.HIDDEN_STATE: tf.constant([[1.]])}),
      },
      edge_sets={
          "writes": tfgnn.EdgeSet.from_fields(
              sizes=tf.constant([1]),
              adjacency=tfgnn.Adjacency.from_indices(
                  ("author", tf.constant([0])),
                  ("paper", tf.constant([0])))),
          "cites": tfgnn.EdgeSet.from_fields(
              sizes=tf.constant([1]),
              adjacency=tfgnn.Adjacency.from_indices(
                  ("paper", tf.constant([0])),
                  ("paper", tf.constant([0])))),
      })


if __name__ == "__main__":
  tf.test.main()
