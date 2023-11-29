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
"""Binary to create a TensorFlow GNN saved model."""
from collections.abc import Sequence

from absl import app
from absl import flags
import tensorflow as tf
from tensorflow_gnn import lite as tfgnn
from tensorflow_gnn.runner.utils import model_export

_FILEPATH = flags.DEFINE_string(
    "filepath",
    None,
    "Path where to save the model.",
    required=True,
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  source = tf.keras.Input([], dtype=tf.int32, name="source")
  target = tf.keras.Input([], dtype=tf.int32, name="target")
  h = tf.keras.Input([4], name="hidden_state")

  gt = tfgnn.homogeneous(source, target, node_features={tfgnn.HIDDEN_STATE: h})
  def fn(inputs, **unused_kwargs):
    return tf.keras.layers.Dense(
        2,
        kernel_initializer="ones")(inputs[tfgnn.HIDDEN_STATE])
  outputs = tfgnn.keras.layers.MapFeatures(None, fn, None)(gt)
  outputs = tfgnn.keras.layers.Pool(
      tfgnn.CONTEXT,
      "mean",
      node_set_name="nodes")(outputs)
  model = tf.keras.Model((source, target, h), outputs)

  model_export.export_model(model, _FILEPATH.value)


if __name__ == "__main__":
  app.run(main)
