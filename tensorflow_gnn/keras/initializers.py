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
"""Functionality related to Keras Initializer objects."""

import tensorflow as tf


def clone_initializer(initializer):
  """Clones an initializer to ensure a new default seed.

  Users can specify initializers for trainable weights by `Initializer` objects
  or various other types understood by `tf.keras.initializers.get()`, namely
  `str` with the name, `dict` with the config, or `None`.

  As of TensorFlow 2.10, `Initializer` objects are stateless and fix their
  random seed (even if not explicitly specified) at creation time, so that
  all calls to them return the same sequence of numbers. To achieve
  independent initializations of the various model weights, user-specified
  initializers must be cloned for each weight before passing them to Keras.
  This way, each of them gets a separate seed (unless explicitly overriden).

  This helper function clones `Initializer` obejcts and passes through all other
  forms of specifying an initializer. TF-GNN's modeling code applies it before
  passing user-specified initaializers to Keras. User code that calls Keras
  directly and passes an initializer more than once is advised to wrap it with
  this function as well.

  Example:

  ```
  def build_graph_update(units, initializer):
    def dense(units):  # Called for multiple node sets and edge sets.
      tf.keras.layers.Dense(
          units, activation="relu",
          kernel_initializer=tfgnn.keras.clone_initializer(initializer))

    gnn_builder = tfgnn.keras.ConvGNNBuilder(
        lambda edge_set_name, receiver_tag: tfgnn.keras.layers.SimpleConv(
            dense(units), receiver_tag=receiver_tag),
        lambda node_set_name: tfgnn.keras.layers.NextStateFromConcat(
            dense(units)),
        receiver_tag=tfgnn.TARGET)
  return gnn_builder.Convolve()
  ```

  Args:
    initializer: An initializer specification as understood by Keras.

  Returns:
    A new `Initializer` object with the same config as `initializer`,
    or `initializer` unchanged if if was not an `Initializer` object.
  """
  if isinstance(initializer, tf.keras.initializers.Initializer):
    return initializer.__class__.from_config(initializer.get_config())
  return initializer
