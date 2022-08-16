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
"""Model templates."""
from typing import Callable, Sequence

import tensorflow as tf
import tensorflow_gnn as tfgnn


class ModelFromInitAndUpdates:
  """A model_fn for Trainer that concats `init` and a sequence of `updates`.

  This class is initialized with a user-supplied callback for hidden state
  initialization (e.g., a `tfgnn.keras.layers.MapFeatures` layer) and a sequence
  of user-supplied callbacks for hidden state updates (e.g., a list of
  GraphUpdate layers). When the Trainer calls it on the `graph_tensor_spec` for
  the preprocessed input, it builds a Keras Model that accepts a Keras
  GraphTensor of that type and performs hidden state initialization and
  state updates in order.

  This class is mostly useful as syntactic sugar for a Gin config, to specify
  a model_fn by object initialization in Gin instead of spelling out
  `def model_fn ...` in Python.
  """

  def __init__(self, *,
               init: Callable[..., tfgnn.GraphTensor],
               updates: Sequence[Callable[..., tfgnn.GraphTensor]]):
    """Initializes with callbacks for state initialization and updates.

    Args:
      init: A callable that maps a scalar Keras GraphTensor matching
        `graph_tensor_spec` with preprocessed model inputs to a Keras
        GraphTensor with a `tfgnn.HIDDEN_STATE` feature on each node set,
        suitable for use with `updates`.
      updates: A list of callables that each map a Keras GraphTensor with a
        `tfgnn.HIDDEN_STATE` feature on each node set to another such
        Keras GraphTensor with updated hidden states.
    """
    self._init = init
    self._updates = updates

  def __call__(self,
               graph_tensor_spec: tfgnn.GraphTensorSpec) -> tf.keras.Model:
    """Returns a model for inputs matching `graph_tensor_spec`."""
    graph = inputs = tf.keras.layers.Input(type_spec=graph_tensor_spec)
    graph = self._init(graph)
    for update in self._updates:
      graph = update(graph)
    return tf.keras.Model(inputs, graph)
