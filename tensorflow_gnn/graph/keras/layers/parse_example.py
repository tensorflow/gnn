"""The Keras wrapper for tfgnn.parse_example() and related functionality."""

import tensorflow as tf

from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import graph_tensor_io as io


# Function dispatch does not work for extension types outside TF (b/205710036)
# so this needs an explicit wrapper for use in the Keras functional API.
@tf.keras.utils.register_keras_serializable(package="GNN")
class ParseExample(tf.keras.layers.Layer):
  """Applies tfgnn.parse_example(graph_tensor_spec, _) to a batch of strings."""

  def __init__(self, graph_tensor_spec: gt.GraphTensorSpec, **kwargs):
    super().__init__(**kwargs)
    self._graph_tensor_spec = graph_tensor_spec

  def get_config(self):
    return dict(graph_tensor_spec=self._graph_tensor_spec,
                **super().get_config())

  def call(self, inputs):
    return io.parse_example(self._graph_tensor_spec, inputs)


@tf.keras.utils.register_keras_serializable(package="GNN")
class ParseSingleExample(tf.keras.layers.Layer):
  """Applies tfgnn.parse_single_example(graph_tensor_spec, _)."""

  def __init__(self, graph_tensor_spec: gt.GraphTensorSpec, **kwargs):
    super().__init__(**kwargs)
    self._graph_tensor_spec = graph_tensor_spec

  def get_config(self):
    return dict(graph_tensor_spec=self._graph_tensor_spec,
                **super().get_config())

  def call(self, inputs):
    return io.parse_single_example(self._graph_tensor_spec, inputs)
