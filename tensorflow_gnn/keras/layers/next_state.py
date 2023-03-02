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
"""The NextState layers.

The NextState layers encapsulate various ways in which the state on an EdgeSet,
a NodeSet or the Context can be computed from its old state (if any) and
various side inputs. Typically, these NextState layers are passed to the
initializer of an EdgeSetUpdate, NodeSetUpdate or ContextUpdate layer.
For example, users of a NodeSetUpdate can choose between a SimpleStateUpdate
with a feed-forward network and a SkipConnectionStateUpdate that adds a
skip connection from the old to the new state.

Users of the TF-GNN library can define their own NextState layers.
Any Keras layer that can be called like the following prototypical examples
can be used as a NextState layer; it is not required to inherit from a
particular base class other than tf.keras.layers.Layer.

The input to a NextState layer is a tuple of three nests of tensors, all shaped
like features of the updated graph piece, and all meant to be included in the
computation of the next state. (The caller needs to pick the right inputs and
pool or broadcast them, as needed, before passing them to a NextState layer.)

The first position in the tuple has the inputs from the updated graph piece
itself, either a single tensor or a dict of tensors keyed by feature names.
It depends on the particular NextState layer whether it distinguishes further
between the tensor(s) with the state of the updated graph piece and other
features of it.

The second and third positions on the tuple hold nests of tensors with inputs
from the other related graph pieces. The exact format varies between edge sets,
node sets and context (see below). For each related graph piece, there is
either a single feature or a dict of features.

```python
class NextStateForEdgeSet(tf.keras.layers.Layer):
  def call(self, inputs: Tuple[
      FieldOrFields, # From the edges themselves.
      Mapping[IncidentNodeTag, FieldOrFields],  # From incident nodes.
      FieldOrFields]]  # From context.
  ) -> FieldOrFields:
    edge_input, incident_node_inputs, context_input = inputs
    raise NotImplementedError()

class NextStateForNodeSet(tf.keras.layers.Layer):
  def call(self, inputs: Tuple[
      FieldOrFields,  # From the nodes themselves.
      Mapping[EdgeSetName, FieldOrFields],  # From the incident edges.
      FieldOrFields]  # From context.
  ) -> FieldOrFields:
    node_input, edge_inputs, context_input = inputs
    raise NotImplementedError()

class NextStateForContext(tf.keras.layers.Layer):
  def call(self, inputs: Tuple[
      FieldOrFields,  # From context itself.
      Mapping[NodeSetName, FieldOrFields],  # From nodes.
      Mapping[EdgeSetName, FieldOrFields]]  # From edges.
  ) -> FieldOrFields:
    context_input, node_inputs, edge_inputs = inputs
    raise NotImplementedError()
```

A NextState layer that only cares about the distinction between the updated
graph piece and all others, but not whether the updated graph piece is
an EdgeSet, a NodeSet or the Context, can conform to all three prototypes
as follows:

```python
class NextState(NextStateForEdgeSet, NextStateForNodeSet, NextStateForContext):
  def call(self, inputs: Tuple[
      FieldOrFields,  # From the updated graph piece.
      FieldsNest, FieldsNest]  # From related graph pieces of other types.
  ) -> FieldOrFields:
    self_inputs, *other_inputs = inputs
    flat_other_inputs = tf.nest.flatten(other_inputs)
    raise NotImplementedError("add implementation here")
```

The names of the prototypical classes above are used in type annotations
for the benefit of human readers, but pytype cannot actually check the
interface requirements beyond being any Keras layer.
"""

from typing import Any, Optional, Tuple

import tensorflow as tf

from tensorflow_gnn.graph import graph_constants as const


# See module docstring.
NextStateForEdgeSet = tf.keras.layers.Layer
NextStateForNodeSet = tf.keras.layers.Layer
NextStateForContext = tf.keras.layers.Layer
NextState = tf.keras.layers.Layer


@tf.keras.utils.register_keras_serializable(package="GNN")
class NextStateFromConcat(tf.keras.layers.Layer):
  """Computes a new state by concatenating inputs and applying a Keras Layer.

  This layer flattens all inputs into a list (forgetting their origin),
  concatenates them and sends them through a user-supplied feed-forward network.

  Init args:
    transformation: Required. A Keras Layer to transform the combined inputs
      into the new state.

  Call returns:
    The result of transformation.
  """

  def __init__(self,
               transformation: tf.keras.layers.Layer,
               **kwargs):
    super().__init__(**kwargs)
    self._transformation = transformation

  def get_config(self):
    return dict(transformation=self._transformation,
                **super().get_config())

  def call(
      self, inputs: Tuple[
          const.FieldOrFields, const.FieldsNest, const.FieldsNest
      ]) -> const.FieldOrFields:
    net = tf.nest.flatten(inputs)
    net = tf.concat(net, axis=-1)
    net = self._transformation(net)
    return net


@tf.keras.utils.register_keras_serializable(package="GNN")
class ResidualNextState(tf.keras.layers.Layer):
  """Updates a state with a residual block.

  This layer concatenates all inputs, sends them through a user-supplied
  transformation, forms a skip connection by adding back the state of the
  updated graph piece, and finally applies an activation function.
  In other words, the user-supplied transformation is a residual block
  that modifies the state. The output shape of the residual block must match
  the shape of the state that gets updated so that they can be added.

  If the initial state of the graph piece that is being updated has size 0,
  the skip connection is omitted. This avoids the need to special-case, say,
  latent node sets in modeling code applied to different node sets.

  Init args:
    residual_block: Required. A Keras Layer to transform the concatenation
      of all inputs into a delta that gets added to the state.
    activation: An activation function (none by default), as understood by
      `tf.keras.layers.Activation`. This activation function is applied after
      the residual block and the addition. If using this, typically the
      residual block does not have an activation function on its last layer,
      or vice versa.
    skip_connection_feature_name: Controls which input from the updated graph
      piece is added back after the residual block. If the input from the
      updated graph piece is a single tensor, that tensor is used, and this arg
      must not be set. If the input is a dict, this key is used; if unset, it
      defaults to `tfgnn.HIDDEN_STATE`.

  Call returns:
    A tensor to use as the new state.
  """

  def __init__(
      self,
      residual_block: tf.keras.layers.Layer,
      *,
      activation: Any = None,
      skip_connection_feature_name: Optional[const.FieldName] = None,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self._residual_block = residual_block
    if isinstance(activation, tf.keras.layers.Layer):
      self._activation = activation
    else:
      self._activation = tf.keras.layers.Activation(activation)
    self._skip_connection_feature_name = skip_connection_feature_name

  def get_config(self):
    return dict(
        residual_block=self._residual_block,
        activation=self._activation,
        skip_connection_feature_name=self._skip_connection_feature_name,
        **super().get_config())

  def call(
      self, inputs: Tuple[
          const.FieldOrFields, const.FieldsNest, const.FieldsNest
      ]) -> const.FieldOrFields:
    # Extract the feature for a skip connection.
    self_input = inputs[0]
    if isinstance(self_input, (tf.Tensor, tf.RaggedTensor)):
      if self._skip_connection_feature_name is not None:
        raise KeyError(
            "ResidualNextState() invoked with explicit skip connection feature"
            f" '{self._skip_connection_feature_name}', but the "
            " updated graph piece does not have this key (it's not a"
            f" dictionary) {self_input}."
        )
      skip_connection_feature = self_input
      skip_connection_msg = "single input"
    else:
      if self._skip_connection_feature_name is None:
        skip_connection_feature_name = const.HIDDEN_STATE
      else:
        skip_connection_feature_name = self._skip_connection_feature_name
      try:
        skip_connection_feature = self_input[skip_connection_feature_name]
      except KeyError as e:
        raise KeyError(
            "ResidualNextState() could not find the "
            f"skip connection feature '{skip_connection_feature_name}' "
            f"in the features of the updated graph piece: {list(self_input)}"
        ) from e
      skip_connection_msg = (
          f"input feature '{skip_connection_feature_name}'")

    # Compute the state update.
    net = tf.nest.flatten(inputs)
    net = tf.concat(net, axis=-1)
    net = self._residual_block(net)
    if skip_connection_feature.shape[1:].num_elements() == 0:
      tf.get_logger().warning(
          "ResidualNextState() called on empty input state (latent node set?); "
          "will omit residual link.")
    elif not skip_connection_feature.shape.is_compatible_with(net.shape):
      raise ValueError(
          "A ResidualNextState() requires an update_fn whose "
          "output has the same shape as the input state, but got "
          f"output shape {net.shape.as_list()} vs "
          f"input shape {skip_connection_feature.shape.as_list()} "
          f"from {skip_connection_msg}.")
    else:
      net = tf.add(net, skip_connection_feature)
    net = self._activation(net)
    return net


@tf.keras.utils.register_keras_serializable(package="GNN")
class SingleInputNextState(tf.keras.layers.Layer):
  """Replaces a state from a single input.

  In a NodeSetUpdate, it replaces the node state with a single edge set input.
  For an EdgeSetUpdate, it replaces the edge_state with the incident node set's
  input. For a ContextUpdate, it replaces the context state with a single node
  set input.

  Call returns:
    A tensor to use as the new state.
  """

  def call(self, inputs):
    unused_first_input, second_input, unused_third_input = inputs
    if tf.nest.flatten(unused_third_input):
      raise ValueError(
          "GraphPieceUpdate should only pass 2 inputs to SingleInputNextState"
          )

    try:
      single_input, = second_input.values()
    except ValueError as e:
      raise ValueError(
          "Second input to SingleInputNextState had multiple values."
          " This layer should take only a single input."
      ) from e
    return single_input
