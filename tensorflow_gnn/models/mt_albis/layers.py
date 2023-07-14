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
"""The Layer classes of Model Template "Albis"."""

from typing import Any, Callable, Collection, Literal, Mapping, Optional, Tuple, Union

import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn.models import gat_v2
from tensorflow_gnn.models import multi_head_attention


def MtAlbisSimpleConv(  # To be called like a class initializer.  pylint: disable=invalid-name
    units: int,
    *,
    receiver_tag: tfgnn.IncidentNodeTag,
    reduce_type: str = "mean",
    activation: Union[str, Callable[..., Any]] = "relu",
    edge_dropout_rate: float = 0.0,
    use_receiver_state: bool = True,
    edge_feature_name: Optional[tfgnn.FieldName] = None,
    kernel_initializer: Any = "glorot_uniform",
    kernel_regularizer: Any = None,
    name: Optional[str] = None,
) -> tf.keras.layers.Layer:
  """Returns a Layer object for the non-attention flavor of Conv in MtAlbis.

  See MtAlbisGraphUpdate for the user-facing documentation.

  Args:
    units: The dimension of the message computed on each edge.
    receiver_tag: one of `tfgnn.SOURCE` or `tfgnn.TARGET`. The messages are
      pooled for the nodes at that endpoint of edges.
      If left unset for init, the tag must be passed at call time.
    reduce_type: Controls how messages are aggregated on an EdgeSet for each
      receiver node; defaults to `"mean"`. Can be any reduce_type understood by
      `tfgnn.pool()`, including concatenations like `"mean|max"` (but mind the
      increased dimension of the result and the growing number of model weights
      in the next-state layer).
    activation: The nonlinearity used on each message before pooling.
      This can be specified as a Keras layer, a tf.keras.activations.*
      function, or a string understood by `tf.keras.layers.Activation`.
    edge_dropout_rate: Can be set to a dropout rate for entire edges:
      with the given probability, the entire message of an edge is dropped,
      as if the edge were not present in the graph.
    use_receiver_state: Controls whether the receiver node state is used in
      computing each edge's message.
    edge_feature_name: Optionally, the name of an edge feature to include in
      message computation on edges.
    kernel_initializer: Can be set to a `kernel_initializer` as understood
      by `tf.keras.layers.Dense` etc.
      An `Initializer` object gets cloned before use to ensure a fresh seed,
      if not set explicitly. For more, see `tfgnn.keras.clone_initializer()`.
    kernel_regularizer: Can be set to a `kernel_regularizer` as understood
      by `tf.keras.layers.Dense` etc.
    name: Optionally, a Layer.name for the returned object.
  """
  message_fn = tf.keras.Sequential([
      tf.keras.layers.Dense(
          units,
          activation=activation,
          use_bias=True,
          kernel_initializer=tfgnn.keras.clone_initializer(kernel_initializer),
          bias_initializer="zeros",
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=None,  # Intentionally different from VanillaMPNN.
      ),
      tfgnn.keras.layers.ItemDropout(edge_dropout_rate)])
  return tfgnn.keras.layers.SimpleConv(
      message_fn=message_fn,
      reduce_type=reduce_type,
      receiver_tag=receiver_tag,
      receiver_feature=(tfgnn.HIDDEN_STATE if use_receiver_state else None),
      sender_edge_feature=edge_feature_name,
      name=name)


@tf.keras.utils.register_keras_serializable(package="GNN>models>mt_albis")
class MtAlbisNextNodeState(tf.keras.layers.Layer):
  """Computes a new node state in the Model Template "Albis".

  This NextState layer is meant for use in `tfgnn.keras.layers.NodeSetUpdate`
  to computes new hidden states for a NodeSet from a combination of the old
  node state with the pooled messages received by the NodeSet. (If the
  `NodeSetUpdate` provides a context input, that is used as well.)

  Init args:
    units: The dimension of the computed node states.
    next_state_type: `"dense"` or `"residual"`. With the latter, a residual
      link is added from the old to the new node state, which requires that all
      input node states already have size `units` (unless their size is 0, as
      for latent node sets, in which case the residual link is omitted).
    dropout_rate: Can be set to a dropout rate for entries in the output
      node states.
    normalization_type: controls the normalization of output node states.
      By default (`"layer"`), LayerNormalization is used. Can be set to
      `"none"`, or to `"batch"` for BatchNormalization.
    batch_normalization_momentum: If `normalization_type="batch"`, sets the
      `BatchNormalization(momentum=...)` parameter. Ignored otherwise.
    edge_set_combine_type: `"concat"` or `"sum"`. Controls how pooled messages
      from various edge sets are combined as inputs to the next-state
      computation.
    activation: The nonlinearity applied to the output. This can be specified
      as a Keras layer, a tf.keras.activations.* function, or a string
      understood by `tf.keras.layers.Activation`.
    kernel_initializer: Can be set to a `kernel_initializer` as understood
      by `tf.keras.layers.Dense` etc.
      An `Initializer` object gets cloned before use to ensure a fresh seed,
      if not set explicitly. For more, see `tfgnn.keras.clone_initializer()`.
    kernel_regularizer: Can be set to a `kernel_regularizer` as understood
      by `tf.keras.layers.Dense` etc.
  """

  def __init__(
      self,
      units: int,
      *,
      next_state_type: str = "dense",
      dropout_rate: float = 0.0,
      normalization_type: Literal["layer", "batch", "none"] = "layer",
      batch_normalization_momentum: float = 0.99,
      edge_set_combine_type: str = "concat",
      activation: Union[str, Callable[..., Any]] = "relu",
      kernel_initializer: Any = "glorot_uniform",
      kernel_regularizer: Any = None,
      **kwargs):
    super().__init__(**kwargs)
    self._next_state_type = next_state_type
    self._dense = tf.keras.layers.Dense(
        units,
        activation=activation,
        use_bias=True,
        kernel_initializer=tfgnn.keras.clone_initializer(
            tf.keras.initializers.get(kernel_initializer)),
        bias_initializer="zeros",
        kernel_regularizer=tf.keras.regularizers.get(kernel_regularizer),
        bias_regularizer=None,  # Intentionally different from VanillaMPNN.
        name="dense")
    self._dropout = tf.keras.layers.Dropout(dropout_rate)
    self._normalization_type = normalization_type
    self._batch_normalization_momentum = batch_normalization_momentum
    if normalization_type == "none":
      self._normalization = tf.keras.layers.Layer(name="no_norm")
    elif normalization_type == "layer":
      self._normalization = tf.keras.layers.LayerNormalization()
    elif normalization_type == "batch":
      self._normalization = tf.keras.layers.BatchNormalization(
          momentum=batch_normalization_momentum)
    else:
      raise ValueError(f"Unknown normalization_type: '{normalization_type}'")
    self._edge_set_combine_type = edge_set_combine_type
    self._activation = tf.keras.activations.get(activation)

  def get_config(self):
    return dict(
        units=self._dense.units,
        next_state_type=self._next_state_type,
        dropout_rate=self._dropout.rate,
        normalization_type=self._normalization_type,
        batch_normalization_momentum=self._batch_normalization_momentum,
        edge_set_combine_type=self._edge_set_combine_type,
        activation=self._dense.activation,
        # Regularizers and initializers need explicit serialization here
        # (and deserialization in __init__ via .get()) due to b/238163789.
        kernel_initializer=tf.keras.initializers.serialize(
            self._dense.kernel_initializer),
        kernel_regularizer=tf.keras.regularizers.serialize(
            self._dense.kernel_regularizer),
        **super().get_config())

  def call(
      self,
      inputs: Tuple[tfgnn.FieldOrFields,
                    Mapping[tfgnn.EdgeSetName, tfgnn.FieldOrFields],
                    tfgnn.FieldOrFields],
  ) -> tfgnn.FieldOrFields:
    input_state, edge_set_inputs, context_input = inputs
    flat_inputs = []
    # Collect the previous state of the updated node set.
    input_state = _require_single_tensor(input_state, "input state")
    flat_inputs.append(input_state)
    # Collect and combine pooled messages (conv results) from edge sets.
    edge_input = self._combine_edge_inputs(edge_set_inputs)
    edge_input = self._dropout(edge_input)
    flat_inputs.append(edge_input)
    # Collect a context input, if any. (Empty Mapping means none.)
    if isinstance(context_input, Mapping) and not context_input:
      pass
    else:
      context_input = _require_single_tensor(context_input,
                                             "input from context")
      context_input = self._dropout(context_input)
      flat_inputs.append(context_input)

    net = tf.concat(flat_inputs, axis=-1)
    net = self._dense(net)
    net = self._dropout(net)

    if (self._next_state_type == "residual" and
        input_state.shape[1:].num_elements() != 0):
      if not input_state.shape.is_compatible_with(net.shape):
        raise ValueError(
            f"MtAlbisNextNodeState(next_state_type={self._next_state_type}) "
            "requires an input that matches the requested output shape, "
            f"but got input shape {input_state.shape.as_list()} vs "
            f"output shape {net.shape.as_list()}.")
      net = tf.add(net, input_state)

    net = self._normalization(net)
    return net

  def _combine_edge_inputs(self, edge_set_inputs):
    inputs_list = [_require_single_tensor(v, f"input from edge set {k}")
                   for k, v in sorted(edge_set_inputs.items())]
    return tfgnn.combine_values(inputs_list, self._edge_set_combine_type)


def _require_single_tensor(x: tfgnn.FieldOrFields, name) -> tfgnn.Field:
  """Raises ValueError for a Mapping where a single tensor is expected."""
  if isinstance(x, Mapping):
    raise ValueError(
        f"MtAlbisNextNodeState expects a single tensor as {name}, got {x}")
  return x  # Returning with narrowed type annotation helps pytype.


def MtAlbisGraphUpdate(  # To be called like a class initializer.  pylint: disable=invalid-name
    # LINT.IfChange(MtAlbisGraphUpdate_args)
    *,  # To be called like a class initializer.  pylint: disable=invalid-name
    units: int,
    message_dim: int,
    receiver_tag: tfgnn.IncidentNodeTag,
    node_set_names: Optional[Collection[tfgnn.NodeSetName]] = None,
    # TODO(b/261835577): Can edge_feature be set for some EdgeSets only?
    edge_feature_name: Optional[tfgnn.FieldName] = None,
    attention_type: Literal["none", "multi_head", "gat_v2"] = "none",
    attention_edge_set_names: Optional[Collection[tfgnn.EdgeSetName]] = None,
    attention_num_heads: int = 4,
    simple_conv_reduce_type: str = "mean",
    simple_conv_use_receiver_state: bool = True,
    state_dropout_rate: float = 0.0,
    edge_dropout_rate: float = 0.0,
    l2_regularization: float = 0.0,
    kernel_initializer: Any = "glorot_uniform",
    # TODO(b/265755923): Should normalization_type apply to Convs, too?
    # TODO(b/265755923): Relative order of relu, dropout and normalization.
    normalization_type: Literal["layer", "batch", "none"] = "layer",
    batch_normalization_momentum: float = 0.99,
    next_state_type: Literal["dense", "residual"] = "dense",
    # TODO(b/265755968): Support "mean", maybe retire "sum".
    edge_set_combine_type: Literal["concat", "sum"] = "concat"
    # LINT.ThenChange(./config_dict.py:graph_update_get_config_dict)
) -> tf.keras.layers.Layer:
  """Returns GraphUpdate layer for message passing with Model Template "Albis".

  The TF-GNN Model Template "Albis" provides a small selection of field-tested
  GNN architectures through the unified interface of this class.

  Args:
    units: The dimension of node states in the output GraphTensor.
    message_dim: The dimension of messages computed transiently on each edge.
    receiver_tag: One of `tfgnn.SOURCE` or `tfgnn.TARGET`. The messages are
      sent to the nodes at this endpoint of edges.
    node_set_names: Optionally, the names of NodeSets to update. By default,
      all NodeSets are updated that receive from at least one EdgeSet.
    edge_feature_name: Optionally, the name of an edge feature to include in
      message computation on edges.
    attention_type: `"none"`, `"multi_head"`, or `"gat_v2"`. Selects whether
      messages are pooled with data-dependent weights computed by a trained
      attention mechansim.
    attention_edge_set_names: If set, edge sets other than those named here
      will be treated as if `attention_type="none"` regardless.
    attention_num_heads: For attention_types `"multi_head"` or `"gat_v2"`,
      the number of attention heads.
    simple_conv_reduce_type: For attention_type `"none"`, controls how messages
      are aggregated on an EdgeSet for each receiver node. Defaults to `"mean"`;
      other recommened values are the concatenations `"mean|sum"`, `"mean|max"`,
      and `"mean|sum|max"` (but mind the increased output dimension and the
      corresponding increase in the number of weights in the next-state layer).
      Technically, can be set to any reduce_type understood by `tfgnn.pool()`.
    simple_conv_use_receiver_state: For attention_type `"none"`, controls
      whether the receiver node state is used in computing each edge's message
      (in addition to the sender node state and possibly an `edge feature`).
    state_dropout_rate: The dropout rate applied to the pooled and combined
      messages from all edges, to the optional input from context, and to the
      new node state. This is conventional dropout, independently for each
      dimension of the network's hidden state. (Unlike VanillaMPNN, dropout
      is applied to messages after pooling.)
    edge_dropout_rate: Can be set to a dropout rate for entire edges during
      message computation: with the given probability, the entire message of
      an edge is dropped, as if the edge were not present in the graph.
    l2_regularization: The coefficient of L2 regularization for trained weights.
      (Unlike VanillaMPNN, this is not applied to biases.)
    kernel_initializer: Can be set to a `kernel_initializer` as understood
      by `tf.keras.layers.Dense` etc.
      An `Initializer` object gets cloned before use to ensure a fresh seed,
      if not set explicitly. For more, see `tfgnn.keras.clone_initializer()`.
    normalization_type: controls the normalization of output node states.
      By default (`"layer"`), LayerNormalization is used. Can be set to
      `"none"`, or to `"batch"` for BatchNormalization.
    batch_normalization_momentum: If `normalization_type="batch"`, sets the
      `BatchNormalization(momentum=...)` parameter. Ignored otherwise.
    next_state_type: `"dense"` or `"residual"`. With the latter, a residual
      link is added from the old to the new node state, which requires that all
      input node states already have size `units` (unless their size is 0, as
      for latent node sets, in which case the residual link is omitted).
    edge_set_combine_type: `"concat"` or `"sum"`. Controls how pooled messages
      from various edge sets are combined as inputs to the NextState layer
      that updates the node states. Defaults to `"concat"`, which gives the
      pooled messages from each edge set separate weights in the NextState
      layer, namely `units * message_dim * num_incident_edge_sets` per node set.
      Setting this to `"sum"` adds up the pooled messages into a single
      vector before passing them into the NextState layer, which requires just
      `units * message_dim` weights per node set.
  """
  def needs_attention(edge_set_name):
    if attention_type == "none":
      return False
    elif attention_edge_set_names is None:
      return True
    else:
      return edge_set_name in attention_edge_set_names

  kernel_regularizer = tf.keras.regularizers.l2(l2_regularization)
  def convolutions_factory(edge_set_name, *, receiver_tag):
    if not needs_attention(edge_set_name):
      return MtAlbisSimpleConv(
          message_dim,
          receiver_tag=receiver_tag,
          reduce_type=simple_conv_reduce_type,
          edge_dropout_rate=edge_dropout_rate,
          use_receiver_state=simple_conv_use_receiver_state,
          edge_feature_name=edge_feature_name,
          kernel_initializer=kernel_initializer,  # Cloned by the layer.
          kernel_regularizer=kernel_regularizer)

    if message_dim % attention_num_heads:
      raise ValueError("message_dim must be divisible by attention_num_heads, "
                       f"got {message_dim} and {attention_num_heads}.")
    per_head_channels = message_dim // attention_num_heads
    if attention_type == "multi_head":
      return multi_head_attention.MultiHeadAttentionConv(
          num_heads=attention_num_heads,
          per_head_channels=per_head_channels,
          receiver_tag=receiver_tag,
          edge_dropout=edge_dropout_rate,
          kernel_initializer=kernel_initializer,  # Cloned by the layer.
          kernel_regularizer=kernel_regularizer,
          sender_edge_feature=edge_feature_name)
    elif attention_type == "gat_v2":
      return gat_v2.GATv2Conv(
          num_heads=attention_num_heads,
          per_head_channels=per_head_channels,
          receiver_tag=receiver_tag,
          edge_dropout=edge_dropout_rate,
          kernel_initializer=kernel_initializer,  # Cloned by the layer.
          kernel_regularizer=kernel_regularizer,
          sender_edge_feature=edge_feature_name)
    else:
      raise ValueError(f"Unknown attention_type: '{attention_type}'")

  def nodes_next_state_factory(node_set_name):
    del node_set_name  # Unused.
    return MtAlbisNextNodeState(
        units,
        next_state_type=next_state_type,
        dropout_rate=state_dropout_rate,
        normalization_type=normalization_type,
        batch_normalization_momentum=batch_normalization_momentum,
        edge_set_combine_type=edge_set_combine_type,
        activation="relu",
        kernel_initializer=kernel_initializer,  # Cloned by the layer.
        kernel_regularizer=kernel_regularizer)

  gnn_builder = tfgnn.keras.ConvGNNBuilder(
      convolutions_factory, nodes_next_state_factory, receiver_tag=receiver_tag)
  return gnn_builder.Convolve(node_set_names)
