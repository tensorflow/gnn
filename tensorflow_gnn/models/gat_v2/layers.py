"""Contains a Graph Attention Network v2 and associated layers."""
from typing import Any, Callable, Collection, Mapping, Optional, Union

import tensorflow as tf
import tensorflow_gnn as tfgnn


@tf.keras.utils.register_keras_serializable(package="GNN>models>gat_v2")
class GATv2Conv(tfgnn.keras.layers.AnyToAnyConvolutionBase):
  """The multi-head attention from Graph Attention Networks v2 (GATv2).

  GATv2 (https://arxiv.org/abs/2105.14491) improves upon the popular
  GAT architecture (https://arxiv.org/abs/1710.10903) by allowing the network
  to compute a more expressive "dynamic" instead of just "static" attention,
  each of whose heads is described by Equations (7), (3) and (4) in
  https://arxiv.org/abs/2105.14491.

  Example: GATv2-style attention on incoming edges whose result is
  concatenated with the old node state and passed through a Dense layer
  to compute the new node state.

  ```python
  dense = tf.keras.layers.Dense
  graph = tfgnn.keras.layers.GraphUpdate(
      node_sets={"paper": tfgnn.keras.layers.NodeSetUpdate(
          {"cites": tfgnn.keras.layers.GATv2Conv(
               message_dim, receiver_tag=tfgnn.TARGET)},
          tfgnn.keras.layers.NextStateFromConcat(dense(node_state_dim)))}
  )(graph)
  ```

  This layer implements the multi-head attention of GATv2 with the following
  generalizations:

    * This implementation of GATv2 attends only to edges that are explicitly
      stored in the input GraphTensor. Attention of a node to itself is
      enabled or disabled by storing or not storing an explicit loop in the
      edge set. The example above uses a separate layer to combine the old
      node state with the attention result to form the new node state.
    * Attention values can be computed from a sender node state that gets
      broadcast onto the edge (see arg `sender_node_feature`), from an edge
      feature (see arg `sender_edge_feature`), or from their concatenation
      (by setting both arguments). This choice is used in place of the sender
      node state $h_j$ in the defining equations cited above.
    * This layer can be used with `receiver_tag=tfgnn.CONTEXT` to perform a
      convolution to the context, with graph components as receivers and the
      containment in graph components used in lieu of edges.
    * An `edge_dropout` option is provided.

  This layer can also be configured to do attention pooling from edges to
  context or to receiver nodes (without regard for source nodes) by setting
  `sender_node_feature=None` and setting `sender_edge_feature=...` to the
  applicable edge feature name (e.g., `tfgnn.DEFAULT_FEATURE_NAME`).

  Like the Keras Dense layer, if the input features have rank greater than 2,
  this layer computes a point-wise attention along the last axis of the inputs.
  For example, if the input features have shape `[num_nodes, 2, 4, 1]`, then it
  will perform an identical computation on each of the `num_nodes * 2 * 4` input
  values.

  Init args:
    num_heads: The number of attention heads.
    per_head_channels: The number of channels for each attention head. This
      means that the final output size will be per_head_channels * num_heads.
    receiver_tag: one of `tfgnn.SOURCE`, `tfgnn.TARGET` or `tfgnn.CONTEXT`.
      The results of attention are aggregated for this graph piece.
      If set to `tfgnn.SOURCE` or `tfgnn.TARGET`, the layer can be called for
      an edge set and will aggregate results at the specified endpoint of the
      edges.
      If set to `tfgnn.CONTEXT`, the layer can be called for an edge set or
      node set.
      If left unset for init, the tag must be passed at call time.
    receiver_feature: Can be set to override `tfgnn.DEFAULT_FEATURE_NAME`
      for use as the receiver's input feature to attention. (The attention key
      is derived from this input.)
    sender_node_feature: Can be set to override `tfgnn.DEFAULT_FEATURE_NAME`
      for use as the input feature from sender nodes to attention.
      IMPORANT: Must be set to `None` for use with `receiver_tag=tfgnn.CONTEXT`
      on an edge set, or for pooling from edges without sender node states.
    sender_edge_feature: Can be set to a feature name of the edge set to select
      it as an input feature. By default, this set to `None`, which disables
      this input.
      IMPORTANT: Must be set for use with `receiver_tag=tfgnn.CONTEXT`
      on an edge set.
    use_bias: If true, a bias term is added to the transformations of query and
      value inputs.
    edge_dropout: Can be set to a dropout rate for edge dropout. (When pooling
      nodes to context, it's the node's membership in a graph component that
      is dropped out.)
    attention_activation: The nonlinearity used on the transformed inputs
      before multiplying with the trained weights of the attention layer.
      This can be specified as a Keras layer, a tf.keras.activations.*
      function, or a string understood by tf.keras.layers.Activation().
      Defaults to "leaky_relu", which in turn defaults to a negative slope
      of `alpha=0.2`.
    activation: The nonlinearity applied to the final result of attention,
      specified in the same ways as attention_activation.
    kernel_initializer: Can be set to a `kerner_initializer` as understood
      by `tf.keras.layers.Dense` etc.
  """

  def __init__(self,
               *,
               num_heads: int,
               per_head_channels: int,
               receiver_tag: Optional[tfgnn.IncidentNodeOrContextTag] = None,
               receiver_feature: tfgnn.FieldName = tfgnn.HIDDEN_STATE,
               sender_node_feature: Optional[
                   tfgnn.FieldName] = tfgnn.HIDDEN_STATE,
               sender_edge_feature: Optional[tfgnn.FieldName] = None,
               use_bias: bool = True,
               edge_dropout: float = 0.,
               attention_activation: Union[str,
                                           Callable[..., Any]] = "leaky_relu",
               activation: Union[str, Callable[..., Any]] = "relu",
               kernel_initializer: Union[
                   None, str, tf.keras.initializers.Initializer] = None,
               **kwargs):
    kwargs.setdefault("name", "gat_v2_conv")
    super().__init__(
        receiver_tag=receiver_tag,
        receiver_feature=receiver_feature,
        sender_node_feature=sender_node_feature,
        sender_edge_feature=sender_edge_feature,
        extra_receiver_ops={"softmax": tfgnn.softmax},
        **kwargs)
    if not self.takes_receiver_input:
      raise ValueError("Receiver feature cannot be None")

    if num_heads <= 0:
      raise ValueError(f"Number of heads {num_heads} must be greater than 0.")
    self._num_heads = num_heads

    if per_head_channels <= 0:
      raise ValueError(
          f"Per-head channels {per_head_channels} must be greater than 0.")
    self._per_head_channels = per_head_channels

    self._use_bias = use_bias

    if not 0 <= edge_dropout < 1:
      raise ValueError(f"Edge dropout {edge_dropout} must be in [0, 1).")
    self._edge_dropout = edge_dropout
    if self._edge_dropout > 0:
      self._edge_dropout_layer = tf.keras.layers.Dropout(self._edge_dropout)
    else:
      self._edge_dropout_layer = None

    self._attention_activation = tf.keras.activations.get(attention_activation)
    self._activation = tf.keras.activations.get(activation)
    self._kernel_initializer = kernel_initializer

    # Create the transformations for the query input in all heads.
    self._w_query = tf.keras.layers.Dense(
        per_head_channels * num_heads,
        kernel_initializer=kernel_initializer,
        # This bias gets added to the attention features but not the outputs.
        use_bias=use_bias,
        name="query")

    # Create the transformations for value input from sender nodes and edges.
    if self.takes_sender_node_input:
      self._w_sender_node = tf.keras.layers.Dense(
          per_head_channels * num_heads,
          kernel_initializer=kernel_initializer,
          # This bias gets added to the attention features and the outputs.
          use_bias=use_bias,
          name="value_node")
    else:
      self._w_sender_node = None

    if self.takes_sender_edge_input:
      self._w_sender_edge = tf.keras.layers.Dense(
          per_head_channels * num_heads,
          kernel_initializer=kernel_initializer,
          # This bias would be redundant with self._w_sender_node.
          use_bias=use_bias and self._w_sender_node is None,
          name="value_edge")
    else:
      self._w_sender_edge = None

    if self._w_sender_node is None and self._w_sender_edge is None:
      raise ValueError("GATv2Attention initialized with no inputs.")

    # Create attention logits layers, one for each head. Note that we can't
    # use a single Dense layer that outputs `num_heads` units because we need
    # to apply a different attention function a_k to its corresponding
    # W_k-transformed features.
    self._attention_logits_fn = tf.keras.layers.experimental.EinsumDense(
        "...ik,ki->...i",
        output_shape=(None, num_heads, 1),  # TODO(b/205825425): (num_heads,)
        kernel_initializer=kernel_initializer,
        name="attn_logits")

  def get_config(self):
    return dict(
        num_heads=self._num_heads,
        per_head_channels=self._per_head_channels,
        use_bias=self._use_bias,
        edge_dropout=self._edge_dropout,
        attention_activation=self._attention_activation,
        activation=self._activation,
        kernel_initializer=self._kernel_initializer,
        **super().get_config())

  def convolve(self, *,
               sender_node_input: Optional[tf.Tensor],
               sender_edge_input: Optional[tf.Tensor],
               receiver_input: Optional[tf.Tensor],
               broadcast_from_sender_node: Callable[[tf.Tensor], tf.Tensor],
               broadcast_from_receiver: Callable[[tf.Tensor], tf.Tensor],
               pool_to_receiver: Callable[..., tf.Tensor],
               extra_receiver_ops: Optional[
                   Mapping[str, Callable[..., Any]]] = None,
               **kwargs) -> tf.Tensor:
    """Overridden internal method of the base class."""
    # Form the attention query for each head.
    # [num_items, *extra_dims, num_heads, channels_per_head]
    assert receiver_input is not None, "__init__() should have checked this."
    query = broadcast_from_receiver(self._split_heads(self._w_query(
        receiver_input)))

    # Form the attention value by transforming the configured inputs
    # and adding up the transformed values.
    # [num_items, *extra_dims, num_heads, channels_per_head]
    value_terms = []
    if sender_node_input is not None:
      value_terms.append(broadcast_from_sender_node(
          self._split_heads(self._w_sender_node(sender_node_input))))
    if sender_edge_input is not None:
      value_terms.append(
          self._split_heads(self._w_sender_edge(sender_edge_input)))
    assert value_terms, "Internal error: no values, __init__ should catch this."
    value = tf.add_n(value_terms)

    # Compute the features from which attention logits are computed.
    # [num_items, *extra_dims, num_heads, channels_per_head]
    attention_features = self._attention_activation(query + value)

    # Compute the attention logits and softmax to get the coefficients.
    # [num_items, *extra_dims, num_heads, 1]
    logits = tf.expand_dims(self._attention_logits_fn(attention_features), -1)
    attention_coefficients = extra_receiver_ops["softmax"](logits)

    if self._edge_dropout_layer is not None:
      # If requested, add layer with dropout to the normalized attention
      # coefficients, as is done in the original GAT paper. This should
      # have the same effect as edge dropout.
      # Also, note that `keras.layers.Dropout` upscales the remaining values,
      # which should maintain the sum-up-to-1 per node in expectation.
      attention_coefficients = self._edge_dropout_layer(attention_coefficients,
                                                        **kwargs)

    # Apply the attention coefficients to the transformed query.
    # [num_items, *extra_dims, num_heads, per_head_channels]
    messages = value * attention_coefficients
    # Take the sum of the weighted values, which equals the weighted average.
    # Receivers without incoming senders get the empty sum 0.
    # [num_receivers, *extra_dims, num_heads, per_head_channels]
    pooled_messages = pool_to_receiver(messages, reduce_type="sum")
    # Apply the nonlinearity.
    pooled_messages = self._activation(pooled_messages)
    pooled_messages = self._merge_heads(pooled_messages)

    return pooled_messages

  # The following helpers map forth and back between tensors with...
  #  - a separate heads dimension: shape [..., num_heads, channels_per_head],
  #  - all heads concatenated:    shape [..., num_heads * channels_per_head].

  def _split_heads(self, tensor):
    extra_dims = tensor.shape[1:-1]  # Possibly empty.
    if not extra_dims.is_fully_defined():
      raise ValueError(
          "GATv2Attention requires non-ragged Tensors as inputs, "
          "and GraphTensor requires these to have statically known "
          f"dimensions except the first, but got {tensor.shape}")
    new_shape = (-1, *extra_dims, self._num_heads, self._per_head_channels)
    return tf.reshape(tensor, new_shape)

  def _merge_heads(self, tensor):
    num_merged = 2
    extra_dims = tensor.shape[1 : -num_merged]  # Possibly empty.
    merged_dims = tensor.shape[-num_merged:]
    if not extra_dims.is_fully_defined() or not merged_dims.is_fully_defined():
      raise ValueError(
          f"Unexpected unknown dimensions in shape {tensor.shape}")
    new_shape = (-1, *extra_dims, merged_dims.num_elements())
    return tf.reshape(tensor, new_shape)


def GATv2EdgePool(*,  # To be called like a class initializer.  pylint: disable=invalid-name
                  num_heads: int,
                  per_head_channels: int,
                  receiver_tag: Optional[tfgnn.IncidentNodeOrContextTag] = None,
                  receiver_feature: tfgnn.FieldName = tfgnn.HIDDEN_STATE,
                  sender_feature: tfgnn.FieldName = tfgnn.HIDDEN_STATE,
                  **kwargs):
  """Returns a layer for pooling edges with GATv2-style attention.

  When initialized with receiver_tag SOURCE or TARGET, the returned layer can
  be called on an edge set to compute the weighted sum of edge states at the
  given endpoint. The weights are computed by the method of Graph Attention
  Networks v2 (GATv2), except that edge states, not node states broadcast from
  the edges' other endpoint, are used as input values to attention.

  When initialized with receiver_tag CONTEXT, the returned layer can be called
  on an edge set to do the analogous pooling of edge states to context.

  NOTE: This layer cannot pool node states. For that, use `gat_v2.GATv2Conv`.

  Args:
    num_heads: The number of attention heads.
    per_head_channels: The number of channels for each attention head. This
      means that the final output size will be per_head_channels * num_heads.
    receiver_tag: The results of attention are aggregated for this graph piece.
      If set to `tfgnn.CONTEXT`, the layer can be called for an edge set or
      node set.
      If set to an IncidentNodeTag (e.g., `tfgnn.SOURCE` or `tfgnn.TARGET`),
      the layer can be called for an edge set and will aggregate results at
      the specified endpoint of the edges.
      If left unset, the tag must be passed when calling the layer.
    receiver_feature: By default, the default state feature of the receiver
      is used to compute the attention query. A different feature name can be
      selected by setting this argument.
    sender_feature: By default, the default state feature of the edge set is
      used to compute the attention values. A different feature name can be
      selected by setting this argument.
    **kwargs: Any other option for GATv2Conv, except sender_node_feature,
      which is set to None.
  """
  if kwargs.pop("sender_node_feature", None) is not None:
    raise TypeError("GATv2EdgePool() got an unexpected keyword argument "
                    "'sender_node_feature'. Did you mean GATv2Conv()?")
  kwargs.setdefault("name", "gat_v2_edge_pool")
  return GATv2Conv(
      num_heads=num_heads,
      per_head_channels=per_head_channels,
      receiver_tag=receiver_tag,
      receiver_feature=receiver_feature,
      sender_edge_feature=sender_feature,
      sender_node_feature=None,
      **kwargs)


# TODO(b/236941740): a systematic solution for adding loops.
def GATv2HomGraphUpdate(
    *,  # To be called like a class initializer.  pylint: disable=invalid-name
    num_heads: int,
    per_head_channels: int,
    receiver_tag: tfgnn.IncidentNodeOrContextTag,
    feature_name: str = tfgnn.HIDDEN_STATE,
    name: str = "gat_v2",
    **kwargs):
  """Returns a GraphUpdate layer with a Graph Attention Network V2 (GATv2).

  The returned layer performs one update step of a Graph Attention Network v2
  (GATv2) from https://arxiv.org/abs/2105.14491 on a GraphTensor that stores
  a homogeneous graph.
  For heterogeneous graphs with multiple node sets and edge sets, users are
  advised to consider a GraphUpdate with one or more GATv2Conv objects
  instead, such as the GATv2MPNNGraphUpdate.

  > IMPORTANT: This implementation of GAT attends only to edges that are
  > explicitly stored in the input GraphTensor. Attention of a node to itself
  > requires having an explicit loop in the edge set.

  Args:
    num_heads: The number of attention heads.
    per_head_channels: The number of channels for each attention head. This
      means that the final output size will be per_head_channels * num_heads.
    receiver_tag: one of `tfgnn.SOURCE` or `tfgnn.TARGET`.
    feature_name: The feature name of node states; defaults to
      `tfgnn.HIDDEN_STATE`.
    name: Optionally, a name for the layer returned.
    **kwargs: Any optional arguments to GATv2Conv, see there.
  """
  # Build a GraphUpdate for the target node set of the given edge_set_name.
  # That needs to be deferred until we see a GraphTensorSpec that tells us
  # the node_set_name.
  def deferred_init_callback(spec: tfgnn.GraphTensorSpec):
    tfgnn.check_homogeneous_graph_tensor(spec, "GATv2HomGraphUpdate")
    edge_set_name, = spec.edge_sets_spec.keys()
    node_set_name = spec.edge_sets_spec[
        edge_set_name].adjacency_spec.node_set_name(receiver_tag)
    node_set_updates = {
        node_set_name: tfgnn.keras.layers.NodeSetUpdate(
            {edge_set_name: GATv2Conv(
                num_heads=num_heads, per_head_channels=per_head_channels,
                receiver_tag=receiver_tag,
                sender_node_feature=feature_name, receiver_feature=feature_name,
                **kwargs)},
            next_state=tfgnn.keras.layers.SingleInputNextState(),
            node_input_feature=feature_name)}
    return dict(node_sets=node_set_updates)
  return tfgnn.keras.layers.GraphUpdate(
      deferred_init_callback=deferred_init_callback, name=name)


# DEPRECATED.
def GATv2GraphUpdate(*,  # To be called like a class initializer.  pylint: disable=invalid-name
                     num_heads: int,
                     per_head_channels: int,
                     edge_set_name: str,
                     feature_name: str = tfgnn.HIDDEN_STATE,
                     name: str = "gat_v2",
                     **kwargs):
  del edge_set_name  # Must be the only one anyways.
  return GATv2HomGraphUpdate(
      num_heads=num_heads, per_head_channels=per_head_channels,
      receiver_tag=tfgnn.TARGET, feature_name=feature_name, name=name, **kwargs)


def GATv2MPNNGraphUpdate(  # To be called like a class initializer.  pylint: disable=invalid-name
    *,
    units: int,
    message_dim: int,
    num_heads: int,
    receiver_tag: tfgnn.IncidentNodeOrContextTag,
    node_set_names: Optional[Collection[tfgnn.NodeSetName]] = None,
    edge_feature: Optional[tfgnn.FieldName] = None,
    l2_regularization: float = 0.0,
    edge_dropout_rate: float = 0.0,
    state_dropout_rate: float = 0.0,
    attention_activation: Union[str, Callable[..., Any]] = "leaky_relu",
    conv_activation: Union[str, Callable[..., Any]] = "relu",
    activation: Union[str, Callable[..., Any]] = "relu",
    kernel_initializer: Union[
        None, str, tf.keras.initializers.Initializer] = "glorot_uniform",
    ) -> tf.keras.layers.Layer:
  """Returns a GraphUpdate layer for message passing with GATv2 pooling.

  The returned layer performs one round of message passing between the nodes
  of a heterogeneous GraphTensor, using `gat_v2.GATv2Conv` to compute the
  messages and their pooling with attention, followed by a dense layer to
  compute the new node states from a concatenation of the old node state and
  all pooled messages.

  Args:
    units: The dimension of output hidden states for each node.
    message_dim: The dimension of messages (attention values) computed on
      each edge.  Must be divisible by `num_heads`.
    num_heads: The number of attention heads used by GATv2. `message_dim`
      must be divisible by this number.
    receiver_tag: one of `tfgnn.TARGET` or `tfgnn.SOURCE`, to select the
      incident node of each edge that receives the message.
    node_set_names: The names of node sets to update. If unset, updates all
      that are on the receiving end of any edge set.
    edge_feature: Can be set to a feature name of the edge set to select
      it as an input feature. By default, this set to `None`, which disables
      this input.
    l2_regularization: The coefficient of L2 regularization for weights and
      biases.
    edge_dropout_rate: The edge dropout rate applied during attention pooling
      of edges.
    state_dropout_rate: The dropout rate applied to the resulting node states.
    attention_activation: The nonlinearity used on the transformed inputs
      before multiplying with the trained weights of the attention layer.
      This can be specified as a Keras layer, a tf.keras.activations.*
      function, or a string understood by tf.keras.layers.Activation().
      Defaults to "leaky_relu", which in turn defaults to a negative slope
      of `alpha=0.2`.
    conv_activation: The nonlinearity applied to the result of attention on one
      edge set, specified in the same ways as attention_activation.
    activation: The nonlinearity applied to the new node states computed by
      this graph update.
    kernel_initializer: Can be set to a `kerner_initializer` as understood
      by `tf.keras.layers.Dense` etc.

  Returns:
    A GraphUpdate layer for use on a scalar GraphTensor with
    `tfgnn.HIDDEN_STATE` features on the node sets.
  """
  if message_dim % num_heads:
    raise ValueError("message_dim must be divisible by num_heads, "
                     f"got {message_dim} and {num_heads}.")
  per_head_channels = message_dim // num_heads

  def dense(units):  # pylint: disable=invalid-name
    regularizer = tf.keras.regularizers.l2(l2_regularization)
    return tf.keras.Sequential([
        tf.keras.layers.Dense(
            units,
            activation=activation,
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer="zeros",
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer),
        tf.keras.layers.Dropout(state_dropout_rate)])

  # pylint: disable=g-long-lambda
  gnn_builder = tfgnn.keras.ConvGNNBuilder(
      lambda edge_set_name, receiver_tag: GATv2Conv(
          num_heads=num_heads, per_head_channels=per_head_channels,
          edge_dropout=edge_dropout_rate, receiver_tag=receiver_tag,
          sender_edge_feature=edge_feature,
          attention_activation=attention_activation, activation=conv_activation,
          kernel_initializer=kernel_initializer),
      lambda node_set_name: tfgnn.keras.layers.NextStateFromConcat(
          dense(units)),
      receiver_tag=receiver_tag)
  return gnn_builder.Convolve(node_set_names)
