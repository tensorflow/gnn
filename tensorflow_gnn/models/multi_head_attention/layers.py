"""Contains a Multi-Head Attention and associated layers."""
from typing import Any, Callable, Collection, Mapping, Optional, Union
import warnings

import tensorflow as tf
import tensorflow_gnn as tfgnn


@tf.keras.utils.register_keras_serializable(
    package="GNN>models>multi_head_attention")
class MultiHeadAttentionConv(tfgnn.keras.layers.AnyToAnyConvolutionBase):
  r"""Transformer-style (dot-product) multi-head attention on GNNs.

  The [Graph Transformer](https://arxiv.org/abs/2012.09699) introduces
  [transformer-style multi-head attention](https://arxiv.org/abs/1706.03762)
  to GNN. This class describes a layer of computing such multi-head attention
  (without position encoding, and without the subsequent feed-forward network).
  Please see tensorflow_gnn/models/multi_head_attention/README.md for more
  details. For the regular sequential transformer attention, please see
  `tf.keras.layers.MultiHeadAttention` instead.

  This attention is formuated differently depending on the presence of
  edge features:
    1. When edge features are NOT considered, this layer is exactly the same
       as Graph Transformer attention, where the receiver node feature is seen
       as 'query' and the sender node feature is 'key' and 'value':

        $$Q = h_v, K = V = h_u, \text{where} \enspace u \in N(v)$$

    2. When edge features are considered, this layer still uses the
       receiver node feature as 'query', but uses the concatenation of the
       sender node feature and edge feature as 'key' and 'value':

        $$Q = h_v, K = V = [h_u||e_{uv}], \text{where} \enspace u \in N(v)$$

        which is different from Graph Transformer.

  Then, similar to what is done in "Attention is all you need",
  for attention head $i$, the attention output $O_i$ is computed as:

    $$O_i = Softmax((Q W_Q^i)(K W_K^i)^T/ \sqrt{d})V$$

  where $d$ is the per-head channel width and the denominator term is
  scaling of attention scores proposed by
  [Vaswani&al., 2017](https://arxiv.org/abs/1706.03762).

  Users are able to remove the scaling of attention scores (score_scaling=False)
  or add an activation on the transformed query (controled by
  `attention_activation`). However, we recommend to remove the scaling when
  using an `attention_activation` since activating both of them may lead to
  degrated accuracy. One can also customize the transformation kernels with
  different intializers, regularizers as well as the use of bias terms, using
  the other arguments.

  Example: Transformer-style attention on neighbors along incoming edges
  whose result is concatenated with the old node state and passed through
  a Dense layer to compute the new node state.
  ```
  dense = tf.keras.layers.Dense
  graph = tfgnn.keras.layers.GraphUpdate(
      node_sets={"paper": tfgnn.keras.layers.NodeSetUpdate(
          {"cites": tfgnn.keras.layers.MultiHeadAttentionConv(
               message_dim, receiver_tag=tfgnn.TARGET)},
          tfgnn.keras.layers.NextStateFromConcat(dense(node_state_dim)))}
  )(graph)
  ```

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
    receiver_feature: Can be set to override `tfgnn.HIDDEN_STATE`
      for use as the receiver's input feature to attention. (The attention key
      is derived from this input.)
    sender_node_feature: Can be set to override `tfgnn.HIDDEN_STATE`
      for use as the input feature from sender nodes to attention.
      IMPORANT: Must be set to `None` for use with `receiver_tag=tfgnn.CONTEXT`
      on an edge set, or for pooling from edges without sender node states.
    sender_edge_feature: Can be set to a feature name of the edge set to select
      it as an input feature. By default, this set to `None`, which disables
      this input.
      IMPORTANT: Must be set for use with `receiver_tag=tfgnn.CONTEXT`
      on an edge set.
    use_bias: If true, bias terms are added to the transformations of query,
      key and value inputs.
    edge_dropout: Can be set to a dropout rate for edge dropout. (When pooling
      nodes to context, it's the node's membership in a graph component that
      is dropped out.)
    attention_activation: The nonlinearity used on the transformed inputs
      (query) before multiplying with the trained weights of the attention
      layer. This can be specified as a Keras layer, a tf.keras.activations.*
      function, or a string understood by tf.keras.layers.Activation().
      Defaults to None.
    activation: The nonlinearity applied to the final result of attention,
      specified in the same ways as attention_activation.
    kernel_initializer: Can be set to a `kernel_initializer` as understood
      by tf.keras.layers.Dense etc.
    kernel_regularizer: Can be set to a `kernel_regularizer` as understood
      by tf.keras.layers.Dense etc.
    score_scaling: If true, the attention scores are divided by the square root
      of per_head_channels.
  """

  def __init__(
      self,
      *,
      num_heads: int,
      per_head_channels: int,
      receiver_tag: Optional[tfgnn.IncidentNodeOrContextTag] = None,
      receiver_feature: tfgnn.FieldName = tfgnn.HIDDEN_STATE,
      sender_node_feature: Optional[tfgnn.FieldName] = tfgnn.HIDDEN_STATE,
      sender_edge_feature: Optional[tfgnn.FieldName] = None,
      use_bias: bool = True,
      edge_dropout: float = 0.,
      attention_activation: Optional[Union[str, Callable[..., Any]]] = None,
      activation: Union[str, Callable[..., Any]] = "relu",
      kernel_initializer: Union[None, str,
                                tf.keras.initializers.Initializer] = None,
      kernel_regularizer: Union[None, str,
                                tf.keras.regularizers.Regularizer] = None,
      score_scaling: bool = True,
      **kwargs):
    kwargs.setdefault("name", "multi_head_attention_conv")
    super().__init__(
        receiver_tag=receiver_tag,
        receiver_feature=receiver_feature,
        sender_node_feature=sender_node_feature,
        sender_edge_feature=sender_edge_feature,
        extra_receiver_ops={
            "softmax": tfgnn.softmax,
        },
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

    # Check for conflicting options.
    if attention_activation is not None and score_scaling:
      warnings.warn("using both activation on transformed inputs and"
                    "score scaling may lead to degraded accuracy,"
                    "please consider only one of them.")

    # Check for valid inputs.
    if (not self.takes_sender_node_input and
        not self.takes_sender_edge_input):
      raise ValueError("MultiHeadAttentionConv initialized with no inputs.")

    self._attention_activation = tf.keras.activations.get(attention_activation)
    self._activation = tf.keras.activations.get(activation)
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self._score_scaling = score_scaling

    # Create the transformations for the query input in all heads.
    self._w_query = tf.keras.layers.Dense(
        per_head_channels * num_heads,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        # This bias gets added to the attention features but not the outputs.
        use_bias=use_bias,
        name="query")

    # Create the transformations for key input
    # from sender nodes and edges.
    if self.takes_sender_node_input:
      self._w_sender_node_to_key = tf.keras.layers.Dense(
          per_head_channels * num_heads,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          use_bias=use_bias,
          name="key_node")
    else:
      self._w_sender_node_to_key = None
    if self.takes_sender_edge_input:
      self._w_sender_edge_to_key = tf.keras.layers.Dense(
          per_head_channels * num_heads,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          # This bias would be redundant with self._w_sender_node.
          use_bias=use_bias and self._w_sender_node_to_key is None,
          name="key_edge")
    else:
      self._w_sender_edge_to_key = None

    self._w_value = tf.keras.layers.Dense(
        per_head_channels * num_heads,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        use_bias=use_bias,
        name="value")

  def get_config(self):
    return dict(
        num_heads=self._num_heads,
        per_head_channels=self._per_head_channels,
        use_bias=self._use_bias,
        edge_dropout=self._edge_dropout,
        attention_activation=self._attention_activation,
        activation=self._activation,
        kernel_initializer=tf.keras.initializers.serialize(
            self._kernel_initializer
        ),
        kernel_regularizer=tf.keras.regularizers.serialize(  # b/238163789
            self._kernel_regularizer
        ),
        score_scaling=self._score_scaling,
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

    # Form the attention query for each head.
    # [num_items, *extra_dims, num_heads, channels_per_head]
    assert receiver_input is not None, "__init__() should have checked this."
    queries = broadcast_from_receiver(
        self._split_heads(
            self._w_query(receiver_input)))

    # Maybe add an activation to the queries.
    if self._attention_activation is not None:
      queries = self._attention_activation(queries)

    # Form the keys  by transforming the configured inputs
    # and adding up the transformed values.
    keys = []
    if sender_node_input is not None:
      keys.append(broadcast_from_sender_node(
          self._split_heads(self._w_sender_node_to_key(sender_node_input))))
    if sender_edge_input is not None:
      keys.append(self._split_heads(
          self._w_sender_edge_to_key(sender_edge_input)))
    keys = tf.add_n(keys)

    # Dot-product of queries and keys to produce the attention coefficients.
    # [num_items, *extra_dims, num_heads, 1]
    attention_coefficients = tf.reduce_sum(
        queries * keys, axis=-1, keepdims=True)
    if self._score_scaling:
      attention_coefficients *= tf.math.rsqrt(
          tf.cast(self._per_head_channels, tf.float32))

    attention_coefficients = extra_receiver_ops["softmax"](
        attention_coefficients)

    if self._edge_dropout_layer is not None:
      # If requested, add layer with dropout to the normalized attention
      # coefficients. This should have the same effect as edge dropout.
      # Also, note that `keras.layers.Dropout` upscales the remaining values,
      # which should maintain the sum-up-to-1 per node in expectation.
      attention_coefficients = self._edge_dropout_layer(attention_coefficients,
                                                        **kwargs)

    # Form the values and multiply them with the attention coefficients.
    values = []
    if sender_node_input is not None:
      values.append(broadcast_from_sender_node(sender_node_input))
    if sender_edge_input is not None:
      values.append(sender_edge_input)
    # [num_items, *extra_dims, num_heads, per_head_channels]
    values = tf.concat(values, axis=-1)

    # First project the values, then compute the weighted combination.
    values = self._split_heads(self._w_value(values))
    messages = values * attention_coefficients
    pooled_messages = pool_to_receiver(messages, reduce_type="sum")

    # Apply the nonlinearity on the final result.
    # [num_receivers, *extra_dims, num_heads, per_head_channels]
    pooled_messages = self._activation(pooled_messages)
    pooled_messages = self._merge_heads(pooled_messages)

    return pooled_messages

  # The following helpers map back and forth between tensors with...
  #  - a separate heads dimension: shape [..., num_heads, channels_per_head],
  #  - all heads concatenated:    shape [..., num_heads * channels_per_head].

  def _split_heads(self, tensor):
    extra_dims = tensor.shape[1:-1]  # Possibly empty.
    if not extra_dims.is_fully_defined():
      raise ValueError(
          "MultiHeadAttentionConv requires non-ragged Tensors as inputs, "
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


def MultiHeadAttentionEdgePool(
    *,  # To be called like a class initializer.  pylint: disable=invalid-name
    num_heads: int,
    per_head_channels: int,
    receiver_tag: Optional[tfgnn.IncidentNodeOrContextTag] = None,
    receiver_feature: tfgnn.FieldName = tfgnn.HIDDEN_STATE,
    sender_feature: tfgnn.FieldName = tfgnn.HIDDEN_STATE,
    **kwargs):
  """Returns a layer for pooling edges with Transformer-style Multi-Head Attention.

  When initialized with receiver_tag SOURCE or TARGET, the returned layer can
  be called on an edge set to compute the weighted sum of edge states at the
  given endpoint. The weights are computed by the method of Transformer-style
  Multi-Head Attention, except that edge states, not node states broadcast from
  the edges' other endpoint, are used as input values to attention.

  When initialized with receiver_tag CONTEXT, the returned layer can be called
  on an edge set to do the analogous pooling of edge states to context.

  NOTE: This layer cannot pool node states.
        For that, use MultiHeadAttentionConv.

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
    **kwargs: Any other option for MultiHeadAttentionConv,
      except sender_node_feature, which is set to None.
  """
  if kwargs.pop("sender_node_feature", None) is not None:
    raise TypeError(
        "MultiHeadAttentionEdgePool() got an unexpected keyword argument"
        "'sender_node_feature'. Did you mean MultiHeadAttentionConv()?")
  kwargs.setdefault("name", "multi_head_attention_edge_pool")
  return MultiHeadAttentionConv(
      num_heads=num_heads,
      per_head_channels=per_head_channels,
      receiver_tag=receiver_tag,
      receiver_feature=receiver_feature,
      sender_edge_feature=sender_feature,
      sender_node_feature=None,
      **kwargs)


# TODO(b/236941740): a systematic solution for adding loops.
def MultiHeadAttentionHomGraphUpdate(
    *,  # To be called like a class initializer.  pylint: disable=invalid-name
    num_heads: int,
    per_head_channels: int,
    receiver_tag: tfgnn.IncidentNodeOrContextTag,
    feature_name: str = tfgnn.HIDDEN_STATE,
    name: str = "multi_head_attention",
    **kwargs):
  """Returns a GraphUpdate layer with a transformer-style multihead attention.

  The returned layer performs one update step of a Transformer-style
  Multi-Head Attention (but without the Feed Forward Network) on a GraphTensor
   that stores a homogeneous graph.

  For heterogeneous graphs with multiple node sets and edge sets, users are
  advised to consider a GraphUpdate with one or more MultiHeadAttentionConv
  objects instead, such as the MultiHeadAttentionMPNNGraphUpdate (see it for
  more details).

  > IMPORTANT: This implementation of MultiHeadAttention attends only to edges
  > that are explicitly stored in the input GraphTensor. Attention of a node to
  > itself requires having an explicit loop in the edge set.

  Args:
    num_heads: The number of attention heads.
    per_head_channels: The number of channels for each attention head. This
      means that the final output size will be per_head_channels * num_heads.
    receiver_tag: one of `tfgnn.SOURCE` or `tfgnn.TARGET`.
    feature_name: The feature name of node states; defaults to
      `tfgnn.HIDDEN_STATE`.
    name: Optionally, a name for the layer returned.
    **kwargs: Any optional arguments to MultiHeadAttentionConv, see there.
  """
  # Build a GraphUpdate for the target node set of the given edge_set_name.
  # That needs to be deferred until we see a GraphTensorSpec that tells us
  # the node_set_name.
  def deferred_init_callback(spec: tfgnn.GraphTensorSpec):
    tfgnn.check_homogeneous_graph_tensor(spec,
                                         "MultiHeadAttentionHomGraphUpdate")
    edge_set_name, = spec.edge_sets_spec.keys()
    node_set_name = spec.edge_sets_spec[
        edge_set_name].adjacency_spec.node_set_name(receiver_tag)
    node_set_updates = {
        node_set_name: tfgnn.keras.layers.NodeSetUpdate(
            {edge_set_name: MultiHeadAttentionConv(
                num_heads=num_heads, per_head_channels=per_head_channels,
                receiver_tag=receiver_tag,
                sender_node_feature=feature_name, receiver_feature=feature_name,
                **kwargs)},
            next_state=tfgnn.keras.layers.SingleInputNextState(),
            node_input_feature=feature_name)}
    return dict(node_sets=node_set_updates)
  return tfgnn.keras.layers.GraphUpdate(
      deferred_init_callback=deferred_init_callback, name=name)


def MultiHeadAttentionMPNNGraphUpdate(  # To be called like a class initializer.  pylint: disable=invalid-name
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
    attention_activation: Optional[Union[str, Callable[..., Any]]] = None,
    conv_activation: Union[str, Callable[..., Any]] = "relu",
    activation: Union[str, Callable[..., Any]] = "relu",
    kernel_initializer: Union[
        None, str, tf.keras.initializers.Initializer] = "glorot_uniform",
    ) -> tf.keras.layers.Layer:
  """Returns a GraphUpdate layer for message passing with MultiHeadAttention pooling.

  The returned layer performs one round of message passing between the nodes
  of a heterogeneous GraphTensor, using
  `multi_head_attention.MultiHeadAttentionConv` to compute the messages and
  their pooling with attention, followed by a dense layer to compute the new
  node states from a concatenation of the old node state and all pooled
  messages, analogous to TF-GNN's `vanilla_mpnn.VanillaMPNNGraphUpdate` and
  `gat_v2.GATv2MPNNGraphUpdate`.

  Args:
    units: The dimension of output hidden states for each node.
    message_dim: The dimension of messages (attention values) computed on
      each edge.  Must be divisible by `num_heads`.
    num_heads: The number of attention heads used by MultiHeadAttention.
      `message_dim` must be divisible by this number.
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
      Defaults to None.
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
      lambda edge_set_name, receiver_tag: MultiHeadAttentionConv(
          num_heads=num_heads, per_head_channels=per_head_channels,
          edge_dropout=edge_dropout_rate, receiver_tag=receiver_tag,
          sender_edge_feature=edge_feature,
          attention_activation=attention_activation, activation=conv_activation,
          kernel_initializer=kernel_initializer),
      lambda node_set_name: tfgnn.keras.layers.NextStateFromConcat(
          dense(units)),
      receiver_tag=receiver_tag)
  return gnn_builder.Convolve(node_set_names)
