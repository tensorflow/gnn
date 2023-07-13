# pyformat: mode=yapf
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
"""Contains a Multi-Head Attention and associated layers."""
from typing import Any, Callable, Collection, Literal, Mapping, Optional, Union
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
  and produces concatenated multi-head outputs (without positional encoding,
  clamping in Softmax, linear transformation for multi-head outputs, the
  feed-forward network, the residual connections and normalization layers).
  Please see tensorflow_gnn/models/multi_head_attention/README.md for more
  details. For the regular sequential transformer attention, please see
  `tf.keras.layers.MultiHeadAttention` instead.

  This attention is formuated differently depending on the presence of
  edge features:

    1. When edge features are NOT considered, this layer is exactly the same
       as Graph Transformer attention, where the receiver node feature is seen
       as 'query' and the sender node feature is 'key' and 'value':

        $$Q_v = h_v, K_u = V_u = h_u, \text{where} \enspace u \in N(v)$$

    2. When edge features are considered, this layer still uses the
       receiver node feature as 'query', but uses the concatenation of the
       sender node feature and edge feature as 'key' and 'value':

        $$Q_v = h_v, K_u = V_u = [h_u||e_{uv}],
        \text{where} \enspace u \in N(v)$$

  Then, similar to what is done in "Attention is all you need" and what is
  described in Equations (4) and (5) of "Graph Transformer", the attention
  output $O^k_v$ from head $k$ for receiver node $v$ is computed as

    $$O^k_v = \sum_{u \in N(v)} \alpha^k_{uv} V_u W_V^k$$

  with attention weights

    $$(\alpha^k_{uv} \mid u \in N(v))
      = Softmax((Q_v W_Q^k)(K_u W_K^k)^T \mid u \in N(v)) / \sqrt{d}$$

  where the softmax is taken over all neighbors $u$ along edges $(u,v)$ into $v$
  and $d$ is the dimension of keys and queries as projected by $W_K$ and $W_Q$.
  The final output for node $v$ is the concatenation over all heads, that is

    $$O_v = ||_k O^k_v$$.

  Note that in the context of graph, only nodes with edges connected are
  attended to each other, which means we do NOT compute $N^2$ pairs of scores
  as the original Transformer-style Attention.

  Users are able to remove the scaling of attention scores
  (`score_scaling="none"`) or add an activation on the transformed query
  (controlled by `attention_activation`). However, we recommend to remove the
  scaling when using an `attention_activation` since activating both of them may
  lead to degraded accuracy. One can also customize the transformation kernels
  with different initializers, regularizers as well as the use of bias terms,
  using the other arguments.

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

  For now, there is a variant that modifies the inputs transformation part and
  could potentially be beneficial:

      1. (transform_keys is False) Instead of projecting both queries and
        keys when computing attention weights, we only project the queries
        because the two linear projections can be collapsed to a single
        projection:

          $$ (Q_v W_Q^k)(K_u W_K^k)^T
            = Q_v (W_Q^k {W_K^k}^T) K_u^T
            = Q_v W_{QK}^k K_u^T $$

        where $d$ is the key width. (Following "Attention is all you need",
        this scaling is meant to achieve unit variance of the results, assuming
        that $Q_v W_{QK}^k$ has unit variance due to the initialization of
        $Q_v W_{QK}^k$.)

        NOTE: The single projection matrix behaves differently in
        gradient-descent training than the product of two matrices.

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
      IMPORTANT: Must be set to `None` for use with `receiver_tag=tfgnn.CONTEXT`
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
    inputs_dropout: Dropout rate for random dropout on the inputs to this
      convolution layer, i.e. the receiver, sender node, and sender edge inputs.
    attention_activation: The nonlinearity used on the transformed inputs
      (query, and keys if `transform_keys` is `True`) before computing the
      attention scores. This can be specified as a Keras layer, a
      tf.keras.activations.* function, or a string understood by
      `tf.keras.layers.Activation`. Defaults to None.
    activation: The nonlinearity applied to the final result of attention,
      specified in the same ways as attention_activation.
    kernel_initializer: Can be set to a `kernel_initializer` as understood
      by `tf.keras.layers.Dense` etc.
      An `Initializer` object gets cloned before use to ensure a fresh seed,
      if not set explicitly. For more, see `tfgnn.keras.clone_initializer()`.
    kernel_regularizer: Can be set to a `kernel_regularized` as understood
      by `tf.keras.layers.Dense` etc.
    transform_keys: If true, transform both queries and keys inputs. Otherwise,
      only queries are transformed since the two transformations on queries and
      keys are equivalent to one. (The presence of transformations on values is
      independent of this arg.)
    score_scaling: One of either `"none"`, `"rsqrt_dim"`, or
      `"trainable_sigmoid"`. If set to `"rsqrt_dim"`, the attention scores are
      divided by the square root of the dimension of keys (i.e.,
      `per_head_channels` if `transform_keys=True`, otherwise whatever the
      dimension of combined sender inputs is). If set to `"trainable_sigmoid"`,
      the scores are scaled with `sigmoid(x)`, where `x` is a trainable weight
      of the model that is initialized to `-5.0`, which initially makes all the
      attention weights equal and slowly ramps up as the other weights in the
      layer converge. Defaults to `"rsqrt_dim"`.
    transform_values_after_pooling: By default, each attention head applies
      the value transformation, then pools with attention coefficients.
      Setting this option pools inputs with attention coefficients, then applies
      the transformation. This is mathematically equivalent but can be faster
      or slower to compute, depending on the platform and the dataset.
      IMPORTANT: Toggling this option breaks checkpoint compatibility.
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
      inputs_dropout: float = 0.,
      attention_activation: Optional[Union[str, Callable[..., Any]]] = None,
      activation: Union[str, Callable[..., Any]] = "relu",
      kernel_initializer: Any = None,
      kernel_regularizer: Any = None,
      transform_keys: bool = True,
      score_scaling: Literal["none", "rsqrt_dim",
                             "trainable_sigmoid"] = "rsqrt_dim",
      transform_values_after_pooling: bool = False,
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

    # Create dropout layers. Note that if the dropout rate is zero, then the
    # layer will just be a pass-through.
    self._edge_dropout_layer = tf.keras.layers.Dropout(edge_dropout)
    self._inputs_dropout_layer = tf.keras.layers.Dropout(inputs_dropout)

    # Check for conflicting options.
    if attention_activation is not None and score_scaling != "none":
      warnings.warn(
          "using both an activation on transformed inputs and score scaling "
          "may lead to degraded accuracy if the activation function restricts "
          "the range of the values, e.g. 'tanh' which restricts the values to "
          "the range [-1, 1], Please consider using only one of them.")

    # Check for valid inputs.
    if (not self.takes_sender_node_input and not self.takes_sender_edge_input):
      raise ValueError("MultiHeadAttentionConv initialized with no inputs.")

    self._attention_activation = tf.keras.activations.get(attention_activation)
    self._activation = tf.keras.activations.get(activation)
    # IMPORTANT: Use with tfgnn.keras.clone_initializer(), b/268648226.
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self._transform_keys = transform_keys
    self._score_scaling = score_scaling
    self._transform_values_after_pooling = transform_values_after_pooling

    # The creation of queries transfomations is deferred to the first call of
    # `Convolve()` (see there).
    self._w_query = None

    # Create the transformations for keys inputs from sender nodes and edges.
    # No transformations will be created if we only transform queries.
    if not self._transform_keys:
      self._w_sender_node_to_key = None
      self._w_sender_edge_to_key = None
    else:
      if self.takes_sender_node_input:
        self._w_sender_node_to_key = tf.keras.layers.Dense(
            per_head_channels * num_heads,
            kernel_initializer=tfgnn.keras.clone_initializer(
                self._kernel_initializer),
            kernel_regularizer=kernel_regularizer,
            use_bias=use_bias,
            name="key_node")
      else:
        self._w_sender_node_to_key = None
      if self.takes_sender_edge_input:
        self._w_sender_edge_to_key = tf.keras.layers.Dense(
            per_head_channels * num_heads,
            kernel_initializer=tfgnn.keras.clone_initializer(
                self._kernel_initializer),
            kernel_regularizer=kernel_regularizer,
            # This bias would be redundant with self._w_sender_node_to_key.
            use_bias=use_bias and self._w_sender_node_to_key is None,
            name="key_edge")
      else:
        self._w_sender_edge_to_key = None

    if not self._transform_values_after_pooling:
      if self.takes_sender_node_input:
        self._w_sender_node_to_value = tf.keras.layers.Dense(
            per_head_channels * num_heads,
            kernel_initializer=tfgnn.keras.clone_initializer(
                self._kernel_initializer),
            kernel_regularizer=kernel_regularizer,
            use_bias=use_bias,
            name="value_node")
      else:
        self._w_sender_node_to_value = None
      if self.takes_sender_edge_input:
        self._w_sender_edge_to_value = tf.keras.layers.Dense(
            per_head_channels * num_heads,
            kernel_initializer=tfgnn.keras.clone_initializer(
                self._kernel_initializer),
            kernel_regularizer=kernel_regularizer,
            # This bias would be redundant with self._w_sender_node_to_value.
            use_bias=use_bias and self._w_sender_node_to_value is None,
            name="value_edge")
      else:
        self._w_sender_edge_to_value = None
    else:
      self._w_sender_pooled_to_value = tf.keras.layers.EinsumDense(
          equation="...hv,hvc->...hc",
          output_shape=(num_heads, per_head_channels),
          bias_axes="hc" if use_bias else None,
          kernel_initializer=tfgnn.keras.clone_initializer(
              self._kernel_initializer),
          kernel_regularizer=kernel_regularizer,
          name="value_pooled")

    if self._score_scaling == "trainable_sigmoid":
      self._score_scaling_weight = None

  def get_config(self):
    return dict(
        num_heads=self._num_heads,
        per_head_channels=self._per_head_channels,
        use_bias=self._use_bias,
        edge_dropout=self._edge_dropout_layer.rate,
        inputs_dropout=self._inputs_dropout_layer.rate,
        # All forms of activation functions can be returned as-is:
        # - A Keras Layer is serialized and deserialized recursively through
        #   its own get_config/from config methods. It's best to not try and
        #   simulate that recursive process here.
        # - A str with a name may be passed to __init__, but __init__ anyways
        #   calls .get() to turn it into a function from tf.keras.activations.*.
        # - A function from tf.keras.activations.* is automatically serialized
        #   and deserialized as its name, and then converted to a function by
        #   __init__. (Activation functions that require to save a hparam,
        #  such as LeakyReLU, are Layer objects, not functions.)
        attention_activation=self._attention_activation,
        activation=self._activation,
        # Regularizers and initializers need explicit serialization here
        # (and deserialization in __init__ via .get()) due to b/238163789.
        kernel_initializer=tf.keras.initializers.serialize(
            self._kernel_initializer),
        kernel_regularizer=tf.keras.regularizers.serialize(
            self._kernel_regularizer),
        transform_keys=self._transform_keys,
        score_scaling=self._score_scaling,
        transform_values_after_pooling=self._transform_values_after_pooling,
        **super().get_config())

  def convolve(self,
               *,
               sender_node_input: Optional[tf.Tensor],
               sender_edge_input: Optional[tf.Tensor],
               receiver_input: Optional[tf.Tensor],
               broadcast_from_sender_node: Callable[[tf.Tensor], tf.Tensor],
               broadcast_from_receiver: Callable[[tf.Tensor], tf.Tensor],
               pool_to_receiver: Callable[..., tf.Tensor],
               extra_receiver_ops: Optional[Mapping[str, Callable[...,
                                                                  Any]]] = None,
               **kwargs) -> tf.Tensor:

    # Apply dropout on the inputs.
    receiver_input = self._inputs_dropout_layer(receiver_input)
    if sender_node_input is not None:
      sender_node_input = self._inputs_dropout_layer(sender_node_input)
    if sender_edge_input is not None:
      sender_edge_input = self._inputs_dropout_layer(sender_edge_input)

    # Determine the width of transformed queries and create transfomations.
    # If transform_keys is true, queries will be transformed to
    # self._per_head_channels. Otherwise, transform the queries to match
    # the width of raw sender inputs (keys).
    if self._w_query is None:
      with tf.init_scope():
        if not self._transform_keys:
          keys_width = 0
          if sender_node_input is not None:
            keys_width += sender_node_input.shape[-1]
          if sender_edge_input is not None:
            keys_width += sender_edge_input.shape[-1]
          self._w_query = tf.keras.layers.Dense(
              keys_width * self._num_heads,
              kernel_initializer=tfgnn.keras.clone_initializer(
                  self._kernel_initializer),
              kernel_regularizer=self._kernel_regularizer,
              use_bias=self._use_bias,
              name="query")
        else:
          self._w_query = tf.keras.layers.Dense(
              self._per_head_channels * self._num_heads,
              kernel_initializer=tfgnn.keras.clone_initializer(
                  self._kernel_initializer),
              kernel_regularizer=self._kernel_regularizer,
              use_bias=self._use_bias,
              name="query")
    assert self._w_query is not None

    # Form the attention query for each head.
    # If transform_keys is true, it has the shape:
    # [num_items, *extra_dims, num_heads, channels_per_head]
    # Otherwise, the shape is: [num_items, *extra_dims, num_heads, keys_width].
    assert receiver_input is not None, "__init__() should have checked this."
    queries = self._w_query(receiver_input)
    queries = self._attention_activation(queries)
    queries = broadcast_from_receiver(self._split_heads(queries))

    # Form the attention key for each head.
    # If transform_keys is true, the pieces of keys inputs are transformed to
    # [num_items, *extra_dims, num_heads, channels_per_head] and the results
    # are added, which allows transformation for the piece from the nodes before
    # broadcasting it and equals to first concatenating the pieces and
    # then transforming them to channels_per_head.
    # If transform_keys is false, the pieces of keys inputs are concatenated on
    # last axis with a shape [num_items, *extra_dims, num_heads, keys_width].
    keys = []
    if not self._transform_keys:
      if sender_node_input is not None:
        keys.append(
            tf.expand_dims(
                broadcast_from_sender_node(sender_node_input), axis=-2))
      if sender_edge_input is not None:
        keys.append(tf.expand_dims(sender_edge_input, axis=-2))
      keys = tf.concat(keys, axis=-1)
    else:
      if sender_node_input is not None and sender_edge_input is None:
        # In this special case, we can apply the attention_activation first
        # and then broadcast its results.
        keys = broadcast_from_sender_node(
            self._split_heads(
                self._attention_activation(
                    self._w_sender_node_to_key(sender_node_input))))
      else:
        # In the general case, the attention_activation (if any) comes last.
        if sender_node_input is not None:
          keys.append(
              broadcast_from_sender_node(
                  self._split_heads(
                      self._w_sender_node_to_key(sender_node_input))))
        if sender_edge_input is not None:
          keys.append(
              self._split_heads(self._w_sender_edge_to_key(sender_edge_input)))
        keys = tf.add_n(keys)
        keys = self._attention_activation(keys)

    # Dot-product of queries and keys to produce the attention coefficients.
    # [num_items, *extra_dims, num_heads, 1]
    attention_coefficients = tf.expand_dims(
        tf.einsum("...j,...j->...", queries, keys), axis=-1)

    # Optionally scale the attention scores.
    if self._score_scaling == "none":
      pass
    elif self._score_scaling == "rsqrt_dim":
      attention_coefficients *= tf.math.rsqrt(
          tf.cast(tf.shape(keys)[-1], tf.float32))
    elif self._score_scaling == "trainable_sigmoid":
      if self._score_scaling_weight is None:
        self._score_scaling_weight = self.add_weight(
            name="score_scaling",
            shape=[],
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(-5.0),
            trainable=True,
        )
      attention_coefficients *= tf.keras.activations.sigmoid(
          self._score_scaling_weight)
    else:
      raise ValueError("Unknown value MultiHeadAttentionConv("
                       f"score_scaling='{self._score_scaling}')")

    attention_coefficients = extra_receiver_ops["softmax"](
        attention_coefficients)

    # Add layer with dropout to the normalized attention coefficients. This
    # should have the same effect as edge dropout. Also, note that
    # `keras.layers.Dropout` upscales the remaining values, which should
    # maintain the sum-up-to-1 per node in expectation.
    attention_coefficients = self._edge_dropout_layer(attention_coefficients,
                                                      **kwargs)

    # Compute the pooled values by
    #   * transforming the inputs and
    #   * computing their weighted sum according to the attention coefficients.
    # These two operations are linear, so, mathematically, they can be applied
    # in either order. It depends on input/output dimensions, the ratio of
    # num_items to num_receivers and the platform which one is faster.
    if not self._transform_values_after_pooling:
      # Option 1: First transform the inputs, then pool the values.
      # The transformation is split into the terms for node inputs and edge
      # inputs, so that each node input is transformed once before broadcasting.
      #
      # Compute the transformed inputs.
      # [num_items, *extra_dims, num_heads, per_head_channels]
      value_terms = []
      if sender_node_input is not None:
        value_terms.append(
            broadcast_from_sender_node(
                self._split_heads(
                    self._w_sender_node_to_value(sender_node_input))))
      if sender_edge_input is not None:
        value_terms.append(
            self._split_heads(self._w_sender_edge_to_value(sender_edge_input)))
      values = tf.add_n(value_terms)
      # Compute the weighed sum.
      # [num_receivers, *extra_dims, num_heads, per_head_channels]
      weighted_values = values * attention_coefficients
      pooled_values = pool_to_receiver(weighted_values, reduce_type="sum")
    else:
      # Option 2: First pool the inputs, then apply the value transformation.
      # This reduces the number of transformations from num_items to
      # num_receivers.
      #
      # Collect the inputs for each item (same for all heads).
      # [num_items, *extra_dims, 1, input_channels]
      input_parts = []
      if sender_node_input is not None:
        input_parts.append(broadcast_from_sender_node(sender_node_input))
      if sender_edge_input is not None:
        input_parts.append(sender_edge_input)
      value_inputs = tf.expand_dims(tf.concat(input_parts, axis=-1), axis=-2)
      # Compute the weighed sum.
      # [num_receivers, *extra_dims, num_heads, input_channels]
      weighted_inputs = value_inputs * attention_coefficients
      pooled_inputs = pool_to_receiver(weighted_inputs, reduce_type="sum")
      # Apply the transformation.
      # [num_receivers, *extra_dims, num_heads, per_head_channels]
      pooled_values = self._w_sender_pooled_to_value(pooled_inputs)

    # Apply the nonlinearity on the final result.
    pooled_values = self._activation(pooled_values)
    # Merge heads for output.
    pooled_values = self._merge_heads(pooled_values)

    return pooled_values

  # The following helpers map back and forth between tensors with...
  #  - a separate heads dimension: shape [..., num_heads, channels_per_head],
  #  - all heads concatenated:    shape [..., num_heads * channels_per_head].

  def _split_heads(self, tensor):
    assert tensor.shape[-1] is not None
    assert tensor.shape[-1] % self._num_heads == 0, (
        f"{tensor.shape[-1]} not"
        f"divisible by {self._num_heads}")
    per_head_channels = tensor.shape[-1] // self._num_heads
    extra_dims = tensor.shape[1:-1]  # Possibly empty.
    if not extra_dims.is_fully_defined():
      raise ValueError(
          "MultiHeadAttentionConv requires non-ragged Tensors as inputs, "
          "and GraphTensor requires these to have statically known "
          f"dimensions except the first, but got {tensor.shape}")
    new_shape = (-1, *extra_dims, self._num_heads, per_head_channels)
    return tf.reshape(tensor, new_shape)

  def _merge_heads(self, tensor):
    num_merged = 2
    extra_dims = tensor.shape[1:-num_merged]  # Possibly empty.
    merged_dims = tensor.shape[-num_merged:]
    if not extra_dims.is_fully_defined() or not merged_dims.is_fully_defined():
      raise ValueError(f"Unexpected unknown dimensions in shape {tensor.shape}")
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
      If set to `tfgnn.CONTEXT`, the layer can be called for an edge set or node
      set. If set to an IncidentNodeTag (e.g., `tfgnn.SOURCE` or
      `tfgnn.TARGET`), the layer can be called for an edge set and will
      aggregate results at the specified endpoint of the edges. If left unset,
      the tag must be passed when calling the layer.
    receiver_feature: By default, the default state feature of the receiver is
      used to compute the attention query. A different feature name can be
      selected by setting this argument.
    sender_feature: By default, the default state feature of the edge set is
      used to compute the attention values. A different feature name can be
      selected by setting this argument.
    **kwargs: Any other option for MultiHeadAttentionConv, except
      sender_node_feature, which is set to None.
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


# TODO(b/286015280): a systematic solution for adding loops.
def MultiHeadAttentionHomGraphUpdate(
    *,  # To be called like a class initializer.  pylint: disable=invalid-name
    num_heads: int,
    per_head_channels: int,
    receiver_tag: tfgnn.IncidentNodeTag,
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
    node_set_name, edge_set_name = tfgnn.get_homogeneous_node_and_edge_set_name(
        spec, "MultiHeadAttentionHomGraphUpdate")
    node_set_updates = {
        node_set_name:
            tfgnn.keras.layers.NodeSetUpdate(
                {
                    edge_set_name:
                        MultiHeadAttentionConv(
                            num_heads=num_heads,
                            per_head_channels=per_head_channels,
                            receiver_tag=receiver_tag,
                            sender_node_feature=feature_name,
                            receiver_feature=feature_name,
                            **kwargs)  # kernel_initializer cloned by layer.
                },
                next_state=tfgnn.keras.layers.SingleInputNextState(),
                node_input_feature=feature_name)
    }
    return dict(node_sets=node_set_updates)

  return tfgnn.keras.layers.GraphUpdate(
      deferred_init_callback=deferred_init_callback, name=name)


def MultiHeadAttentionMPNNGraphUpdate(  # To be called like a class initializer.  pylint: disable=invalid-name
    # LINT.IfChange(MultiHeadAttentionMPNNGraphUpdate_args)
    *,
    units: int,
    message_dim: int,
    num_heads: int,
    receiver_tag: tfgnn.IncidentNodeTag,
    node_set_names: Optional[Collection[tfgnn.NodeSetName]] = None,
    edge_feature: Optional[tfgnn.FieldName] = None,
    l2_regularization: float = 0.0,
    edge_dropout_rate: float = 0.0,
    state_dropout_rate: float = 0.0,
    attention_activation: Optional[Union[str, Callable[..., Any]]] = None,
    conv_activation: Union[str, Callable[..., Any]] = "relu",
    activation: Union[str, Callable[..., Any]] = "relu",
    kernel_initializer: Any = "glorot_uniform",
    # LINT.ThenChange(./config_dict.py:graph_update_get_config_dict)
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
    message_dim: The dimension of messages (attention values) computed on each
      edge.  Must be divisible by `num_heads`.
    num_heads: The number of attention heads used by MultiHeadAttention.
      `message_dim` must be divisible by this number.
    receiver_tag: one of `tfgnn.TARGET` or `tfgnn.SOURCE`, to select the
      incident node of each edge that receives the message.
    node_set_names: The names of node sets to update. If unset, updates all that
      are on the receiving end of any edge set.
    edge_feature: Can be set to a feature name of the edge set to select it as
      an input feature. By default, this set to `None`, which disables this
      input.
    l2_regularization: The coefficient of L2 regularization for weights and
      biases.
    edge_dropout_rate: The edge dropout rate applied during attention pooling of
      edges.
    state_dropout_rate: The dropout rate applied to the resulting node states.
    attention_activation: The nonlinearity used on the transformed inputs before
      multiplying with the trained weights of the attention layer. This can be
      specified as a Keras layer, a tf.keras.activations.* function, or a string
      understood by `tf.keras.layers.Activation`. Defaults to None.
    conv_activation: The nonlinearity applied to the result of attention on one
      edge set, specified in the same ways as attention_activation.
    activation: The nonlinearity applied to the new node states computed by this
      graph update.
    kernel_initializer: Can be set to a `kernel_initializer` as understood
      by `tf.keras.layers.Dense` etc.
      An `Initializer` object gets cloned before use to ensure a fresh seed,
      if not set explicitly. For more, see `tfgnn.keras.clone_initializer()`.

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
            kernel_initializer=tfgnn.keras.clone_initializer(
                kernel_initializer),
            bias_initializer="zeros",
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer),
        tf.keras.layers.Dropout(state_dropout_rate)
    ])

  # pylint: disable=g-long-lambda
  gnn_builder = tfgnn.keras.ConvGNNBuilder(
      lambda edge_set_name, receiver_tag: MultiHeadAttentionConv(
          num_heads=num_heads,
          per_head_channels=per_head_channels,
          edge_dropout=edge_dropout_rate,
          receiver_tag=receiver_tag,
          sender_edge_feature=edge_feature,
          attention_activation=attention_activation,
          activation=conv_activation,
          kernel_initializer=kernel_initializer),  # Cloned by the layer.
      lambda node_set_name: tfgnn.keras.layers.NextStateFromConcat(
          dense(units)),
      receiver_tag=receiver_tag)
  return gnn_builder.Convolve(node_set_names)
