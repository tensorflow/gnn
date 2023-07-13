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
"""Contains the Heterogeneous Graph Transformer layer implementation.

This file contains an implementation of HGT from Hu et al. 2020.
"""
import collections
from typing import Any, Callable, Union

import tensorflow as tf
import tensorflow_gnn as tfgnn


@tf.keras.utils.register_keras_serializable(package='GNN>models>hgt')
class HGTGraphUpdate(tf.keras.layers.Layer):
  """Graph Update Layer for Heterogeneous Graph Transformers (HGT).

  This layer performs one round of state updates for the
  [Heterogeneous Graph Transformer](https://arxiv.org/abs/2003.01332) by
  Hu et al. (2020).

  This implementation combines the description of the algorithm in the paper
  with some details resolved from the authors' linked implementation. In
  practice this means splitting per-relation weights per attention head
  and introducing a learnable skip connection parameter. This implementation
  also assumes that the paper's notion of "edge type" is equivalent to
  TFGNN "edge_set_name". Edge features are ignored as in the paper.

  Init args:
    num_heads: The number of attention heads.
    per_head_channels: The dimensionality of each attention head
    receiver_tag: `tfgnn.TARGET` or `tfgnn.SOURCE`, (Source and target are
      conventional names for the incident nodes of a directed edge, data flow in
      a GNN may happen in either direction.)
    use_weighted_skip: If True, applies a learnable weighted average
      parameter to combine the inputs with the Transformer output. This is
      implemented in the reference implementation for HGT, even though the paper
      describes a plain addition.
    dropout_rate: Fraction of final node activations that are dropped out
    use_layer_norm: If True, applies layer normalization to the resulting
      node transformation.
    kernel_initializer: Can be set to a `kernel_initializer` as understood
      by `tf.keras.layers.Dense` etc.
      An `Initializer` object gets cloned before use to ensure a fresh seed,
      if not set explicitly. For more, see `tfgnn.keras.clone_initializer()`.
    use_bias: If True, bias terms are added to the transformations of query,
      key, message, and aggregation inputs.
    name: Optionally, a name for the layer returned.
    **kwargs: Any optional arguments to HgtGraphUpdate.
  """

  def __init__(
      # LINT.IfChange(HGTGraphUpdate_args)
      self,
      *,
      num_heads: int,
      per_head_channels: int,
      receiver_tag: tfgnn.IncidentNodeTag,
      use_weighted_skip: bool = True,
      dropout_rate: float = 0.2,
      use_layer_norm: bool = True,
      kernel_initializer: Any = None,
      use_bias: bool = True,
      activation: Union[str, Callable[..., Any]] = 'gelu',
      feature_name: str = tfgnn.HIDDEN_STATE,
      **kwargs,
      # LINT.ThenChange(./config_dict.py:graph_update_get_config_dict)
  ):
    super().__init__(**kwargs)
    self._is_initialized = False
    self._num_heads = num_heads
    self._per_head_channels = per_head_channels
    self._receiver_tag = receiver_tag
    self._use_weighted_skip = use_weighted_skip
    self._dropout_rate = dropout_rate
    self._use_layer_norm = use_layer_norm
    # IMPORTANT: Use with tfgnn.keras.clone_initializer(), b/268648226.
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._use_bias = use_bias
    self._activation = tf.keras.activations.get(activation)
    self._feature_name = feature_name

  def get_config(self):
    return dict(
        num_heads=self._num_heads,
        per_head_channels=self._per_head_channels,
        receiver_tag=self._receiver_tag,
        use_weighted_skip=self._use_weighted_skip,
        dropout_rate=self._dropout_rate,
        use_layer_norm=self._use_layer_norm,
        kernel_initializer=tf.keras.initializers.serialize(  # b/238163789
            self._kernel_initializer),
        use_bias=self._use_bias,
        activation=self._activation,
        feature_name=self._feature_name,
        **super().get_config(),
    )

  def _init_from_spec(self, spec: tfgnn.GraphTensorSpec):
    units = self._num_heads * self._per_head_channels
    receiver_tag = self._receiver_tag
    sender_tag = tfgnn.reverse_tag(receiver_tag)

    edge_sets_spec = {k: v for k, v in spec.edge_sets_spec.items()
                      if not tfgnn.get_aux_type_prefix(k)}
    self._receivers = {
        edge_set_spec.adjacency_spec.node_set_name(receiver_tag)
        for edge_set_spec in edge_sets_spec.values()}
    self._senders = {
        edge_set_spec.adjacency_spec.node_set_name(sender_tag)
        for edge_set_spec in edge_sets_spec.values()}
    bad_node_set_names = {
        name for name in list(self._receivers) + list(self._receivers)
        if tfgnn.get_aux_type_prefix(name)}
    if bad_node_set_names:
      raise ValueError(
          f'Auxiliary node sets {bad_node_set_names} '
          f'are incident to non-auxiliary edge sets.')

    self._dropout = tf.keras.layers.Dropout(self._dropout_rate)

    self._key_projections = {}
    self._message_projections = {}
    for node_set_name in self._senders:
      self._key_projections[node_set_name] = tf.keras.layers.Dense(
          units,
          kernel_initializer=tfgnn.keras.clone_initializer(
              self._kernel_initializer),
          use_bias=self._use_bias,
          name=f'key_node_{node_set_name}')
      self._message_projections[node_set_name] = tf.keras.layers.Dense(
          units,
          kernel_initializer=tfgnn.keras.clone_initializer(
              self._kernel_initializer),
          use_bias=self._use_bias,
          name=f'message_node_{node_set_name}')

    self._query_projections = {}
    self._aggr_projections = {}
    self._skip_connection_weights = {}
    self._norms = {}
    self._is_state_size_constant = {}
    for node_set_name in self._receivers:
      self._query_projections[node_set_name] = tf.keras.layers.Dense(
          units,
          kernel_initializer=tfgnn.keras.clone_initializer(
              self._kernel_initializer),
          use_bias=self._use_bias,
          name=f'query_node_{node_set_name}')
      self._aggr_projections[node_set_name] = tf.keras.layers.Dense(
          units,
          kernel_initializer=tfgnn.keras.clone_initializer(
              self._kernel_initializer),
          use_bias=self._use_bias,
          name=f'aggr_node_{node_set_name}',
      )
      if self._use_layer_norm:
        self._norms[node_set_name] = tf.keras.layers.LayerNormalization(
            name=f'normalization_{node_set_name}'
        )
      else:
        self._norms[node_set_name] = tf.keras.layers.Layer(
            name='no_normalization'
        )
      feature_shape = spec.node_sets_spec[
          node_set_name][self._feature_name].shape
      output_shape = tf.TensorShape([
          *feature_shape[:-1],
          self._num_heads * self._per_head_channels,
      ])
      self._is_state_size_constant[node_set_name] = (
          feature_shape.is_compatible_with(output_shape)
      )
      if self._is_state_size_constant[node_set_name]:
        if self._use_weighted_skip:
          self._skip_connection_weights[node_set_name] = self.add_weight(
              shape=(),
              initializer='ones',
              trainable=True,
              name=f'skip_{node_set_name}',
          )
      elif feature_shape[1:].num_elements() != 0:
        raise ValueError(
            'The input features need to either be empty or the '
            f'same shape as the output. However, inputs are {feature_shape} '
            f'and outputs will be {output_shape} for NodeSet {node_set_name}.'
        )

    # This code treats each EdgeSet as a distinct edge type (see docstring)
    # and hence uses edge_set_name as the key here.
    self._edge_type_attention_projections = {}
    self._edge_type_message_projections = {}
    self._edge_type_priors = {}
    for edge_set_name in edge_sets_spec:
      self._edge_type_attention_projections[
          edge_set_name] = tf.keras.layers.EinsumDense(
              equation='...jk,jkl->...jl',
              output_shape=(self._num_heads, self._per_head_channels),
              kernel_initializer=tfgnn.keras.clone_initializer(
                  self._kernel_initializer),
              name=f'attention_edge_{edge_set_name}'
          )
      self._edge_type_message_projections[
          edge_set_name] = tf.keras.layers.EinsumDense(
              equation='...jk,jkl->...jl',
              output_shape=(self._num_heads, self._per_head_channels),
              kernel_initializer=tfgnn.keras.clone_initializer(
                  self._kernel_initializer),
              name=f'message_edge_{edge_set_name}'
          )
      self._edge_type_priors[edge_set_name] = self.add_weight(
          shape=(self._num_heads),
          initializer='ones',
          trainable=True,
          name=f'priors_{edge_set_name}',
      )

  # The following helpers map back and forth between tensors with...
  #  - a separate heads dimension: shape [..., num_heads, channels_per_head],
  #  - all heads concatenated:    shape [..., num_heads * channels_per_head].

  def _split_heads(self, tensor):
    assert tensor.shape[-1] is not None
    assert (
        tensor.shape[-1] % self._num_heads == 0
    ), f'{tensor.shape[-1]} not divisible by {self._num_heads}'
    per_head_channels = self._per_head_channels
    extra_dims = tensor.shape[1:-1]  # Possibly empty.
    if not extra_dims.is_fully_defined():
      raise ValueError(
          'HGT requires non-ragged Tensors as inputs, '
          'and GraphTensor requires these to have statically known '
          f'dimensions except the first, but got {tensor.shape}'
      )
    new_shape = (-1, *extra_dims, self._num_heads, per_head_channels)
    return tf.reshape(tensor, new_shape)

  def _merge_heads(self, tensor):
    num_merged = 2
    extra_dims = tensor.shape[1:-num_merged]  # Possibly empty.
    merged_dims = tensor.shape[-num_merged:]
    if not extra_dims.is_fully_defined() or not merged_dims.is_fully_defined():
      raise ValueError(f'Unexpected unknown dimensions in shape {tensor.shape}')
    new_shape = (-1, *extra_dims, merged_dims.num_elements())
    return tf.reshape(tensor, new_shape)

  def _graph_update(self, graph: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
    receiver_tag = self._receiver_tag
    sender_tag = tfgnn.reverse_tag(receiver_tag)

    # Compute keys and messages for senders
    keys_by_sender = {}
    messages_by_sender = {}
    for node_set_name in self._senders:
      x = graph.node_sets[node_set_name][self._feature_name]
      keys_by_sender[node_set_name] = self._split_heads(
          self._key_projections[node_set_name](x)
      )
      messages_by_sender[node_set_name] = self._split_heads(
          self._message_projections[node_set_name](x)
      )

    # Compute queries for receivers
    queries_by_receiver = {}
    for node_set_name in self._receivers:
      x = graph.node_sets[node_set_name][self._feature_name]
      queries_by_receiver[node_set_name] = self._split_heads(
          self._query_projections[node_set_name](x),
      )

    # Broadcast the scores and messages over the edge sets
    messages_by_edge_set = {}
    scores_by_receiver = collections.defaultdict(dict)
    rsqrt_dim = tf.math.rsqrt(tf.cast(self._per_head_channels, tf.float32))
    for edge_set_name, edge_set in graph.edge_sets.items():
      if tfgnn.get_aux_type_prefix(edge_set_name):
        continue
      sender_name = edge_set.adjacency.node_set_name(sender_tag)
      receiver_name = edge_set.adjacency.node_set_name(receiver_tag)

      messages = self._edge_type_message_projections[edge_set_name](
          messages_by_sender[sender_name])
      messages_by_edge_set[edge_set_name] = tfgnn.broadcast_node_to_edges(
          graph,
          edge_set_name,
          sender_tag,
          feature_value=messages,
      )
      keys = tfgnn.broadcast_node_to_edges(
          graph,
          edge_set_name,
          sender_tag,
          feature_value=keys_by_sender[sender_name],
      )
      queries = self._edge_type_attention_projections[edge_set_name](
          queries_by_receiver[receiver_name])
      queries = tfgnn.broadcast_node_to_edges(
          graph,
          edge_set_name,
          receiver_tag,
          feature_value=queries,
      )
      scores = (tf.einsum('...i,...i->...', queries, keys)
                * self._edge_type_priors[edge_set_name]
                * rsqrt_dim)
      scores_by_receiver[receiver_name][edge_set_name] = scores

    # Apply softmax to the scores to get the attention coefficients
    coefficients_by_receiver = {}
    for node_set_name in self._receivers:
      edge_set_names, scores = zip(*scores_by_receiver[node_set_name].items())
      coefficients = tfgnn.softmax(graph, receiver_tag,
                                   edge_set_name=edge_set_names,
                                   feature_value=scores)
      coefficients_by_receiver[node_set_name] = dict(zip(edge_set_names,
                                                         coefficients))

    # Scale the messages and pool them to the receiver nodes
    pooled_messages_by_receiver = collections.defaultdict(list)
    for edge_set_name, messages_on_edge in messages_by_edge_set.items():
      edge_set = graph.edge_sets[edge_set_name]
      receiver_name = edge_set.adjacency.node_set_name(receiver_tag)
      coefficients_on_edge = coefficients_by_receiver[receiver_name][
          edge_set_name]
      scaled_messages = (
          messages_on_edge * coefficients_on_edge[..., tf.newaxis])
      pooled_messages = tfgnn.pool_edges_to_node(
          graph,
          edge_set_name,
          receiver_tag,
          reduce_type='sum',
          feature_value=scaled_messages,
      )
      pooled_messages_by_receiver[receiver_name].append(
          self._merge_heads(pooled_messages))

    updated_node_features = {}
    # Update the receiver node states
    for node_set_name in self._receivers:
      node_set = graph.node_sets[node_set_name]
      res = tf.add_n(pooled_messages_by_receiver[node_set_name])
      res = self._aggr_projections[node_set_name](self._activation(res))
      res = self._dropout(res)
      # Shapes should be the same in order to add a residual connection
      # Otherwise, the features are empty (like in latent features) or the
      # initialization function would have thrown an error
      if self._is_state_size_constant[node_set_name]:
        if self._use_weighted_skip:
          alpha = tf.sigmoid(self._skip_connection_weights[node_set_name])
          res = res * alpha + node_set[self._feature_name] * (1 - alpha)
        else:
          res = res + node_set[self._feature_name]
      features = graph.node_sets[node_set_name].get_features_dict()  # Copy
      features[self._feature_name] = self._norms[node_set_name](res)
      updated_node_features[node_set_name] = features

    return graph.replace_features(node_sets=updated_node_features)

  def call(self, graph: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
    tfgnn.check_scalar_graph_tensor(graph, 'HGTGraphUpdate')
    if not self._is_initialized:
      with tf.init_scope():
        self._init_from_spec(graph.spec)
        self._is_initialized = True
    assert self._is_initialized
    return self._graph_update(graph)
