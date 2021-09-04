"""Contains a Graph Attention Network v2 and associated layers."""
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import graph_tensor_ops as ops
from tensorflow_gnn.graph import normalization_ops


class GATv2(tf.keras.layers.Layer):
  """Simple Graph Attention Network V2 (GATv2).

  Based off of https://arxiv.org/abs/2105.14491, the GATv2 brings strict
  improvements over the original GAT (https://arxiv.org/abs/1710.10903) by
  allowing the network to compute a more expressived "dynamic" instead of
  just "static" attention. See the above papers for more details.
  """

  def __init__(self,
               num_heads: int,
               per_head_channels: int,
               edge_set_name: str,
               feature_name: str = const.DEFAULT_STATE_NAME,
               output_feature_name: str = const.DEFAULT_STATE_NAME,
               use_bias: bool = True,
               edge_dropout: float = 0.,
               query_kernel_initializer: Optional[
                   tf.keras.initializers.Initializer] = None,
               key_kernel_initializer: Optional[
                   tf.keras.initializers.Initializer] = None,
               attention_kernel_initializers: Optional[
                   tf.keras.initializers.Initializer] = None,
               **kwargs):
    """Initializes the Graph Attention Network v2.

    Like the Keras Dense module, if the input features have rank greater than 2,
    this computes a point-wise GAT along the last axis of the inputs. For
    example, if the input features is [num_nodes, 2, 4, 1], then this will
    perform an identical GAT layer on each of the num_nodes * 2 * 4 input
    values.

    Args:
      num_heads: The number of attention heads.
      per_head_channels: The number of channels for each attention head. This
        means that the final output size will be per_head_channels * num_heads.
      edge_set_name: The edge set name indexing the nodes to run over.
      feature_name: The name of the feature to run over.
      output_feature_name: The name for the feature where the results will be
        stored in the returned GraphTensor.
      use_bias: If true, add a bias term to the transformation of the input
        features. For consistency with the GATv2 paper's code at
        https://github.com/tech-srl/how_attentive_are_gats/blob/main/gatv2_conv_PyG.py,
          a bias term is not used in the attention weights.
      edge_dropout: The percentage (between 0 and 1) of edges to randomly drop
        during training.
      query_kernel_initializer: An initializer for the `query` part of the
        linear transformation of the input.
      key_kernel_initializer: An initializer for the `key` part of the linear
        transformation of the input.
      attention_kernel_initializers: Initializers for the attention logit
        function weights. Note that this will be equally partitioned into
        separate weights for each head.
      **kwargs: Any extra arguments to pass to the super class's init.

    Raises:
      ValueError if the `softmax` reduce operation is not registered.
      ValueError if num_heads is less than 1.
      ValueError if per_head_channels is less than 1.
    """
    super().__init__(**kwargs)

    if num_heads <= 0:
      raise ValueError(f'Number of heads {num_heads} must be greater than 0.')
    if per_head_channels <= 0:
      raise ValueError(
          f'Per-head channels {per_head_channels} must be greater than 0.')

    self._num_heads = num_heads
    self._per_head_channels = per_head_channels
    self._use_bias = use_bias
    self._edge_set_name = edge_set_name
    self._feature_name = feature_name
    self._output_feature_name = output_feature_name
    self._edge_dropout = edge_dropout
    self._query_kernel_initializer = query_kernel_initializer
    self._key_kernel_initializer = key_kernel_initializer
    self._attention_kernel_initializers = attention_kernel_initializers

    # Decompose W into W_query (left) and W_key (right). See impl for details.
    self._w_query = tf.keras.layers.Dense(
        per_head_channels * num_heads,
        kernel_initializer=query_kernel_initializer,
        use_bias=use_bias,
        name='gatv2_query')
    self._w_key = tf.keras.layers.Dense(
        per_head_channels * num_heads,
        kernel_initializer=key_kernel_initializer,
        use_bias=use_bias,
        name='gatv2_key')

    # Multi-head attention pooling.
    self._attention_pool = GATv2AttentionPool(
        num_heads,
        per_head_channels,
        edge_dropout=edge_dropout,
        tag=const.TARGET,
        edge_set_name=self._edge_set_name,
        attention_kernel_initializers=attention_kernel_initializers)

  def call(self, graph: gt.GraphTensor, training=False) -> gt.GraphTensor:
    """Runs a single GATv2 layer on the input graph.

    Args:
      graph: The input, which should be a scalar GraphTensor, i.e. its batch
        dimension has been merged to the components dimension.
      training: True iff we are training, not evaluating. Used to enable
        training-specific features like dropout.

    Returns:
      A new GraphTensor after the GATv2 has been applied.

    Raises:
      ValueError if the input GraphTensor is not a scalar.
      ValueError if the edge_set_name is not in the GraphTensor's edge sets.
    """
    if graph.shape.rank != 0:
      raise ValueError(
          f'Input GraphTensor must be a scalar, but had rank {graph.shape.rank}'
      )

    if self._edge_set_name not in graph.edge_sets:
      raise ValueError(f'Edge {self._edge_set_name} not in Graph edge sets')
    adjacency = graph.edge_sets[self._edge_set_name].adjacency

    query_node = graph.node_sets[adjacency.source_name]
    key_node = graph.node_sets[adjacency.target_name]

    # Decompose W*[query || key] into W_Left * query + W_Right * key,
    # since we'll need W_Left * query later. See GATv2AttentionPool for details.
    # Note that we are performing the transformation before broadcasting.
    # [num_nodes, opt_extra_dims, per_head_channels * num_heads]
    query = self._w_query(query_node[self._feature_name])
    key = self._w_key(key_node[self._feature_name])

    # Broadcast these features to get them ready for the pooling layer.
    # [n_edges, opt_extra_dims, per_head_channels * num_heads]
    query_broadcasted = ops.broadcast_node_to_edges(
        graph, self._edge_set_name, const.SOURCE, feature_value=query)
    key_broadcasted = ops.broadcast_node_to_edges(
        graph, self._edge_set_name, const.TARGET, feature_value=key)

    # Compute attention pooling to get the output features.
    pooled = self._attention_pool((graph, query_broadcasted, key_broadcasted),
                                  training=training)

    # Add these features to the GraphTensor.
    features = graph.node_sets[adjacency.target_name].get_features_dict()
    features[self._output_feature_name] = pooled
    return graph.replace_features(node_sets={adjacency.target_name: features})

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'num_heads': self._num_heads,
        'per_head_channels': self._per_head_channels,
        'edge_set_name': self._edge_set_name,
        'feature_name': self._feature_name,
        'output_feature_name': self._output_feature_name,
        'use_bias': self._use_bias,
        'edge_dropout': self._edge_dropout,
        'query_kernel_initializer': self._query_kernel_initializer,
        'key_kernel_initializer': self._key_kernel_initializer,
        'attention_kernel_initializers': self._attention_kernel_initializers,
    })
    return config


class GATv2AttentionPool(tf.keras.layers.Layer):
  """GATv2 multi-head attention pooling.

  Implements the pooling layer describe in https://arxiv.org/abs/2105.14491
  Equations (3) and (4). That is, given the edge values, this layer computes the
  attention coefficients and multiplies them by the edges, and aggregates these
  by summing them on a per-node basis.
  """

  def __init__(self,
               num_heads: int,
               per_head_channels: int,
               edge_dropout: float = 0.,
               attention_kernel_initializers: Optional[
                   tf.keras.initializers.Initializer] = None,
               tag: Optional[const.IncidentNodeTag] = None,
               edge_set_name: Optional[const.EdgeSetName] = None,
               **kwargs):
    """Initializes the GATv2 multi-head attention pooling layer.

    Like the Keras Dense module, if the input features have rank greater than
    2, this computes Pooling along the last axis of the inputs. See the
    documentation for the GATv2 class for more details.

    Args:
      num_heads: The number of attention heads.
      per_head_channels: The number of channels for each attention head. This
        means that the final output size will be per_head_channels * num_heads.
      edge_dropout: The percentage (between 0 and 1) of edges to randomly drop
        during training.
      attention_kernel_initializers: Initializers for the attention logit
        function weights. Note that this will be equally partitioned into
        separate weights for each head.
      tag: The incident node tag to pool to.
      edge_set_name: If set, the feature will be pooled from this edge set to
        the given destination.
      **kwargs: Any extra arguments to pass to the super class's init.

    Raises:
      ValueError if edge_dropout is less than 0 or greater than 1.
      NotImplementedError if `tag` == 'CONTEXT', as Context pooling is not yet
        supported.

    Returns:
       A pooling layer that can be used in a Keras model.
    """
    super().__init__(**kwargs)

    if tag == const.CONTEXT:
      raise NotImplementedError('Context pooling not currently supported.')
    self._tag = tag
    self._edge_set_name = edge_set_name

    self._per_head_channels = per_head_channels
    self._num_heads = num_heads
    if not 0 <= edge_dropout < 1:
      raise ValueError(f'Edge dropout {edge_dropout} must be in [0, 1).')
    self._edge_dropout = edge_dropout

    # Create attention logits layers, one for each head. Note that we can't
    # use a single Dense layer that outputs `num_heads` units because we need
    # to apply a different attention function a_k to its corresponding
    # W_k-transformed features.
    self._attention_logits_fn = tf.keras.layers.experimental.EinsumDense(
        '...ik,ki->...i',
        output_shape=(None, num_heads, 1),
        kernel_initializer=attention_kernel_initializers,
        name='attn_logits_fn')
    self._attention_kernel_initializers = attention_kernel_initializers

  def call(self,
           inputs: Tuple[gt.GraphTensor, const.Field, const.Field],
           training=False) -> const.Field:
    """Compute attention pooling over the given queries and keys.

    The query and key features already have been transformed by W_query (
    left) and W_key (right), respectively. See implementation for more details.

    Args:
      inputs: A tuple containing the following items: (1) The GraphTensor to
        read from. (2) The value of the broadcasted query feature. (3) The value
        of the broadcasted key feature.
      training: True iff we are training, not evaluating. Used to enable
        training-specific features like dropout.

    Returns:
      A tensor with the pooled feature value.
    """
    graph, query_broadcasted, key_broadcasted = inputs
    # Extract the broadcasted query and key features.
    adjacency = graph.edge_sets[self._edge_set_name].adjacency
    query_node_set = graph.node_sets[adjacency.source_name]

    # Per the doc comments, we support features that have extra dimensions. This
    # block determines if those extra dimensions exist, and adds them to the
    # `reshape` shapes if so.
    features_shape = query_broadcasted.shape.as_list()
    edges_and_extra_dims = [-1]  # Use -1 as the edges dimension.
    if len(features_shape) > 2:
      edges_and_extra_dims.extend(features_shape[1:-1])
    # [n_edges, opt_extra_dims, num_heads, per_head_channels]
    query_broadcasted = tf.reshape(
        query_broadcasted,
        (*edges_and_extra_dims, self._num_heads, self._per_head_channels))
    key_broadcasted = tf.reshape(
        key_broadcasted,
        (*edges_and_extra_dims, self._num_heads, self._per_head_channels))

    # Recall that the algorithm calls for W*[query || key]. However,
    # we actually need just the transformed query in Equation (4) of
    # https://arxiv.org/pdf/2105.14491.pdf (the paper is unclear on this
    # point). To do this, we previously decomposed:
    # W*[query || key] = W_query * query + W_key * key
    # and now we recompose this to get W*[query || key].
    # [n_edges, opt_extra_dims, num_heads, per_head_channels]
    features = query_broadcasted + key_broadcasted

    # Compute the attention logits and softmax to get the coefficients.
    # [n_edges, opt_extra_dims, num_heads, 1]
    logits = tf.expand_dims(self._attention_logits_fn(features), -1)
    attention_coefficients = normalization_ops.softmax_edges_per_node(
        graph, self._edge_set_name, self._tag, feature_value=logits)

    if training:
      # Apply dropout to the normalized attention coefficients, as is done in
      # the original GAT paper. This should have the same effect as edge
      # dropout. Also, note that tf.nn.dropout upscales the remaining values,
      # which should maintain the sum-up-to-1 per node in expectation.
      attention_coefficients = tf.nn.dropout(attention_coefficients,
                                             self._edge_dropout)

    # Apply the attention coefficients to the transformed query.
    # [n_edges, opt_extra_dims, num_heads, per_head_channels]
    messages = query_broadcasted * attention_coefficients
    # Take the sum of the weighted values, which equals the weighted average,
    # and add a nonlinearity.
    pooled_h = tf.nn.relu(
        ops.pool_edges_to_node(
            graph,
            self._edge_set_name,
            self._tag,
            'sum',
            feature_value=messages))

    # Reshape to get to [nodes, opt_extra_dims, per_head_channels * num_heads]
    num_nodes = tf.reduce_sum(query_node_set.sizes)
    out_reshape = [num_nodes]
    if len(features_shape) > 2:
      out_reshape.extend(features_shape[1:-1])
    out_reshape.append(self._per_head_channels * self._num_heads)
    return tf.reshape(pooled_h, out_reshape)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'num_heads': self._num_heads,
        'per_head_channels': self._per_head_channels,
        'edge_dropout': self._edge_dropout,
        'attention_kernel_initializers': self._attention_kernel_initializers,
        'tag': self._tag,
        'edge_set_name': self._edge_set_name,
    })
    return config
