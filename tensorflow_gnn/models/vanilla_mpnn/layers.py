"""Contains the Vanilla MPNN layers."""
from typing import Collection, Optional, Union

import tensorflow as tf
import tensorflow_gnn as tfgnn


def VanillaMPNNGraphUpdate(  # To be called like a class initializer.  pylint: disable=invalid-name
    *,
    units: int,
    message_dim: int,
    receiver_tag: tfgnn.IncidentNodeOrContextTag,
    node_set_names: Optional[Collection[tfgnn.NodeSetName]] = None,
    edge_feature: Optional[tfgnn.FieldName] = None,
    reduce_type: str = "sum",
    l2_regularization: float = 0.0,
    dropout_rate: float = 0.0,
    kernel_initializer: Union[
        None, str, tf.keras.initializers.Initializer] = "glorot_uniform",
) -> tf.keras.layers.Layer:
  r"""Returns a GraphUpdate layer for a Vanilla MPNN.

  The returned layer performs one round of node state updates with a
  Message Passing Neural Network that uses a single dense layer to
  compute messages and update node states.

  For each edge set E, the pooled messages for node v are computed as follows
  from its neighbors N_E(v), that is, the other endpoints of those edges
  that have v at the endpoint identified by `receiver_tag`.

  $$m_E = \text{reduce}(
      \text{ReLU}(W_{\text{msg}} (h_v || h_u || x_{(u,v)}))
      \text{ for all } u \in N_E(v)).$$

  The inputs are, in this order: the `tfgnn.HIDDEN_STATE` features of the
  receiver and sender node as well as the named `edge_feature`, if any.
  The reduction happens with the specified `reduce_type`, e.g., sum or mean.

  The new hidden state at node v is computed as follows from the old node
  state and the pooled messages from all incident node sets E_1, E_2, ...:

  $$h_v := \text{ReLU}(
      W_{\text{state}} (h_v || m_{E_1} || m_{E_2} || \ldots)).$$

  Args:
    units: The dimension of output hidden states for each node.
    message_dim: The dimension of messages (attention values) computed on
      each edge.  Must be divisible by `num_heads`.
    receiver_tag: one of `tfgnn.TARGET` or `tfgnn.SOURCE`, to select the
      incident node of each edge that receives the message.
    node_set_names: The names of node sets to update. If unset, updates all
      that are on the receiving end of any edge set.
    edge_feature: Can be set to a feature name of the edge set to select
      it as an input feature. By default, this set to `None`, which disables
      this input.
    reduce_type: How to pool the messages from edges to receiver nodes.
      Can be any name from `tfgnn.get_registered_reduce_operation_names()`,
      defaults to "sum".
    l2_regularization: The coefficient of L2 regularization for weights and
      biases.
    dropout_rate: The dropout rate applied to messages on each edge and to the
      new node state.
    kernel_initializer: Can be set to a `kerner_initializer` as understood
      by `tf.keras.layers.Dense` etc.

  Returns:
    A GraphUpdate layer for use on a scalar GraphTensor with
    `tfgnn.HIDDEN_STATE` features on the node sets.
  """
  def dense(units):  # pylint: disable=invalid-name
    regularizer = tf.keras.regularizers.l2(l2_regularization)
    return tf.keras.Sequential([
        tf.keras.layers.Dense(
            units,
            activation="relu",
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer="zeros",
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer),
        tf.keras.layers.Dropout(dropout_rate)])

  # pylint: disable=g-long-lambda
  gnn_builder = tfgnn.keras.ConvGNNBuilder(
      lambda edge_set_name, receiver_tag: tfgnn.keras.layers.SimpleConv(
          dense(message_dim), reduce_type, receiver_tag=receiver_tag,
          sender_edge_feature=edge_feature),
      lambda node_set_name: tfgnn.keras.layers.NextStateFromConcat(
          dense(units)),
      receiver_tag=receiver_tag)
  return gnn_builder.Convolve(node_set_names)
