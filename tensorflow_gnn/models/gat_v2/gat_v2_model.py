"""An example use of GATv2Conv."""
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import gat_v2


def pass_messages_with_gat_v2(
    graph,
    *,
    receiver_tag: tfgnn.IncidentNodeTag,
    num_heads: int,
    message_dim: int,
    h_next_dim: int,
    num_message_passing: int,
    l2_regularization: float = 0.0,
    edge_dropout_rate: float = 0.0,
    state_dropout_rate: float = 0.0) -> tfgnn.GraphTensor:
  """Performs message passing with GATv2 pooling.

  Args:
    graph: a scalar GraphTensor with `tfgnn.DEFAULT_STATE_NAME` features on all
       node sets.
    receiver_tag: one of `tfgnn.TARGET` or `tfgnn.SOURCE`, to select the
      incident node of each edge that receives the message.
    num_heads: The number of attention heads used by GATv2.
    message_dim: The dimension of messages computed on each edge.
    h_next_dim: The dimension of hidden states computed for each node.
    num_message_passing: The number of rounds of message passing along edges.
    l2_regularization: The coefficient of L2 regularization for weights and
      biases.
    edge_dropout_rate: The edge dropout rate applied during attention pooling
      of edges.
    state_dropout_rate: The dropout rate applied to the resulting node states.

  Returns:
    A scalar GraphTensor with `tfgnn.DEFAULT_STATE_NAME` features on all node
    sets that have been updated by the specified rounds of message passing.
  """
  if message_dim % num_heads:
    raise ValueError("message_dim must be divisible by num_heads, "
                     f"got {message_dim} and {num_heads}.")
  per_head_channels = message_dim // num_heads
  # pylint: disable=g-long-lambda
  gnn_builder = tfgnn.keras.ConvGNNBuilder(
      lambda edge_set_name, receiver_tag: gat_v2.GATv2Conv(
          num_heads=num_heads, per_head_channels=per_head_channels,
          edge_dropout=edge_dropout_rate, receiver_tag=receiver_tag),
      lambda node_set_name: tfgnn.keras.layers.NextStateFromConcat(
          _dense_layer(h_next_dim, l2_regularization=l2_regularization,
                       dropout_rate=state_dropout_rate)),
      receiver_tag=receiver_tag)

  for _ in range(num_message_passing):
    graph = gnn_builder.Convolve()(graph)
  return graph


def _dense_layer(units,
                 *,
                 linear: bool = False,
                 l2_regularization: float,
                 dropout_rate: float):
  """Returns a feed-forward network with the given number of output units."""
  regularizer = tf.keras.regularizers.l2(l2_regularization)
  if linear:
    activation = None
  else:
    activation = tf.keras.layers.ReLU()
  return tf.keras.Sequential([
      tf.keras.layers.Dense(
          units,
          activation=activation,
          use_bias=True,
          kernel_initializer="glorot_uniform",
          bias_initializer="zeros",
          kernel_regularizer=regularizer,
          bias_regularizer=regularizer),
      tf.keras.layers.Dropout(dropout_rate)])
