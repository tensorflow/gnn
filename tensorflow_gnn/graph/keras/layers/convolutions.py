"""The most elementary convolutions, and associated tooling."""

from typing import Callable, Optional

import tensorflow as tf

from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor_ops as ops
from tensorflow_gnn.graph.keras.layers import convolution_base


@tf.keras.utils.register_keras_serializable(package="GNN")
class SimpleConvolution(convolution_base.AnyToAnyConvolutionBase):
  """A convolution layer that applies a passed-in message_fn.

  This layer can compute a convolution over an edge set by applying the
  passed-in message_fn for all edges on the concatenated inputs from some or all
  of: the edge itself, the sender node, and the receiver node, followed by
  pooling to the receiver node.

  Alternatively, depending on init arguments, it can perform the equivalent
  computation from nodes to context, edges to incident nodes, or edges to
  context, with the calling conventions described in the docstring for
  tfgnn.keras.layers.AnyToAnyConvolutionBase.

  Example: Using a SimpleConvolution in an MPNN-style graph update with a
  single-layer network to compute "sum"-pooled message on each edge from
  concatenated source and target states. (The result is then fed into the
  next-state layer, which concatenates the old node state and applies another
  single-layer network.)

  ```
  dense = tf.keras.layers.Dense  # ...or some fancier feed-forward network.
  graph = tfgnn.keras.layers.GraphUpdate(
      node_sets={"paper": tfgnn.keras.layers.NodeSetUpdate(
          {"cites": tfgnn.keras.layers.SimpleConvolution(
               dense(message_dim, "relu"), "sum", receiver_tag=tfgnn.TARGET)},
          tfgnn.keras.layers.NextStateFromConcat(dense(state_dim, "relu")))}
  )(graph)
  ```

  Init args:
    message_fn: A Keras layer that computes the individual messages from the
      combined input features (see combine_type).
    reduce_type: Specifies how to pool the messages to receivers. Defaults to
      "sum", can be any name from tfgnn.get_registered_reduce_operation_names().
    combine_type: a string understood by tfgnn.combine_values(), to specify how
      the inputs are combined before passing them to the message_fn. Defaults
      to "concat", which concatenates inputs along the last axis.
    receiver_tag:  one of `tfgnn.SOURCE`, `tfgnn.TARGET` or `tfgnn.CONTEXT`.
      Selects the receiver of the pooled messages.
      If set to `tfgnn.SOURCE` or `tfgnn.TARGET`, the layer can be called for
      an edge set and will pool results at the specified endpoint of the edges.
      If set to `tfgnn.CONTEXT`, the layer can be called for an edge set or node
      set and will pool results for the context (i.e., per graph component).
      If left unset for init, the tag must be passed at call time.
    receiver_feature: Can be set to override `tfgnn.DEFAULT_FEATURE_NAME`
      for use as the input feature from the receiver. Passing `None` disables
      input from the receiver.
    sender_node_feature: Can be set to override `tfgnn.DEFAULT_FEATURE_NAME`
      for use as the input feature from sender nodes. Passing `None` disables
      input from the sender node.
      IMPORANT: Must be set to `None` for use with `receiver_tag=tfgnn.CONTEXT`
      on an edge set, or for pooling from edges without sender node states.
    sender_edge_feature: Can be set to a feature name of the edge set to select
      it as an input feature. By default, this set to `None`, which disables
      this input.
      IMPORTANT: Must be set for use with `receiver_tag=tfgnn.CONTEXT` on an
      edge set.

  Call returns:
    A Tensor whose leading dimension is indexed by receivers, with the
    pooled messages for each receiver.
  """

  def __init__(
      self,
      message_fn: tf.keras.layers.Layer,
      reduce_type: str = "sum",
      *,
      combine_type: str = "concat",
      receiver_tag: const.IncidentNodeTag = const.TARGET,
      receiver_feature: const.FieldName = const.DEFAULT_STATE_NAME,
      sender_node_feature: Optional[
          const.FieldName] = const.DEFAULT_STATE_NAME,
      sender_edge_feature: Optional[const.FieldName] = None,
      **kwargs):

    # TODO(b/215486977): Remove this compat logic for older arguments.
    sender_edge_feature = _get_compat_sender_edge_feature_arg(
        sender_edge_feature, kwargs)
    sender_node_feature, receiver_feature = _get_compat_node_feature_args(
        sender_node_feature, receiver_feature, receiver_tag, kwargs)

    super().__init__(
        receiver_tag=receiver_tag,
        receiver_feature=receiver_feature,
        sender_node_feature=sender_node_feature,
        sender_edge_feature=sender_edge_feature,
        **kwargs)

    self._message_fn = message_fn
    self._reduce_type = reduce_type
    self._combine_type = combine_type

  def get_config(self):
    return dict(
        message_fn=self._message_fn,
        reduce_type=self._reduce_type,
        combine_type=self._combine_type,
        **super().get_config())

  def convolve(self, *,
               sender_node_input: Optional[tf.Tensor],
               sender_edge_input: Optional[tf.Tensor],
               receiver_input: Optional[tf.Tensor],
               broadcast_from_sender_node: Callable[[tf.Tensor], tf.Tensor],
               broadcast_from_receiver: Callable[[tf.Tensor], tf.Tensor],
               pool_to_receiver: Callable[..., tf.Tensor],
               training: bool) -> tf.Tensor:
    # Collect inputs, suitably broadcast.
    inputs = []
    if sender_edge_input is not None:
      inputs.append(sender_edge_input)
    if sender_node_input is not None:
      inputs.append(broadcast_from_sender_node(sender_node_input))
    if receiver_input is not None:
      inputs.append(broadcast_from_receiver(receiver_input))
    # Combine inputs.
    combined_input = ops.combine_values(inputs, self._combine_type)

    # Compute the result.
    messages = self._message_fn(combined_input)
    pooled_messages = pool_to_receiver(messages, reduce_type=self._reduce_type)
    return pooled_messages


def _get_compat_sender_edge_feature_arg(sender_edge_feature, init_kwargs):
  """Handles the deprecated alias edge_input_feature for sender_edge_feature."""
  absent = ()
  edge_input_feature = init_kwargs.pop("edge_input_feature", absent)
  if edge_input_feature is absent:
    return sender_edge_feature  # Unchanged.
  # Raise ValueError if both the old and the new arg are set.
  # (We're missing the corner case of sender_edge_feature being set explicitly
  # to its default.)
  if sender_edge_feature is not None:
    raise ValueError(
        "Set sender_edge_feature only; edge_input_feature is deprecated.")

  sender_edge_feature = edge_input_feature
  return sender_edge_feature


def _get_compat_node_feature_args(sender_node_feature, receiver_feature,
                                  receiver_tag, init_kwargs):
  """Handles the deprecated argument node_input_tags."""
  absent = ()
  node_input_tags = init_kwargs.pop("node_input_tags", absent)
  if node_input_tags is absent:
    return sender_node_feature, receiver_feature  # Unchanged.
  # Raise ValueError if both the old arg and one of the new args is set.
  # (We're missing the corner case of a new arg being set explicitly
  # to its default.)
  if not sender_node_feature == receiver_feature == const.DEFAULT_STATE_NAME:
    raise ValueError(
        "Set sender_node_feature and/or receiver_feature only, "
        "node_input_tags is deprecated.")

  # Convert old to new args, with the following imperfections:
  # - Incident node tags beyond SOURCE and TARGET are unsupported.
  # - The order of concatenated inputs and hence the order of weights in the
  #   first layer of message_fn is preserved only for the following cases:
  #    - for len(node_input_tags) < 2, irrespective of receiver_tag,
  #    - for the default arguments receiver_tag == tfgnn.TARGET and
  #      node_input_tags == [tfgnn.SOURCE, tfgnn.TARGET],
  #    - for receiver_tag == tfgnn.SOURCE and the unusually reversed
  #      node_input_tags == [tfgnn.TARGET, tfgnn.SOURCE].
  receiver_feature = sender_node_feature = None
  for tag in node_input_tags:
    if tag not in [const.SOURCE, const.TARGET]:
      raise ValueError(f"Unsupported node_input_tag {tag}")
    if tag == receiver_tag:
      receiver_feature = const.DEFAULT_STATE_NAME
    else:
      sender_node_feature = const.DEFAULT_STATE_NAME
  return sender_node_feature, receiver_feature
