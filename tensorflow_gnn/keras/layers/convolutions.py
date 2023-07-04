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
"""The most elementary convolutions, and associated tooling."""

from typing import Any, Callable, Optional

import tensorflow as tf

from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor_ops as ops
from tensorflow_gnn.keras.layers import convolution_base


@tf.keras.utils.register_keras_serializable(package="GNN")
class SimpleConv(convolution_base.AnyToAnyConvolutionBase):
  """A convolution layer that applies a passed-in message_fn.

  This layer can compute a convolution over an edge set by applying the
  passed-in message_fn for all edges on the concatenated inputs from some or all
  of: the edge itself, the sender node, and the receiver node, followed by
  pooling to the receiver node.

  Alternatively, depending on init arguments, it can perform the equivalent
  computation from nodes to context, edges to incident nodes, or edges to
  context, with the calling conventions described in the docstring for
  tfgnn.keras.layers.AnyToAnyConvolutionBase.

  Example: Using a SimpleConv in an MPNN-style graph update with a
  single-layer network to compute "sum"-pooled message on each edge from
  concatenated source and target states. (The result is then fed into the
  next-state layer, which concatenates the old node state and applies another
  single-layer network.)

  ```python
  dense = tf.keras.layers.Dense  # ...or some fancier feed-forward network.
  graph = tfgnn.keras.layers.GraphUpdate(
      node_sets={"paper": tfgnn.keras.layers.NodeSetUpdate(
          {"cites": tfgnn.keras.layers.SimpleConv(
               dense(message_dim, "relu"), "sum", receiver_tag=tfgnn.TARGET)},
          tfgnn.keras.layers.NextStateFromConcat(dense(state_dim, "relu")))}
  )(graph)
  ```

  Init args:
    message_fn: A Keras layer that computes the individual messages from the
      combined input features (see combine_type).
    reduce_type: Specifies how to pool the messages to receivers. Defaults to
      `"sum"`, can be any reduce_type understood by `tfgnn.pool()`, including
      concatenations like `"sum|max"` (but mind the increased dimension of the
      result and the growing number of model weights in the next-state layer).
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
    receiver_feature: Can be set to override `tfgnn.HIDDEN_STATE` for use as
      the input feature from the receiver. Passing `None` disables input from
      the receiver.
    sender_node_feature: Can be set to override `tfgnn.HIDDEN_STATE` for use as
      the input feature from sender nodes. Passing `None` disables input from
      the sender node.
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
      receiver_feature: Optional[const.FieldName] = const.HIDDEN_STATE,
      sender_node_feature: Optional[
          const.FieldName] = const.HIDDEN_STATE,
      sender_edge_feature: Optional[const.FieldName] = None,
      **kwargs):
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
               extra_receiver_ops: Any = None,
               training: bool) -> tf.Tensor:
    assert extra_receiver_ops is None, "Internal error: bad super().__init__()"
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
