"""The most elementary convolutions, and associated tooling."""

from typing import Optional, Sequence

import tensorflow as tf

from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import graph_tensor_ops as ops
from tensorflow_gnn.graph.keras.layers import graph_update
from tensorflow_gnn.graph.keras.layers import next_state as next_state_lib


@tf.keras.utils.register_keras_serializable(package="GNN")
class ConvolutionFromEdgeSetUpdate(tf.keras.layers.Layer):
  """Wraps an EdgeSetUpdate as a Convolution.

  This layer can be used in `NodeSetUpdate(edge_set_inputs={...})` to perform
  the per-edge computation of `edge_set_update` without merging the result
  into the GraphTensor.

  Init args:
    edge_set_update: An EdgeSetUpdate layer (or custom reimplementation) that
      computes new edge states from the input graph tensor. Its results are
      pooled for the receiver node set and returned from this layer.
    receiver_tag: This layer's result is obtained by pooling the per-edge
      results at this endpoint of each edge (by default, `tfgnn.TARGET`).
    reduce_type: Specifies how to pool the per-edge results to each edge's
      receiver node. Defaults to "sum", can be set to any name from
      tfgnn.get_registered_reduce_operation_names().

  Call returns:
    A tensor or dict of tensors with the result of edge_set_update, pooled for
    the receiver node set. (Typically, the caller combines this with results
    from other edge sets to compute a node set update at the receiving end.)
  """

  def __init__(self,
               edge_set_update: graph_update.EdgeSetUpdateLayer,
               *,
               receiver_tag: const.IncidentNodeTag = const.TARGET,
               reduce_type: str = "sum",
               **kwargs):
    if "destination_tag" in kwargs:  # TODO(b/192858913): Remove in 2022.
      raise TypeError(
          "Argument destination_tag has been renamed to receiver_tag")
    super().__init__(**kwargs)
    self._edge_set_update = edge_set_update
    self._receiver_tag = receiver_tag
    self._reduce_type = reduce_type

  def get_config(self):
    return dict(
        edge_set_update=self._edge_set_update,
        receiver_tag=self._receiver_tag,
        reduce_type=self._reduce_type,
        **super().get_config())

  def call(self, graph: gt.GraphTensor,
           edge_set_name: const.EdgeSetName) -> const.FieldOrFields:
    messages = self._edge_set_update(graph, edge_set_name=edge_set_name)
    def pool(feature_value):
      return ops.pool_edges_to_node(
          graph, edge_set_name, self._receiver_tag, self._reduce_type,
          feature_value=feature_value)
    pooled_messages = tf.nest.map_structure(pool, messages)
    return pooled_messages


@tf.keras.utils.register_keras_serializable(package="GNN")
class SimpleConvolution(ConvolutionFromEdgeSetUpdate):
  """A convolution layer that applies message_fn on each edge.

  This layer that can be used in NodeSetInput({edge_set_name: ...}) to provide
  pooled messages from an edge set as input to the node set's state update.

  Init args:
    message_fn: A Keras layer that takes input features concatenated into a
      single tensor and computes the message of each edge.
      Input and output tensors are shaped like edge features. (See `reduce_type`
      and `receiver_tag` about the pooling that happens afterwards.)
    reduce_type: Specifies how to pool the per-edge results to each edge's
      receiver node. Defaults to "sum", can be set to any name from
      tfgnn.get_registered_reduce_operation_names().
    node_input_tags: The incident nodes of each edge whose states are used
      as an input, specified as IncidentNodeTags (tfgnn.SOURCE and tfgnn.TARGET
      by default).
    edge_input_feature: Can be set to a feature name of the EdgeSet (or a
      sequence of those) for use as additional inputs to message_fn.
      By default, no edge features are used.
    receiver_tag: This layer's result is obtained by pooling the per-edge
      results at this endpoint of each edge. The default is `tfgnn.TARGET`,
      but it is perfectly reasonable to do a convolution towards the
      `tfgnn.SOURCE` instead. (Source and target are conventional names for
      the incident nodes of a directed edge, data flow in a GNN may happen
      in either direction.)

  Call returns:
    A tensor or dict of tensors with the result of edge_set_update, pooled for
    the receiver node set.
  """

  def __init__(
      self,
      message_fn: tf.keras.layers.Layer,
      reduce_type: str = "sum",
      *,
      node_input_tags: Sequence[const.IncidentNodeTag] = (
          const.SOURCE, const.TARGET),
      edge_input_feature: Optional[const.FieldNameOrNames] = None,
      receiver_tag: const.IncidentNodeTag = const.TARGET):
    next_state = next_state_lib.NextStateFromConcat(transformation=message_fn)
    edge_set_update = graph_update.EdgeSetUpdate(
        next_state,
        node_input_tags=node_input_tags,
        edge_input_feature=edge_input_feature,
        context_input_feature=None)
    super().__init__(
        edge_set_update, receiver_tag=receiver_tag,
        reduce_type=reduce_type)
