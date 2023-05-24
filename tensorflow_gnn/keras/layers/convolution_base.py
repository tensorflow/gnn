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
"""The AnyToAnyConvolutionBase class and associated tooling."""

import abc
from typing import Any, Callable, Mapping, Optional

import tensorflow as tf

from tensorflow_gnn.graph import broadcast_ops
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import pool_ops
from tensorflow_gnn.graph import tag_utils


class AnyToAnyConvolutionBase(tf.keras.layers.Layer, abc.ABC):
  """Convenience base class for convolutions to nodes or to context.

  This base class simplifies the implementation of graph convolutions as Keras
  Layers. Instead of subclassing Layer directly and implementing `call()` with a
  GraphTensor input, users can subclass this class and implement the abstract
  `convolve()` method, which is invoked with the relevant tensors unpacked from
  the graph, and with callbacks to broadcast and pool values across the relevant
  parts of the graph structure (see its docstring for more).  The resulting
  subclass can perform convolutions from nodes to nodes, optionally with a side
  input from edges, or the equivalent computations from nodes to context,
  from edges to context, or from edges into incident nodes.

  Here is a minimal example:

  ```python
  @tf.keras.utils.register_keras_serializable(package="MyGNNProject")
  class ExampleConvolution(tfgnn.keras.layers.AnyToAnyConvolutionBase):

    def __init__(self, units, **kwargs):
      super().__init__(**kwargs)
      self._message_fn = tf.keras.layers.Dense(units, "relu")

    def get_config(self):
      return dict(units=self._message_fn.units, **super().get_config())

    def convolve(
        self, *,
        sender_node_input, sender_edge_input, receiver_input,
        broadcast_from_sender_node, broadcast_from_receiver, pool_to_receiver,
        training):
      inputs = []
      if sender_node_input is not None:
        inputs.append(broadcast_from_sender_node(sender_node_input))
      if sender_edge_input is not None:
        inputs.append(sender_edge_input)
      if receiver_input is not None:
        inputs.append(broadcast_from_receiver(receiver_input))
      messages = self._message_fn(tf.concat(inputs, axis=-1))
      return pool_to_receiver(messages, reduce_type="sum")
  ```

  The resulting subclass can be applied to a GraphTensor in four ways:

   1. Convolutions from nodes.

      a) The classic case: convolutions over an edge set.
         ```
           sender_node -> sender_edge <->> receiver (node)
         ```
         A message is computed for each edge of the edge set, depending on
         inputs from the sender node, the receiver node (single arrowheads),
         and/or the edge itself. The messages are aggregated at each receiver
         node from its incident edges (double arrowhead).

      b) Convolutions from a node set to context.
         ```
           sender_node <->> receiver (context)
         ```
         Instead of the EdgeSet as in case (1a), there is a NodeSet, which
         tracks the containment of its nodes in graph components. Instead of
         a receiver node, there is the per-component graph context.
         A message is computed for each node of the node set, depending on
         inputs from the node itself and possibly its context. The messages are
         aggregated for each component from the nodes contained in it.


   2. Pooling from edges.

      a) Pooling from an edge set to an incident node set.
         ```
           sender_edge <->> receiver (node)
         ```
         This works like a convolution in case (1a) with the side input for the
         edge feature switched on and the main input from the sender node
         switched off.

      b) Pooling from an edge set to context.
         ```
           sender_edge <->> receiver (context)
         ```
         Like in case (1b), the receiver is the context, and senders are
         connected to it by containment in a graph component. Unlike case (1b),
         but similar to case (2a), the sender input comes from an edge set,
         not a node set.

  Case (1) is solved directly by subclasses of this class with default init args
  `sender_node_feature=tfgnn.HIDDEN_STATE` and `sender_edge_feature=None`.
  The sub-cases are distinguised by passing `receiver_tag=tfgnn.SOURCE` or
  `tfgnn.TARGET` for (a) and `tfgnn.CONTEXT` for (b).
  The side input from edges can be activated in case (1a) by setting the init
  arg `sender_edge_feature=`.

  Case (2) is solved indirectly by wrapping a subclass to invert the selection
  for sender inputs: nodes off, edges on. The sub-cases (a) and (b) are again
  distinguished by the `receiver_tag`. TF-GNN avoids the term "Convolution" for
  this operation and calls it "EdgePool" instead, to emphasize that sender nodes
  are not involved. Whenever it makes sense to repurpose a convolution as an
  EdgePool operation, we recommend that the convolution is accompanied by a
  wrapper function in the following style so that the proper term "edge pool"
  can be used in code:

  ```python
  def ExampleEdgePool(*args, sender_feature=tfgnn.HIDDEN_STATE, **kwargs):
    return ExampleConvolution(*args, sender_node_feature=None,
                              sender_edge_feature=sender_feature, **kwargs)
  ```

  Repurposing convolutions for pooling is especially useful for those tha
  implement some attention algorithm: it may have been originally designed for
  attention to neighbor nodes, but in this way we can reuse the same code for
  attention to incident edges, or to all nodes/edges in a graph component.
  """

  def __init__(self,
               *,
               receiver_tag: Optional[const.IncidentNodeOrContextTag] = None,
               receiver_feature: Optional[const.FieldName] = const.HIDDEN_STATE,
               sender_node_feature: Optional[
                   const.FieldName] = const.HIDDEN_STATE,
               sender_edge_feature: Optional[const.FieldName] = None,
               extra_receiver_ops: Optional[
                   Mapping[str, Callable[..., Any]]] = None,
               **kwargs):
    """Initializes the AnyToAnyConvolutionBase of a convolution layer.

    Args:
      receiver_tag: one of `tfgnn.SOURCE`, `tfgnn.TARGET` or `tfgnn.CONTEXT`.
        The results are aggregated for this graph piece.
        If set to `tfgnn.SOURCE` or `tfgnn.TARGET`, the layer can be called for
        an edge set and will aggregate results at the specified endpoint of the
        edges.
        If set to `tfgnn.CONTEXT`, the layer can be called for an edge set or a
        node set and will aggregate results for context (per graph component).
        If left unset for init, the tag must be passed at call time.
      receiver_feature: The name of the feature that is read from the receiver
        graph piece and passed as convolve(receiver_input=...).
      sender_node_feature: The name of the feature that is read from the sender
        nodes, if any, and passed as convolve(sender_node_input=...).
        NOTICE this must be `None` for use with `receiver_tag=tfgnn.CONTEXT`
        on an edge set, or for pooling from edges without sender node states.
      sender_edge_feature: The name of the feature that is read from the sender
        edges, if any, and passed as convolve(sender_edge_input=...).
        NOTICE this must not be `None` for use with `receiver_tag=tfgnn.CONTEXT`
        on an edge set.
      extra_receiver_ops: A str-keyed dictionary of Python callables that are
        wrapped to bind some arguments and then passed on to `convolve()`.
        Sample usage: `extra_receiver_ops={"softmax": tfgnn.softmax}`.
        The values passed in this dict must be callable as follows, with two
        positional arguments:

        ```python
        f(graph, receiver_tag, node_set_name=..., feature_value=..., ...)
        f(graph, receiver_tag, edge_set_name=..., feature_value=..., ...)
        ```

        The wrapped callables seen by `convolve()` can be called like

        ```python
        wrapped_f(feature_value, ...)
        ```

        The first three arguments of `f` are set to the input GraphTensor of
        the layer and the tag/name pair required by `tfgnn.broadcast()` and
        `tfgnn.pool()` to move values between the receiver and the messages that
        are computed inside the convolution. The sole positional argument of
        `wrapped_f()` is passed to `f()`  as `feature_value=`, and any keyword
        arguments are forwarded.
      **kwargs: Forwarded to the base class tf.keras.layers.Layer.
    """
    super().__init__(**kwargs)
    self._receiver_tag = receiver_tag
    self._receiver_feature = receiver_feature
    self._sender_node_feature = sender_node_feature
    self._sender_edge_feature = sender_edge_feature
    self._extra_receiver_ops = (None if extra_receiver_ops is None
                                else dict(extra_receiver_ops))

  def get_config(self):
    """Returns config with features and tag managed by AnyToAnyConvolutionBase.

    AnyToAnyConvolutionBase.get_config() returns a dict that includes:
      - its initializer arguments to control what the convolution is run on,
        that is, the `receiver_tag` and all `*_feature` names.
      - the configuration of its base class.

    Usually, a subclass accepts these args in their `__init__()` method
    and forward them verbatim to `super().__init__()`. Correspondingly, its
    `get_config()` will get them from `super().get_config()` and does not need
    to insert them itself (cf. `ExampleConvolution.get_config()` in the usage
    example in the docstring of class AnyToAnyConvolutionBase).

    The init arg `extra_receiver_ops` is not returned here, because it is not a
    free parameter: The subclass initializer sets it when calling
    `AnyToAnyConvolutionBase.__init__()` to do whatever `convolve()` needs;
    that works the same way whether the initializer is called the usual way
    from user code or as part of initializing from a config. (Besides, generic
    Python callables are unsuitable for serialization.)
    """
    return dict(
        receiver_tag=self._receiver_tag,
        receiver_feature=self._receiver_feature,
        sender_node_feature=self._sender_node_feature,
        sender_edge_feature=self._sender_edge_feature,
        **super().get_config())

  @property
  def takes_receiver_input(self) -> bool:
    """If `False`, all calls to convolve() will get `receiver_input=None`."""
    return self._receiver_feature is not None

  @property
  def takes_sender_node_input(self) -> bool:
    """If `False`, all calls to convolve() will get `sender_node_input=None`."""
    return self._sender_node_feature is not None

  @property
  def takes_sender_edge_input(self) -> bool:
    """If `False`, all calls to convolve() will get `sender_edge_input=None`."""
    return self._sender_edge_feature is not None

  def call(self, graph: gt.GraphTensor, *,
           edge_set_name: Optional[gt.EdgeSetName] = None,
           node_set_name: Optional[gt.NodeSetName] = None,
           receiver_tag: Optional[const.IncidentNodeOrContextTag] = None,
           training: Optional[bool] = False) -> tf.Tensor:
    # pylint: disable=g-long-lambda

    # Normalize inputs.
    class_name = self.__class__.__name__
    gt.check_scalar_graph_tensor(graph, class_name)
    receiver_tag = _get_init_or_call_arg(class_name, "receiver_tag",
                                         self._receiver_tag, receiver_tag)

    # Find the receiver graph piece (NodeSet or Context), the EdgeSet (if any)
    # and the sender NodeSet (if any) with its broadcasting function.
    if receiver_tag == const.CONTEXT:
      if (edge_set_name is None) + (node_set_name is None) != 1:
        raise ValueError(
            "Must pass exactly one of edge_set_name, node_set_name "
            "for receiver_tag CONTEXT.")
      if edge_set_name is not None:
        # Pooling from EdgeSet to Context; no node set involved.
        name_kwarg = dict(edge_set_name=edge_set_name)
        edge_set = graph.edge_sets[edge_set_name]
        sender_node_set = None
        broadcast_from_sender_node = None
      else:
        # Pooling from NodeSet to Context, no EdgeSet involved.
        name_kwarg = dict(node_set_name=node_set_name)
        edge_set = None
        sender_node_set = graph.node_sets[node_set_name]
        # Values are computed per sender node, no need to broadcast
        broadcast_from_sender_node = lambda feature_value: feature_value
      receiver_piece = graph.context
    else:
      # Convolving from nodes to nodes.
      if edge_set_name is None or node_set_name is not None:
        raise ValueError("Must pass edge_set_name, not node_set_name")
      name_kwarg = dict(edge_set_name=edge_set_name)
      edge_set = graph.edge_sets[edge_set_name]
      sender_node_tag = tag_utils.reverse_tag(receiver_tag)
      sender_node_set = graph.node_sets[
          edge_set.adjacency.node_set_name(sender_node_tag)]
      broadcast_from_sender_node = (
          lambda feature_value: broadcast_ops.broadcast_node_to_edges(
              graph, edge_set_name, sender_node_tag,
              feature_value=feature_value))
      receiver_piece = graph.node_sets[
          edge_set.adjacency.node_set_name(receiver_tag)]

    # Set up the broadcast/pool ops for the receiver, plus any ops requested
    # by the subclass. The tag and name arguments conveniently encode the
    # distinction between operating over edge/node, node/context or
    # edge/context.
    def bind_receiver_args(fn):
      return lambda feature_value, **kwargs: fn(
          graph, receiver_tag, **name_kwarg,
          feature_value=feature_value, **kwargs)
    broadcast_from_receiver = bind_receiver_args(broadcast_ops.broadcast_v2)
    pool_to_receiver = bind_receiver_args(pool_ops.pool_v2)
    if self._extra_receiver_ops is None:
      extra_receiver_ops_kwarg = {}  # Pass no argument for this.
    else:
      extra_receiver_ops_kwarg = dict(
          extra_receiver_ops={name: bind_receiver_args(fn)
                              for name, fn in self._extra_receiver_ops.items()})

    # Set up the inputs.
    receiver_input = sender_node_input = sender_edge_input = None
    if self._receiver_feature is not None:
      receiver_input = receiver_piece[self._receiver_feature]
    if None not in [sender_node_set, self._sender_node_feature]:
      sender_node_input = sender_node_set[self._sender_node_feature]
    if None not in [edge_set, self._sender_edge_feature]:
      sender_edge_input = edge_set[self._sender_edge_feature]

    return self.convolve(
        sender_node_input=sender_node_input,
        sender_edge_input=sender_edge_input,
        receiver_input=receiver_input,
        broadcast_from_sender_node=broadcast_from_sender_node,
        broadcast_from_receiver=broadcast_from_receiver,
        pool_to_receiver=pool_to_receiver,
        **extra_receiver_ops_kwarg,
        training=training)

  @abc.abstractmethod
  def convolve(self, *,
               sender_node_input: Optional[tf.Tensor],
               sender_edge_input: Optional[tf.Tensor],
               receiver_input: Optional[tf.Tensor],
               broadcast_from_sender_node: Callable[[tf.Tensor], tf.Tensor],
               broadcast_from_receiver: Callable[[tf.Tensor], tf.Tensor],
               pool_to_receiver: Callable[..., tf.Tensor],
               extra_receiver_ops: Optional[
                   Mapping[str, Callable[..., tf.Tensor]]] = None,
               training: bool) -> tf.Tensor:
    """Returns the convolution result.

    The Tensor inputs to this function still have their original shapes
    and need to be broadcast such that the leading dimension is indexed
    by the items in the graph for which messages are computed (usually edges;
    except when convolving from nodes to context). In the end, values have to be
    pooled from there into a Tensor with a leading dimension indexed by
    receivers, see `pool_to_receiver`.

    Args:
      sender_node_input: The input Tensor from the sender NodeSet, or `None`.
        If self.takes_sender_node_input is `False`, this arg will be `None`.
        (If it is `True`, that depends on how this layer gets called.)
        See also broadcast_from_sender_node.
      sender_edge_input: The input Tensor from the sender EdgeSet, or `None`.
        If self.takes_sender_edge_input is `False`, this arg will be `None`.
        (If it is `True`, it depends on how this layer gets called.)
        If present, this Tensor is already indexed by the items for which
        messages are computed.
      receiver_input: The input Tensor from the receiver NodeSet or Context,
        or None. If self.takes_receiver_input is `False`, this arg will be
        `None`. (If it is `True`, it depends on how this layer gets called.)
        See broadcast_from_receiver.
      broadcast_from_sender_node: A function that broadcasts a Tensor indexed
        like sender_node_input to a Tensor indexed by the items for which
        messages are computed.
      broadcast_from_receiver: Call this as `broadcast_from_receiver(value)`
        to broadcast a Tensor indexed like receiver_input to a Tensor indexed
        by the items for which messages are computed.
      pool_to_receiver: Call this as `pool_to_receiver(value, reduce_type=...)`
        to pool an item-indexed Tensor to a receiver-indexed tensor, using
        a reduce_type understood by tfgnn.pool(), such as "sum".
      extra_receiver_ops: The extra_receiver_ops passed to init, see there,
        wrapped so that they can be called directly on a feature value.
        If init did not receive extra_receiver_ops, convolve() will not receive
        this argument, so subclass implementors not using it can omit it.
      training: The `training` boolean that was passed to Layer.call(). If true,
        the result is computed for training rather than inference. For example,
        calls to tf.nn.dropout() are usually conditioned on this flag.
        By contrast, calling another Keras layer (like tf.keras.layers.Dropout)
        does not require forwarding this arg, Keras does that automatically.

    Returns:
      A Tensor whose leading dimension is indexed by receivers, with the
      result of the convolution for each receiver.
    """
    raise NotImplementedError("To be implemented by the concrete subclass.")


def _get_init_or_call_arg(class_name, arg_name, init_value, call_value):
  """Returns unified value for arg that can be set at init or call time."""
  if call_value is None:
    if init_value is None:
      raise ValueError(
          f"{class_name} requires {arg_name} to be set at init or call time")
    return init_value
  else:
    if init_value not in [None, call_value]:
      raise ValueError(
          f"{class_name}(..., {arg_name}={init_value})"
          f"was called with contradictory value {arg_name}={call_value}")
    return call_value
