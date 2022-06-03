description: Convenience base class for convolutions to nodes or to context.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfgnn.keras.layers.AnyToAnyConvolutionBase" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="convolve"/>
</div>

# tfgnn.keras.layers.AnyToAnyConvolutionBase

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/convolution_base.py#L14-L378">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Convenience base class for convolutions to nodes or to context.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.AnyToAnyConvolutionBase(
    *,
    receiver_tag: Optional[const.IncidentNodeOrContextTag] = None,
    receiver_feature: Optional[const.FieldName] = const.HIDDEN_STATE,
    sender_node_feature: Optional[const.FieldName] = const.HIDDEN_STATE,
    sender_edge_feature: Optional[const.FieldName] = None,
    extra_receiver_ops: Optional[Mapping[str, Callable[..., Any]]] = None,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

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
<a href="../../../tfgnn.md#TARGET"><code>tfgnn.TARGET</code></a> for (a) and <a href="../../../tfgnn.md#CONTEXT"><code>tfgnn.CONTEXT</code></a> for (b).
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

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`receiver_tag`
</td>
<td>
one of <a href="../../../tfgnn.md#SOURCE"><code>tfgnn.SOURCE</code></a>, <a href="../../../tfgnn.md#TARGET"><code>tfgnn.TARGET</code></a> or <a href="../../../tfgnn.md#CONTEXT"><code>tfgnn.CONTEXT</code></a>.
The results are aggregated for this graph piece.
If set to <a href="../../../tfgnn.md#SOURCE"><code>tfgnn.SOURCE</code></a> or <a href="../../../tfgnn.md#TARGET"><code>tfgnn.TARGET</code></a>, the layer can be called for
an edge set and will aggregate results at the specified endpoint of the
edges.
If set to <a href="../../../tfgnn.md#CONTEXT"><code>tfgnn.CONTEXT</code></a>, the layer can be called for an edge set or a
node set and will aggregate results for context (per graph component).
If left unset for init, the tag must be passed at call time.
</td>
</tr><tr>
<td>
`receiver_feature`
</td>
<td>
The name of the feature that is read from the receiver
graph piece and passed as convolve(receiver_input=...).
</td>
</tr><tr>
<td>
`sender_node_feature`
</td>
<td>
The name of the feature that is read from the sender
nodes, if any, and passed as convolve(sender_node_input=...).
NOTICE this must be `None` for use with `receiver_tag=tfgnn.CONTEXT`
on an edge set, or for pooling from edges without sender node states.
</td>
</tr><tr>
<td>
`sender_edge_feature`
</td>
<td>
The name of the feature that is read from the sender
edges, if any, and passed as convolve(sender_edge_input=...).
NOTICE this must not be `None` for use with `receiver_tag=tfgnn.CONTEXT`
on an edge set.
</td>
</tr><tr>
<td>
`extra_receiver_ops`
</td>
<td>
A str-keyed dictionary of Python callables that are
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
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Forwarded to the base class tf.keras.layers.Layer.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`takes_receiver_input`
</td>
<td>
If `False`, all calls to convolve() will get `receiver_input=None`.
</td>
</tr><tr>
<td>
`takes_sender_edge_input`
</td>
<td>
If `False`, all calls to convolve() will get `sender_edge_input=None`.
</td>
</tr><tr>
<td>
`takes_sender_node_input`
</td>
<td>
If `False`, all calls to convolve() will get `sender_node_input=None`.
</td>
</tr>
</table>



## Methods

<h3 id="convolve"><code>convolve</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/convolution_base.py#L321-L378">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>convolve(
    *,
    sender_node_input: Optional[tf.Tensor],
    sender_edge_input: Optional[tf.Tensor],
    receiver_input: Optional[tf.Tensor],
    broadcast_from_sender_node: Callable[[tf.Tensor], tf.Tensor],
    broadcast_from_receiver: Callable[[tf.Tensor], tf.Tensor],
    pool_to_receiver: Callable[..., tf.Tensor],
    extra_receiver_ops: Optional[Mapping[str, Callable[..., tf.Tensor]]] = None,
    training: bool
) -> tf.Tensor
</code></pre>

Returns the convolution result.

The Tensor inputs to this function still have their original shapes
and need to be broadcast such that the leading dimension is indexed
by the items in the graph for which messages are computed (usually edges;
except when convolving from nodes to context). In the end, values have to be
pooled from there into a Tensor with a leading dimension indexed by
receivers, see `pool_to_receiver`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`sender_node_input`
</td>
<td>
The input Tensor from the sender NodeSet, or `None`.
If self.takes_sender_node_input is `False`, this arg will be `None`.
(If it is `True`, that depends on how this layer gets called.)
See also broadcast_from_sender_node.
</td>
</tr><tr>
<td>
`sender_edge_input`
</td>
<td>
The input Tensor from the sender EdgeSet, or `None`.
If self.takes_sender_edge_input is `False`, this arg will be `None`.
(If it is `True`, it depends on how this layer gets called.)
If present, this Tensor is already indexed by the items for which
messages are computed.
</td>
</tr><tr>
<td>
`receiver_input`
</td>
<td>
The input Tensor from the receiver NodeSet or Context,
or None. If self.takes_receiver_input is `False`, this arg will be
`None`. (If it is `True`, it depends on how this layer gets called.)
See broadcast_from_receiver.
</td>
</tr><tr>
<td>
`broadcast_from_sender_node`
</td>
<td>
A function that broadcasts a Tensor indexed
like sender_node_input to a Tensor indexed by the items for which
messages are computed.
</td>
</tr><tr>
<td>
`broadcast_from_receiver`
</td>
<td>
Call this as `broadcast_from_receiver(value)`
to broadcast a Tensor indexed like receiver_input to a Tensor indexed
by the items for which messages are computed.
</td>
</tr><tr>
<td>
`pool_to_receiver`
</td>
<td>
Call this as `pool_to_receiver(value, reduce_type=...)`
to pool an item-indexed Tensor to a receiver-indexed tensor, using
a reduce_type understood by tfgnn.pool(), such as "sum".
</td>
</tr><tr>
<td>
`extra_receiver_ops`
</td>
<td>
The extra_receiver_ops passed to init, see there,
wrapped so that they can be called directly on a feature value.
If init did not receive extra_receiver_ops, convolve() will not receive
this argument, so subclass implementors not using it can omit it.
</td>
</tr><tr>
<td>
`training`
</td>
<td>
The `training` boolean that was passed to Layer.call(). If true,
the result is computed for training rather than inference. For example,
calls to tf.nn.dropout() are usually conditioned on this flag.
By contrast, calling another Keras layer (like tf.keras.layers.Dropout)
does not require forwarding this arg, Keras does that automatically.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Tensor whose leading dimension is indexed by receivers, with the
result of the convolution for each receiver.
</td>
</tr>

</table>





