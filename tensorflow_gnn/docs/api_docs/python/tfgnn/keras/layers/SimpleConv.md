# tfgnn.keras.layers.SimpleConv

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/convolutions.py#L26-L142">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

A convolution layer that applies a passed-in message_fn.

Inherits From:
[`AnyToAnyConvolutionBase`](../../../tfgnn/keras/layers/AnyToAnyConvolutionBase.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.SimpleConv(
    message_fn: tf.keras.layers.Layer,
    reduce_type: str = &#x27;sum&#x27;,
    *,
    combine_type: str = &#x27;concat&#x27;,
    receiver_tag: const.IncidentNodeTag = const.TARGET,
    receiver_feature: const.FieldName = const.HIDDEN_STATE,
    sender_node_feature: Optional[const.FieldName] = const.HIDDEN_STATE,
    sender_edge_feature: Optional[const.FieldName] = None,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

This layer can compute a convolution over an edge set by applying the passed-in
message_fn for all edges on the concatenated inputs from some or all of: the
edge itself, the sender node, and the receiver node, followed by pooling to the
receiver node.

Alternatively, depending on init arguments, it can perform the equivalent
computation from nodes to context, edges to incident nodes, or edges to context,
with the calling conventions described in the docstring for
tfgnn.keras.layers.AnyToAnyConvolutionBase.

Example: Using a SimpleConv in an MPNN-style graph update with a single-layer
network to compute "sum"-pooled message on each edge from concatenated source
and target states. (The result is then fed into the next-state layer, which
concatenates the old node state and applies another single-layer network.)

```python
dense = tf.keras.layers.Dense  # ...or some fancier feed-forward network.
graph = tfgnn.keras.layers.GraphUpdate(
    node_sets={"paper": tfgnn.keras.layers.NodeSetUpdate(
        {"cites": tfgnn.keras.layers.SimpleConv(
             dense(message_dim, "relu"), "sum", receiver_tag=tfgnn.TARGET)},
        tfgnn.keras.layers.NextStateFromConcat(dense(state_dim, "relu")))}
)(graph)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Init args</h2></th></tr>

<tr>
<td>
`message_fn`<a id="message_fn"></a>
</td>
<td>
A Keras layer that computes the individual messages from the
combined input features (see combine_type).
</td>
</tr><tr>
<td>
`reduce_type`<a id="reduce_type"></a>
</td>
<td>
Specifies how to pool the messages to receivers. Defaults to
"sum", can be any name from tfgnn.get_registered_reduce_operation_names().
</td>
</tr><tr>
<td>
`combine_type`<a id="combine_type"></a>
</td>
<td>
a string understood by tfgnn.combine_values(), to specify how
the inputs are combined before passing them to the message_fn. Defaults
to "concat", which concatenates inputs along the last axis.
</td>
</tr><tr>
<td>
`receiver_tag`<a id="receiver_tag"></a>
</td>
<td>
 one of <a href="../../../tfgnn.md#SOURCE"><code>tfgnn.SOURCE</code></a>, <a href="../../../tfgnn.md#TARGET"><code>tfgnn.TARGET</code></a> or <a href="../../../tfgnn.md#CONTEXT"><code>tfgnn.CONTEXT</code></a>.
Selects the receiver of the pooled messages.
If set to <a href="../../../tfgnn.md#SOURCE"><code>tfgnn.SOURCE</code></a> or <a href="../../../tfgnn.md#TARGET"><code>tfgnn.TARGET</code></a>, the layer can be called for
an edge set and will pool results at the specified endpoint of the edges.
If set to <a href="../../../tfgnn.md#CONTEXT"><code>tfgnn.CONTEXT</code></a>, the layer can be called for an edge set or node
set and will pool results for the context (i.e., per graph component).
If left unset for init, the tag must be passed at call time.
</td>
</tr><tr>
<td>
`receiver_feature`<a id="receiver_feature"></a>
</td>
<td>
Can be set to override <a href="../../../tfgnn.md#HIDDEN_STATE"><code>tfgnn.HIDDEN_STATE</code></a> for use as
the input feature from the receiver. Passing `None` disables input from
the receiver.
</td>
</tr><tr>
<td>
`sender_node_feature`<a id="sender_node_feature"></a>
</td>
<td>
Can be set to override <a href="../../../tfgnn.md#HIDDEN_STATE"><code>tfgnn.HIDDEN_STATE</code></a> for use as
the input feature from sender nodes. Passing `None` disables input from
the sender node.
IMPORANT: Must be set to `None` for use with `receiver_tag=tfgnn.CONTEXT`
on an edge set, or for pooling from edges without sender node states.
</td>
</tr><tr>
<td>
`sender_edge_feature`<a id="sender_edge_feature"></a>
</td>
<td>
Can be set to a feature name of the edge set to select
it as an input feature. By default, this set to `None`, which disables
this input.
IMPORTANT: Must be set for use with `receiver_tag=tfgnn.CONTEXT` on an
edge set.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Call returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A Tensor whose leading dimension is indexed by receivers, with the
pooled messages for each receiver.
</td>
</tr>

</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr> <td> `receiver_tag`<a id="receiver_tag"></a> </td> <td> one of
<a href="../../../tfgnn.md#SOURCE"><code>tfgnn.SOURCE</code></a>,
<a href="../../../tfgnn.md#TARGET"><code>tfgnn.TARGET</code></a> or
<a href="../../../tfgnn.md#CONTEXT"><code>tfgnn.CONTEXT</code></a>. The results
are aggregated for this graph piece. If set to
<a href="../../../tfgnn.md#SOURCE"><code>tfgnn.SOURCE</code></a> or
<a href="../../../tfgnn.md#TARGET"><code>tfgnn.TARGET</code></a>, the layer can
be called for an edge set and will aggregate results at the specified endpoint
of the edges. If set to
<a href="../../../tfgnn.md#CONTEXT"><code>tfgnn.CONTEXT</code></a>, the layer
can be called for an edge set or a node set and will aggregate results for
context (per graph component). If left unset for init, the tag must be passed at
call time. </td> </tr><tr> <td> `receiver_feature`<a id="receiver_feature"></a>
</td> <td> The name of the feature that is read from the receiver graph piece
and passed as convolve(receiver_input=...). </td> </tr><tr> <td>
`sender_node_feature`<a id="sender_node_feature"></a> </td> <td> The name of the
feature that is read from the sender nodes, if any, and passed as
convolve(sender_node_input=...). NOTICE this must be `None` for use with
`receiver_tag=tfgnn.CONTEXT` on an edge set, or for pooling from edges without
sender node states. </td> </tr><tr> <td>
`sender_edge_feature`<a id="sender_edge_feature"></a> </td> <td> The name of the
feature that is read from the sender edges, if any, and passed as
convolve(sender_edge_input=...). NOTICE this must not be `None` for use with
`receiver_tag=tfgnn.CONTEXT` on an edge set. </td> </tr><tr> <td>
`extra_receiver_ops`<a id="extra_receiver_ops"></a> </td> <td> A str-keyed
dictionary of Python callables that are wrapped to bind some arguments and then
passed on to `convolve()`. Sample usage: `extra_receiver_ops={"softmax":
tfgnn.softmax}`. The values passed in this dict must be callable as follows,
with two positional arguments:

```python
f(graph, receiver_tag, node_set_name=..., feature_value=..., ...)
f(graph, receiver_tag, edge_set_name=..., feature_value=..., ...)
```

The wrapped callables seen by `convolve()` can be called like

```python
wrapped_f(feature_value, ...)
```

The first three arguments of `f` are set to the input GraphTensor of
the layer and the tag/name pair required by <a href="../../../tfgnn/broadcast.md"><code>tfgnn.broadcast()</code></a> and
<a href="../../../tfgnn/pool.md"><code>tfgnn.pool()</code></a> to move values between the receiver and the messages that
are computed inside the convolution. The sole positional argument of
`wrapped_f()` is passed to `f()`  as `feature_value=`, and any keyword
arguments are forwarded.
</td>
</tr><tr>
<td>
`**kwargs`<a id="**kwargs"></a>
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
`takes_receiver_input`<a id="takes_receiver_input"></a>
</td>
<td>
If `False`, all calls to convolve() will get `receiver_input=None`.
</td>
</tr><tr>
<td>
`takes_sender_edge_input`<a id="takes_sender_edge_input"></a>
</td>
<td>
If `False`, all calls to convolve() will get `sender_edge_input=None`.
</td>
</tr><tr>
<td>
`takes_sender_node_input`<a id="takes_sender_node_input"></a>
</td>
<td>
If `False`, all calls to convolve() will get `sender_node_input=None`.
</td>
</tr>
</table>

## Methods

<h3 id="convolve"><code>convolve</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/convolutions.py#L120-L142">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>convolve(
    *,
    sender_node_input: Optional[tf.Tensor],
    sender_edge_input: Optional[tf.Tensor],
    receiver_input: Optional[tf.Tensor],
    broadcast_from_sender_node: Callable[[tf.Tensor], tf.Tensor],
    broadcast_from_receiver: Callable[[tf.Tensor], tf.Tensor],
    pool_to_receiver: Callable[..., tf.Tensor],
    training: bool
) -> tf.Tensor
</code></pre>

Returns the convolution result.

The Tensor inputs to this function still have their original shapes and need to
be broadcast such that the leading dimension is indexed by the items in the
graph for which messages are computed (usually edges; except when convolving
from nodes to context). In the end, values have to be pooled from there into a
Tensor with a leading dimension indexed by receivers, see `pool_to_receiver`.

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
