# tfgnn.keras.layers.Broadcast

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/graph_ops.py#L715-L803">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Broadcasts a GraphTensor feature.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.Broadcast(
    tag: Optional[IncidentNodeOrContextTag] = None,
    *,
    edge_set_name: Union[Sequence[EdgeSetName], EdgeSetName, None] = None,
    node_set_name: Union[Sequence[NodeSetName], NodeSetName, None] = None,
    feature_name: Optional[FieldName] = None,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

This layer accepts a complete GraphTensor and returns a tensor (or tensors) with
the broadcast feature value.

There are two kinds of broadcast that this layer can be used for:

*   From a node set to an edge set (or multiple edge sets). This is selected by
    specifying the receiver edge set(s) as `edge_set_name=...` and the sender by
    tag `tgnn.SOURCE` or
    <a href="../../../tfgnn.md#TARGET"><code>tfgnn.TARGET</code></a> relative to
    the edge set(s). The node set name is implied. (In case of multiple edge
    sets, it must agree between all of them.) The result is a tensor (or list of
    tensors) shaped like an edge feature in which each edge has a copy of the
    feature that is present at its `SOURCE` or `TARGET` node. From a node's
    point of view, `SOURCE` means broadcast to outgoing edges, and `TARGET`
    means broadcast to incoming edges.
*   From the context to one (or more) node sets or one (or more) edge sets. This
    is selected by specifying the receiver(s) as either `node_set_name=...` or
    `edge_set_name=...` and the sender by tag
    <a href="../../../tfgnn.md#CONTEXT"><code>tfgnn.CONTEXT</code></a>. The
    result is a tensor (or list of tensors) shaped like a node/edge feature in
    which each node/edge has a copy of the context feature from the graph
    component it belongs to. (For more on components, see
    <a href="../../../tfgnn/GraphTensor.md#merge_batch_to_components"><code>tfgnn.GraphTensor.merge_batch_to_components()</code></a>.)

Both the initialization of and the call to this layer accept arguments to set
the `tag`, `node_set_name`/`edge_set_name`, and the `feature_name`. The call
arguments take effect for that call only and can supply missing values, but they
are not allowed to contradict initialization arguments. The feature name can be
left unset to select
<a href="../../../tfgnn.md#HIDDEN_STATE"><code>tfgnn.HIDDEN_STATE</code></a>.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Init args</h2></th></tr>

<tr>
<td>
<code>tag</code><a id="tag"></a>
</td>
<td>
Can be set to one of <a href="../../../tfgnn.md#SOURCE"><code>tfgnn.SOURCE</code></a>, <a href="../../../tfgnn.md#TARGET"><code>tfgnn.TARGET</code></a> or <a href="../../../tfgnn.md#CONTEXT"><code>tfgnn.CONTEXT</code></a>
to select the sender from which feature values are broadcast.
</td>
</tr><tr>
<td>
<code>edge_set_name</code><a id="edge_set_name"></a>
</td>
<td>
If set, the feature will be broadcast to this edge set
(or this sequence of edge sets) from the sender given by <code>tag</code>.
Mutually exclusive with <code>node_set_name</code>.
</td>
</tr><tr>
<td>
<code>node_set_name</code><a id="node_set_name"></a>
</td>
<td>
If set, the feature will be broadcast to this node set
(or sequence of node sets). The sender must be selected as
<code>tag=tfgn.CONTEXT</code>. Mutually exclusive with <code>edge_set_name</code>.
</td>
</tr><tr>
<td>
<code>feature_name</code><a id="feature_name"></a>
</td>
<td>
The name of the feature to read. If unset (also in call),
the <a href="../../../tfgnn.md#HIDDEN_STATE"><code>tfgnn.HIDDEN_STATE</code></a> feature will be read.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Call args</h2></th></tr>

<tr>
<td>
<code>graph</code><a id="graph"></a>
</td>
<td>
The scalar <a href="../../../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a> to read from.
</td>
</tr><tr>
<td>
<code>tag</code><a id="tag"></a>
</td>
<td>
Same meaning as for init. Must be passed to init, or to call,
  or to both (with the same value).
edge_set_name, node_set_name: Same meaning as for init. One of them must
  be passed to init, or to call, or to both (with the same value).
</td>
</tr><tr>
<td>
<code>feature_name</code><a id="feature_name"></a>
</td>
<td>
Same meaning as for init. If passed to both, the value must
be the same. If passed to neither, <a href="../../../tfgnn.md#HIDDEN_STATE"><code>tfgnn.HIDDEN_STATE</code></a> is used.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tensor (or list of tensors) with the feature values broadcast to the
requested receivers.
</td>
</tr>

</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
<code>feature_name</code><a id="feature_name"></a>
</td>
<td>
Returns the feature_name argument to init, or None if unset.
</td>
</tr><tr>
<td>
<code>location</code><a id="location"></a>
</td>
<td>
Returns dict of kwarg to init with the node or edge set name.
</td>
</tr><tr>
<td>
<code>tag</code><a id="tag"></a>
</td>
<td>
Returns the tag argument to init, or None if unset.
</td>
</tr>
</table>
