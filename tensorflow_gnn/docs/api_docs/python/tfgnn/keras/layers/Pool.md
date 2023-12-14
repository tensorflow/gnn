# tfgnn.keras.layers.Pool

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/graph_ops.py#L806-L919">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Pools a GraphTensor feature.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.Pool(
    tag: Optional[IncidentNodeOrContextTag] = None,
    reduce_type: Optional[str] = None,
    *,
    edge_set_name: Union[Sequence[EdgeSetName], EdgeSetName, None] = None,
    node_set_name: Union[Sequence[NodeSetName], NodeSetName, None] = None,
    feature_name: Optional[FieldName] = None,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

This layer accepts a complete GraphTensor and returns a tensor with the
result of pooling some feature.

There are two kinds of pooling that this layer can be used for:

*   From an edge set (or multiple edge sets) to a single node set. This is
    selected by specifying the sender as `edge_set_name=...` and the receiver
    with tag `tgnn.SOURCE` or
    <a href="../../../tfgnn.md#TARGET"><code>tfgnn.TARGET</code></a>; the
    corresponding node set name is implied. (In case of multiple edge sets, it
    must be the same for all.) The result is a tensor shaped like a node feature
    in which each node has the aggregated feature values from the edges of the
    edge set(s) that have it as their `SOURCE` or `TARGET`, resp.; that is, the
    outgoing or incoming edges of the node.
*   From one (or more) node sets or one (or more) edge sets to the context. This
    is selected by specifying the sender as either `node_set_name=...` or
    `edge_set_name=...` and the receiver with tag
    <a href="../../../tfgnn.md#CONTEXT"><code>tfgnn.CONTEXT</code></a>. The
    result is a tensor shaped like a context feature in which the entry for each
    graph component has the aggregated feature values from its nodes/edges in
    the selected node or edge set(s). (For more on components, see
    <a href="../../../tfgnn/GraphTensor.md#merge_batch_to_components"><code>tfgnn.GraphTensor.merge_batch_to_components()</code></a>.)

Feature values are aggregated into a single value by a reduction function like
`"sum"` or `"mean|max_no_inf"` as described for `tfgnn.pool()`; see there for
more details.

Both the initialization of and the call to this layer accept arguments for the
receiver `tag`, the `node_set_name`/`edge_set_name`, the `reduce_type` and the
`feature_name`. The call arguments take effect for that call only and can supply
missing values, but they are not allowed to contradict initialization arguments.
The feature name can be left unset to select
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
to select the receiver.
</td>
</tr><tr>
<td>
<code>reduce_type</code><a id="reduce_type"></a>
</td>
<td>
Can be set to any <code>reduce_type</code> understood by <a href="../../../tfgnn/pool.md"><code>tfgnn.pool()</code></a>.
</td>
</tr><tr>
<td>
<code>edge_set_name</code><a id="edge_set_name"></a>
</td>
<td>
If set, the feature will be pooled from this edge set
(or this sequence of edge sets) to the receiver given by <code>tag</code>.
Mutually exclusive with <code>node_set_name</code>.
</td>
</tr><tr>
<td>
<code>node_set_name</code><a id="node_set_name"></a>
</td>
<td>
If set, the feature will be pooled from this node set
(or sequence of node sets). The receiver must be selected as
<code>tag=tfgnn.CONTEXT</code>. Mutually exclusive with <code>edge_set_name</code>.
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
<code>reduce_type</code><a id="reduce_type"></a>
</td>
<td>
Same meaning as for init. Must be passed to init, or to call,
or to both (with the same value).
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
A tensor with the pooled feature value.
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
<code>reduce_type</code><a id="reduce_type"></a>
</td>
<td>
Returns the reduce_type argument to init, or None if unset.
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
