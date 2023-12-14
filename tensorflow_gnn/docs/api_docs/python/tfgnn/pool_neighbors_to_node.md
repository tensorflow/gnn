# tfgnn.pool_neighbors_to_node

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_ops.py#L1134-L1198">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Aggregates (pools) neighbor node values along one or more edge sets.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.pool_neighbors_to_node(
    graph_tensor: GraphTensor,
    edge_set_name: Union[Sequence[EdgeSetName], EdgeSetName],
    to_tag: IncidentNodeTag,
    *,
    reduce_type: str,
    from_tag: Optional[IncidentNodeTag] = None,
    feature_value: Optional[Field] = None,
    feature_name: Optional[FieldName] = None
) -> Field
</code></pre>

<!-- Placeholder for "Used in" -->

This is a helper function that first broadcasts feature values from the nodes at
the `from_tag` endpoints (the "neighbors") of each given edge set and then pools
those values at the `to_tag` endpoints (the "nodes").

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>graph_tensor</code><a id="graph_tensor"></a>
</td>
<td>
A scalar GraphTensor.
</td>
</tr><tr>
<td>
<code>edge_set_name</code><a id="edge_set_name"></a>
</td>
<td>
The name of the edge set through which values are pooled, or
a non-empty sequence of such names. It is required that all edge sets
connect the same <code>from_tag</code> and <code>to_tag</code> node sets.
</td>
</tr><tr>
<td>
<code>to_tag</code><a id="to_tag"></a>
</td>
<td>
The incident node of each edge at which values are aggregated,
identified by its tag in the edge set.
</td>
</tr><tr>
<td>
<code>reduce_type</code><a id="reduce_type"></a>
</td>
<td>
A pooling operation name like <code>"sum"</code> or <code>"mean"</code>, or a
<code>|</code>-separated combination of these; see <a href="../tfgnn/pool.md"><code>tfgnn.pool()</code></a>.
</td>
</tr><tr>
<td>
<code>from_tag</code><a id="from_tag"></a>
</td>
<td>
The incident node of each edge from which values are aggregated.
Required for hypergraphs. For ordinary graphs, defaults to the opposite of
<code>to_tag</code>.
</td>
</tr><tr>
<td>
<code>feature_value</code><a id="feature_value"></a>
</td>
<td>
A ragged or dense neighbor feature value. Has a shape
<code>[num_sender_nodes, *feature_shape]</code>, where <code>num_sender_nodes</code> is the
number of sender nodes and <code>feature_shape</code> is the shape of the feature
value for each sender node.
</td>
</tr><tr>
<td>
<code>feature_name</code><a id="feature_name"></a>
</td>
<td>
An neighbors feature name.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The sender nodes values pooled to each receiver node. Has a shape
<code>[num_receiver_nodes, *feature_shape]</code>, where <code>num_receiver_nodes</code> is the
number of receiver nodes and <code>feature_shape</code> is not affected.
</td>
</tr>

</table>
