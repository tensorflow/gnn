# tfgnn.pool_neighbors_to_node_feature

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_ops.py#L1201-L1255">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Aggregates (pools) sender node feature to receiver nodes feature.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.pool_neighbors_to_node_feature(
    graph_tensor: GraphTensor,
    edge_set_name: Union[Sequence[EdgeSetName], EdgeSetName],
    to_tag: IncidentNodeTag,
    *,
    reduce_type: str,
    feature_name: const.FieldName,
    to_feature_name: Optional[const.FieldName] = None,
    from_tag: Optional[IncidentNodeTag] = None
) -> gt.GraphTensor
</code></pre>

<!-- Placeholder for "Used in" -->

Similar to the `pool_neighbors_to_node` but results in the graph tensor with
updated receiver feature.

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
<code>feature_name</code><a id="feature_name"></a>
</td>
<td>
An neighbors feature name to pool values from.
</td>
</tr><tr>
<td>
<code>to_feature_name</code><a id="to_feature_name"></a>
</td>
<td>
A receiver feature name to write pooled values to. Defaults
to the feature_name.
</td>
</tr><tr>
<td>
<code>from_tag</code><a id="from_tag"></a>
</td>
<td>
The incident node of each edge from which values are aggregated.
Optional for regular graphs, required for hypergraphs.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Copy of the input graph tensor with updated <code>receiver_feature_name</code> for the
receiver node set.
</td>
</tr>

</table>
