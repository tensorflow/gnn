# tfgnn.pool_edges_to_context

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/pool_ops.py#L151-L204">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Aggregates (pools) edge values to graph context.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.pool_edges_to_context(
    graph_tensor: GraphTensor,
    edge_set_name: EdgeSetName,
    reduce_type: str = &#x27;sum&#x27;,
    *,
    feature_value: Optional[Field] = None,
    feature_name: Optional[FieldName] = None
) -> Field
</code></pre>

<!-- Placeholder for "Used in" -->

Given a particular edge set (identified by `edge_set_name`), this operation
reduces edge features to their corresponding graph component. For example,
setting `reduce_type="sum"` computes the sum over the edge features of each
graph, while `reduce_type="sum|mean"` would compute the concatenation of their
sum and mean along the innermost axis, in this order.

For the converse operation of broadcasting from context to edges, see
<a href="../tfgnn/broadcast_context_to_edges.md"><code>tfgnn.broadcast_context_to_edges()</code></a>.
For a generalization beyond a single edge set, see
<a href="../tfgnn/pool.md"><code>tfgnn.pool()</code></a>.

The feature to fetch edge values from is provided either by name (using
`feature_name`) and found in the graph tensor itself, or provided explicitly
(using `feature_value`). One of `feature_value` or `feature_name` must be
specified.

(Note that in most cases the `feature_value` form will be used, because in a
regular convolution, we will first broadcast over edges and combine the result
of that with this function or a pooling over the nodes.)

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`graph_tensor`<a id="graph_tensor"></a>
</td>
<td>
A scalar GraphTensor.
</td>
</tr><tr>
<td>
`edge_set_name`<a id="edge_set_name"></a>
</td>
<td>
An edge set name.
</td>
</tr><tr>
<td>
`reduce_type`<a id="reduce_type"></a>
</td>
<td>
A pooling operation name, like `"sum"` or `"mean"`, or a
`|`-separated combination of these; see <a href="../tfgnn/pool.md"><code>tfgnn.pool()</code></a>.
</td>
</tr><tr>
<td>
`feature_value`<a id="feature_value"></a>
</td>
<td>
A ragged or dense edge feature value. Has a shape
`[num_edges, *feature_shape]`, where `num_edges` is the number of edges in
the `edge_set_name` edge set and `feature_shape` is the shape of the
feature value for each edge.
</td>
</tr><tr>
<td>
`feature_name`<a id="feature_name"></a>
</td>
<td>
An edge feature name.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A node value pooled to graph context. Has a shape `[num_components,
*feature_shape]`, where `num_components` is the number of components in a
graph and `feature_shape` is not affected.
</td>
</tr>

</table>

