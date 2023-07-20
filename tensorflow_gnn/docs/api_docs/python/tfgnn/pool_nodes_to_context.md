# tfgnn.pool_nodes_to_context

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/pool_ops.py#L99-L148">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Aggregates (pools) node values to graph context.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.pool_nodes_to_context(
    graph_tensor: GraphTensor,
    node_set_name: NodeSetName,
    reduce_type: str = &#x27;sum&#x27;,
    *,
    feature_value: Optional[Field] = None,
    feature_name: Optional[FieldName] = None
) -> Field
</code></pre>

<!-- Placeholder for "Used in" -->

Given a particular node set (identified by `node_set_name`), this operation
reduces node features to their corresponding graph component. For example,
setting `reduce_type="sum"` computes the sum over the node features of each
graph, while `reduce_type="sum|mean"` would compute the concatenation of their
sum and mean along the innermost axis, in this order.

For the converse operation of broadcasting from context to nodes, see
<a href="../tfgnn/broadcast_context_to_nodes.md"><code>tfgnn.broadcast_context_to_nodes()</code></a>.
For a generalization beyond a single node set, see
<a href="../tfgnn/pool.md"><code>tfgnn.pool()</code></a>.

The feature to fetch node values from is provided either by name (using
`feature_name`) and found in the graph tensor itself, or provided explicitly
(using `feature_value`). One of `feature_value` or `feature_name` must be
specified.

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
`node_set_name`<a id="node_set_name"></a>
</td>
<td>
A node set name.
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
A ragged or dense node feature value. Has a shape
`[num_nodes, *feature_shape]`, where `num_nodes` is the number of nodes in
the `node_set_name` node set and `feature_shape` is the shape of the
feature value for each node.
</td>
</tr><tr>
<td>
`feature_name`<a id="feature_name"></a>
</td>
<td>
A node feature name.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Node value pooled to graph context. Has a shape `[num_components,
*feature_shape]`, where `num_components` is the number of components in a
graph and `feature_shape` is not affected.
</td>
</tr>

</table>

