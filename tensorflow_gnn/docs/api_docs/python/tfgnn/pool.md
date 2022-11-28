# tfgnn.pool

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_ops.py#L387-L435">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Pools values from edges to nodes, or from nodes or edges to context.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.pool(
    graph_tensor: <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>,
    to_tag: <a href="../tfgnn/IncidentNodeOrContextTag.md"><code>tfgnn.IncidentNodeOrContextTag</code></a>,
    *,
    edge_set_name: Optional[EdgeSetName] = None,
    node_set_name: Optional[NodeSetName] = None,
    reduce_type: str = &#x27;sum&#x27;,
    feature_value: Optional[Field] = None,
    feature_name: Optional[FieldName] = None
) -> <a href="../tfgnn/Field.md"><code>tfgnn.Field</code></a>
</code></pre>

<!-- Placeholder for "Used in" -->

This function pools to context if `to_tag=tfgnn.CONTEXT` and pools from edges to
incident nodes if `to_tag` is an ordinary node tag like
<a href="../tfgnn.md#SOURCE"><code>tfgnn.SOURCE</code></a> or
<a href="../tfgnn.md#TARGET"><code>tfgnn.TARGET</code></a>. Most user code will
not need this flexibility and can directly call one of the underlying functions
pool_edges_to_node, pool_nodes_to_context or pool_edges_to_context.

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
`to_tag`<a id="to_tag"></a>
</td>
<td>
Values are pooled to context if this is <a href="../tfgnn.md#CONTEXT"><code>tfgnn.CONTEXT</code></a> or to the
incident node on each edge with this tag.
</td>
</tr><tr>
<td>
`edge_set_name`<a id="edge_set_name"></a>
</td>
<td>
The name of the edge set from which values are pooled.
</td>
</tr><tr>
<td>
`node_set_name`<a id="node_set_name"></a>
</td>
<td>
The name of the node set from which values are pooled.
Can only be set with `to_tag=tfgnn.CONTEXT`. Either edge_set_name or
node_set_name must be set.
</td>
</tr><tr>
<td>
`reduce_type`<a id="reduce_type"></a>
</td>
<td>
As for the underlying pool_*() function: a pooling operation
name. Defaults to 'sum'.
</td>
</tr><tr>
<td>
`feature_value`<a id="feature_value"></a>
</td>
<td>
As for the underlying pool_*() function.
</td>
</tr><tr>
<td>
`feature_name`<a id="feature_name"></a>
</td>
<td>
As for the underlying pool_*() function.
Exactly one of feature_name or feature_value must be set.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The result of the underlying pool_*() function.
</td>
</tr>

</table>
