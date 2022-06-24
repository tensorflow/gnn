# tfgnn.broadcast

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_ops.py#L324-L369">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Broadcasts values from nodes to edges, or from context to nodes or edges.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.broadcast(
    graph_tensor: <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>,
    from_tag: <a href="../tfgnn/IncidentNodeOrContextTag.md"><code>tfgnn.IncidentNodeOrContextTag</code></a>,
    *,
    edge_set_name: Optional[EdgeSetName] = None,
    node_set_name: Optional[NodeSetName] = None,
    feature_value: Optional[Field] = None,
    feature_name: Optional[FieldName] = None
) -> <a href="../tfgnn/Field.md"><code>tfgnn.Field</code></a>
</code></pre>

<!-- Placeholder for "Used in" -->

This function broadcasts from context if `from_tag=tfgnn.CONTEXT` and broadcasts
from incident nodes to edges if `from_tag` is an ordinary node tag like
<a href="../tfgnn.md#SOURCE"><code>tfgnn.SOURCE</code></a> or
<a href="../tfgnn.md#TARGET"><code>tfgnn.TARGET</code></a>. Most user code will
not need this flexibility and can directly call one of the underlying functions
`broadcast_node_to_edges()`, `broadcast_context_to_nodes()`, or
`broadcast_context_to_edges()`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`graph_tensor`
</td>
<td>
A scalar GraphTensor.
</td>
</tr><tr>
<td>
`from_tag`
</td>
<td>
Values are broadcast from context if this is <a href="../tfgnn.md#CONTEXT"><code>tfgnn.CONTEXT</code></a> or
from the incident node on each edge with this tag.
</td>
</tr><tr>
<td>
`edge_set_name`
</td>
<td>
The name of the edge set to which values are broadcast.
</td>
</tr><tr>
<td>
`node_set_name`
</td>
<td>
The name of the node set to which values are broadcast.
Can only be set with `from_tag=tfgnn.CONTEXT`. Either edge_set_name or
node_set_name must be set.
</td>
</tr><tr>
<td>
`feature_value`
</td>
<td>
As for the underlying broadcast_*() function.
</td>
</tr><tr>
<td>
`feature_name`
</td>
<td>
As for the underlying broadcast_*() function.
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
The result of the underlying broadcast_*() function.
</td>
</tr>

</table>
