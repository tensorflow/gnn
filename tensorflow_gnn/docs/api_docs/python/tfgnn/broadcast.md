# tfgnn.broadcast

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/broadcast_ops.py#L181-L258">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Broadcasts values from nodes to edges, or from context to nodes or edges.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.broadcast(
    graph_tensor: GraphTensor,
    from_tag: IncidentNodeOrContextTag,
    *,
    edge_set_name: Union[Sequence[EdgeSetName], EdgeSetName, None] = None,
    node_set_name: Union[Sequence[NodeSetName], NodeSetName, None] = None,
    feature_value: Optional[Field] = None,
    feature_name: Optional[FieldName] = None
) -> Union[list[Field], Field]
</code></pre>

<!-- Placeholder for "Used in" -->

This function broadcasts a feature value from context to nodes or edges if
called with `from_tag=tfgnn.CONTEXT`, or from incident nodes to edges if called
with `from_tag` set to an ordinary node tag like
<a href="../tfgnn.md#SOURCE"><code>tfgnn.SOURCE</code></a> or
<a href="../tfgnn.md#TARGET"><code>tfgnn.TARGET</code></a>.

The `edge_set_name` (or `node_set_name`, when broadcasting from context) can be
set to the name of a single destination, or to a list of names of multiple
destinations.

Functionally, there is no difference to calling the underlying functions
`broadcast_node_to_edges()`, `broadcast_context_to_nodes()`, or
`broadcast_context_to_edges()` directly on individual edge sets or node sets.
However, the more generic API of this function provides the proper mirror image
of <a href="../tfgnn/pool.md"><code>tfgnn.pool()</code></a>, which comes in
handy for some algorithms.

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
`from_tag`<a id="from_tag"></a>
</td>
<td>
Values are broadcast from context if this is <a href="../tfgnn.md#CONTEXT"><code>tfgnn.CONTEXT</code></a> or
from the incident node on each edge with this tag.
</td>
</tr><tr>
<td>
`edge_set_name`<a id="edge_set_name"></a>
</td>
<td>
The name of the edge set to which values are broadcast, or
a non-empty sequence of such names. Unless `from_tag=tfgnn.CONTEXT`,
all named edge sets must have the same incident node set at the given tag.
</td>
</tr><tr>
<td>
`node_set_name`<a id="node_set_name"></a>
</td>
<td>
The name of the node set to which values are broadcast,
or a non-empty sequence of such names. Can only be passed together with
`from_tag=tfgnn.CONTEXT`. Exactly one of edge_set_name or node_set_name
must be set.
</td>
</tr><tr>
<td>
`feature_value`<a id="feature_value"></a>
</td>
<td>
A tensor of shape `[num_items, *feature_shape]` from which
the broadcast values are taken. The first dimension indexes the items
from which the broadcast is done (that is, the nodes of the common node
set identified by `from_tag`, or the graph components in the context).
</td>
</tr><tr>
<td>
`feature_name`<a id="feature_name"></a>
</td>
<td>
The name of a feature stored in the graph, for use instead of
feature_value. Exactly one of feature_name or feature_value must be set.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The result of broadcasting to the specified edge set(s) or node set(s).
If a single name was specified, the result is is a single tensor.
If a list of names was specified, the result is a list of tensors,
with parallel indices.
</td>
</tr>

</table>
