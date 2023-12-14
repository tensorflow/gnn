# tfgnn.broadcast_node_to_edges

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/broadcast_ops.py#L36-L81">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Broadcasts values from nodes to incident edges.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.broadcast_node_to_edges(
    graph_tensor: GraphTensor,
    edge_set_name: EdgeSetName,
    node_tag: IncidentNodeTag,
    *,
    feature_value: Optional[Field] = None,
    feature_name: Optional[FieldName] = None
) -> Field
</code></pre>

<!-- Placeholder for "Used in" -->

Given a particular edge set (identified by `edge_set_name` name), this
operation collects node features from the specific incident node of each edge
(as indicated by `node_tag`). For example, setting `node_tag=tfgnn.SOURCE` and
`reduce_type='sum'` gathers the source node features over each edge. (See the
corresponding `pool_edges_to_node()` mirror operation).

The feature to fetch node values from is provided either by name (using
`feature_name`) and found in the graph tensor itself, or provided explicitly
(using `feature_value`) in which case its shape has to be compatible with the
shape prefix of the node set being gathered from. One of `feature_value`
or `feature_name` must be specified.

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
The name of the edge set to which values are broadcast.
</td>
</tr><tr>
<td>
<code>node_tag</code><a id="node_tag"></a>
</td>
<td>
The incident side of each edge from which values are broadcast,
specified by its tag in the edge set (e.g. <a href="../tfgnn.md#SOURCE"><code>tfgnn.SOURCE</code></a>,
<a href="../tfgnn.md#TARGET"><code>tfgnn.TARGET</code></a>).
</td>
</tr><tr>
<td>
<code>feature_value</code><a id="feature_value"></a>
</td>
<td>
A ragged or dense source node feature values. Has a shape
<code>[num_nodes, *feature_shape]</code>, where <code>num_nodes</code> is the number of nodes in
the incident node set and feature_shape is the shape of the feature value
for each node.
</td>
</tr><tr>
<td>
<code>feature_name</code><a id="feature_name"></a>
</td>
<td>
A source node feature name.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Source node value broadcast to corresponding edges. Has a shape <code>[num_edges,
*feature_shape]</code>, where <code>num_edges</code> is the number of edges in the
<code>edge_set_name</code> edge set and <code>feature_shape</code> is not affected.
</td>
</tr>

</table>

