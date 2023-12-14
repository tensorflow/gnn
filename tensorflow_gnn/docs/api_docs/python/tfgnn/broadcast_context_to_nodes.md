# tfgnn.broadcast_context_to_nodes

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/broadcast_ops.py#L84-L121">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Broadcasts a context value to the `node_set` nodes.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.broadcast_context_to_nodes(
    graph_tensor: GraphTensor,
    node_set_name: NodeSetName,
    *,
    feature_value: Optional[Field] = None,
    feature_name: Optional[FieldName] = None
) -> Field
</code></pre>

<!-- Placeholder for "Used in" -->

Given a particular node set (as identified by `node_set_name`), this operation
collects context features from the corresponding graphs to each node. See the
corresponding `pool_nodes_to_context()` mirror operation).

The context feature to fetch values from is provided either by name (using
`feature_name`) and found in the graph tensor itself, or provided explicitly
(using `feature_value`) in which case its shape has to be compatible with the
shape prefix of the node set being gathered from. One of `feature_value` or
`feature_name` must be specified.

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
<code>node_set_name</code><a id="node_set_name"></a>
</td>
<td>
A node set name.
</td>
</tr><tr>
<td>
<code>feature_value</code><a id="feature_value"></a>
</td>
<td>
A ragged or dense graph context feature value. Has a shape
<code>[num_components, *feature_shape]</code>, where <code>num_components</code> is the number
of components in a graph and <code>feature_shape</code> is the shape of the feature
value for each component.
</td>
</tr><tr>
<td>
<code>feature_name</code><a id="feature_name"></a>
</td>
<td>
A context feature name.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Graph context value broadcast to the <code>node_set</code> nodes. Has a shape
<code>[num_nodes, *feature_shape]</code>, where <code>num_nodes</code> is the number of nodes in
the <code>node_set_name</code> node set and <code>feature_shape</code> is not affected.
</td>
</tr>

</table>
