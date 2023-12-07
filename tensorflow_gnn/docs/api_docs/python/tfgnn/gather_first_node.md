# tfgnn.gather_first_node

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_ops.py#L175-L226">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Gathers feature value from the first node of each graph component.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.gather_first_node(
    graph_tensor: GraphTensor,
    node_set_name: NodeSetName,
    *,
    feature_value: Optional[Field] = None,
    feature_name: Optional[FieldName] = None
) -> Field
</code></pre>

<!-- Placeholder for "Used in" -->

Given a particular node set (identified by `node_set_name`), this operation will
gather the given feature from the first node of each graph component.

This is often used for rooted graphs created by sampling around the
neighborhoods of seed nodes in a large graph: by convention, each seed node is
the first node of its component in the respective node set, and this operation
reads out the information it has accumulated there. (In other node sets, the
first node may be arbitrary -- or nonexistant, in which case this operation must
not be used and may raise an error at runtime.)

The feature to fetch node values from is provided either by name (using
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
A seed node set name.
</td>
</tr><tr>
<td>
<code>feature_value</code><a id="feature_value"></a>
</td>
<td>
A ragged or dense node feature value. Has a shape
<code>[num_nodes, *feature_shape]</code>, where <code>num_nodes</code> is the number of nodes in
the <code>node_set_name</code> node set and <code>feature_shape</code> is the shape of the
feature value for each node.
</td>
</tr><tr>
<td>
<code>feature_name</code><a id="feature_name"></a>
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
A tensor of gathered feature values, one for each graph component, like a
context feature. Has a shape <code>[num_components, *feature_shape]</code>, where
<code>num_components</code> is the number of components in a graph and <code>feature_shape</code>
is not affected.
</td>
</tr>

</table>
