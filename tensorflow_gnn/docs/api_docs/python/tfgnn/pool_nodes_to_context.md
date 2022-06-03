description: Aggregates (pools) node values to graph context.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfgnn.pool_nodes_to_context" />
<meta itemprop="path" content="Stable" />
</div>

# tfgnn.pool_nodes_to_context

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_ops.py#L232-L273">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Aggregates (pools) node values to graph context.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.pool_nodes_to_context(
    graph_tensor: <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>,
    node_set_name: NodeSetName,
    reduce_type: str = &#x27;sum&#x27;,
    *,
    feature_value: Optional[Field] = None,
    feature_name: Optional[FieldName] = None
) -> <a href="../tfgnn/Field.md"><code>tfgnn.Field</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

Given a particular node set (identified by `node_set_name`), this operation
reduces node features to their corresponding graph component. For example,
setting `reduce_type='sum'` computes the sum over the node features of each
graph. (See the corresponding `broadcast_context_to_nodes()` mirror
operation).

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
`graph_tensor`
</td>
<td>
A scalar GraphTensor.
</td>
</tr><tr>
<td>
`node_set_name`
</td>
<td>
A node set name.
</td>
</tr><tr>
<td>
`reduce_type`
</td>
<td>
A pooling operation name, like 'sum', 'mean' or 'max'. For the
list of supported values use `get_registered_reduce_operation_names()`.
You may `register_reduce_operation()` to register new ops.
</td>
</tr><tr>
<td>
`feature_value`
</td>
<td>
A ragged or dense node feature value. Has a shape
`[num_nodes, *feature_shape]`, where `num_nodes` is the number of nodes in
the `node_set_name` node set and `feature_shape` is the shape of the
feature value for each node.
</td>
</tr><tr>
<td>
`feature_name`
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

