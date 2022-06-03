description: Aggregates (pools) edge values to graph context.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfgnn.pool_edges_to_context" />
<meta itemprop="path" content="Stable" />
</div>

# tfgnn.pool_edges_to_context

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_ops.py#L276-L321">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Aggregates (pools) edge values to graph context.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.pool_edges_to_context(
    graph_tensor: <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>,
    edge_set_name: EdgeSetName,
    reduce_type: str = &#x27;sum&#x27;,
    *,
    feature_value: Optional[Field] = None,
    feature_name: Optional[FieldName] = None
) -> <a href="../tfgnn/Field.md"><code>tfgnn.Field</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

Given a particular edge set (identified by `edge_set_name`), this operation
reduces edge features to their corresponding graph component. For example,
setting `reduce_type='sum'` computes the sum over the edge features of each
graph. (See the corresponding `broadcast_context_to_edges()` mirror
operation).

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
`graph_tensor`
</td>
<td>
A scalar GraphTensor.
</td>
</tr><tr>
<td>
`edge_set_name`
</td>
<td>
An edge set name.
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
A ragged or dense edge feature value. Has a shape
`[num_edges, *feature_shape]`, where `num_edges` is the number of edges in
the `edge_set_name` edge set and `feature_shape` is the shape of the
feature value for each edge.
</td>
</tr><tr>
<td>
`feature_name`
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

