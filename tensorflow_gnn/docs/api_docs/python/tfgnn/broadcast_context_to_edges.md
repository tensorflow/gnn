description: Broadcasts a context value to the edge_set edges.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfgnn.broadcast_context_to_edges" />
<meta itemprop="path" content="Stable" />
</div>

# tfgnn.broadcast_context_to_edges

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_ops.py#L193-L229">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Broadcasts a context value to the `edge_set` edges.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.broadcast_context_to_edges(
    graph_tensor: <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>,
    edge_set_name: EdgeSetName,
    *,
    feature_value: Optional[Field] = None,
    feature_name: Optional[FieldName] = None
) -> <a href="../tfgnn/Field.md"><code>tfgnn.Field</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

Given a particular edge set (as identified by `edge_set_name`), this operation
collects context features from the corresponding graphs to each edge. See the
corresponding `pool_edges_to_context()` mirror operation).

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
`feature_value`
</td>
<td>
A ragged or dense graph context feature value. Has a shape
`[num_components, *feature_shape]`, where `num_components` is the number
of components in a graph and `feature_shape` is the shape of the feature
value for each component.
</td>
</tr><tr>
<td>
`feature_name`
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
Graph context value broadcast to the `edge_set` edges. Has a shape
`[num_edges, *feature_shape]`, where `num_edges` is the number of edges in
the `edge_set_name` edge set and `feature_shape` is not affected.
</td>
</tr>

</table>

