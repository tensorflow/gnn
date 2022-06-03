description: Aggregates (pools) edge values to incident nodes.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfgnn.pool_edges_to_node" />
<meta itemprop="path" content="Stable" />
</div>

# tfgnn.pool_edges_to_node

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_ops.py#L94-L151">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Aggregates (pools) edge values to incident nodes.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.pool_edges_to_node(
    graph_tensor: <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>,
    edge_set_name: EdgeSetName,
    node_tag: IncidentNodeTag,
    reduce_type: str = &#x27;sum&#x27;,
    *,
    feature_value: Optional[Field] = None,
    feature_name: Optional[FieldName] = None
) -> <a href="../tfgnn/Field.md"><code>tfgnn.Field</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

Given a particular edge set (identified by `edge_set_name` name), this
operation reduces edge features at the specific incident node of each edge (as
indicated by `node_tag`). For example, setting `node_tag=tfgnn.TARGET` and
`reduce_type='sum'` computes the sum over the incoming edge features at each
node. (See the corresponding `broadcast_node_to_edges()` mirror operation).

The feature to fetch edge values from is provided either by name (using
`feature_name`) and found in the graph tensor itself, or provided explicitly
(using `feature_value`) in which case its shape has to be compatible with the
shape prefix of the edge set being gathered from. One of `feature_value`
or `feature_name` must be specified.

(Note that in most cases the `feature_value` form will be used, because in a
regular convolution, we will first broadcast over edges and combine the result
of that with this function.)

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
The name of the edge set from which values are pooled.
</td>
</tr><tr>
<td>
`node_tag`
</td>
<td>
The incident node of each edge at which values are aggregated,
identified by its tag in the edge set.
</td>
</tr><tr>
<td>
`reduce_type`
</td>
<td>
A pooling operation name, like 'sum', 'mean' or 'max'. For the
list of supported values use `get_registered_reduce_operation_names()`.
You may use `register_reduce_operation()` to register new ops.
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
The edge values pooled to each incident node. Has a shape `[num_nodes,
*feature_shape]`, where `num_nodes` is the number of nodes in the incident
node set and `feature_shape` is not affected.
</td>
</tr>

</table>

