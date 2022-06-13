description: Broadcasts values from nodes to incident edges.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfgnn.broadcast_node_to_edges" />
<meta itemprop="path" content="Stable" />
</div>

# tfgnn.broadcast_node_to_edges

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_ops.py#L47-L91">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Broadcasts values from nodes to incident edges.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.broadcast_node_to_edges(
    graph_tensor: <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>,
    edge_set_name: EdgeSetName,
    node_tag: IncidentNodeTag,
    *,
    feature_value: Optional[Field] = None,
    feature_name: Optional[FieldName] = None
) -> <a href="../tfgnn/Field.md"><code>tfgnn.Field</code></a>
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
The name of the edge set to which values are broadcast.
</td>
</tr><tr>
<td>
`node_tag`
</td>
<td>
The incident side of each edge from which values are broadcast,
specified by its tag in the edge set (e.g. <a href="../tfgnn.md#SOURCE"><code>tfgnn.SOURCE</code></a>,
<a href="../tfgnn.md#TARGET"><code>tfgnn.TARGET</code></a>).
</td>
</tr><tr>
<td>
`feature_value`
</td>
<td>
A ragged or dense source node feature values. Has a shape
`[num_nodes, *feature_shape]`, where `num_nodes` is the number of nodes in
the incident node set and feature_shape is the shape of the feature value
for each node.
</td>
</tr><tr>
<td>
`feature_name`
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
Source node value broadcast to corresponding edges. Has a shape `[num_edges,
*feature_shape]`, where `num_edges` is the number of edges in the
`edge_set_name` edge set and `feature_shape` is not affected.
</td>
</tr>

</table>

