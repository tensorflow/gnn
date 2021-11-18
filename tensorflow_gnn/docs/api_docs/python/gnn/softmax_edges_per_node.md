description: Softmaxes all the edges in the graph over their incident nodes.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gnn.softmax_edges_per_node" />
<meta itemprop="path" content="Stable" />
</div>

# gnn.softmax_edges_per_node

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/normalization_ops.py#L11-L56">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Softmaxes all the edges in the graph over their incident nodes.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gnn.softmax_edges_per_node(
    graph_tensor: <a href="../gnn/GraphTensor.md"><code>gnn.GraphTensor</code></a>,
    edge_set_name: gt.EdgeSetName,
    node_tag: const.IncidentNodeTag,
    *,
    feature_value: Optional[gt.Field] = None,
    feature_name: Optional[gt.FieldName] = None
) -> <a href="../gnn/Field.md"><code>gnn.Field</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

This function performs a per-edge softmax operation, grouped by all
the edges per node the direction of `node_tag`.

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
`feature_value`
</td>
<td>
A ragged or dense edge feature value.
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
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>
<tr class="alt">
<td colspan="2">
ValueError is `edge_set_name` is not in the `graph_tensor` edges.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The edge values softmaxed per incident node. The dimensions do not change.
</td>
</tr>

</table>

