# tfgnn.softmax

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/normalization_ops.py#L12-L77">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Computes softmax over a many-to-one relationship in a GraphTensor.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.softmax(
    graph_tensor: <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>,
    per_tag: <a href="../tfgnn/IncidentNodeOrContextTag.md"><code>tfgnn.IncidentNodeOrContextTag</code></a>,
    *,
    edge_set_name: Optional[const.EdgeSetName] = None,
    node_set_name: Optional[const.NodeSetName] = None,
    feature_value: Optional[gt.Field] = None,
    feature_name: Optional[gt.FieldName] = None
) -> <a href="../tfgnn/Field.md"><code>tfgnn.Field</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

This function can be used to compute a softmax normalization...

  * of edge values, across the edges with a common incident node at `per_tag`
    (e.g., SOURCE or TARGET);
  * of node values, across all the nodes in the same graph component;
  * of edge values, across all the edges in the same graph component.

For non-scalar values, the softmax function is applied element-wise.

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
`per_tag`
</td>
<td>
tfgnn.CONTEXT for normalization per graph component, or an incident
node tag (e.g., <a href="../tfgnn.md#SOURCE"><code>tfgnn.SOURCE</code></a> or <a href="../tfgnn.md#TARGET"><code>tfgnn.TARGET</code></a>) for normalization per
common incident node.
</td>
</tr><tr>
<td>
`edge_set_name`
</td>
<td>
The name of the edge set on which values are normalized
Exactly one of edge_set_name and node_set_name must be set.
</td>
</tr><tr>
<td>
`node_set_name`
</td>
<td>
The name of the node set on which values are normalized,
allowed only if per_tag is <a href="../tfgnn.md#CONTEXT"><code>tfgnn.CONTEXT</code></a>. See also edge_set_name.
</td>
</tr><tr>
<td>
`feature_value`
</td>
<td>
A ragged or dense tensor with the value; cf. feature_name.
</td>
</tr><tr>
<td>
`feature_name`
</td>
<td>
The name of the feature to be used as input value.
Exactly one of feature_value or feature_name must be set.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if `graph_tensor` does not contain an edge set or node set
of the given name.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The softmaxed values. The dimensions do not change from the input.
</td>
</tr>

</table>

