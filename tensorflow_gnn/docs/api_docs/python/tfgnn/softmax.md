# tfgnn.softmax

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/normalization_ops.py#L36-L108">
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
    edge_set_name: Union[Sequence[EdgeSetName], EdgeSetName, None] = None,
    node_set_name: Union[Sequence[NodeSetName], NodeSetName, None] = None,
    feature_value: Union[Sequence[Field], Field, None] = None,
    feature_name: Optional[FieldName] = None
) -> Union[Sequence[Field], Field]
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
`graph_tensor`<a id="graph_tensor"></a>
</td>
<td>
A scalar GraphTensor.
</td>
</tr><tr>
<td>
`per_tag`<a id="per_tag"></a>
</td>
<td>
tfgnn.CONTEXT for normalization per graph component, or an incident
node tag (e.g., <a href="../tfgnn.md#SOURCE"><code>tfgnn.SOURCE</code></a> or <a href="../tfgnn.md#TARGET"><code>tfgnn.TARGET</code></a>) for normalization per
common incident node.
</td>
</tr><tr>
<td>
`edge_set_name`<a id="edge_set_name"></a>
</td>
<td>
The name of the edge set on which values are normalized,
or a non-empty sequence of such names. Unless `from_tag=tfgnn.CONTEXT`,
all named edge sets must have the same incident node set at the given tag.
</td>
</tr><tr>
<td>
`node_set_name`<a id="node_set_name"></a>
</td>
<td>
The name of the node set on which values are normalized,
or a non-empty sequence of such names. Can only be passed together with
`from_tag=tfgnn.CONTEXT`. Exactly one of edge_set_name or node_set_name
must be set.
</td>
</tr><tr>
<td>
`feature_value`<a id="feature_value"></a>
</td>
<td>
A tensor or list of tensors, parallel to the node_set_names
or edge_set_names, to supply the input values of softmax. Each tensor
has shape `[num_items, *feature_shape]`, where `num_items` is the number
of edges in the given edge set or nodes in the given node set, and
`*feature_shape` is the same across all inputs.
</td>
</tr><tr>
<td>
`feature_name`<a id="feature_name"></a>
</td>
<td>
The name of a feature stored on each graph piece from which
pooling is done, for use instead of an explicity passed feature_value.
Exactly one of feature_name or feature_value must be set.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tensor or a list of tensors with the softmaxed values. The dimensions of
the tensors and the length of the list do not change from the input.
</td>
</tr>

</table>

