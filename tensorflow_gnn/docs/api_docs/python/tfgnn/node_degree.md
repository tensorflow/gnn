# tfgnn.node_degree

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_ops.py#L1173-L1200">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Returns the degree of each node w.r.t. one side of an edge set.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.node_degree(
    graph_tensor: <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>,
    edge_set_name: EdgeSetName,
    node_tag: IncidentNodeTag
) -> <a href="../tfgnn/Field.md"><code>tfgnn.Field</code></a>
</code></pre>

<!-- Placeholder for "Used in" -->
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
`edge_set_name`<a id="edge_set_name"></a>
</td>
<td>
The name of the edge set for which degrees are calculated.
</td>
</tr><tr>
<td>
`node_tag`<a id="node_tag"></a>
</td>
<td>
The side of each edge for which the degrees are calculated,
specified by its tag in the edge set (e.g., <a href="../tfgnn.md#SOURCE"><code>tfgnn.SOURCE</code></a>,
<a href="../tfgnn.md#TARGET"><code>tfgnn.TARGET</code></a>).
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An integer Tensor of shape `[num_nodes]` and dtype equal to `indices_dtype`
of the GraphTensor. Element `i` contains the number of edges in the given
edge set that have node index `i` as their endpoint with the given
`node_tag`. The dimension `num_nodes` is the number of nodes in the
respective node set.
</td>
</tr>

</table>
