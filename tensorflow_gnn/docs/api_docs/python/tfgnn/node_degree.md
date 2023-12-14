# tfgnn.node_degree

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_ops.py#L615-L643">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Returns the degree of each node w.r.t. one side of an edge set.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.node_degree(
    graph_tensor: GraphTensor,
    edge_set_name: EdgeSetName,
    node_tag: IncidentNodeTag
) -> Field
</code></pre>

<!-- Placeholder for "Used in" -->
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
<code>edge_set_name</code><a id="edge_set_name"></a>
</td>
<td>
The name of the edge set for which degrees are calculated.
</td>
</tr><tr>
<td>
<code>node_tag</code><a id="node_tag"></a>
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
An integer Tensor of shape <code>[num_nodes]</code> and dtype equal to <code>indices_dtype</code>
of the GraphTensor. Element <code>i</code> contains the number of edges in the given
edge set that have node index <code>i</code> as their endpoint with the given
<code>node_tag</code>. The dimension <code>num_nodes</code> is the number of nodes in the
respective node set.
</td>
</tr>

</table>
