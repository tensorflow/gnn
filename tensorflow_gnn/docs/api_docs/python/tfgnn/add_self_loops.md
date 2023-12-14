# tfgnn.add_self_loops

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_ops.py#L53-L172">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Adds self-loops for `edge_set_name` EVEN if they already exist.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.add_self_loops(
    graph: GraphTensor, edge_set_name: gt.EdgeSetName
) -> GraphTensor
</code></pre>

<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>graph</code><a id="graph"></a>
</td>
<td>
A scalar GraphTensor.
</td>
</tr><tr>
<td>
<code>edge_set_name</code><a id="edge_set_name"></a>
</td>
<td>
An edge set in <code>graph</code> that has the same node set as source
and target.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A GraphTensor with self-loops added. A self-loop is added at each node,
even if some or all of these nodes already have a loop. All feature tensors
of the edge set are extended to cover the newly added edges with values
that are all zeros (for numeric features), false (for boolean features), or
empty (for string features), respectively.
</td>
</tr>

</table>
