# tfgnn.get_homogeneous_node_and_edge_set_name

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L1907-L1937">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Returns the sole `node_set_name, edge_set_name` or raises `ValueError`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.get_homogeneous_node_and_edge_set_name(
    graph: Union[GraphTensor, GraphTensorSpec],
    name: str = &#x27;This operation&#x27;
) -> tuple[str, str]
</code></pre>

<!-- Placeholder for "Used in" -->

By default, this function ignores auxiliary node sets and edge sets for which
<a href="../tfgnn/get_aux_type_prefix.md"><code>tfgnn.get_aux_type_prefix(set_name)
is not None</code></a> (e.g., those needed for
<a href="../tfgnn/structured_readout.md"><code>tfgnn.structured_readout()</code></a>),
as appropriate for model-building code.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>graph</code><a id="graph"></a>
</td>
<td>
the <code>GraphTensor</code> or <code>GraphTensorSpec</code> to check.
</td>
</tr><tr>
<td>
<code>name</code><a id="name"></a>
</td>
<td>
Optionally, the name of the operation (library function, class, ...)
to mention in the user-visible error message in case an exception is
raised.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tuple <code>node_set_name, edge_set_name</code> with the unique node set and edge
set, resp., in <code>graph</code> for which <a href="../tfgnn/get_aux_type_prefix.md"><code>tfgnn.get_aux_type_prefix(set_name)</code></a> is
<code>None</code>.
</td>
</tr>

</table>
