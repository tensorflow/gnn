# tfgnn.add_readout_from_first_node

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/readout.py#L545-L603">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Adds a readout structure equivalent to
<a href="../tfgnn/gather_first_node.md"><code>tfgnn.gather_first_node()</code></a>.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.add_readout_from_first_node(
    graph: <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>,
    key: str,
    *,
    node_set_name: const.NodeSetName,
    readout_node_set: const.NodeSetName = &#x27;_readout&#x27;
) -> <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>
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
A scalar <code>GraphTensor</code>. If it contains the <code>readout_node_set</code>
already, its size in each graph component must be 1.
</td>
</tr><tr>
<td>
<code>key</code><a id="key"></a>
</td>
<td>
A key, for use with <a href="../tfgnn/structured_readout.md"><code>tfgnn.structured_readout()</code></a>. The input graph
must not already contain auxiliary edge sets for readout with this key.
</td>
</tr><tr>
<td>
<code>node_set_name</code><a id="node_set_name"></a>
</td>
<td>
The name of the node set from which values are to be read
out.
</td>
</tr><tr>
<td>
<code>readout_node_set</code><a id="readout_node_set"></a>
</td>
<td>
The name of the auxiliary node set for readout,
as in <a href="../tfgnn/structured_readout.md"><code>tfgnn.structured_readout()</code></a>.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A modified GraphTensor such that <a href="../tfgnn/structured_readout.md"><code>tfgnn.structured_readout(..., key)</code></a> works
like <a href="../tfgnn/gather_first_node.md"><code>tfgnn.gather_first_node(...)</code></a> on the input graph.
</td>
</tr>

</table>
