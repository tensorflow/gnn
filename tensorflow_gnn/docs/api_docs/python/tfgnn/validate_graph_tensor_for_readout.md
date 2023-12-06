# tfgnn.validate_graph_tensor_for_readout

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/readout.py#L68-L138">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Checks `graph` supports `structured_readout()` from `required_keys`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.validate_graph_tensor_for_readout(
    graph: <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>,
    required_keys: Optional[Sequence[str]] = None,
    *,
    readout_node_set: const.NodeSetName = &#x27;_readout&#x27;
) -> <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>
</code></pre>

<!-- Placeholder for "Used in" -->

This function checks that a
<a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a> contains
correctly connected auxiliary graph pieces (edge sets and node sets) for
structured readout. It does all the checks of
<a href="../tfgnn/validate_graph_tensor_spec_for_readout.md"><code>tfgnn.validate_graph_tensor_spec_for_readout()</code></a>.
Additionally, it checks that the actual tensor values (esp. node indices) are
valid for
<a href="../tfgnn/structured_readout.md"><code>tfgnn.structured_readout(graph,
key)</code></a> for each `key` in `required_keys`. If `required_keys` is unset,
all keys provided in the graph structure are checked.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>graph</code><a id="graph"></a>
</td>
<td>
The graph tensor to check.
</td>
</tr><tr>
<td>
<code>required_keys</code><a id="required_keys"></a>
</td>
<td>
Can be set to a list of readout keys to check. If unset,
checks all keys provided by the graph.
</td>
</tr><tr>
<td>
<code>readout_node_set</code><a id="readout_node_set"></a>
</td>
<td>
Optionally, a non-default name for use as
<a href="../tfgnn/structured_readout.md"><code>tfgnn.structured_readout(..., readout_node_set=...)</code></a>.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The input GraphTensor, unchanged. This helps to put <code>tf.debugging.assert*</code>
ops from this function into a dependency chain.
</td>
</tr>

</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
<code>ValueError</code><a id="ValueError"></a>
</td>
<td>
if the auxiliary graph pieces for readout are malformed in the
<a href="../tfgnn/GraphTensorSpec.md"><code>tfgnn.GraphTensorSpec</code></a>.
</td>
</tr><tr>
<td>
<code>KeyError</code><a id="KeyError"></a>
</td>
<td>
if any of the <code>required_keys</code> is missing.
</td>
</tr><tr>
<td>
<code>tf.errors.InvalidArgumentError</code><a id="tf.errors.InvalidArgumentError"></a>
</td>
<td>
If values in the GraphTensor, notably
node indices of auxiliary edge sets, are incorrect.
</td>
</tr>
</table>
