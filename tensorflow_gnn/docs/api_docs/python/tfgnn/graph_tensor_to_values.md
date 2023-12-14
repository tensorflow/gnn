# tfgnn.graph_tensor_to_values

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_pprint.py#L50-L72">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Convert an eager `GraphTensor` to a mapping of mappings of PODTs.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.graph_tensor_to_values(
    graph: <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>
) -> Dict[str, Any]
</code></pre>



<!-- Placeholder for "Used in" -->

This is used for pretty-printing. Convert your graph tensor with this and run
the result through `pprint.pprint()` or `pprint.pformat()` for display of its
contents.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>graph</code><a id="graph"></a>
</td>
<td>
An eager <code>GraphTensor</code> instance to be pprinted.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A dict of plain-old data types that can be run through <code>pprint.pprint()</code> or
a JSON conversion library.
</td>
</tr>

</table>

