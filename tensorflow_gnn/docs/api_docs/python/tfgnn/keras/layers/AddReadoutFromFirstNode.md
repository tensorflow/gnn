<!-- lint-g3mark -->

# tfgnn.keras.layers.AddReadoutFromFirstNode

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/graph_ops.py#L518-L566">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Adds readout node set equivalent to
<a href="../../../tfgnn/keras/layers/ReadoutFirstNode.md"><code>tfgnn.keras.layers.ReadoutFirstNode</code></a>.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.AddReadoutFromFirstNode(
    key: str,
    *,
    node_set_name: NodeSetName,
    readout_node_set: NodeSetName = &#x27;_readout&#x27;,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Init args</h2></th></tr>

<tr>
<td>
`key`<a id="key"></a>
</td>
<td>
A key, for use with <a href="../../../tfgnn/keras/layers/StructuredReadout.md"><code>tfgnn.keras.layers.StructuredReadout</code></a>. The input
graph must not already contain auxiliary edge sets for readout with this
key.
</td>
</tr><tr>
<td>
`node_set_name`<a id="node_set_name"></a>
</td>
<td>
The name of the node set from which values are to be read
out.
</td>
</tr><tr>
<td>
`readout_node_set`<a id="readout_node_set"></a>
</td>
<td>
The name of the auxiliary node set for readout,
as in <a href="../../../tfgnn/structured_readout.md"><code>tfgnn.structured_readout()</code></a>.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Call args</h2></th></tr>

<tr>
<td>
`graph`<a id="graph"></a>
</td>
<td>
A scalar `GraphTensor`. If it contains the readout_node_set already,
its size in each graph component must be 1.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A modified `GraphTensor` so that <a href="../../../tfgnn/keras/layers/StructuredReadout.md"><code>tfgnn.keras.layers.StructuredReadout(key)</code></a>
acts like <a href="../../../tfgnn/keras/layers/ReadoutFirstNode.md"><code>tfgnn.keras.layers.ReadoutFirstNode(node_set_name=node_set_name)</code></a>
on the input graph.
</td>
</tr>

</table>
