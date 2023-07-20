# tfgnn.get_aux_type_prefix

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L1555-L1587">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Returns type prefix of aux node or edge set names, or `None` if non-aux.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.get_aux_type_prefix(
    set_name: const.SetName
) -> Optional[str]
</code></pre>

<!-- Placeholder for "Used in" -->

Auxiliary node sets and edge sets in a
<a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a> have names
that begin with an underscore `_` (preferred) or any of the characters `#`, `!`,
`%`, `.`, `^`, and `~` (reserved for future use). They store structural
information needed by helper functions like
<a href="../tfgnn/structured_readout.md"><code>tfgnn.structured_readout()</code></a>,
beyond the node sets and edge sets that represent graph data from the
application domain.

By convention, the names of auxiliary graph pieces begin with a type prefix that
contains the leading special character and all letters, digits and underscores
following it.

Users can define their own aux types and handle them in their own code. The
TF-GNN library uses the following types:

*   `"_readout"` and (rarely) `"_shadow"`, for `tfgnn.structured_readout()`.

See the named function(s) for how the respective types of auxiliary edge sets
and node sets are formed.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`set_name`<a id="set_name"></a>
</td>
<td>
The name of a node set or edge set in a <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>,
<a href="../tfgnn/GraphTensorSpec.md"><code>tfgnn.GraphTensorSpec</code></a> or <a href="../tfgnn/GraphSchema.md"><code>tfgnn.GraphSchema</code></a>.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
For an auxiliary node set or edge set, a non-empty prefix that identifies
its type; for other node sets or edge sets, `None`.
</td>
</tr>

</table>
