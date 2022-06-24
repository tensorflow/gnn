# tfgnn.create_graph_spec_from_schema_pb

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/schema_utils.py#L53-L104">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Converts a graph schema proto message to a scalar GraphTensorSpec.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.create_graph_spec_from_schema_pb(
    schema: <a href="../tfgnn/GraphSchema.md"><code>tfgnn.GraphSchema</code></a>,
    indices_dtype: tf.dtypes.DType = gc.default_indices_dtype
) -> <a href="../tfgnn/GraphTensorSpec.md"><code>tfgnn.GraphTensorSpec</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

A `GraphSchema` message contains shape information in a serializable format.
The `GraphTensorSpec` is a runtime object fulfilling the type spec
requirements, that accompanies each `GraphTensor` instance and fulfills much
of the same goal. This function converts the proto to the corresponding type
spec.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`schema`
</td>
<td>
An instance of the graph schema proto message.
</td>
</tr><tr>
<td>
`indices_dtype`
</td>
<td>
A `tf.dtypes.DType` for GraphTensor edge set source and
target indices, node and edge sets sizes.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `GraphTensorSpec` specification for the scalar graph tensor (of rank 0).
</td>
</tr>

</table>

