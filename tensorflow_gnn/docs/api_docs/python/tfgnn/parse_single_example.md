# tfgnn.parse_single_example

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_io.py#L103-L129">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Parses a single serialized Example proto into a single `GraphTensor`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.parse_single_example(
    spec: <a href="../tfgnn/GraphTensorSpec.md"><code>tfgnn.GraphTensorSpec</code></a>,
    serialized: tf.Tensor,
    prefix: Optional[str] = None,
    validate: bool = True
) -> <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

Like `parse_example()`, but for a single graph tensor.
See <a href="../tfgnn/parse_example.md"><code>tfgnn.parse_example()</code></a> for reference.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`spec`<a id="spec"></a>
</td>
<td>
A graph tensor type specification.
</td>
</tr><tr>
<td>
`serialized`<a id="serialized"></a>
</td>
<td>
A scalar string tensor with a serialized Example proto
containing a graph tensor object with the `spec` type spec.
</td>
</tr><tr>
<td>
`prefix`<a id="prefix"></a>
</td>
<td>
An optional prefix string over all the features. You may use
this if you are encoding other data in the same protocol buffer.
</td>
</tr><tr>
<td>
`validate`<a id="validate"></a>
</td>
<td>
A boolean indicating whether or not to validate that the input
fields form a valid `GraphTensor`. Defaults to `True`.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A graph tensor object with a matching type spec.
</td>
</tr>

</table>

