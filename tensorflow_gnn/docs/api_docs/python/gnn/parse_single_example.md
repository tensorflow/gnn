description: Parses a single serialized Example proto into a single GraphTensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gnn.parse_single_example" />
<meta itemprop="path" content="Stable" />
</div>

# gnn.parse_single_example

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_io.py#L86-L112">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Parses a single serialized Example proto into a single `GraphTensor`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gnn.parse_single_example(
    spec: <a href="../gnn/GraphTensorSpec.md"><code>gnn.GraphTensorSpec</code></a>,
    serialized: tf.Tensor,
    prefix: Optional[str] = None,
    validate: bool = True
) -> <a href="../gnn/GraphTensor.md"><code>gnn.GraphTensor</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

Like `parse_example()`, but for a single graph tensor. See
`tensorflow_gnn.parse_example()` for reference.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`spec`
</td>
<td>
A graph tensor type specification.
</td>
</tr><tr>
<td>
`serialized`
</td>
<td>
A scalar string tensor with a serialized Example proto
containing a graph tensor object with the `spec` type spec.
</td>
</tr><tr>
<td>
`prefix`
</td>
<td>
An optional prefix string over all the features. You may use
this if you are encoding other data in the same protocol buffer.
</td>
</tr><tr>
<td>
`validate`
</td>
<td>
A boolean indicating whether or not to validate that the input
fields form a valid `GraphTensor`. Defaults to True.
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

