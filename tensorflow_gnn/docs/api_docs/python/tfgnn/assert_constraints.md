# tfgnn.assert_constraints

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/schema_validation.py#L249-L267">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Validate the shape constaints of a graph's features at runtime.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.assert_constraints(
    graph: <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>
) -> tf.Operation
</code></pre>



<!-- Placeholder for "Used in" -->

This code returns a TensorFlow op with debugging assertions that ensure the
parsed data has valid shape constraints for a graph. This can be instantiated
in your TensorFlow graph while debugging if you believe that your data may be
incorrectly shaped, or simply applied to a manually produced dataset to ensure
that those constraints have been applied correctly.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`graph`
</td>
<td>
An instance of a `GraphTensor`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A list of check operations.
</td>
</tr>

</table>

