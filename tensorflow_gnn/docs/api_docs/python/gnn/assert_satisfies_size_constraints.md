description: Raises InvalidArgumentError if graph_tensor exceeds
size_constraints.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gnn.assert_satisfies_size_constraints" />
<meta itemprop="path" content="Stable" />
</div>

# gnn.assert_satisfies_size_constraints

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/padding_ops.py#L196-L235">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Raises InvalidArgumentError if graph_tensor exceeds size_constraints.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`gnn.assert_satisfies_total_sizes`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gnn.assert_satisfies_size_constraints(
    graph_tensor: <a href="../gnn/GraphTensor.md"><code>gnn.GraphTensor</code></a>,
    size_constraints: <a href="../gnn/SizeConstraints.md"><code>gnn.SizeConstraints</code></a>
)
</code></pre>

<!-- Placeholder for "Used in" -->

This function can be used as follows:

```
  with tf.control_dependencies([
    assert_satisfies_size_constraints(graph_tensor, size_constraints)]):
    # Use graph_tensor after sizes have been checked.
```

Conceptually, that means this function is like standard tensorflow assertions,
like tf.debugging.Assert(satisfies_size_constraints(...)), but with the
following important advantages: - This functions logs a detailed message which
size constraint is violated. - This function works around a TensorFlow issue to
make sure the assertion is executed before the ops it guards, even in the
presence of conflicting attempts to eliminate constant subexpressions.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`graph_tensor`
</td>
<td>
a graph tensor to check against target total sizes.
</td>
</tr><tr>
<td>
`size_constraints`
</td>
<td>
target total sizes for each graph piece.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Validation operations to execute within a tf.control_dependencies.
</td>
</tr>

</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`tf.errors.InvalidArgumentError`
</td>
<td>
if input graph tensor could not be padded to
the `size_constraints`.
</td>
</tr>
</table>
