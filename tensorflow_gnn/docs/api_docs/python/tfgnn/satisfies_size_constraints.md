# tfgnn.satisfies_size_constraints

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/padding_ops.py#L188-L213">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Returns whether the input `graph_tensor` satisfies `total_sizes`.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tfgnn.satisfies_total_sizes`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.satisfies_size_constraints(
    graph_tensor: <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>,
    total_sizes: <a href="../tfgnn/SizeConstraints.md"><code>tfgnn.SizeConstraints</code></a>
) -> tf.Tensor
</code></pre>

<!-- Placeholder for "Used in" -->
<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>graph_tensor</code><a id="graph_tensor"></a>
</td>
<td>
a graph tensor to check against target total sizes.
</td>
</tr><tr>
<td>
<code>total_sizes</code><a id="total_sizes"></a>
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
A scalar boolean tensor equal to <code>True</code> if the <code>graph_tensor</code> statisifies
<code>total_sizes</code>, and <code>False</code> if not.
</td>
</tr>

</table>
