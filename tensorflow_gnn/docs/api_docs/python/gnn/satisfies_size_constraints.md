description: Returns whether the input graph_tensor satisfies total_sizes.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gnn.satisfies_size_constraints" />
<meta itemprop="path" content="Stable" />
</div>

# gnn.satisfies_size_constraints

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/padding_ops.py#L169-L193">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Returns whether the input `graph_tensor` satisfies `total_sizes`.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`gnn.satisfies_total_sizes`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gnn.satisfies_size_constraints(
    graph_tensor: <a href="../gnn/GraphTensor.md"><code>gnn.GraphTensor</code></a>,
    total_sizes: <a href="../gnn/SizeConstraints.md"><code>gnn.SizeConstraints</code></a>
) -> tf.Tensor
</code></pre>

<!-- Placeholder for "Used in" -->
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
`total_sizes`
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
A scalar boolean tensor equal to True if the `graph_tensor` statisifies
`total_sizes`, and False if not.
</td>
</tr>

</table>
