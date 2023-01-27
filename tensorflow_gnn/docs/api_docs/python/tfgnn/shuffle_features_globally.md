# tfgnn.shuffle_features_globally

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_ops.py#L974-L1008">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Shuffles context, node set and edge set features of a scalar GraphTensor.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.shuffle_features_globally(
    graph_tensor: <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>,
    *,
    seed: Optional[int] = None
) -> <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`graph_tensor`<a id="graph_tensor"></a>
</td>
<td>
A scalar GraphTensor.
</td>
</tr><tr>
<td>
`seed`<a id="seed"></a>
</td>
<td>
A seed for random uniform shuffle.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A scalar GraphTensor `result` with the same graph structure as the input,
but randomly shuffled feature tensors. More precisely, the result satisfies
`result.node_sets[ns][ft][i] = graph_tensor.node_sets[ns][ft][sigma(i)]`
for all node set names `ns`, all feature names `ft` and all indices `i`
in `range(n)`, where `n` is the total_size of the node set and `sigma`
is a permutation of `range(n)`. Moreover, the result satisfies the
the analogous equations for all features of all edge sets and the context.
The permutation `sigma` is drawn uniformly at random, independently for
each graph piece and each feature(!). That is, separate features are
permuted differently, and features on any one item (edge, node, component)
can form combinations not seen on an input item.
</td>
</tr>

</table>
