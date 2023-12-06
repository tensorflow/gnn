# tfgnn.shuffle_features_globally

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_ops.py#L401-L442">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Shuffles context, node set and edge set features of a scalar GraphTensor.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.shuffle_features_globally(
    graph_tensor: GraphTensor, *, seed: Optional[int] = None
) -> GraphTensor
</code></pre>

<!-- Placeholder for "Used in" -->

NOTE(b/277938756): This operation is not available in TFLite (last checked for
TF 2.12).

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>graph_tensor</code><a id="graph_tensor"></a>
</td>
<td>
A scalar GraphTensor.
</td>
</tr><tr>
<td>
<code>seed</code><a id="seed"></a>
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
A scalar GraphTensor <code>result</code> with the same graph structure as the input,
but randomly shuffled feature tensors. More precisely, the result satisfies
<code>result.node_sets[ns][ft][i] = graph_tensor.node_sets[ns][ft][sigma(i)]</code>
for all node set names <code>ns</code> (including auxiliary node sets), all feature
names <code>ft</code> and all indices <code>i</code> in <code>range(n)</code>, where <code>n</code> is the total_size
of the node set and <code>sigma</code> is a permutation of <code>range(n)</code>.
Moreover, the result satisfies the the analogous equations for all features
of all edge sets (including auxiliary edge sets) and the context.
The permutation <code>sigma</code> is drawn uniformly at random, independently for
each graph piece and each feature(!). That is, separate features are
permuted differently, and features on any one item (edge, node, component)
can form combinations not seen on an input item.
</td>
</tr>

</table>
