# tfgnn.reorder_nodes

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_ops.py#L445-L542">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Reorders nodes within node sets according to indices.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.reorder_nodes(
    graph_tensor: GraphTensor,
    node_indices: Mapping[gt.NodeSetName, tf.Tensor],
    *,
    validate: bool = True
) -> GraphTensor
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
A scalar GraphTensor.
</td>
</tr><tr>
<td>
<code>node_indices</code><a id="node_indices"></a>
</td>
<td>
A mapping from node sets name to new nodes indices (positions
within the node set). Each index is an arbitrary permutation of
<code>tf.range(num_nodes)</code>, where <code>index[i]</code> is an index of an original node
to be placed at position <code>i</code>.
</td>
</tr><tr>
<td>
<code>validate</code><a id="validate"></a>
</td>
<td>
If True, checks that <code>node_indices</code> are valid permutations.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A scalar GraphTensor with randomly shuffled nodes within <code>node_sets</code>.
</td>
</tr>

</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
<code>ValueError</code><a id="ValueError"></a>
</td>
<td>
If <code>node_sets</code> contains non existing node set names.
</td>
</tr><tr>
<td>
<code>ValueError</code><a id="ValueError"></a>
</td>
<td>
If indices are not <code>rank=1</code> <code>tf.int32</code> or <code>tf.int64</code> tensors.
</td>
</tr><tr>
<td>
<code>InvalidArgumentError</code><a id="InvalidArgumentError"></a>
</td>
<td>
if an index shape is not <code>[num_nodes]</code>.
</td>
</tr><tr>
<td>
<code>InvalidArgumentError</code><a id="InvalidArgumentError"></a>
</td>
<td>
if an index is not a permutation of
<code>tf.range(num_nodes)</code>. Only if validate is set to True.
</td>
</tr>
</table>
