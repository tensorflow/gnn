# tfgnn.reorder_nodes

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_ops.py#L449-L551">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

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
`graph_tensor`<a id="graph_tensor"></a>
</td>
<td>
A scalar GraphTensor.
</td>
</tr><tr>
<td>
`node_indices`<a id="node_indices"></a>
</td>
<td>
A mapping from node sets name to new nodes indices (positions
within the node set). Each index is an arbitrary permutation of
`tf.range(num_nodes)`, where `index[i]` is an index of an original node
to be placed at position `i`.
</td>
</tr><tr>
<td>
`validate`<a id="validate"></a>
</td>
<td>
If True, checks that `node_indices` are valid permutations.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A scalar GraphTensor with randomly shuffled nodes within `node_sets`.
</td>
</tr>

</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
If `node_sets` contains non existing node set names.
</td>
</tr><tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
If indices are not `rank=1` `tf.int32` or `tf.int64` tensors.
</td>
</tr><tr>
<td>
`InvalidArgumentError`<a id="InvalidArgumentError"></a>
</td>
<td>
if an index shape is not `[num_nodes]`.
</td>
</tr><tr>
<td>
`InvalidArgumentError`<a id="InvalidArgumentError"></a>
</td>
<td>
if an index is not a permutation of
`tf.range(num_nodes)`. Only if validate is set to True.
</td>
</tr>
</table>
