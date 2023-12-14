# tfgnn.shuffle_nodes

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_ops.py#L545-L612">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Randomly reorders nodes of given node sets, within each graph component.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.shuffle_nodes(
    graph_tensor: GraphTensor,
    *,
    node_sets: Optional[Collection[gt.NodeSetName]] = None,
    seed: Optional[int] = None
) -> GraphTensor
</code></pre>

<!-- Placeholder for "Used in" -->

The order of edges does not change; only their adjacency is modified to match
the new order of shuffled nodes. The order of graph components (as created by
`merge_graph_to_components()`) does not change, nodes are shuffled separately
within each component.

Auxiliary node sets are not shuffled, unless they are explicitly included in
`node_sets`. Not shuffling is the correct behavior for the auxiliary node sets
used by
<a href="../tfgnn/structured_readout.md"><code>tfgnn.structured_readout()</code></a>.

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
<code>node_sets</code><a id="node_sets"></a>
</td>
<td>
An optional collection of node sets names to shuffle. If None,
all node sets are shuffled.  Should not overlap with <code>shuffle_indices</code>.
</td>
</tr><tr>
<td>
<code>seed</code><a id="seed"></a>
</td>
<td>
Optionally, a fixed seed for random uniform shuffle.
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
If <code>node_sets</code> containes non existing node set names.
</td>
</tr>
</table>
