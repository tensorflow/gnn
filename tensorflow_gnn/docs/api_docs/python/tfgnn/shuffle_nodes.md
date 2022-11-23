# tfgnn.shuffle_nodes

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_ops.py#L1116-L1170">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Randomly reorders nodes of given node sets, within each graph component.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.shuffle_nodes(
    graph_tensor: <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>,
    *,
    node_sets: Optional[Collection[gt.NodeSetName]] = None,
    seed: Optional[int] = None
) -> <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>
</code></pre>

<!-- Placeholder for "Used in" -->

The order of edges does not change; only their adjacency is modified to match
the new order of shuffled nodes. The order of graph components (as created by
`merge_graph_to_components()`) does not change, nodes are shuffled separatelty
within each component.

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
`node_sets`<a id="node_sets"></a>
</td>
<td>
An optional collection of node sets names to shuffle. If None,
all node sets are shuffled.  Should not overlap with `shuffle_indices`.
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
If `node_sets` containes non existing node set names.
</td>
</tr>
</table>
