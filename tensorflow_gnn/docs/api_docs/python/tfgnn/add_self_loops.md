# tfgnn.add_self_loops

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_ops.py#L448-L563">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Adds self-loops for edge with name `edge_set_name` EVEN if already exist.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.add_self_loops(
    graph: <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>,
    edge_set_name: gt.EdgeSetName,
    *,
    edge_feature_initializer: _EdgeFeatureInitializer = _zero_edge_feat_init
) -> <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>
</code></pre>

<!-- Placeholder for "Used in" -->

Edge `edge_set_name` must connect pair of nodes of the same node set.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`graph`<a id="graph"></a>
</td>
<td>
GraphTensor without self-loops. NOTE: If it has self-loops, then
another round if self-loops will be added.
</td>
</tr><tr>
<td>
`edge_set_name`<a id="edge_set_name"></a>
</td>
<td>
Must connect node pairs of the same node set.
</td>
</tr><tr>
<td>
`edge_feature_initializer`<a id="edge_feature_initializer"></a>
</td>
<td>
initializes edge features for the self-loop edges.
It defaults to initializing features of new edges to tf.zeros.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
GraphTensor with self-loops added.
</td>
</tr>

</table>
