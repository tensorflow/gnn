description: Returns a GraphUpdater layer with a Graph Attention Network V2
(GATv2).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gat_v2.GATv2GraphUpdate" />
<meta itemprop="path" content="Stable" />
</div>

# gat_v2.GATv2GraphUpdate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/gat_v2/layers.py#L346-L398">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Returns a GraphUpdater layer with a Graph Attention Network V2 (GATv2).

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gat_v2.GATv2GraphUpdate(
    *,
    num_heads: int,
    per_head_channels: int,
    edge_set_name: str,
    feature_name: str = tfgnn.HIDDEN_STATE,
    name: str = &#x27;gat_v2&#x27;,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

The returned layer performs one update step of a Graph Attention Network v2
(GATv2) from https://arxiv.org/abs/2105.14491 on an edge set of a GraphTensor.
It is best suited for graphs that have just that one edge set. For heterogeneous
graphs with multiple node sets and edge sets, users are advised to consider a
GraphUpdate with one or more GATv2Conv objects instead.

This implementation of GAT attends only to edges that are explicitly stored in
the input GraphTensor. Attention of a node to itself requires having an explicit
loop in the edge set.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`num_heads`
</td>
<td>
The number of attention heads.
</td>
</tr><tr>
<td>
`per_head_channels`
</td>
<td>
The number of channels for each attention head. This
means that the final output size will be per_head_channels * num_heads.
</td>
</tr><tr>
<td>
`edge_set_name`
</td>
<td>
A GATv2 update happens on this edge set and its incident
node set(s) of the input GraphTensor.
</td>
</tr><tr>
<td>
`feature_name`
</td>
<td>
The feature name of node states; defaults to
`tfgnn.HIDDEN_STATE`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Optionally, a name for the layer returned.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Any optional arguments to GATv2Conv, see there.
</td>
</tr>
</table>
