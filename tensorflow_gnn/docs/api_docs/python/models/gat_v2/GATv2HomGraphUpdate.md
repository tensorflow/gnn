# gat_v2.GATv2HomGraphUpdate

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/gat_v2/layers.py#L416-L470">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Returns a GraphUpdate layer with a Graph Attention Network V2 (GATv2).

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gat_v2.GATv2HomGraphUpdate(
    *,
    num_heads: int,
    per_head_channels: int,
    receiver_tag: tfgnn.IncidentNodeTag,
    feature_name: str = tfgnn.HIDDEN_STATE,
    heads_merge_type: str = &#x27;concat&#x27;,
    name: str = &#x27;gat_v2&#x27;,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

The returned layer performs one update step of a Graph Attention Network v2
(GATv2) from https://arxiv.org/abs/2105.14491 on a GraphTensor that stores a
homogeneous graph. For heterogeneous graphs with multiple node sets and edge
sets, users are advised to consider a GraphUpdate with one or more GATv2Conv
objects instead, such as the GATv2MPNNGraphUpdate.

> IMPORTANT: This implementation of GAT attends only to edges that are
> explicitly stored in the input GraphTensor. Attention of a node to itself
> requires having an explicit loop in the edge set.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`num_heads`<a id="num_heads"></a>
</td>
<td>
The number of attention heads.
</td>
</tr><tr>
<td>
`per_head_channels`<a id="per_head_channels"></a>
</td>
<td>
The number of channels for each attention head. This
means that the final output size will be per_head_channels * num_heads.
</td>
</tr><tr>
<td>
`receiver_tag`<a id="receiver_tag"></a>
</td>
<td>
one of `tfgnn.SOURCE` or `tfgnn.TARGET`.
</td>
</tr><tr>
<td>
`feature_name`<a id="feature_name"></a>
</td>
<td>
The feature name of node states; defaults to
`tfgnn.HIDDEN_STATE`.
</td>
</tr><tr>
<td>
`heads_merge_type`<a id="heads_merge_type"></a>
</td>
<td>
"concat" or "mean". Gets passed to GATv2Conv, which uses
it to combine all heads into layer's output.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
Optionally, a name for the layer returned.
</td>
</tr><tr>
<td>
`**kwargs`<a id="**kwargs"></a>
</td>
<td>
Any optional arguments to GATv2Conv, see there.
</td>
</tr>
</table>
