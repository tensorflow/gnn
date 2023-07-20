# multi_head_attention.MultiHeadAttentionHomGraphUpdate

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/multi_head_attention/layers.py#L622-L681">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Returns a GraphUpdate layer with a transformer-style multihead attention.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>multi_head_attention.MultiHeadAttentionHomGraphUpdate(
    *,
    num_heads: int,
    per_head_channels: int,
    receiver_tag: tfgnn.IncidentNodeTag,
    feature_name: str = tfgnn.HIDDEN_STATE,
    name: str = &#x27;multi_head_attention&#x27;,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

The returned layer performs one update step of a Transformer-style Multi-Head
Attention (but without the Feed Forward Network) on a GraphTensor that stores a
homogeneous graph.

For heterogeneous graphs with multiple node sets and edge sets, users are
advised to consider a GraphUpdate with one or more MultiHeadAttentionConv
objects instead, such as the MultiHeadAttentionMPNNGraphUpdate (see it for more
details).

> IMPORTANT: This implementation of MultiHeadAttention attends only to edges
> that are explicitly stored in the input GraphTensor. Attention of a node to
> itself requires having an explicit loop in the edge set.

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
Any optional arguments to MultiHeadAttentionConv, see there.
</td>
</tr>
</table>
