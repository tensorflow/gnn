# gat_v2.GATv2HomGraphUpdate

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/gat_v2/layers.py#L431-L487">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

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

The layer returned by this function can be restored from config by
`tf.keras.models.load_model()` when saved as part of a Keras model using
`save_format="tf"`.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>num_heads</code><a id="num_heads"></a>
</td>
<td>
The number of attention heads.
</td>
</tr><tr>
<td>
<code>per_head_channels</code><a id="per_head_channels"></a>
</td>
<td>
The number of channels for each attention head. This
means that the final output size will be per_head_channels * num_heads.
</td>
</tr><tr>
<td>
<code>receiver_tag</code><a id="receiver_tag"></a>
</td>
<td>
one of <code>tfgnn.SOURCE</code> or <code>tfgnn.TARGET</code>.
</td>
</tr><tr>
<td>
<code>feature_name</code><a id="feature_name"></a>
</td>
<td>
The feature name of node states; defaults to
<code>tfgnn.HIDDEN_STATE</code>.
</td>
</tr><tr>
<td>
<code>heads_merge_type</code><a id="heads_merge_type"></a>
</td>
<td>
"concat" or "mean". Gets passed to GATv2Conv, which uses
it to combine all heads into layer's output.
</td>
</tr><tr>
<td>
<code>name</code><a id="name"></a>
</td>
<td>
Optionally, a name for the layer returned.
</td>
</tr><tr>
<td>
<code>**kwargs</code><a id="**kwargs"></a>
</td>
<td>
Any optional arguments to GATv2Conv, see there.
</td>
</tr>
</table>
