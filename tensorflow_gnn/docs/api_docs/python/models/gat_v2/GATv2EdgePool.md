# gat_v2.GATv2EdgePool

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/gat_v2/layers.py#L372-L427">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Returns a layer for pooling edges with GATv2-style attention.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gat_v2.GATv2EdgePool(
    *,
    num_heads: int,
    per_head_channels: int,
    receiver_tag: Optional[tfgnn.IncidentNodeOrContextTag] = None,
    receiver_feature: tfgnn.FieldName = tfgnn.HIDDEN_STATE,
    sender_feature: tfgnn.FieldName = tfgnn.HIDDEN_STATE,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

When initialized with receiver_tag SOURCE or TARGET, the returned layer can be
called on an edge set to compute the weighted sum of edge states at the given
endpoint. The weights are computed by the method of Graph Attention Networks v2
(GATv2), except that edge states, not node states broadcast from the edges'
other endpoint, are used as input values to attention.

When initialized with receiver_tag CONTEXT, the returned layer can be called on
an edge set to do the analogous pooling of edge states to context.

NOTE: This layer cannot pool node states. For that, use
<a href="../gat_v2/GATv2Conv.md"><code>gat_v2.GATv2Conv</code></a>.

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
The results of attention are aggregated for this graph piece.
If set to <code>tfgnn.CONTEXT</code>, the layer can be called for an edge set or
node set.
If set to an IncidentNodeTag (e.g., <code>tfgnn.SOURCE</code> or <code>tfgnn.TARGET</code>),
the layer can be called for an edge set and will aggregate results at
the specified endpoint of the edges.
If left unset, the tag must be passed when calling the layer.
</td>
</tr><tr>
<td>
<code>receiver_feature</code><a id="receiver_feature"></a>
</td>
<td>
By default, the default state feature of the receiver
is used to compute the attention query. A different feature name can be
selected by setting this argument.
</td>
</tr><tr>
<td>
<code>sender_feature</code><a id="sender_feature"></a>
</td>
<td>
By default, the default state feature of the edge set is
used to compute the attention values. A different feature name can be
selected by setting this argument.
</td>
</tr><tr>
<td>
<code>**kwargs</code><a id="**kwargs"></a>
</td>
<td>
Any other option for GATv2Conv, except sender_node_feature,
which is set to None.
</td>
</tr>
</table>
