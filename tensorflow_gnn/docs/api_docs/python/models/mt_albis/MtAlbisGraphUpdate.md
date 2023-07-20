# mt_albis.MtAlbisGraphUpdate

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/mt_albis/layers.py#L239-L389">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Returns GraphUpdate layer for message passing with Model Template "Albis".

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mt_albis.MtAlbisGraphUpdate(
    *,
    units: int,
    message_dim: int,
    receiver_tag: tfgnn.IncidentNodeTag,
    node_set_names: Optional[Collection[tfgnn.NodeSetName]] = None,
    edge_feature_name: Optional[tfgnn.FieldName] = None,
    attention_type: Literal['none', 'multi_head', 'gat_v2'] = &#x27;none&#x27;,
    attention_edge_set_names: Optional[Collection[tfgnn.EdgeSetName]] = None,
    attention_num_heads: int = 4,
    simple_conv_reduce_type: str = &#x27;mean&#x27;,
    simple_conv_use_receiver_state: bool = True,
    state_dropout_rate: float = 0.0,
    edge_dropout_rate: float = 0.0,
    l2_regularization: float = 0.0,
    kernel_initializer: Any = &#x27;glorot_uniform&#x27;,
    normalization_type: Literal['layer', 'batch', 'none'] = &#x27;layer&#x27;,
    batch_normalization_momentum: float = 0.99,
    next_state_type: Literal['dense', 'residual'] = &#x27;dense&#x27;,
    edge_set_combine_type: Literal['concat', 'sum'] = &#x27;concat&#x27;
) -> tf.keras.layers.Layer
</code></pre>

<!-- Placeholder for "Used in" -->

The TF-GNN Model Template "Albis" provides a small selection of field-tested GNN
architectures through the unified interface of this class.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`units`<a id="units"></a>
</td>
<td>
The dimension of node states in the output GraphTensor.
</td>
</tr><tr>
<td>
`message_dim`<a id="message_dim"></a>
</td>
<td>
The dimension of messages computed transiently on each edge.
</td>
</tr><tr>
<td>
`receiver_tag`<a id="receiver_tag"></a>
</td>
<td>
One of `tfgnn.SOURCE` or `tfgnn.TARGET`. The messages are
sent to the nodes at this endpoint of edges.
</td>
</tr><tr>
<td>
`node_set_names`<a id="node_set_names"></a>
</td>
<td>
Optionally, the names of NodeSets to update. By default,
all NodeSets are updated that receive from at least one EdgeSet.
</td>
</tr><tr>
<td>
`edge_feature_name`<a id="edge_feature_name"></a>
</td>
<td>
Optionally, the name of an edge feature to include in
message computation on edges.
</td>
</tr><tr>
<td>
`attention_type`<a id="attention_type"></a>
</td>
<td>
`"none"`, `"multi_head"`, or `"gat_v2"`. Selects whether
messages are pooled with data-dependent weights computed by a trained
attention mechansim.
</td>
</tr><tr>
<td>
`attention_edge_set_names`<a id="attention_edge_set_names"></a>
</td>
<td>
If set, edge sets other than those named here
will be treated as if `attention_type="none"` regardless.
</td>
</tr><tr>
<td>
`attention_num_heads`<a id="attention_num_heads"></a>
</td>
<td>
For attention_types `"multi_head"` or `"gat_v2"`,
the number of attention heads.
</td>
</tr><tr>
<td>
`simple_conv_reduce_type`<a id="simple_conv_reduce_type"></a>
</td>
<td>
For attention_type `"none"`, controls how messages
are aggregated on an EdgeSet for each receiver node. Defaults to `"mean"`;
other recommened values are the concatenations `"mean|sum"`, `"mean|max"`,
and `"mean|sum|max"` (but mind the increased output dimension and the
corresponding increase in the number of weights in the next-state layer).
Technically, can be set to any reduce_type understood by `tfgnn.pool()`.
</td>
</tr><tr>
<td>
`simple_conv_use_receiver_state`<a id="simple_conv_use_receiver_state"></a>
</td>
<td>
For attention_type `"none"`, controls
whether the receiver node state is used in computing each edge's message
(in addition to the sender node state and possibly an `edge feature`).
</td>
</tr><tr>
<td>
`state_dropout_rate`<a id="state_dropout_rate"></a>
</td>
<td>
The dropout rate applied to the pooled and combined
messages from all edges, to the optional input from context, and to the
new node state. This is conventional dropout, independently for each
dimension of the network's hidden state. (Unlike VanillaMPNN, dropout
is applied to messages after pooling.)
</td>
</tr><tr>
<td>
`edge_dropout_rate`<a id="edge_dropout_rate"></a>
</td>
<td>
Can be set to a dropout rate for entire edges during
message computation: with the given probability, the entire message of
an edge is dropped, as if the edge were not present in the graph.
</td>
</tr><tr>
<td>
`l2_regularization`<a id="l2_regularization"></a>
</td>
<td>
The coefficient of L2 regularization for trained weights.
(Unlike VanillaMPNN, this is not applied to biases.)
</td>
</tr><tr>
<td>
`kernel_initializer`<a id="kernel_initializer"></a>
</td>
<td>
Can be set to a `kernel_initializer` as understood
by `tf.keras.layers.Dense` etc.
An `Initializer` object gets cloned before use to ensure a fresh seed,
if not set explicitly. For more, see `tfgnn.keras.clone_initializer()`.
</td>
</tr><tr>
<td>
`normalization_type`<a id="normalization_type"></a>
</td>
<td>
controls the normalization of output node states.
By default (`"layer"`), LayerNormalization is used. Can be set to
`"none"`, or to `"batch"` for BatchNormalization.
</td>
</tr><tr>
<td>
`batch_normalization_momentum`<a id="batch_normalization_momentum"></a>
</td>
<td>
If `normalization_type="batch"`, sets the
`BatchNormalization(momentum=...)` parameter. Ignored otherwise.
</td>
</tr><tr>
<td>
`next_state_type`<a id="next_state_type"></a>
</td>
<td>
`"dense"` or `"residual"`. With the latter, a residual
link is added from the old to the new node state, which requires that all
input node states already have size `units` (unless their size is 0, as
for latent node sets, in which case the residual link is omitted).
</td>
</tr><tr>
<td>
`edge_set_combine_type`<a id="edge_set_combine_type"></a>
</td>
<td>
`"concat"` or `"sum"`. Controls how pooled messages
from various edge sets are combined as inputs to the NextState layer
that updates the node states. Defaults to `"concat"`, which gives the
pooled messages from each edge set separate weights in the NextState
layer, namely `units * message_dim * num_incident_edge_sets` per node set.
Setting this to `"sum"` adds up the pooled messages into a single
vector before passing them into the NextState layer, which requires just
`units * message_dim` weights per node set.
</td>
</tr>
</table>
