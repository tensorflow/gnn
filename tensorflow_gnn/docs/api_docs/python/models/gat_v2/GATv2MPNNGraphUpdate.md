# gat_v2.GATv2MPNNGraphUpdate

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/gat_v2/layers.py#L487-L581">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Returns a GraphUpdate layer for message passing with GATv2 pooling.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gat_v2.GATv2MPNNGraphUpdate(
    *,
    units: int,
    message_dim: int,
    num_heads: int,
    heads_merge_type: str = &#x27;concat&#x27;,
    receiver_tag: tfgnn.IncidentNodeOrContextTag,
    node_set_names: Optional[Collection[tfgnn.NodeSetName]] = None,
    edge_feature: Optional[tfgnn.FieldName] = None,
    l2_regularization: float = 0.0,
    edge_dropout_rate: float = 0.0,
    state_dropout_rate: float = 0.0,
    attention_activation: Union[str, Callable[..., Any]] = &#x27;leaky_relu&#x27;,
    conv_activation: Union[str, Callable[..., Any]] = &#x27;relu&#x27;,
    activation: Union[str, Callable[..., Any]] = &#x27;relu&#x27;,
    kernel_initializer: Union[None, str, tf.keras.initializers.Initializer] = &#x27;glorot_uniform&#x27;
) -> tf.keras.layers.Layer
</code></pre>

<!-- Placeholder for "Used in" -->

The returned layer performs one round of message passing between the nodes of a
heterogeneous GraphTensor, using
<a href="../gat_v2/GATv2Conv.md"><code>gat_v2.GATv2Conv</code></a> to compute
the messages and their pooling with attention, followed by a dense layer to
compute the new node states from a concatenation of the old node state and all
pooled messages.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`units`<a id="units"></a>
</td>
<td>
The dimension of output hidden states for each node.
</td>
</tr><tr>
<td>
`message_dim`<a id="message_dim"></a>
</td>
<td>
The dimension of messages (attention values) computed on
each edge.  Must be divisible by `num_heads`.
</td>
</tr><tr>
<td>
`num_heads`<a id="num_heads"></a>
</td>
<td>
The number of attention heads used by GATv2. `message_dim`
must be divisible by this number.
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
`receiver_tag`<a id="receiver_tag"></a>
</td>
<td>
one of `tfgnn.TARGET` or `tfgnn.SOURCE`, to select the
incident node of each edge that receives the message.
</td>
</tr><tr>
<td>
`node_set_names`<a id="node_set_names"></a>
</td>
<td>
The names of node sets to update. If unset, updates all
that are on the receiving end of any edge set.
</td>
</tr><tr>
<td>
`edge_feature`<a id="edge_feature"></a>
</td>
<td>
Can be set to a feature name of the edge set to select
it as an input feature. By default, this set to `None`, which disables
this input.
</td>
</tr><tr>
<td>
`l2_regularization`<a id="l2_regularization"></a>
</td>
<td>
The coefficient of L2 regularization for weights and
biases.
</td>
</tr><tr>
<td>
`edge_dropout_rate`<a id="edge_dropout_rate"></a>
</td>
<td>
The edge dropout rate applied during attention pooling
of edges.
</td>
</tr><tr>
<td>
`state_dropout_rate`<a id="state_dropout_rate"></a>
</td>
<td>
The dropout rate applied to the resulting node states.
</td>
</tr><tr>
<td>
`attention_activation`<a id="attention_activation"></a>
</td>
<td>
The nonlinearity used on the transformed inputs
before multiplying with the trained weights of the attention layer.
This can be specified as a Keras layer, a tf.keras.activations.*
function, or a string understood by tf.keras.layers.Activation().
Defaults to "leaky_relu", which in turn defaults to a negative slope
of `alpha=0.2`.
</td>
</tr><tr>
<td>
`conv_activation`<a id="conv_activation"></a>
</td>
<td>
The nonlinearity applied to the result of attention on one
edge set, specified in the same ways as attention_activation.
</td>
</tr><tr>
<td>
`activation`<a id="activation"></a>
</td>
<td>
The nonlinearity applied to the new node states computed by
this graph update.
</td>
</tr><tr>
<td>
`kernel_initializer`<a id="kernel_initializer"></a>
</td>
<td>
Can be set to a `kerner_initializer` as understood
by `tf.keras.layers.Dense` etc.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A GraphUpdate layer for use on a scalar GraphTensor with
`tfgnn.HIDDEN_STATE` features on the node sets.
</td>
</tr>

</table>
