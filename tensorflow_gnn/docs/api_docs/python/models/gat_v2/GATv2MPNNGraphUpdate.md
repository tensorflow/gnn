# gat_v2.GATv2MPNNGraphUpdate

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/gat_v2/layers.py#L504-L604">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Returns a GraphUpdate layer for message passing with GATv2 pooling.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gat_v2.GATv2MPNNGraphUpdate(
    *,
    units: int,
    message_dim: int,
    num_heads: int,
    heads_merge_type: str = &#x27;concat&#x27;,
    receiver_tag: tfgnn.IncidentNodeTag,
    node_set_names: Optional[Collection[tfgnn.NodeSetName]] = None,
    edge_feature: Optional[tfgnn.FieldName] = None,
    l2_regularization: float = 0.0,
    edge_dropout_rate: float = 0.0,
    state_dropout_rate: float = 0.0,
    attention_activation: Union[str, Callable[..., Any]] = &#x27;leaky_relu&#x27;,
    conv_activation: Union[str, Callable[..., Any]] = &#x27;relu&#x27;,
    activation: Union[str, Callable[..., Any]] = &#x27;relu&#x27;,
    kernel_initializer: Any = &#x27;glorot_uniform&#x27;
) -> tf.keras.layers.Layer
</code></pre>

<!-- Placeholder for "Used in" -->

The returned layer performs one round of message passing between the nodes of a
heterogeneous GraphTensor, using
<a href="../gat_v2/GATv2Conv.md"><code>gat_v2.GATv2Conv</code></a> to compute
the messages and their pooling with attention, followed by a dense layer to
compute the new node states from a concatenation of the old node state and all
pooled messages.

The layer returned by this function can be restored from config by
`tf.keras.models.load_model()` when saved as part of a Keras model using
`save_format="tf"`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>units</code><a id="units"></a>
</td>
<td>
The dimension of output hidden states for each node.
</td>
</tr><tr>
<td>
<code>message_dim</code><a id="message_dim"></a>
</td>
<td>
The dimension of messages (attention values) computed on
each edge.  Must be divisible by <code>num_heads</code>.
</td>
</tr><tr>
<td>
<code>num_heads</code><a id="num_heads"></a>
</td>
<td>
The number of attention heads used by GATv2. <code>message_dim</code>
must be divisible by this number.
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
<code>receiver_tag</code><a id="receiver_tag"></a>
</td>
<td>
one of <code>tfgnn.TARGET</code> or <code>tfgnn.SOURCE</code>, to select the
incident node of each edge that receives the message.
</td>
</tr><tr>
<td>
<code>node_set_names</code><a id="node_set_names"></a>
</td>
<td>
The names of node sets to update. If unset, updates all
that are on the receiving end of any edge set.
</td>
</tr><tr>
<td>
<code>edge_feature</code><a id="edge_feature"></a>
</td>
<td>
Can be set to a feature name of the edge set to select
it as an input feature. By default, this set to <code>None</code>, which disables
this input.
</td>
</tr><tr>
<td>
<code>l2_regularization</code><a id="l2_regularization"></a>
</td>
<td>
The coefficient of L2 regularization for weights and
biases.
</td>
</tr><tr>
<td>
<code>edge_dropout_rate</code><a id="edge_dropout_rate"></a>
</td>
<td>
The edge dropout rate applied during attention pooling
of edges.
</td>
</tr><tr>
<td>
<code>state_dropout_rate</code><a id="state_dropout_rate"></a>
</td>
<td>
The dropout rate applied to the resulting node states.
</td>
</tr><tr>
<td>
<code>attention_activation</code><a id="attention_activation"></a>
</td>
<td>
The nonlinearity used on the transformed inputs
before multiplying with the trained weights of the attention layer.
This can be specified as a Keras layer, a tf.keras.activations.*
function, or a string understood by <code>tf.keras.layers.Activation()</code>.
Defaults to "leaky_relu", which in turn defaults to a negative slope
of <code>alpha=0.2</code>.
</td>
</tr><tr>
<td>
<code>conv_activation</code><a id="conv_activation"></a>
</td>
<td>
The nonlinearity applied to the result of attention on one
edge set, specified in the same ways as attention_activation.
</td>
</tr><tr>
<td>
<code>activation</code><a id="activation"></a>
</td>
<td>
The nonlinearity applied to the new node states computed by
this graph update.
</td>
</tr><tr>
<td>
<code>kernel_initializer</code><a id="kernel_initializer"></a>
</td>
<td>
Can be set to a <code>kernel_initializer</code> as understood
by <code>tf.keras.layers.Dense</code> etc.
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
<code>tfgnn.HIDDEN_STATE</code> features on the node sets.
</td>
</tr>

</table>
