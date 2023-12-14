# vanilla_mpnn.VanillaMPNNGraphUpdate

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/vanilla_mpnn/layers.py#L22-L119">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Returns a GraphUpdate layer for a Vanilla MPNN.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>vanilla_mpnn.VanillaMPNNGraphUpdate(
    *,
    units: int,
    message_dim: int,
    receiver_tag: tfgnn.IncidentNodeTag,
    node_set_names: Optional[Collection[tfgnn.NodeSetName]] = None,
    edge_feature: Optional[tfgnn.FieldName] = None,
    reduce_type: str = &#x27;sum&#x27;,
    l2_regularization: float = 0.0,
    dropout_rate: float = 0.0,
    kernel_initializer: Any = &#x27;glorot_uniform&#x27;,
    use_layer_normalization: bool = False
) -> tf.keras.layers.Layer
</code></pre>

<!-- Placeholder for "Used in" -->

The returned layer performs one round of node state updates with a Message
Passing Neural Network that uses a single dense layer to compute messages and
update node states.

For each edge set E, the pooled messages for node v are computed as follows from
its neighbors N_E(v), that is, the other endpoints of those edges that have v at
the endpoint identified by `receiver_tag`.

$$m_E = \text{reduce}(
    \text{ReLU}(W_{\text{msg}} (h_v || h_u || x_{(u,v)}))
    \text{ for all } u \in N_E(v)).$$

The inputs are, in this order: the `tfgnn.HIDDEN_STATE` features of the receiver
and sender node as well as the named `edge_feature`, if any. The reduction
happens with the specified `reduce_type`, e.g., sum or mean.

The new hidden state at node v is computed as follows from the old node state
and the pooled messages from all incident node sets E_1, E_2, ...:

$$h_v := \text{ReLU}(
    W_{\text{state}} (h_v || m_{E_1} || m_{E_2} || \ldots)).$$

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
The dimension of messages computed on each edge.
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
<code>reduce_type</code><a id="reduce_type"></a>
</td>
<td>
How to pool the messages from edges to receiver nodes; defaults
to <code>"sum"</code>. Can be any reduce_type understood by <code>tfgnn.pool()</code>, including
concatenations like <code>"sum|max"</code> (but mind the increased dimension of the
result and the growing number of model weights in the next-state layer).
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
<code>dropout_rate</code><a id="dropout_rate"></a>
</td>
<td>
The dropout rate applied to messages on each edge and to the
new node state.
</td>
</tr><tr>
<td>
<code>kernel_initializer</code><a id="kernel_initializer"></a>
</td>
<td>
Can be set to a <code>kernel_initializer</code> as understood
by <code>tf.keras.layers.Dense</code> etc.
An <code>Initializer</code> object gets cloned before use to ensure a fresh seed,
if not set explicitly. For more, see <code>tfgnn.keras.clone_initializer()</code>.
</td>
</tr><tr>
<td>
<code>use_layer_normalization</code><a id="use_layer_normalization"></a>
</td>
<td>
Flag to determine whether to apply layer
normalization to the new node state.
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
