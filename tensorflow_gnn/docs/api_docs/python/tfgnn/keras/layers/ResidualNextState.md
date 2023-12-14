# tfgnn.keras.layers.ResidualNextState

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/next_state.py#L148-L257">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Updates a state with a residual block.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.ResidualNextState(
    residual_block: tf.keras.layers.Layer,
    *,
    activation: Any = None,
    skip_connection_feature_name: Optional[const.FieldName] = None,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

This layer concatenates all inputs, sends them through a user-supplied
transformation, forms a skip connection by adding back the state of the updated
graph piece, and finally applies an activation function. In other words, the
user-supplied transformation is a residual block that modifies the state. The
output shape of the residual block must match the shape of the state that gets
updated so that they can be added.

If the initial state of the graph piece that is being updated has size 0, the
skip connection is omitted. This avoids the need to special-case, say, latent
node sets in modeling code applied to different node sets.

This layer can be restored from config by `tf.keras.models.load_model()` when
saved as part of a Keras model using `save_format="tf"`.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Init args</h2></th></tr>

<tr>
<td>
<code>residual_block</code><a id="residual_block"></a>
</td>
<td>
Required. A Keras Layer to transform the concatenation
of all inputs into a delta that gets added to the state.
</td>
</tr><tr>
<td>
<code>activation</code><a id="activation"></a>
</td>
<td>
An activation function (none by default), as understood by
<code>tf.keras.layers.Activation</code>. This activation function is applied after
the residual block and the addition. If using this, typically the
residual block does not have an activation function on its last layer,
or vice versa.
</td>
</tr><tr>
<td>
<code>skip_connection_feature_name</code><a id="skip_connection_feature_name"></a>
</td>
<td>
Controls which input from the updated graph
piece is added back after the residual block. If the input from the
updated graph piece is a single tensor, that tensor is used, and this arg
must not be set. If the input is a dict, this key is used; if unset, it
defaults to <a href="../../../tfgnn.md#HIDDEN_STATE"><code>tfgnn.HIDDEN_STATE</code></a>.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Call returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tensor to use as the new state.
</td>
</tr>

</table>
