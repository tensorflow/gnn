# tfgnn.keras.layers.ResidualNextState

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/next_state.py#L145-L228">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Updates a state with a residual block.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.ResidualNextState(
    residual_block: tf.keras.layers.Layer,
    *,
    activation: Any = None,
    skip_connection_feature_name: const.FieldName = const.HIDDEN_STATE,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This layer concatenates all inputs, sends them through a user-supplied
transformation, forms a skip connection by adding back the state of the
updated graph piece, and finally applies an activation function.
In other words, the user-supplied transformation is a residual block
that modifies the state.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Init args</h2></th></tr>

<tr>
<td>
`residual_block`<a id="residual_block"></a>
</td>
<td>
Required. A Keras Layer to transform the concatenation
of all inputs into a delta that gets added to the state. Notice that
the activation function is applied after the residual_block and the
addition, so typically the residual_block does *not* use an activation
function in its last layer.
</td>
</tr><tr>
<td>
`activation`<a id="activation"></a>
</td>
<td>
An activation function (none by default),
as understood by tf.keras.layers.Activation.
</td>
</tr><tr>
<td>
`skip_connection_feature_name`<a id="skip_connection_feature_name"></a>
</td>
<td>
Controls which input from the updated graph
piece is added back after the residual block. If the input from the
updated graph piece is a single tensor, that one is used. If it is
a dict, this key is used; defaults to <a href="../../../tfgnn.md#HIDDEN_STATE"><code>tfgnn.HIDDEN_STATE</code></a>.
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
