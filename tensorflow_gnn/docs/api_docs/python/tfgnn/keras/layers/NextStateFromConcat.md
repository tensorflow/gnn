# tfgnn.keras.layers.NextStateFromConcat

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/next_state.py#L110-L145">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Computes a new state by concatenating inputs and applying a Keras Layer.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.NextStateFromConcat(
    transformation: tf.keras.layers.Layer, **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

This layer flattens all inputs into a list (forgetting their origin),
concatenates them and sends them through a user-supplied feed-forward network.

This layer can be restored from config by `tf.keras.models.load_model()` when
saved as part of a Keras model using `save_format="tf"`.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Init args</h2></th></tr>

<tr>
<td>
<code>transformation</code><a id="transformation"></a>
</td>
<td>
Required. A Keras Layer to transform the combined inputs
into the new state.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Call returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The result of transformation.
</td>
</tr>

</table>
