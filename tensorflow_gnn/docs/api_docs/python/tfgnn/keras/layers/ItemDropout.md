# tfgnn.keras.layers.ItemDropout

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/item_dropout.py#L22-L77">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Dropout of feature values for entire edges, nodes or components.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.ItemDropout(
    rate: float, seed: Optional[int] = None, **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

This Layer class wraps `tf.keras.layers.Dropout` to perform edge dropout or node
dropout (or "component dropout", which is rarely useful) on Tensors shaped like
features of a **scalar** GraphTensor.

This layer can be restored from config by `tf.keras.models.load_model()` when
saved as part of a Keras model using `save_format="tf"`.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Init args</h2></th></tr>

<tr>
<td>
<code>rate</code><a id="rate"></a>
</td>
<td>
The dropout rate, forwarded to <code>tf.keras.layers.Dropout</code>.
</td>
</tr><tr>
<td>
<code>seed</code><a id="seed"></a>
</td>
<td>
The random seed, forwarded <code>tf.keras.layers.Dropout</code>.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Call args</h2></th></tr>

<tr>
<td>
<code>x</code><a id="x"></a>
</td>
<td>
A float Tensor of shape <code>[num_items, *feature_dims]</code>. This is the shape
of node features or edge features (or context features) in a *scalar**
GraphTensor. Across calls, all inputs must have the same known rank.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Call returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A Tensor <code>y</code> with the same shape and dtype as the input <code>x</code>.
In non-training mode, the output is the same as the input: <code>y == x</code>.
In training mode, each row <code>y[i]</code> is either zeros (with probability <code>rate</code>)
or a scaled-up copy of the input row: <code>y[i] = x[i] * 1./(1-rate)</code>.
This is similar to ordinary dropout, except all or none of the feature
values for each item are dropped out.
</td>
</tr>

</table>
