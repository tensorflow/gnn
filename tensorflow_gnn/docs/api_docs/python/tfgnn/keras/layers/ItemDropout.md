<!-- lint-g3mark -->

# tfgnn.keras.layers.ItemDropout

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/item_dropout.py#L22-L74">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

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

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Init args</h2></th></tr>

<tr>
<td>
`rate`<a id="rate"></a>
</td>
<td>
The dropout rate, forwarded to `tf.keras.layers.Dropout`.
</td>
</tr><tr>
<td>
`seed`<a id="seed"></a>
</td>
<td>
The random seed, forwarded `tf.keras.layers.Dropout`.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Call args</h2></th></tr>

<tr>
<td>
`x`<a id="x"></a>
</td>
<td>
A float Tensor of shape `[num_items, *feature_dims]`. This is the shape
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
A Tensor `y` with the same shape and dtype as the input `x`.
In non-training mode, the output is the same as the input: `y == x`.
In training mode, each row `y[i]` is either zeros (with probability `rate`)
or a scaled-up copy of the input row: `y[i] = x[i] * 1./(1-rate)`.
This is similar to ordinary dropout, except all or none of the feature
values for each item are dropped out.
</td>
</tr>

</table>
