# tfgnn.keras.layers.PadToTotalSizes

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/padding_ops.py#L23-L64">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Applies tfgnn.pad_to_total_sizes() to a GraphTensor.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.PadToTotalSizes(
    sizes_constraints: <a href="../../../tfgnn/SizeConstraints.md"><code>tfgnn.SizeConstraints</code></a>,
    *,
    validate: bool = True,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This Keras layer maps a GraphTensor to a GraphTensor by calling
<a href="../../../tfgnn/pad_to_total_sizes.md"><code>tfgnn.pad_to_total_sizes()</code></a> with the additional arguments, notably
`sizes_constraints`, passed at initialization time. See that function
for detailed documentation.

Serialization to a Keras model config requires the `sizes_constraints` to
contain Python integers or eager Tensors, not symbolic Tensors.

