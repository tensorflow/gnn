# tfgnn.keras.layers.PadToTotalSizes

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/padding_ops.py#L23-L66">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

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

This layer can be restored from config by `tf.keras.models.load_model()` when
saved as part of a Keras model using `save_format="tf"`. Serialization to a
Keras model config requires the `sizes_constraints` to contain Python integers
or eager Tensors, not symbolic Tensors.
