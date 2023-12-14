# tfgnn.keras.layers.MakeEmptyFeature

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/map_features.py#L362-L421">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Returns an empty feature with a shape that fits the input graph piece.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.MakeEmptyFeature(
    trainable=True, name=None, dtype=None, dynamic=False, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This layer can be restored from config by `tf.keras.models.load_model()` when
saved as part of a Keras model using `save_format="tf"`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Init args</h2></th></tr>

<tr>
<td>
<code>dtype</code><a id="dtype"></a>
</td>
<td>
the tf.DType to use for the result, defaults to tf.float32.
</td>
</tr><tr>
<td>
<code>**kwargs</code><a id="**kwargs"></a>
</td>
<td>
Other arguments for the tf.keras.layers.Layer base class.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Call args</h2></th></tr>

<tr>
<td>
<code>graph_piece</code><a id="graph_piece"></a>
</td>
<td>
a Context, NodeSet or EdgeSet from a GraphTensor.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Call returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A potentially ragged tensor of shape [*graph_shape, (num_items), 0] where
graph_shape is the shape of the graph_piece and its containing GraphTensor,
(num_items) is the number of edges, nodes or components contained in the
graph piece, and 0 is the feature dimension that makes this an empty tensor.
In particular, if graph_shape == [], meaning graph_piece is from a scalar
GraphTensor, the result is a Tensor of shape [graph_piece.total_size, 0].
</td>
</tr>

</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">TPU compatibility</h2></th></tr>
<tr class="alt">
<td colspan="2">
If graph_shape == [], the shape of the result is static (as required)
if graph_piece.spec.total_size is not None. That, however, requires the
presence of other features on the same graph piece from which its static
total_size can be inferred. Therefore, to create an empty hidden state for
a latent graph piece (one without input features), this layer must be used
already in dataset preprocessing, before padding inputs to fixed sizes.
</td>
</tr>

</table>
