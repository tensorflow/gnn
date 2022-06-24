# tfgnn.keras.layers.MakeEmptyFeature

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/map_features.py#L326-L376">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns an empty feature with a shape that fits the input graph piece.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.MakeEmptyFeature(
    trainable=True, name=None, dtype=None, dynamic=False, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Init args:


* <b>`dtype`</b>: the tf.DType to use for the result, defaults to tf.float32.
* <b>`**kwargs`</b>: Other arguments for the tf.keras.layers.Layer base class.


#### Call args:


* <b>`graph_piece`</b>: a Context, NodeSet or EdgeSet from a GraphTensor.


#### Call returns:

A potentially ragged tensor of shape [*graph_shape, (num_items), 0] where
graph_shape is the shape of the graph_piece and its containing GraphTensor,
(num_items) is the number of edges, nodes or components contained in the
graph piece, and 0 is the feature dimension that makes this an empty tensor.
In particular, if graph_shape == [], meaning graph_piece is from a scalar
GraphTensor, the result is a Tensor of shape [graph_piece.total_size, 0].
That shape is constant (for use on TPU) if graph_piece.spec.total_size is
not None.


