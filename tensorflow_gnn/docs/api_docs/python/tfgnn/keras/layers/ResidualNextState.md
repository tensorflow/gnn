description: Updates a state with a residual block.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfgnn.keras.layers.ResidualNextState" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# tfgnn.keras.layers.ResidualNextState

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/next_state.py#L131-L214">
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

#### Init args:


* <b>`residual_block`</b>: Required. A Keras Layer to transform the concatenation
  of all inputs into a delta that gets added to the state. Notice that
  the activation function is applied after the residual_block and the
  addition, so typically the residual_block does *not* use an activation
  function in its last layer.
* <b>`activation`</b>: An activation function (none by default),
  as understood by tf.keras.layers.Activation.
* <b>`skip_connection_feature_name`</b>: Controls which input from the updated graph
  piece is added back after the residual block. If the input from the
  updated graph piece is a single tensor, that one is used. If it is
  a dict, this key is used; defaults to <a href="../../../tfgnn.md#HIDDEN_STATE"><code>tfgnn.HIDDEN_STATE</code></a>.


#### Call returns:

A tensor to use as the new state.


