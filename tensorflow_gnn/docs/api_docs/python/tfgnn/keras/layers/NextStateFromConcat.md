description: Computes a new state by concatenating inputs and applying a Keras Layer.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfgnn.keras.layers.NextStateFromConcat" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# tfgnn.keras.layers.NextStateFromConcat

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/next_state.py#L96-L128">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Computes a new state by concatenating inputs and applying a Keras Layer.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.NextStateFromConcat(
    transformation: tf.keras.layers.Layer, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This layer flattens all inputs into a list (forgetting their origin),
concatenates them and sends them through a user-supplied feed-forward network.

#### Init args:


* <b>`transformation`</b>: Required. A Keras Layer to transform the combined inputs
  into the new state.


#### Call returns:

The result of transformation.


