# tfgnn.keras.clone_initializer

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/initializers.py#L20-L67">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Clones an initializer to ensure a new default seed.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.clone_initializer(
    initializer
)
</code></pre>

<!-- Placeholder for "Used in" -->

Users can specify initializers for trainable weights by `Initializer` objects or
various other types understood by `tf.keras.initializers.get()`, namely `str`
with the name, `dict` with the config, or `None`.

As of TensorFlow 2.10, `Initializer` objects are stateless and fix their random
seed (even if not explicitly specified) at creation time, so that all calls to
them return the same sequence of numbers. To achieve independent initializations
of the various model weights, user-specified initializers must be cloned for
each weight before passing them to Keras. This way, each of them gets a separate
seed (unless explicitly overriden).

This helper function clones `Initializer` obejcts and passes through all other
forms of specifying an initializer. TF-GNN's modeling code applies it before
passing user-specified initaializers to Keras. User code that calls Keras
directly and passes an initializer more than once is advised to wrap it with
this function as well.

#### Example:

```
def build_graph_update(units, initializer):
  def dense(units):  # Called for multiple node sets and edge sets.
    tf.keras.layers.Dense(
        units, activation="relu",
        kernel_initializer=tfgnn.keras.clone_initializer(initializer))

  gnn_builder = tfgnn.keras.ConvGNNBuilder(
      lambda edge_set_name, receiver_tag: tfgnn.keras.layers.SimpleConv(
          dense(units), receiver_tag=receiver_tag),
      lambda node_set_name: tfgnn.keras.layers.NextStateFromConcat(
          dense(units)),
      receiver_tag=tfgnn.TARGET)
return gnn_builder.Convolve()
```

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`initializer`<a id="initializer"></a>
</td>
<td>
An initializer specification as understood by Keras.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A new `Initializer` object with the same config as `initializer`,
or `initializer` unchanged if if was not an `Initializer` object.
</td>
</tr>

</table>
