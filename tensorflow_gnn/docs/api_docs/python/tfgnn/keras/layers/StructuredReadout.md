# tfgnn.keras.layers.StructuredReadout

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/graph_ops.py#L318-L412">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Reads out a feature value from select nodes (or edges) in a graph.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.StructuredReadout(
    key: Optional[str] = None,
    *,
    feature_name: str = const.HIDDEN_STATE,
    readout_node_set: NodeSetName = &#x27;_readout&#x27;,
    validate: bool = True,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

This Keras layer wraps the
<a href="../../../tfgnn/structured_readout.md"><code>tfgnn.structured_readout()</code></a>
function. It addresses the need to read out final hidden states from a GNN
computation to make predictions for some nodes (or edges) of interest. Its
typical usage looks as follows:

```python
input_graph = tf.keras.Input(type_spec=graph_tensor_spec)
graph = SomeGraphUpdate(...)(input_graph)  # Run your GNN here.
seed_node_states = tfgnn.keras.layers.StructuredReadout("seed")(graph)
logits = tf.keras.layers.Dense(num_classes)(seed_node_states)
model = tf.keras.Model(inputs, logits)
```

...where `"seed"` is a key defined by the readout structure. There can be
multiple of those. For example, a link prediction model could read out
`"source"` and `"target"` node states from the graph.

Please see the documentation of
<a href="../../../tfgnn/structured_readout.md"><code>tfgnn.structured_readout()</code></a>
for the auxiliary node set and edge sets that make up the readout structure
which encodes how values are read out from the graph. Whenever these are
available, it is strongly recommended to make use of them with this layer and
avoid the older
<a href="../../../tfgnn/keras/layers/ReadoutFirstNode.md"><code>tfgnn.keras.layers.ReadoutFirstNode</code></a>.

Note that this layer returns a tensor shaped like a feature of the `"_readout"`
node set but not actually stored on it. To store it there, see
<a href="../../../tfgnn/keras/layers/ReadoutNamedIntoFeature.md"><code>tfgnn.keras.layers.StructuredReadoutIntoFeature</code></a>.
To retrieve a feature unchanged, see
<a href="../../../tfgnn/keras/layers/Readout.md"><code>tfgnn.keras.layers.Readout</code></a>.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Init args</h2></th></tr>

<tr>
<td>
`key`<a id="key"></a>
</td>
<td>
A string key to select between possibly multiple named readouts
(such as `"source"` and `"target"` for link prediction). Can be fixed
in init, or selected for each call.
</td>
</tr><tr>
<td>
`feature_name`<a id="feature_name"></a>
</td>
<td>
The name of the feature to read. If unset (also in call),
<a href="../../../tfgnn.md#HIDDEN_STATE"><code>tfgnn.HIDDEN_STATE</code></a> will be read.
</td>
</tr><tr>
<td>
`readout_node_set`<a id="readout_node_set"></a>
</td>
<td>
A string, defaults to `"_readout"`. This is used as the
name for the readout node set and as a name prefix for its edge sets.
</td>
</tr><tr>
<td>
`validate`<a id="validate"></a>
</td>
<td>
Setting this to false disables the validity checks for the
auxiliary edge sets. This is stronlgy discouraged, unless great care is
taken to run <a href="../../../tfgnn/validate_graph_tensor_for_readout.md"><code>tfgnn.validate_graph_tensor_for_readout()</code></a> earlier on
structurally unchanged GraphTensors.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Call args</h2></th></tr>

<tr>
<td>
`graph`<a id="graph"></a>
</td>
<td>
The scalar GraphTensor to read from.
</td>
</tr><tr>
<td>
`key`<a id="key"></a>
</td>
<td>
Same meaning as for init. Must be passed to init, or to call,
or to both (with the same value).
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tensor of read-out feature values, shaped like a feature of the
`readout_node_set`.
</td>
</tr>

</table>
