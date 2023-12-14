# runner.NodeBinaryClassification

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/tasks/classification.py#L547-L594">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Node binary (or multi-label) classification via structured readout.

Inherits From: [`Task`](../runner/Task.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>runner.NodeBinaryClassification(
    key: str = &#x27;seed&#x27;,
    units: int = 1,
    *,
    feature_name: str = tfgnn.HIDDEN_STATE,
    readout_node_set: tfgnn.NodeSetName = &#x27;_readout&#x27;,
    validate: bool = True,
    name: str = &#x27;classification_logits&#x27;,
    label_fn: Optional[LabelFn] = None,
    label_feature_name: Optional[str] = None
)
</code></pre>

<!-- Placeholder for "Used in" -->
<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>key</code><a id="key"></a>
</td>
<td>
A string key to select between possibly multiple named readouts.
</td>
</tr><tr>
<td>
<code>units</code><a id="units"></a>
</td>
<td>
The units for the classification head. (Typically <code>1</code> for binary
classification and the number of labels for multi-label classification.)
</td>
</tr><tr>
<td>
<code>feature_name</code><a id="feature_name"></a>
</td>
<td>
The name of the feature to read. If unset,
<code>tfgnn.HIDDEN_STATE</code> will be read.
</td>
</tr><tr>
<td>
<code>readout_node_set</code><a id="readout_node_set"></a>
</td>
<td>
A string, defaults to <code>"_readout"</code>. This is used as the
name for the readout node set and as a name prefix for its edge sets.
</td>
</tr><tr>
<td>
<code>validate</code><a id="validate"></a>
</td>
<td>
Setting this to false disables the validity checks for the
auxiliary edge sets. This is stronlgy discouraged, unless great care is
taken to run <code>tfgnn.validate_graph_tensor_for_readout()</code> earlier on
structurally unchanged GraphTensors.
</td>
</tr><tr>
<td>
<code>name</code><a id="name"></a>
</td>
<td>
The classification head's layer name. To control the naming of saved
model outputs see the runner model exporters (e.g.,
<code>KerasModelExporter</code>).
</td>
</tr><tr>
<td>
<code>label_fn</code><a id="label_fn"></a>
</td>
<td>
A label extraction function. This function mutates the input
<code>GraphTensor</code>. Mutually exclusive with <code>label_feature_name</code>.
</td>
</tr><tr>
<td>
<code>label_feature_name</code><a id="label_feature_name"></a>
</td>
<td>
A label feature name for readout from the auxiliary
'_readout' node set. Readout does not mutate the input <code>GraphTensor</code>.
Mutually exclusive with <code>label_fn</code>.
</td>
</tr>
</table>

## Methods

<h3 id="gather_activations"><code>gather_activations</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/tasks/classification.py#L346-L363">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gather_activations(
    inputs: GraphTensor
) -> Field
</code></pre>

Gather activations from auxiliary node (and edge) sets.

<h3 id="losses"><code>losses</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/tasks/classification.py#L179-L180">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>losses() -> interfaces.Losses
</code></pre>

Returns arbitrary task specific losses.

<h3 id="metrics"><code>metrics</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/tasks/classification.py#L182-L188">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>metrics() -> interfaces.Metrics
</code></pre>

Returns arbitrary task specific metrics.

<h3 id="predict"><code>predict</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/tasks/classification.py#L139-L153">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>predict(
    inputs: tfgnn.GraphTensor
) -> interfaces.Predictions
</code></pre>

Apply a linear head for classification.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<code>inputs</code>
</td>
<td>
A <code>tfgnn.GraphTensor</code> for classification.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The classification logits.
</td>
</tr>

</table>

<h3 id="preprocess"><code>preprocess</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/tasks/classification.py#L155-L162">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>preprocess(
    inputs: GraphTensor
) -> tuple[GraphTensor, Field]
</code></pre>

Preprocesses a scalar (after `merge_batch_to_components`) `GraphTensor`.

This function uses the Keras functional API to define non-trainable
transformations of the symbolic input `GraphTensor`, which get executed during
dataset preprocessing in a `tf.data.Dataset.map(...)` operation. It has two
responsibilities:

1.  Splitting the training label out of the input for training. It must be
    returned as a separate tensor or mapping of tensors.
2.  Optionally, transforming input features. Some advanced modeling techniques
    require running the same base GNN on multiple different transformations, so
    this function may return a single `GraphTensor` or a non-empty sequence of
    `GraphTensors`. The corresponding base GNN output for each `GraphTensor` is
    provided to the `predict(...)` method.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<code>inputs</code>
</td>
<td>
A symbolic Keras <code>GraphTensor</code> for processing.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tuple of processed <code>GraphTensor</code>(s) and a (one or mapping of) <code>Field</code> to
be used as labels.
</td>
</tr>

</table>
