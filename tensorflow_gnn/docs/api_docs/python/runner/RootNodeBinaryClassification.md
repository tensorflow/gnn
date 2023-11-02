<!-- lint-g3mark -->

# runner.RootNodeBinaryClassification

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/tasks/classification.py#L316-L318">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Classification by root node label.

Inherits From: [`Task`](../runner/Task.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>runner.RootNodeBinaryClassification(
    node_set_name: str, *, state_name: str = tfgnn.HIDDEN_STATE, **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`node_set_name`<a id="node_set_name"></a>
</td>
<td>
The node set containing the root node.
</td>
</tr><tr>
<td>
`state_name`<a id="state_name"></a>
</td>
<td>
The feature name for activations
(typically: tfgnn.HIDDEN_STATE).
</td>
</tr><tr>
<td>
`**kwargs`<a id="**kwargs"></a>
</td>
<td>
Additional keyword arguments.
</td>
</tr>
</table>

## Methods

<h3 id="gather_activations"><code>gather_activations</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/tasks/classification.py#L299-L303">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gather_activations(
    inputs: GraphTensor
) -> Field
</code></pre>

Gather activations from root nodes.

<h3 id="losses"><code>losses</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/tasks/classification.py#L178-L179">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>losses() -> interfaces.Losses
</code></pre>

Returns arbitrary task specific losses.

<h3 id="metrics"><code>metrics</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/tasks/classification.py#L181-L187">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>metrics() -> interfaces.Metrics
</code></pre>

Returns arbitrary task specific metrics.

<h3 id="predict"><code>predict</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/tasks/classification.py#L138-L152">View
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
`inputs`
</td>
<td>
A `tfgnn.GraphTensor` for classification.
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

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/tasks/classification.py#L154-L161">View
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
`inputs`
</td>
<td>
A symbolic Keras `GraphTensor` for processing.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tuple of processed `GraphTensor`(s) and a (one or mapping of) `Field` to
be used as labels.
</td>
</tr>

</table>
