<!-- lint-g3mark -->

# runner.RootNodeMeanSquaredError

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/tasks/regression.py#L343-L344">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Mean squared error task.

Inherits From: [`Task`](../runner/Task.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>runner.RootNodeMeanSquaredError(
    node_set_name: str,
    *,
    units: int = 1,
    state_name: str = tfgnn.HIDDEN_STATE,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`units`<a id="units"></a>
</td>
<td>
The units for the regression head.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
The regression head's layer name. This name typically appears in
the exported model's SignatureDef.
</td>
</tr><tr>
<td>
`label_fn`<a id="label_fn"></a>
</td>
<td>
A label extraction function. This function mutates the input
`GraphTensor`. Mutually exclusive with `label_feature_name`.
</td>
</tr><tr>
<td>
`label_feature_name`<a id="label_feature_name"></a>
</td>
<td>
A label feature name for readout from the auxiliary
'_readout' node set. Readout does not mutate the input `GraphTensor`.
Mutually exclusive with `label_fn`.
</td>
</tr>
</table>

## Methods

<h3 id="gather_activations"><code>gather_activations</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/tasks/regression.py#L147-L150">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gather_activations(
    inputs: GraphTensor
) -> tf.Tensor
</code></pre>

<h3 id="losses"><code>losses</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/tasks/regression.py#L170-L171">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>losses() -> interfaces.Losses
</code></pre>

<h3 id="metrics"><code>metrics</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/tasks/regression.py#L103-L108">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>metrics() -> interfaces.Metrics
</code></pre>

Regression metrics.

<h3 id="predict"><code>predict</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/tasks/regression.py#L74-L88">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>predict(
    inputs: tfgnn.GraphTensor
) -> interfaces.Predictions
</code></pre>

Apply a linear head for regression.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`inputs`
</td>
<td>
A `tfgnn.GraphTensor` for regression.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The regression logits.
</td>
</tr>

</table>

<h3 id="preprocess"><code>preprocess</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/tasks/regression.py#L90-L97">View
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
