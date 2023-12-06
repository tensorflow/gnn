# runner.Task

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/interfaces.py#L124-L222">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Defines a learning objective for a GNN.

<!-- Placeholder for "Used in" -->

A `Task` represents a learning objective for a GNN model and defines all the
non-GNN pieces around the base GNN. Specifically:

1) `preprocess` is expected to return a `GraphTensor` (or `GraphTensor`s) and a
`Field` where (a) the base GNN's output for each `GraphTensor` is passed to
`predict` and (b) the `Field` is used as the training label (for supervised
tasks); 2) `predict` is expected to (a) take the base GNN's output for each
`GraphTensor` returned by `preprocess` and (b) return a tensor with the model's
prediction for this task; 3) `losses` is expected to return callables
(`tf.Tensor`, `tf.Tensor`) -> `tf.Tensor` that accept (`y_true`, `y_pred`) where
`y_true` is produced by some dataset and `y_pred` is the model's prediction from
(2); 4) `metrics` is expected to return callables (`tf.Tensor`, `tf.Tensor`) ->
`tf.Tensor` that accept (`y_true`, `y_pred`) where `y_true` is produced by some
dataset and `y_pred` is the model's prediction from (2).

`Task` can emit multiple outputs in `predict`: in that case we require that (a)
it is a mapping, (b) outputs of `losses` and `metrics` are also mappings with
matching keys, and (c) there is exactly one loss per key (there may be a
sequence of metrics per key). This is done to prevent accidental dropping of
losses (see b/291874188).

No constraints are made on the `predict` method; e.g.: it may append a head with
learnable weights or it may perform tensor computations only. (The entire `Task`
coordinates what that means with respect to dataset—via `preprocess`—,
modeling—via `predict`— and optimization—via `losses`.)

`Task`s are applied in the scope of a training invocation: they are subject to
the executing context of the `Trainer` and should, when needed, override it
(e.g., a global policy, like `tf.keras.mixed_precision.global_policy()` and its
implications over logit and activation layers).

## Methods

<h3 id="losses"><code>losses</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/interfaces.py#L214-L217">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>losses() -> Losses
</code></pre>

Returns arbitrary task specific losses.

<h3 id="metrics"><code>metrics</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/interfaces.py#L219-L222">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>metrics() -> Metrics
</code></pre>

Returns arbitrary task specific metrics.

<h3 id="predict"><code>predict</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/interfaces.py#L190-L212">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>predict(
    *args
) -> Predictions
</code></pre>

Produces prediction outputs for the learning objective.

Overall model composition* makes use of the Keras Functional API
(https://www.tensorflow.org/guide/keras/functional) to map symbolic Keras
`GraphTensor` inputs to symbolic Keras `Field` outputs. Outputs must match the
structure (one or mapping) of labels from `preprocess`.

*) `outputs = predict(GNN(inputs))` where `inputs` are those `GraphTensor`
returned by `preprocess(...)`, `GNN` is the base GNN, `predict` is this method
and `outputs` are the prediction outputs for the learning objective.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<code>*args</code>
</td>
<td>
The symbolic Keras <code>GraphTensor</code> inputs(s). These inputs correspond
(in sequence) to the base GNN output of each <code>GraphTensor</code> returned by
<code>preprocess(...)</code>.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The model's prediction output for this task.
</td>
</tr>

</table>

<h3 id="preprocess"><code>preprocess</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/interfaces.py#L161-L188">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>preprocess(
    inputs: GraphTensor
) -> tuple[OneOrSequenceOf[GraphTensor], OneOrMappingOf[Field]]
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
