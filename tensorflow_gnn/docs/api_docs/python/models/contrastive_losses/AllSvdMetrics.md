<!-- lint-g3mark -->

# contrastive_losses.AllSvdMetrics

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/metrics.py#L331-L342">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Computes multiple metrics for representations using one SVD call.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>contrastive_losses.AllSvdMetrics(
    *args, **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

Refer to <https://arxiv.org/abs/2305.16562> for more details.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`fns`<a id="fns"></a>
</td>
<td>
a mapping from a metric name to a `Callable` that accepts
representations as well as the result of their SVD decomposition.
Currently only singular values are passed.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
Name for the metric class, used for Keras bookkeeping.
</td>
</tr>
</table>

## Methods

<h3 id="merge_state"><code>merge_state</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>merge_state(
    metrics
)
</code></pre>

Merges the state from one or more metrics.

This method can be used by distributed systems to merge the state computed by
different metric instances. Typically the state will be stored in the form of
the metric's weights. For example, a tf.keras.metrics.Mean metric contains a
list of two weight values: a total and a count. If there were two instances of a
tf.keras.metrics.Accuracy that each independently aggregated partial state for
an overall accuracy calculation, these two metric's states could be combined as
follows:

    >>> m1 = tf.keras.metrics.Accuracy()
    >>> _ = m1.update_state([[1], [2]], [[0], [2]])

    >>> m2 = tf.keras.metrics.Accuracy()
    >>> _ = m2.update_state([[3], [4]], [[3], [4]])

    >>> m2.merge_state([m1])
    >>> m2.result().numpy()
    0.75

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`metrics`
</td>
<td>
an iterable of metrics. The metrics must have compatible
state.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If the provided iterable does not contain metrics matching
the metric's required specifications.
</td>
</tr>
</table>

<h3 id="reset_state"><code>reset_state</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/metrics.py#L313-L315">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset_state() -> None
</code></pre>

Resets all of the metric state variables.

This function is called between epochs/steps, when a metric is evaluated during
training.

<h3 id="result"><code>result</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/metrics.py#L327-L328">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>result() -> Mapping[str, tf.Tensor]
</code></pre>

Computes and returns the scalar metric value tensor or a dict of scalars.

Result computation is an idempotent operation that simply calculates the metric
value using the state variables.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A scalar tensor, or a dictionary of scalar tensors.
</td>
</tr>

</table>

<h3 id="update_state"><code>update_state</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/metrics.py#L317-L325">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_state(
    _, y_pred: tf.Tensor, sample_weight=None
) -> None
</code></pre>

Accumulates statistics for the metric.

Note: This function is executed as a graph function in graph mode. This means:
a) Operations on the same resource are executed in textual order. This should
make it easier to do things like add the updated value of a variable to another,
for example. b) You don't need to worry about collecting the update ops to
execute. All update ops added to the graph by this function will be executed. As
a result, code should generally work the same way with graph or eager execution.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`*args`
</td>
<td>

</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
A mini-batch of inputs to the Metric.
</td>
</tr>
</table>
