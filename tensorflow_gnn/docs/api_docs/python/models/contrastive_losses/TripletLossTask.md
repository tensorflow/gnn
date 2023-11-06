<!-- lint-g3mark -->

# contrastive_losses.TripletLossTask

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/tasks.py#L311-L404">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

The triplet loss task.

Inherits From:
[`ContrastiveLossTask`](../contrastive_losses/ContrastiveLossTask.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>contrastive_losses.TripletLossTask(
    *args, margin: float = 1.0, **kwargs
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
Name of the node set for readout.
</td>
</tr><tr>
<td>
`feature_name`<a id="feature_name"></a>
</td>
<td>
Feature name for readout.
</td>
</tr><tr>
<td>
`representations_layer_name`<a id="representations_layer_name"></a>
</td>
<td>
Layer name for uncorrupted representations.
</td>
</tr><tr>
<td>
`corruptor`<a id="corruptor"></a>
</td>
<td>
`Corruptor` instance for creating negative samples. If not
specified, we use `ShuffleFeaturesGlobally` by default.
</td>
</tr><tr>
<td>
`projector_units`<a id="projector_units"></a>
</td>
<td>
`Sequence` of layer sizes for projector network.
Projectors prevent dimensional collapse, but can hinder training for
easy corruptions. For more details, see
https://arxiv.org/abs/2304.12210.
</td>
</tr><tr>
<td>
`seed`<a id="seed"></a>
</td>
<td>
Random seed for the default corruptor (`ShuffleFeaturesGlobally`).
</td>
</tr>
</table>

## Methods

<h3 id="losses"><code>losses</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/tasks.py#L392-L401">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>losses() -> runner.Losses
</code></pre>

Returns arbitrary task specific losses.

<h3 id="make_contrastive_layer"><code>make_contrastive_layer</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/tasks.py#L319-L320">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>make_contrastive_layer() -> tf.keras.layers.Layer
</code></pre>

Returns the layer contrasting clean outputs with the correupted ones.

<h3 id="metrics"><code>metrics</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/tasks.py#L403-L404">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>metrics() -> runner.Metrics
</code></pre>

Returns arbitrary task specific metrics.

<h3 id="predict"><code>predict</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/tasks.py#L350-L390">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>predict(
    *args
) -> runner.Predictions
</code></pre>

Apply a readout head for use with triplet contrastive loss.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`*args`
</td>
<td>
A tuple of (anchor, positive_sample, negative_sample)
`tfgnn.GraphTensor`s.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The positive and negative distance embeddings for triplet loss as produced
by the implementing subclass.
</td>
</tr>

</table>

<h3 id="preprocess"><code>preprocess</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/tasks.py#L329-L348">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>preprocess(
    inputs: GraphTensor
) -> tuple[Sequence[GraphTensor], tfgnn.Field]
</code></pre>

Creates unused pseudo-labels.

The input tensor should have the anchor and positive sample stacked along the
first dimension for each feature within each node set. The corruptor is applied
on the positive sample.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`inputs`
</td>
<td>
The anchor and positive sample stack along the first axis.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Sequence of three graph tensors (anchor, positive_sample,
corrupted_sample) and unused pseudo-labels.
</td>
</tr>

</table>
