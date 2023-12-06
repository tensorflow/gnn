# contrastive_losses.TripletLossTask

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/tasks.py#L318-L411">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

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
<code>node_set_name</code><a id="node_set_name"></a>
</td>
<td>
Name of the node set for readout.
</td>
</tr><tr>
<td>
<code>feature_name</code><a id="feature_name"></a>
</td>
<td>
Feature name for readout.
</td>
</tr><tr>
<td>
<code>representations_layer_name</code><a id="representations_layer_name"></a>
</td>
<td>
Layer name for uncorrupted representations.
</td>
</tr><tr>
<td>
<code>corruptor</code><a id="corruptor"></a>
</td>
<td>
<code>Corruptor</code> instance for creating negative samples. If not
specified, we use <code>ShuffleFeaturesGlobally</code> by default.
</td>
</tr><tr>
<td>
<code>projector_units</code><a id="projector_units"></a>
</td>
<td>
<code>Sequence</code> of layer sizes for projector network.
Projectors prevent dimensional collapse, but can hinder training for
easy corruptions. For more details, see
https://arxiv.org/abs/2304.12210.
</td>
</tr><tr>
<td>
<code>seed</code><a id="seed"></a>
</td>
<td>
Random seed for the default corruptor (<code>ShuffleFeaturesGlobally</code>).
</td>
</tr>
</table>

## Methods

<h3 id="losses"><code>losses</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/tasks.py#L399-L408">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>losses() -> runner.Losses
</code></pre>

Returns arbitrary task specific losses.

<h3 id="make_contrastive_layer"><code>make_contrastive_layer</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/tasks.py#L326-L327">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>make_contrastive_layer() -> tf.keras.layers.Layer
</code></pre>

Returns the layer contrasting clean outputs with the correupted ones.

<h3 id="metrics"><code>metrics</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/tasks.py#L410-L411">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>metrics() -> runner.Metrics
</code></pre>

Returns arbitrary task specific metrics.

<h3 id="predict"><code>predict</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/tasks.py#L357-L397">View
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
<code>*args</code>
</td>
<td>
A tuple of (anchor, positive_sample, negative_sample)
<code>tfgnn.GraphTensor</code>s.
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

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/tasks.py#L336-L355">View
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
<code>inputs</code>
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
