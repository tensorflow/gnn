# runner.DotProductLinkPrediction

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/tasks/link_prediction.py#L162-L171">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Implements edge score as dot product of features of endpoint nodes.

Inherits From: [`Task`](../runner/Task.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>runner.DotProductLinkPrediction(
    *,
    node_feature_name: tfgnn.FieldName = tfgnn.HIDDEN_STATE,
    readout_label_feature_name: str = &#x27;label&#x27;,
    readout_node_set_name: tfgnn.NodeSetName = &#x27;_readout&#x27;
)
</code></pre>

<!-- Placeholder for "Used in" -->
<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>node_feature_name</code><a id="node_feature_name"></a>
</td>
<td>
Name of feature where node state for link-prediction
is read from. The final link prediction score will be:
`score(graph.node_sets[source][node_feature_name],
        graph.node_sets[target][node_feature_name])`
where <code>source</code> and <code>target</code>, respectively, are:
<code>graph.edge_sets[readout_node_set_name+"/source"].adjacency.source_name</code>
and
<code>graph.edge_sets[readout_node_set_name+"/target"].adjacency.source_name</code>
</td>
</tr><tr>
<td>
<code>readout_label_feature_name</code><a id="readout_label_feature_name"></a>
</td>
<td>
The labels for edge connections,
source nodes
<code>graph.edge_sets[readout_node_set_name+"/source"].adjacency.source</code> in
node set <code>graph.node_sets[source]</code> against target nodes
<code>graph.edge_sets[readout_node_set_name+"/target"].adjacency.source</code> in
node set <code>graph.node_sets[source]</code>, must be stored in
<code>graph.node_sets[readout_node_set_name][readout_label_feature_name]</code>.
</td>
</tr><tr>
<td>
<code>readout_node_set_name</code><a id="readout_node_set_name"></a>
</td>
<td>
Determines the readout node-set, which must have
feature <code>readout_label_feature_name</code>, and must receive connections (at
target endpoints) from edge-sets <code>readout_node_set_name+"/source"</code> and
<code>readout_node_set_name+"/target"</code>.
</td>
</tr>
</table>

## Methods

<h3 id="losses"><code>losses</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/tasks/link_prediction.py#L154-L156">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>losses() -> <a href="../runner/Losses.md"><code>runner.Losses</code></a>
</code></pre>

Binary cross-entropy.

<h3 id="metrics"><code>metrics</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/tasks/link_prediction.py#L158-L159">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>metrics() -> <a href="../runner/Metrics.md"><code>runner.Metrics</code></a>
</code></pre>

Returns arbitrary task specific metrics.

<h3 id="predict"><code>predict</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/tasks/link_prediction.py#L144-L152">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>predict(
    graph: tfgnn.GraphTensor
) -> <a href="../runner/Predictions.md"><code>runner.Predictions</code></a>
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

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/tasks/link_prediction.py#L134-L142">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>preprocess(
    gt: tfgnn.GraphTensor
) -> Tuple[tfgnn.GraphTensor, tfgnn.Field]
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
