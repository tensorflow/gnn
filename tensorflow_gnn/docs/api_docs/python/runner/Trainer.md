<!-- lint-g3mark -->

# runner.Trainer

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/interfaces.py#L225-L263">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

A class for training and validation of a Keras model.

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`model_dir`<a id="model_dir"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`strategy`<a id="strategy"></a>
</td>
<td>

</td>
</tr>
</table>

## Methods

<h3 id="train"><code>train</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/interfaces.py#L238-L263">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>train(
    model_fn: Callable[[], tf.keras.Model],
    train_ds_provider: DatasetProvider,
    *,
    epochs: int = 1,
    valid_ds_provider: Optional[DatasetProvider] = None
) -> tf.keras.Model
</code></pre>

Trains a `tf.keras.Model` with optional validation.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`model_fn`
</td>
<td>
Returns a `tf.keras.Model` for use in training and validation.
</td>
</tr><tr>
<td>
`train_ds_provider`
</td>
<td>
A `DatasetProvider` for training. The items of the
`tf.data.Dataset` are pairs `(graph_tensor, label)` that represent one
batch of per-replica training inputs after
`GraphTensor.merge_batch_to_components()` has been applied.
</td>
</tr><tr>
<td>
`epochs`
</td>
<td>
The epochs to train.
</td>
</tr><tr>
<td>
`valid_ds_provider`
</td>
<td>
A `DatasetProvider` for validation. The items of the
`tf.data.Dataset` are pairs `(graph_tensor, label)` that represent one
batch of per-replica training inputs after
`GraphTensor.merge_batch_to_components()` has been applied.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A trained `tf.keras.Model`.
</td>
</tr>

</table>
