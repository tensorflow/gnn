# runner.Trainer

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/interfaces.py#L225-L263">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

A class for training and validation of a Keras model.

<!-- Placeholder for "Used in" -->
<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr> <td> <code>model_dir</code><a id="model_dir"></a> </td> <td>

</td> </tr><tr> <td> <code>strategy</code><a id="strategy"></a> </td> <td>

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
<code>model_fn</code>
</td>
<td>
Returns a <code>tf.keras.Model</code> for use in training and validation.
</td>
</tr><tr>
<td>
<code>train_ds_provider</code>
</td>
<td>
A <code>DatasetProvider</code> for training. The items of the
<code>tf.data.Dataset</code> are pairs <code>(graph_tensor, label)</code> that represent one
batch of per-replica training inputs after
<code>GraphTensor.merge_batch_to_components()</code> has been applied.
</td>
</tr><tr>
<td>
<code>epochs</code>
</td>
<td>
The epochs to train.
</td>
</tr><tr>
<td>
<code>valid_ds_provider</code>
</td>
<td>
A <code>DatasetProvider</code> for validation. The items of the
<code>tf.data.Dataset</code> are pairs <code>(graph_tensor, label)</code> that represent one
batch of per-replica training inputs after
<code>GraphTensor.merge_batch_to_components()</code> has been applied.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A trained <code>tf.keras.Model</code>.
</td>
</tr>

</table>
