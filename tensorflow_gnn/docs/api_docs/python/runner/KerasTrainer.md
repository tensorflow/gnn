# runner.KerasTrainer

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/trainers/keras_fit.py#L57-L316">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Trains using the `tf.keras.Model.fit` training loop.

Inherits From: [`Trainer`](../runner/Trainer.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>runner.KerasTrainer(
    strategy: tf.distribute.Strategy,
    *,
    model_dir: str,
    checkpoint_options: Optional[<a href="../runner/KerasTrainerCheckpointOptions.md"><code>runner.KerasTrainerCheckpointOptions</code></a>] = None,
    backup_dir: Optional[str] = None,
    steps_per_epoch: Optional[int] = None,
    verbose: Union[int, str] = &#x27;auto&#x27;,
    validation_steps: Optional[int] = None,
    validation_per_epoch: Optional[int] = None,
    validation_freq: Optional[int] = None,
    summarize_every_n_steps: Union[int, str] = 500,
    checkpoint_every_n_steps: Union[int, str] = &#x27;epoch&#x27;,
    backup_and_restore: bool = True,
    callbacks: Optional[Sequence[tf.keras.callbacks.Callback]] = None,
    restore_best_weights: Optional[bool] = None,
    options: Optional[<a href="../runner/KerasTrainerOptions.md"><code>runner.KerasTrainerOptions</code></a>] = None
)
</code></pre>

<!-- Placeholder for "Used in" -->
<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>strategy</code><a id="strategy"></a>
</td>
<td>
A <code>tf.distribute.Strategy.</code>
</td>
</tr><tr>
<td>
<code>model_dir</code><a id="model_dir"></a>
</td>
<td>
A model directory for summaries.
</td>
</tr><tr>
<td>
<code>checkpoint_options</code><a id="checkpoint_options"></a>
</td>
<td>
An optional configuration for checkpointing related
configs. If checkpoint_options.checkpoint_dir is unset;
<code>os.path.join(model_dir, "ckpnt")</code> is used.
</td>
</tr><tr>
<td>
<code>backup_dir</code><a id="backup_dir"></a>
</td>
<td>
An optional directory for backup, if unset;
<code>(os.path.join(model_dir, "backup"),)</code> is used.
</td>
</tr><tr>
<td>
<code>steps_per_epoch</code><a id="steps_per_epoch"></a>
</td>
<td>
The number of training steps per epoch. Optional,
if unspecified: epochs are at <code>tf.data.Dataset</code> end.
</td>
</tr><tr>
<td>
<code>verbose</code><a id="verbose"></a>
</td>
<td>
Forwarded to <code>tf.keras.Model.fit()</code>. Possible values are
0 (silent), 1 (print progress bar), 2 (one line per epoch), and
"auto" (default) defers to keras to select verbosity.
</td>
</tr><tr>
<td>
<code>validation_steps</code><a id="validation_steps"></a>
</td>
<td>
The number of steps used during validation. Optional,
if unspecified: the entire validation <code>tf.data.Dataset</code> is evaluated.
</td>
</tr><tr>
<td>
<code>validation_per_epoch</code><a id="validation_per_epoch"></a>
</td>
<td>
The number of validations done per training epoch.
Optional, if unspecified: Perform one validation per training epoch.
Only one of <code>validation_per_epoch</code> and <code>validation_freq</code> can be
specified.
</td>
</tr><tr>
<td>
<code>validation_freq</code><a id="validation_freq"></a>
</td>
<td>
Specifies how many training epochs to run before a new
validation run is performed. Optional, if unspecified: Performs
validation after every training epoch. Only one of
<code>validation_per_epoch</code> and <code>validation_freq</code> can be specified.
</td>
</tr><tr>
<td>
<code>summarize_every_n_steps</code><a id="summarize_every_n_steps"></a>
</td>
<td>
The frequency for writing TensorBoard summaries,
as an integer number of steps, or "epoch" for once per epoch, or
"never".
</td>
</tr><tr>
<td>
<code>checkpoint_every_n_steps</code><a id="checkpoint_every_n_steps"></a>
</td>
<td>
The frequency for writing latest models, as an
integer number of steps, or "epoch" for once per epoch, or "never".
The best model will always be saved after each validation epoch except
when this parameter is set to "never", because the validation metric is
available only after validation epoch.
</td>
</tr><tr>
<td>
<code>backup_and_restore</code><a id="backup_and_restore"></a>
</td>
<td>
Whether to backup and restore (According to
<code>tf.keras.callbacks.BackupAndRestore</code>). The backup
directory is determined by <code>backup_dir</code>.
</td>
</tr><tr>
<td>
<code>callbacks</code><a id="callbacks"></a>
</td>
<td>
Optional additional <code>tf.keras.callbacks.Callback</code> for
<code>tf.keras.Model.fit.</code>
</td>
</tr><tr>
<td>
<code>restore_best_weights</code><a id="restore_best_weights"></a>
</td>
<td>
Requires a <code>checkpoint_every_n_steps</code> other than
"never." Whether to restore the best model weights as determined by
<code>tf.keras.callbacks.ModelCheckpoint</code> after training. If unspecified,
its value is determined at <code>train(...)</code> invocation: <code>True if
valid_ds_provider is not None else False</code>.
</td>
</tr><tr>
<td>
<code>options</code><a id="options"></a>
</td>
<td>
A <code>KerasTrainerOptions.</code>
</td>
</tr>
</table>

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

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/trainers/keras_fit.py#L165-L316">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>train(
    model_fn: Callable[[], tf.keras.Model],
    train_ds_provider: <a href="../runner/DatasetProvider.md"><code>runner.DatasetProvider</code></a>,
    *,
    epochs: int = 1,
    valid_ds_provider: Optional[<a href="../runner/DatasetProvider.md"><code>runner.DatasetProvider</code></a>] = None
) -> tf.keras.Model
</code></pre>

Runs `tf.keras.Model.fit` with the`tf.distribute.Strategy` provided.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<code>model_fn</code>
</td>
<td>
A <code>ModelFn</code>, to be invoked in the <code>tf.distribute.Strategty</code>
scope.
</td>
</tr><tr>
<td>
<code>train_ds_provider</code>
</td>
<td>
A function that returns a <code>tf.data.Dataset</code> for
training.The items of the <code>tf.data.Dataset</code> are pairs
<code>(graph_tensor, label)</code> that represent one batch of per-replica training
inputs after <code>GraphTensor.merge_batch_to_components()</code> has been applied.
</td>
</tr><tr>
<td>
<code>epochs</code>
</td>
<td>
The epochs to train: adjusted for <code>validation_per_epoch.</code>
</td>
</tr><tr>
<td>
<code>valid_ds_provider</code>
</td>
<td>
An optional function that returns a <code>tf.data.Dataset</code>
for validation. The items of the <code>tf.data.Dataset</code> are pairs
<code>(graph_tensor, label)</code> that represent one batch of per-replica training
inputs after <code>GraphTensor.merge_batch_to_components()</code> has been applied.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A trained <code>tf.keras.Model.</code>
</td>
</tr>

</table>
