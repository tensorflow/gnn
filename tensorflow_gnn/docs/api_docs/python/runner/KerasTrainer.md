<!-- lint-g3mark -->

# runner.KerasTrainer

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/trainers/keras_fit.py#L57-L316">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

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
`strategy`<a id="strategy"></a>
</td>
<td>
A `tf.distribute.Strategy.`
</td>
</tr><tr>
<td>
`model_dir`<a id="model_dir"></a>
</td>
<td>
A model directory for summaries.
</td>
</tr><tr>
<td>
`checkpoint_options`<a id="checkpoint_options"></a>
</td>
<td>
An optional configuration for checkpointing related
configs. If checkpoint_options.checkpoint_dir is unset;
`os.path.join(model_dir, "ckpnt")` is used.
</td>
</tr><tr>
<td>
`backup_dir`<a id="backup_dir"></a>
</td>
<td>
An optional directory for backup, if unset;
`(os.path.join(model_dir, "backup"),)` is used.
</td>
</tr><tr>
<td>
`steps_per_epoch`<a id="steps_per_epoch"></a>
</td>
<td>
The number of training steps per epoch. Optional,
if unspecified: epochs are at `tf.data.Dataset` end.
</td>
</tr><tr>
<td>
`verbose`<a id="verbose"></a>
</td>
<td>
Forwarded to `tf.keras.Model.fit()`. Possible values are
0 (silent), 1 (print progress bar), 2 (one line per epoch), and
"auto" (default) defers to keras to select verbosity.
</td>
</tr><tr>
<td>
`validation_steps`<a id="validation_steps"></a>
</td>
<td>
The number of steps used during validation. Optional,
if unspecified: the entire validation `tf.data.Dataset` is evaluated.
</td>
</tr><tr>
<td>
`validation_per_epoch`<a id="validation_per_epoch"></a>
</td>
<td>
The number of validations done per training epoch.
Optional, if unspecified: Perform one validation per training epoch.
Only one of `validation_per_epoch` and `validation_freq` can be
specified.
</td>
</tr><tr>
<td>
`validation_freq`<a id="validation_freq"></a>
</td>
<td>
Specifies how many training epochs to run before a new
validation run is performed. Optional, if unspecified: Performs
validation after every training epoch. Only one of
`validation_per_epoch` and `validation_freq` can be specified.
</td>
</tr><tr>
<td>
`summarize_every_n_steps`<a id="summarize_every_n_steps"></a>
</td>
<td>
The frequency for writing TensorBoard summaries,
as an integer number of steps, or "epoch" for once per epoch, or
"never".
</td>
</tr><tr>
<td>
`checkpoint_every_n_steps`<a id="checkpoint_every_n_steps"></a>
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
`backup_and_restore`<a id="backup_and_restore"></a>
</td>
<td>
Whether to backup and restore (According to
`tf.keras.callbacks.BackupAndRestore`). The backup
directory is determined by `backup_dir`.
</td>
</tr><tr>
<td>
`callbacks`<a id="callbacks"></a>
</td>
<td>
Optional additional `tf.keras.callbacks.Callback` for
`tf.keras.Model.fit.`
</td>
</tr><tr>
<td>
`restore_best_weights`<a id="restore_best_weights"></a>
</td>
<td>
Requires a `checkpoint_every_n_steps` other than
"never." Whether to restore the best model weights as determined by
`tf.keras.callbacks.ModelCheckpoint` after training. If unspecified,
its value is determined at `train(...)` invocation: `True if
valid_ds_provider is not None else False`.
</td>
</tr><tr>
<td>
`options`<a id="options"></a>
</td>
<td>
A `KerasTrainerOptions.`
</td>
</tr>
</table>

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
`model_fn`
</td>
<td>
A `ModelFn`, to be invoked in the `tf.distribute.Strategty`
scope.
</td>
</tr><tr>
<td>
`train_ds_provider`
</td>
<td>
A function that returns a `tf.data.Dataset` for
training.The items of the `tf.data.Dataset` are pairs
`(graph_tensor, label)` that represent one batch of per-replica training
inputs after `GraphTensor.merge_batch_to_components()` has been applied.
</td>
</tr><tr>
<td>
`epochs`
</td>
<td>
The epochs to train: adjusted for `validation_per_epoch.`
</td>
</tr><tr>
<td>
`valid_ds_provider`
</td>
<td>
An optional function that returns a `tf.data.Dataset`
for validation. The items of the `tf.data.Dataset` are pairs
`(graph_tensor, label)` that represent one batch of per-replica training
inputs after `GraphTensor.merge_batch_to_components()` has been applied.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A trained `tf.keras.Model.`
</td>
</tr>

</table>
