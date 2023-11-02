<!-- lint-g3mark -->

# runner.run

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/orchestration.py#L364-L639">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Runs training (and validation) of a model on task(s) with the given data.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>runner.run(
    *,
    train_ds_provider: DatasetProvider,
    model_fn: Callable[[GraphTensorSpec], tf.keras.Model],
    optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
    trainer: Trainer,
    task: OneOrMappingOf[Task],
    loss_weights: Optional[Mapping[str, float]] = None,
    gtspec: GraphTensorSpec,
    global_batch_size: int,
    epochs: int = 1,
    drop_remainder: bool = False,
    export_dirs: Optional[Sequence[str]] = None,
    model_exporters: Optional[Sequence[ModelExporter]] = None,
    feature_processors: Optional[Sequence[GraphTensorProcessorFn]] = None,
    valid_ds_provider: Optional[DatasetProvider] = None,
    train_padding: Optional[GraphTensorPadding] = None,
    valid_padding: Optional[GraphTensorPadding] = None,
    tf_data_service_config: Optional[TFDataServiceConfig] = None,
    steps_per_execution: Optional[int] = None,
    run_eagerly: bool = False
)
</code></pre>

<!-- Placeholder for "Used in" -->

This includes preprocessing the input data, appending any suitable head(s), and
running training (and validation) with the requested distribution strategy.

The input data is processed in multiple stages, starting from the contents of
the datasets provided by `train_ds_provider` and `valid_ds_provider`:

1.  Input examples are batched.
2.  If necessary, input batches are parsed as `GraphTensor` values and merged
    into components (see: `GraphTensor.merge_batch_to_components`).
3.  If set, `train_padding` and `valid_padding`, resp., are applied.
4.  The given `feature_processors` are applied in order for all non-trainable
    feature transformations on CPU (as part of `tf.data.Dataset.map(...)`).
5.  The
    <a href="../runner/Task.md#preprocess"><code>Task.preprocess(...)</code></a>
    method is applied to extract training targets (for supervised learning, that
    means: labels) and optionally transform the value of the preprocessed
    `GraphTensor` into a model input (or multiple model inputs for tasks like
    self-supervised contrastive losses).
6.  If the resulting `GraphTensor`s have any auxillary pieces (as indicated by
    `tfgnn.get_aux_type_prefix(...)`): all features (typically: labels) are
    removed from those graph pieces.

The base GNN (as built by `model_fn`) is run on all results from step (6).
<a href="../runner/Task.md#predict"><code>Task.predict(...)</code></a> is called
on the model outputs that correspond to the one or more graphs requested in step
(5) by
<a href="../runner/Task.md#preprocess"><code>Task.preprocess(...)</code></a>.

Trainable transformations of inputs (notably lookups in trainable embedding
tables) are required to happen inside `model_fn`.

For supervised learning, training labels enter the pipeline as features on the
`GraphTensor` that undergo the `feature_processors` (shared by all `Task`s) and
are read out of the `GraphTensor` by
<a href="../runner/Task.md#preprocess"><code>Task.preprocess(...)</code></a>.

Users are strongly encouraged to take one of the following two approaches to
prevent the leakage of label information into the training:

  - Store labels on the auxiliary `"_readout"` node set and let
    <a href="../runner/Task.md#preprocess"><code>Task.preprocess(...)</code></a>
    read them from there. (For library-supplied `Task`s, that means initializing
    with `label_feature_name="..."`.) If that is not already true for the input
    datasets, the label feature can be moved there by one of the
    `feature_processors`, using `tfgnn.structured_readout_into_feature(...)` or
    a similar helper function.
  - For single-Task training only: Let
    <a href="../runner/Task.md#preprocess"><code>Task.preprocess()</code></a>
    return modified `GraphTensor`s that no longer contain the separately
    returned labels. (Library-supplied Tasks delegate this to the
    `label_fn="..."` passed in initialization.)

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`train_ds_provider`<a id="train_ds_provider"></a>
</td>
<td>
A `DatasetProvider` for training. The `tf.data.Dataset`
is not batched and contains scalar `GraphTensor` values conforming to
`gtspec`, possibly serialized as a `tf.train.Example` proto.
</td>
</tr><tr>
<td>
`model_fn`<a id="model_fn"></a>
</td>
<td>
Returns the base GNN `tf.keras.Model` for use in training and
validation.
</td>
</tr><tr>
<td>
`optimizer_fn`<a id="optimizer_fn"></a>
</td>
<td>
Returns a `tf.keras.optimizers.Optimizer` for use in training.
</td>
</tr><tr>
<td>
`trainer`<a id="trainer"></a>
</td>
<td>
A `Trainer`.
</td>
</tr><tr>
<td>
`task`<a id="task"></a>
</td>
<td>
A `Task` for single-Task training or a `Mapping[str, Task]` for
multi-Task training. In multi-Task training, <a href="../runner/Task.md#preprocess"><code>Task.preprocess(...)</code></a>
must return `GraphTensors` with the same spec as its inputs, only the
values may change (so that there remains a single spec for `model_fn`).
</td>
</tr><tr>
<td>
`loss_weights`<a id="loss_weights"></a>
</td>
<td>
An optional `Mapping[str, float]` for multi-Task training. If
given, this structure must match (with `tf.nest.assert_same_structure`)
the structure of `task`. The mapping contains, for each `task`, a scalar
coefficient to weight the loss contributions of that `task`.
</td>
</tr><tr>
<td>
`gtspec`<a id="gtspec"></a>
</td>
<td>
A `GraphTensorSpec` matching the elements of `train` and `valid`
datasets. If `train` or `valid` contain `tf.string` elements, this
`GraphTensorSpec` is used for parsing; otherwise, `train` or `valid` are
expected to contain `GraphTensor` elements whose relaxed spec matches
`gtspec`.
</td>
</tr><tr>
<td>
`global_batch_size`<a id="global_batch_size"></a>
</td>
<td>
The `tf.data.Dataset` global batch size for both training
and validation.
</td>
</tr><tr>
<td>
`epochs`<a id="epochs"></a>
</td>
<td>
The epochs to train.
</td>
</tr><tr>
<td>
`drop_remainder`<a id="drop_remainder"></a>
</td>
<td>
Whether to drop a `tf.data.Dataset` remainder at batching.
</td>
</tr><tr>
<td>
`export_dirs`<a id="export_dirs"></a>
</td>
<td>
Optional directories for exports (SavedModels); if unset,
default behavior is `os.path.join(model_dir, "export")`.
</td>
</tr><tr>
<td>
`model_exporters`<a id="model_exporters"></a>
</td>
<td>
Zero or more `ModelExporter` for exporting (SavedModels) to
`export_dirs`. If unset, default behavior is `[KerasModelExporter()]`.
</td>
</tr><tr>
<td>
`feature_processors`<a id="feature_processors"></a>
</td>
<td>
A sequence of callables for feature processing with the
Keras functional API. Each callable must accept and return a symbolic
scalar `GraphTensor`. The callables are composed in order and may change
the `GraphTensorSpec` (e.g., add/remove features). The resulting Keras
model is executed on CPU as part of a `tf.data.Dataset.map` operation.
</td>
</tr><tr>
<td>
`valid_ds_provider`<a id="valid_ds_provider"></a>
</td>
<td>
A `DatasetProvider` for validation. The `tf.data.Dataset`
is not batched and contains scalar `GraphTensor` values conforming to
`gtspec`, possibly serialized as a `tf.train.Example` proto.
</td>
</tr><tr>
<td>
`train_padding`<a id="train_padding"></a>
</td>
<td>
`GraphTensor` padding for training. Required if training on
TPU.
</td>
</tr><tr>
<td>
`valid_padding`<a id="valid_padding"></a>
</td>
<td>
`GraphTensor` padding for validation. Required if training on
TPU.
</td>
</tr><tr>
<td>
`tf_data_service_config`<a id="tf_data_service_config"></a>
</td>
<td>
tf.data service speeds-up tf.data input pipeline
runtime reducing input bottlenecks for model training. Particularly for
training on accelerators consider enabling it. For more info please see:
https://www.tensorflow.org/api_docs/python/tf/data/experimental/service.
</td>
</tr><tr>
<td>
`steps_per_execution`<a id="steps_per_execution"></a>
</td>
<td>
The number of batches to run during each training
iteration. If not set, for TPU strategy default to 100 and to `None`
otherwise.
</td>
</tr><tr>
<td>
`run_eagerly`<a id="run_eagerly"></a>
</td>
<td>
Whether to compile the model in eager mode, primarily for
debugging purposes. Note that the symbolic model will still be run twice,
so if you use a `breakpoint()` you will have to Continue twice before you
are in a real eager execution.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `RunResult` object containing models and information about this run.
</td>
</tr>

</table>
