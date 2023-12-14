# runner.run

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/orchestration.py#L364-L639">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

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
6.  If the resulting `GraphTensor`s have any auxiliary pieces (as indicated by
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

*   Store labels on the auxiliary `"_readout"` node set and let
    <a href="../runner/Task.md#preprocess"><code>Task.preprocess(...)</code></a>
    read them from there. (For library-supplied `Task`s, that means initializing
    with `label_feature_name="..."`.) If that is not already true for the input
    datasets, the label feature can be moved there by one of the
    `feature_processors`, using `tfgnn.structured_readout_into_feature(...)` or
    a similar helper function.
*   For single-Task training only: Let
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
<code>train_ds_provider</code><a id="train_ds_provider"></a>
</td>
<td>
A <code>DatasetProvider</code> for training. The <code>tf.data.Dataset</code>
is not batched and contains scalar <code>GraphTensor</code> values conforming to
<code>gtspec</code>, possibly serialized as a <code>tf.train.Example</code> proto.
</td>
</tr><tr>
<td>
<code>model_fn</code><a id="model_fn"></a>
</td>
<td>
Returns the base GNN <code>tf.keras.Model</code> for use in training and
validation.
</td>
</tr><tr>
<td>
<code>optimizer_fn</code><a id="optimizer_fn"></a>
</td>
<td>
Returns a <code>tf.keras.optimizers.Optimizer</code> for use in training.
</td>
</tr><tr>
<td>
<code>trainer</code><a id="trainer"></a>
</td>
<td>
A <code>Trainer</code>.
</td>
</tr><tr>
<td>
<code>task</code><a id="task"></a>
</td>
<td>
A <code>Task</code> for single-Task training or a <code>Mapping[str, Task]</code> for
multi-Task training. In multi-Task training, <a href="../runner/Task.md#preprocess"><code>Task.preprocess(...)</code></a>
must return <code>GraphTensors</code> with the same spec as its inputs, only the
values may change (so that there remains a single spec for <code>model_fn</code>).
</td>
</tr><tr>
<td>
<code>loss_weights</code><a id="loss_weights"></a>
</td>
<td>
An optional <code>Mapping[str, float]</code> for multi-Task training. If
given, this structure must match (with <code>tf.nest.assert_same_structure</code>)
the structure of <code>task</code>. The mapping contains, for each <code>task</code>, a scalar
coefficient to weight the loss contributions of that <code>task</code>.
</td>
</tr><tr>
<td>
<code>gtspec</code><a id="gtspec"></a>
</td>
<td>
A <code>GraphTensorSpec</code> matching the elements of <code>train</code> and <code>valid</code>
datasets. If <code>train</code> or <code>valid</code> contain <code>tf.string</code> elements, this
<code>GraphTensorSpec</code> is used for parsing; otherwise, <code>train</code> or <code>valid</code> are
expected to contain <code>GraphTensor</code> elements whose relaxed spec matches
<code>gtspec</code>.
</td>
</tr><tr>
<td>
<code>global_batch_size</code><a id="global_batch_size"></a>
</td>
<td>
The <code>tf.data.Dataset</code> global batch size for both training
and validation.
</td>
</tr><tr>
<td>
<code>epochs</code><a id="epochs"></a>
</td>
<td>
The epochs to train.
</td>
</tr><tr>
<td>
<code>drop_remainder</code><a id="drop_remainder"></a>
</td>
<td>
Whether to drop a <code>tf.data.Dataset</code> remainder at batching.
</td>
</tr><tr>
<td>
<code>export_dirs</code><a id="export_dirs"></a>
</td>
<td>
Optional directories for exports (SavedModels); if unset,
default behavior is <code>os.path.join(model_dir, "export")</code>.
</td>
</tr><tr>
<td>
<code>model_exporters</code><a id="model_exporters"></a>
</td>
<td>
Zero or more <code>ModelExporter</code> for exporting (SavedModels) to
<code>export_dirs</code>. If unset, default behavior is <code>[KerasModelExporter()]</code>.
</td>
</tr><tr>
<td>
<code>feature_processors</code><a id="feature_processors"></a>
</td>
<td>
A sequence of callables for feature processing with the
Keras functional API. Each callable must accept and return a symbolic
scalar <code>GraphTensor</code>. The callables are composed in order and may change
the <code>GraphTensorSpec</code> (e.g., add/remove features). The resulting Keras
model is executed on CPU as part of a <code>tf.data.Dataset.map</code> operation.
</td>
</tr><tr>
<td>
<code>valid_ds_provider</code><a id="valid_ds_provider"></a>
</td>
<td>
A <code>DatasetProvider</code> for validation. The <code>tf.data.Dataset</code>
is not batched and contains scalar <code>GraphTensor</code> values conforming to
<code>gtspec</code>, possibly serialized as a <code>tf.train.Example</code> proto.
</td>
</tr><tr>
<td>
<code>train_padding</code><a id="train_padding"></a>
</td>
<td>
<code>GraphTensor</code> padding for training. Required if training on
TPU.
</td>
</tr><tr>
<td>
<code>valid_padding</code><a id="valid_padding"></a>
</td>
<td>
<code>GraphTensor</code> padding for validation. Required if training on
TPU.
</td>
</tr><tr>
<td>
<code>tf_data_service_config</code><a id="tf_data_service_config"></a>
</td>
<td>
tf.data service speeds-up tf.data input pipeline
runtime reducing input bottlenecks for model training. Particularly for
training on accelerators consider enabling it. For more info please see:
https://www.tensorflow.org/api_docs/python/tf/data/experimental/service.
</td>
</tr><tr>
<td>
<code>steps_per_execution</code><a id="steps_per_execution"></a>
</td>
<td>
The number of batches to run during each training
iteration. If not set, for TPU strategy default to 100 and to <code>None</code>
otherwise.
</td>
</tr><tr>
<td>
<code>run_eagerly</code><a id="run_eagerly"></a>
</td>
<td>
Whether to compile the model in eager mode, primarily for
debugging purposes. Note that the symbolic model will still be run twice,
so if you use a <code>breakpoint()</code> you will have to Continue twice before you
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
A <code>RunResult</code> object containing models and information about this run.
</td>
</tr>

</table>
