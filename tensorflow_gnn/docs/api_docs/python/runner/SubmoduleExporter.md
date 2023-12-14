# runner.SubmoduleExporter

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/utils/model_export.py#L90-L167">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Exports a Keras submodule.

Inherits From: [`ModelExporter`](../runner/ModelExporter.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>runner.SubmoduleExporter(
    sublayer_name: str,
    *,
    output_names: Optional[Any] = None,
    subdirectory: Optional[str] = None,
    include_preprocessing: bool = False,
    options: Optional[tf.saved_model.SaveOptions] = None
)
</code></pre>

<!-- Placeholder for "Used in" -->

Given a `RunResult`, this exporter creates and exports a submodule with inputs
identical to the trained model and outputs from some intermediate layer (named
`sublayer_name`). For example, with pseudocode:

`trained_model = tf.keras.Sequential([layer1, layer2, layer3, layer4])` and
`SubmoduleExporter(sublayer_name='layer2')`

The exported submodule is:

`submodule = tf.keras.Sequential([layer1, layer2])`

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>sublayer_name</code><a id="sublayer_name"></a>
</td>
<td>
The name of the submodule's final layer.
</td>
</tr><tr>
<td>
<code>output_names</code><a id="output_names"></a>
</td>
<td>
The names for output Tensor(s), see: <code>KerasModelExporter</code>.
</td>
</tr><tr>
<td>
<code>subdirectory</code><a id="subdirectory"></a>
</td>
<td>
An optional subdirectory, if set: submodules are exported
to <code>os.path.join(export_dir, subdirectory)</code>.
</td>
</tr><tr>
<td>
<code>include_preprocessing</code><a id="include_preprocessing"></a>
</td>
<td>
Whether to include any <code>preprocess_model</code>.
</td>
</tr><tr>
<td>
<code>options</code><a id="options"></a>
</td>
<td>
Options for saving to a TensorFlow <code>SavedModel</code>.
</td>
</tr>
</table>

## Methods

<h3 id="save"><code>save</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/utils/model_export.py#L129-L167">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save(
    run_result: <a href="../runner/RunResult.md"><code>runner.RunResult</code></a>,
    export_dir: str
)
</code></pre>

Saves a Keras model submodule.

Importantly: the `run_result.preprocess_model`, if provided, and
`run_result.trained_model` are stacked before any export. Stacking involves the
chaining of the first output of `run_result.preprocess_model` to the only input
of `run_result.trained_model.` The result is a model with the input of
`run_result.preprocess_model` and the output of `run_result.trained_model.`

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<code>run_result</code>
</td>
<td>
A <code>RunResult</code> from training.
</td>
</tr><tr>
<td>
<code>export_dir</code>
</td>
<td>
A destination directory.
</td>
</tr>
</table>
