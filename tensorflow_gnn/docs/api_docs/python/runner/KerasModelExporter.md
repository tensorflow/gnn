# runner.KerasModelExporter

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/utils/model_export.py#L25-L87">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Exports a Keras model (with Keras API) via `tf.keras.models.save_model`.

Inherits From: [`ModelExporter`](../runner/ModelExporter.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>runner.KerasModelExporter(
    *,
    output_names: Optional[Any] = None,
    subdirectory: Optional[str] = None,
    include_preprocessing: bool = True,
    options: Optional[tf.saved_model.SaveOptions] = None,
    use_legacy_model_save: Optional[bool] = None
)
</code></pre>

<!-- Placeholder for "Used in" -->
<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>output_names</code><a id="output_names"></a>
</td>
<td>
By default, each output of the exported model uses the name
of the final Keras layer that created it as its key in the SavedModel
signature. This argument can be set to a single <code>str</code> name or a nested
structure of <code>str</code> names to override the output names. Its nesting
structure must match the exported model's output (as checked by
<code>tf.nest.assert_same_structure</code>). Any <code>None</code> values in <code>output_names</code>
are ignored, leaving that output with its default name.
</td>
</tr><tr>
<td>
<code>subdirectory</code><a id="subdirectory"></a>
</td>
<td>
An optional subdirectory, if set: models are exported to
<code>os.path.join(export_dir, subdirectory).</code>
</td>
</tr><tr>
<td>
<code>include_preprocessing</code><a id="include_preprocessing"></a>
</td>
<td>
Whether to include any <code>preprocess_model.</code>
</td>
</tr><tr>
<td>
<code>options</code><a id="options"></a>
</td>
<td>
Options for saving to a TensorFlow <code>SavedModel</code>.
</td>
</tr><tr>
<td>
<code>use_legacy_model_save</code><a id="use_legacy_model_save"></a>
</td>
<td>
Optional; most users can leave it unset to get a
useful default for export to inference. See <a href="../runner/export_model.md"><code>runner.export_model()</code></a>
for more.
</td>
</tr>
</table>

## Methods

<h3 id="save"><code>save</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/utils/model_export.py#L59-L87">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save(
    run_result: <a href="../runner/RunResult.md"><code>runner.RunResult</code></a>,
    export_dir: str
)
</code></pre>

Exports a Keras model (with Keras API) via tf.keras.models.save_model.

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
