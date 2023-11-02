<!-- lint-g3mark -->

# runner.KerasModelExporter

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/utils/model_export.py#L36-L93">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Exports a Keras model (with Keras API) via `tf.keras.models.save_model`.

Inherits From: [`ModelExporter`](../runner/ModelExporter.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>runner.KerasModelExporter(
    *,
    output_names: Optional[Any] = None,
    subdirectory: Optional[str] = None,
    include_preprocessing: bool = True,
    options: Optional[tf.saved_model.SaveOptions] = None
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`output_names`<a id="output_names"></a>
</td>
<td>
The name(s) for any output Tensor(s). Can be a single `str`
name or a nested structure of `str` names. If a nested structure is
given, it must match the structure of the exported model's output
(as asserted by `tf.nest.assert_same_structure`): model output is
renamed by flattening (`tf.nest.flatten`) and zipping the two
structures. Any `None` values in `output_names` are ignored (leaving
that corresponding atom with its original name).
</td>
</tr><tr>
<td>
`subdirectory`<a id="subdirectory"></a>
</td>
<td>
An optional subdirectory, if set: models are exported to
`os.path.join(export_dir, subdirectory).`
</td>
</tr><tr>
<td>
`include_preprocessing`<a id="include_preprocessing"></a>
</td>
<td>
Whether to include any `preprocess_model.`
</td>
</tr><tr>
<td>
`options`<a id="options"></a>
</td>
<td>
Options for saving to a TensorFlow `SavedModel`.
</td>
</tr>
</table>

## Methods

<h3 id="save"><code>save</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/utils/model_export.py#L65-L93">View
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
`run_result`
</td>
<td>
A `RunResult` from training.
</td>
</tr><tr>
<td>
`export_dir`
</td>
<td>
A destination directory.
</td>
</tr>
</table>
