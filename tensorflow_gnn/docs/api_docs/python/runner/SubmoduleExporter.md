<!-- lint-g3mark -->

# runner.SubmoduleExporter

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/utils/model_export.py#L96-L173">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

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
`sublayer_name`<a id="sublayer_name"></a>
</td>
<td>
The name of the submodule's final layer.
</td>
</tr><tr>
<td>
`output_names`<a id="output_names"></a>
</td>
<td>
The names for output Tensor(s), see: `KerasModelExporter`.
</td>
</tr><tr>
<td>
`subdirectory`<a id="subdirectory"></a>
</td>
<td>
An optional subdirectory, if set: submodules are exported
to `os.path.join(export_dir, subdirectory)`.
</td>
</tr><tr>
<td>
`include_preprocessing`<a id="include_preprocessing"></a>
</td>
<td>
Whether to include any `preprocess_model`.
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

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/utils/model_export.py#L135-L173">View
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
