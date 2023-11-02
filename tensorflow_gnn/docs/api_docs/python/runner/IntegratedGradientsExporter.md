<!-- lint-g3mark -->

# runner.IntegratedGradientsExporter

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/utils/attribution.py#L303-L402">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Exports a Keras model with an additional integrated gradients signature.

Inherits From: [`ModelExporter`](../runner/ModelExporter.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>runner.IntegratedGradientsExporter(
    integrated_gradients_output_name: Optional[str] = None,
    subdirectory: Optional[str] = None,
    random_counterfactual: bool = True,
    steps: int = 32,
    seed: Optional[int] = None,
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
`integrated_gradients_output_name`<a id="integrated_gradients_output_name"></a>
</td>
<td>
The name for the integrated gradients
output tensor. If unset, the tensor will be named by Keras defaults.
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
`random_counterfactual`<a id="random_counterfactual"></a>
</td>
<td>
Whether to use a random uniform counterfactual.
</td>
</tr><tr>
<td>
`steps`<a id="steps"></a>
</td>
<td>
The number of interpolations of the Riemann sum approximation.
</td>
</tr><tr>
<td>
`seed`<a id="seed"></a>
</td>
<td>
An optional random seed.
</td>
</tr><tr>
<td>
`options`<a id="options"></a>
</td>
<td>
Options for saving to SavedModel.
</td>
</tr>
</table>

## Methods

<h3 id="save"><code>save</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/utils/attribution.py#L345-L402">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save(
    run_result: <a href="../runner/RunResult.md"><code>runner.RunResult</code></a>,
    export_dir: str
)
</code></pre>

Exports a Keras model with an additional integrated gradients signature.

Importantly: the `run_result.preprocess_model`, if provided, and
`run_result.trained_model` are stacked before any export. Stacking involves the
chaining of the first output of `run_result.preprocess_model` to the only input
of `run_result.trained_model.` The result is a model with the input of
`run_result.preprocess_model` and the output of `run_result.trained_model.`

Two serving signatures are exported:

'serving_default') The default serving signature (i.e., the `preprocess_model`
input signature), 'integrated_gradients') The integrated gradients signature
(i.e., the `preprocess_model` input signature).

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
