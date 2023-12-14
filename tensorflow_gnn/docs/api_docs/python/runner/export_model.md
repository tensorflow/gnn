# runner.export_model

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/utils/model_export.py#L170-L252">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Exports a Keras model without traces s.t. it is loadable without TF-GNN.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>runner.export_model(
    model: tf.keras.Model,
    export_dir: str,
    *,
    output_names: Optional[Any] = None,
    options: Optional[tf.saved_model.SaveOptions] = None,
    use_legacy_model_save: Optional[bool] = None
) -> None
</code></pre>

<!-- Placeholder for "Used in" -->
<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>model</code><a id="model"></a>
</td>
<td>
Keras model instance to be saved.
</td>
</tr><tr>
<td>
<code>export_dir</code><a id="export_dir"></a>
</td>
<td>
Path where to save the model.
</td>
</tr><tr>
<td>
<code>output_names</code><a id="output_names"></a>
</td>
<td>
Optionally, a nest of <code>str</code> values or <code>None</code> with the same
structure as the outputs of <code>model</code>. A non-<code>None</code> value is used as that
output's key in the SavedModel signature. By default, an output gets
the name of the final Keras layer creating it as its key (matching the
behavior of legacy <code>Model.save(save_format="tf")</code>).
</td>
</tr><tr>
<td>
<code>options</code><a id="options"></a>
</td>
<td>
An optional <code>tf.saved_model.SaveOptions</code> argument.
</td>
</tr><tr>
<td>
<code>use_legacy_model_save</code><a id="use_legacy_model_save"></a>
</td>
<td>
Optional; most users can leave it unset to get a
useful default for export to inference. If set to <code>True</code>, forces the use
of <code>Model.save()</code>, which exports a SavedModel suitable for inference and
potentially also for reloading as a Keras model (depending on its Layers).
If set to <code>False</code>, forces the use of <code>tf.keras.export.ExportArchive</code>,
which is usable as of TensorFlow 2.13 and is advertised as the more
streamlined way of exporting to SavedModel for inference only. Currently,
<code>None</code> behaves like <code>True</code>, but the long-term plan is to migrate towards
<code>False</code>.
</td>
</tr>
</table>
