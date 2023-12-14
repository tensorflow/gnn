# runner.integrated_gradients

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/utils/attribution.py#L230-L300">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Integrated gradients.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>runner.integrated_gradients(
    preprocess_model: tf.keras.Model,
    model: tf.keras.Model,
    *,
    output_name: Optional[str] = None,
    random_counterfactual: bool,
    steps: int,
    seed: Optional[int] = None
) -> tf.types.experimental.ConcreteFunction
</code></pre>

<!-- Placeholder for "Used in" -->

This `tf.function` computes integrated gradients over a `tfgnn.GraphTensor.` The
`tf.function` will be persisted in the ultimate saved model for subsequent
attribution.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>preprocess_model</code><a id="preprocess_model"></a>
</td>
<td>
A <code>tf.keras.Model</code> for preprocessing. This model is
expected to return a tuple (<code>GraphTensor</code>, <code>Tensor</code>) where the
<code>GraphTensor</code> is used to invoke the below <code>model</code> and the tensor is used
used for any loss computation. (Via <code>model.compiled_loss</code>.)
</td>
</tr><tr>
<td>
<code>model</code><a id="model"></a>
</td>
<td>
A <code>tf.keras.Model</code> for integrated gradients.
</td>
</tr><tr>
<td>
<code>output_name</code><a id="output_name"></a>
</td>
<td>
The output <code>Tensor</code> name. If unset, the tensor will be named
by Keras defaults.
</td>
</tr><tr>
<td>
<code>random_counterfactual</code><a id="random_counterfactual"></a>
</td>
<td>
Whether to use a random uniform counterfactual.
</td>
</tr><tr>
<td>
<code>steps</code><a id="steps"></a>
</td>
<td>
The number of interpolations of the Riemann sum approximation.
</td>
</tr><tr>
<td>
<code>seed</code><a id="seed"></a>
</td>
<td>
An option random seed.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A <code>tf.function</code> with the integrated gradients as output.
</td>
</tr>

</table>
