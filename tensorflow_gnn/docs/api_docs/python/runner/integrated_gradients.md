<!-- lint-g3mark -->

# runner.integrated_gradients

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/utils/attribution.py#L230-L300">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

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
`preprocess_model`<a id="preprocess_model"></a>
</td>
<td>
A `tf.keras.Model` for preprocessing. This model is
expected to return a tuple (`GraphTensor`, `Tensor`) where the
`GraphTensor` is used to invoke the below `model` and the tensor is used
used for any loss computation. (Via `model.compiled_loss`.)
</td>
</tr><tr>
<td>
`model`<a id="model"></a>
</td>
<td>
A `tf.keras.Model` for integrated gradients.
</td>
</tr><tr>
<td>
`output_name`<a id="output_name"></a>
</td>
<td>
The output `Tensor` name. If unset, the tensor will be named
by Keras defaults.
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
A `tf.function` with the integrated gradients as output.
</td>
</tr>

</table>
