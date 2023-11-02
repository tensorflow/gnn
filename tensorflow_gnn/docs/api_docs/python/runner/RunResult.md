<!-- lint-g3mark -->

# runner.RunResult

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/interfaces.py#L51-L73">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Holds the return values of `run(...)`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>runner.RunResult(
    preprocess_model: Optional[tf.keras.Model],
    base_model: tf.keras.Model,
    trained_model: tf.keras.Model
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`preprocess_model`<a id="preprocess_model"></a>
</td>
<td>
Keras model containing only the computation for
preprocessing inputs. It is not trained. The model takes serialized
`GraphTensor`s as its inputs and returns preprocessed `GraphTensor`s.
`None` when no preprocess model exists.
</td>
</tr><tr>
<td>
`base_model`<a id="base_model"></a>
</td>
<td>
Keras base GNN (as returned by the user provided `model_fn`).
The model both takes and returns `GraphTensor`s. The model contains
any--but not all--trained weights. The `trained_model` contains all
`base_model` trained weights in addition to any prediction trained
weights.
</td>
</tr><tr>
<td>
`trained_model`<a id="trained_model"></a>
</td>
<td>
Keras model for the e2e GNN. (Base GNN plus any prediction
head(s).) The model takes `preprocess_model` output as its inputs and
returns `Task` predictions as its output. Output matches the structure of
the `Task`: an atom for single- or a mapping for multi- `Task` training.
The model contains all trained weights.
</td>
</tr>
</table>

## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.
