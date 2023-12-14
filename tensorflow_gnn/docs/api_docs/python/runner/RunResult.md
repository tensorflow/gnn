# runner.RunResult

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/interfaces.py#L51-L73">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

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
<code>preprocess_model</code><a id="preprocess_model"></a>
</td>
<td>
Keras model containing only the computation for
preprocessing inputs. It is not trained. The model takes serialized
<code>GraphTensor</code>s as its inputs and returns preprocessed <code>GraphTensor</code>s.
<code>None</code> when no preprocess model exists.
</td>
</tr><tr>
<td>
<code>base_model</code><a id="base_model"></a>
</td>
<td>
Keras base GNN (as returned by the user provided <code>model_fn</code>).
The model both takes and returns <code>GraphTensor</code>s. The model contains
any--but not all--trained weights. The <code>trained_model</code> contains all
<code>base_model</code> trained weights in addition to any prediction trained
weights.
</td>
</tr><tr>
<td>
<code>trained_model</code><a id="trained_model"></a>
</td>
<td>
Keras model for the e2e GNN. (Base GNN plus any prediction
head(s).) The model takes <code>preprocess_model</code> output as its inputs and
returns <code>Task</code> predictions as its output. Output matches the structure of
the <code>Task</code>: an atom for single- or a mapping for multi- <code>Task</code> training.
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
