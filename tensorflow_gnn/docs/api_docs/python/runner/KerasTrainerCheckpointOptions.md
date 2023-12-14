# runner.KerasTrainerCheckpointOptions

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/trainers/keras_fit.py#L37-L54">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Provides Keras Checkpointing related configuration options.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>runner.KerasTrainerCheckpointOptions(
    checkpoint_dir: Optional[str] = None,
    best_checkpoint: str = &#x27;best&#x27;,
    latest_checkpoint: str = &#x27;latest&#x27;
)
</code></pre>

<!-- Placeholder for "Used in" -->
<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
<code>checkpoint_dir</code><a id="checkpoint_dir"></a>
</td>
<td>
Directory path to save checkpoint files.
</td>
</tr><tr>
<td>
<code>best_checkpoint</code><a id="best_checkpoint"></a>
</td>
<td>
Filename for the best checkpoint.
</td>
</tr><tr>
<td>
<code>latest_checkpoint</code><a id="latest_checkpoint"></a>
</td>
<td>
Filename for the latest checkpoint.
</td>
</tr>
</table>

## Methods

<h3 id="best_checkpoint_filepath"><code>best_checkpoint_filepath</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/trainers/keras_fit.py#L50-L51">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>best_checkpoint_filepath() -> str
</code></pre>

<h3 id="latest_checkpoint_filepath"><code>latest_checkpoint_filepath</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/trainers/keras_fit.py#L53-L54">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>latest_checkpoint_filepath() -> str
</code></pre>

<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
best_checkpoint<a id="best_checkpoint"></a>
</td>
<td>
<code>'best'</code>
</td>
</tr><tr>
<td>
checkpoint_dir<a id="checkpoint_dir"></a>
</td>
<td>
<code>None</code>
</td>
</tr><tr>
<td>
latest_checkpoint<a id="latest_checkpoint"></a>
</td>
<td>
<code>'latest'</code>
</td>
</tr>
</table>
