<!-- lint-g3mark -->

# runner.ModelExporter

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/interfaces.py#L107-L121">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Saves a Keras model.

<!-- Placeholder for "Used in" -->

## Methods

<h3 id="save"><code>save</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/interfaces.py#L110-L121">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>save(
    run_result: RunResult, export_dir: str
)
</code></pre>

Saves a Keras model.

All persistence decisions are left to the implementation: e.g., a Keras model
with full API or a simple `tf.train.Checkpoint` may be saved.

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
