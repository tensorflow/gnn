# runner.incrementing_model_dir

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/utils/model_dir.py#L21-L35">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Create, given some `dirname`, an incrementing model directory.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>runner.incrementing_model_dir(
    dirname: str, start: int = 0
) -> str
</code></pre>

<!-- Placeholder for "Used in" -->
<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>dirname</code><a id="dirname"></a>
</td>
<td>
The base directory name.
</td>
</tr><tr>
<td>
<code>start</code><a id="start"></a>
</td>
<td>
The starting integer.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A model directory <code>dirname/n</code> where 'n' is the maximum integer in <code>dirname</code>.
</td>
</tr>

</table>
