# tfgnn.write_schema

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/schema_utils.py#L57-L66">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Write a `GraphSchema` to a text-formatted proto file.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.write_schema(
    schema: <a href="../tfgnn/proto/GraphSchema.md"><code>tfgnn.proto.GraphSchema</code></a>,
    filename: str
)
</code></pre>

<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>schema</code><a id="schema"></a>
</td>
<td>
A <code>GraphSchema</code> instance to write out.
</td>
</tr><tr>
<td>
<code>filename</code><a id="filename"></a>
</td>
<td>
A string, the path to a file to render a text-formatted rendition
of the <code>GraphSchema</code> message to.
</td>
</tr>
</table>
