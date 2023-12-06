# tfgnn.parse_schema

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/schema_utils.py#L30-L40">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Parse a schema from text-formatted protos.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.parse_schema(
    schema_text: Union[bytes, str]
) -> <a href="../tfgnn/proto/GraphSchema.md"><code>tfgnn.proto.GraphSchema</code></a>
</code></pre>

<!-- Placeholder for "Used in" -->
<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>schema_text</code><a id="schema_text"></a>
</td>
<td>
A string containing a text-formatted protocol buffer rendition
of a <code>GraphSchema</code> message.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A <code>GraphSchema</code> instance.
</td>
</tr>

</table>
