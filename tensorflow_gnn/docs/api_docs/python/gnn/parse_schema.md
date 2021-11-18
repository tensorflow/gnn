description: Parse a schema from text-formatted protos.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gnn.parse_schema" />
<meta itemprop="path" content="Stable" />
</div>

# gnn.parse_schema

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/schema_utils.py#L14-L24">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Parse a schema from text-formatted protos.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gnn.parse_schema(
    schema_text: str
) -> <a href="../gnn/GraphSchema.md"><code>gnn.GraphSchema</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`schema_text`
</td>
<td>
A string containing a text-formatted protocol buffer
rendition of a `GraphSchema` message.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `GraphSchema` instance.
</td>
</tr>

</table>

