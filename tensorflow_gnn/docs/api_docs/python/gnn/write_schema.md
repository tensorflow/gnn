description: Write a GraphSchema to a text-formatted proto file.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gnn.write_schema" />
<meta itemprop="path" content="Stable" />
</div>

# gnn.write_schema

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/schema_utils.py#L41-L50">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Write a `GraphSchema` to a text-formatted proto file.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gnn.write_schema(
    schema: <a href="../gnn/GraphSchema.md"><code>gnn.GraphSchema</code></a>,
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
`schema`
</td>
<td>
A `GraphSchema` instance to write out.
</td>
</tr><tr>
<td>
`filename`
</td>
<td>
A string, the path to a file to render a text-formatted
rendition of the `GraphSchema` message to.
</td>
</tr>
</table>

