description: Validates the correctness of a graph schema instance.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gnn.validate_schema" />
<meta itemprop="path" content="Stable" />
</div>

# gnn.validate_schema

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/schema_validation.py#L34-L56">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Validates the correctness of a graph schema instance.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gnn.validate_schema(
    schema: <a href="../gnn/GraphSchema.md"><code>gnn.GraphSchema</code></a>
) -> List[Exception]
</code></pre>



<!-- Placeholder for "Used in" -->

`GraphSchema` configuration messages are created by users in order to describe
the topology of a graph. This function checks various aspects of the schema
for correctness, e.g. prevents usage of reserved feature names, ensures given
shapes are fully-defined, ensures set name references are found, etc.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`schema`
</td>
<td>
An instance of the graph schema.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A list of exceptions describing optional warnings.
Render those to your favorite stream (or ignore).
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValidationError`
</td>
<td>
If a validation check fails.
</td>
</tr>
</table>

