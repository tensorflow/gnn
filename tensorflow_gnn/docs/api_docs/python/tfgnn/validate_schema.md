# tfgnn.validate_schema

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/schema_validation.py#L48-L70">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Validates the correctness of a graph schema instance.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.validate_schema(
    schema: <a href="../tfgnn/GraphSchema.md"><code>tfgnn.GraphSchema</code></a>
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
`schema`<a id="schema"></a>
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
`ValidationError`<a id="ValidationError"></a>
</td>
<td>
If a validation check fails.
</td>
</tr>
</table>
