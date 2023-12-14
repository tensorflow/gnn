# tfgnn.proto.BigQuery

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/proto/graph_schema.proto">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Describes a BigQuery table or SQL statement as datasource of a graph piece.

<!-- Placeholder for "Used in" -->

For detailed documentation, see the comments in the `graph_schema.proto` file.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
<code>read_method</code><a id="read_method"></a>
</td>
<td>
<code>ReadMethod read_method</code>
</td>
</tr><tr>
<td>
<code>reshuffle</code><a id="reshuffle"></a>
</td>
<td>
<code>bool reshuffle</code>
</td>
</tr><tr>
<td>
<code>sql</code><a id="sql"></a>
</td>
<td>
<code>string sql</code>
</td>
</tr><tr>
<td>
<code>table_spec</code><a id="table_spec"></a>
</td>
<td>
<code>TableSpec table_spec</code>
</td>
</tr>
</table>

## Child Classes

[`class TableSpec`](../../tfgnn/proto/BigQuery/TableSpec.md)

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
DIRECT_READ<a id="DIRECT_READ"></a>
</td>
<td>
<code>2</code>
</td>
</tr><tr>
<td>
EXPORT<a id="EXPORT"></a>
</td>
<td>
<code>1</code>
</td>
</tr><tr>
<td>
ReadMethod<a id="ReadMethod"></a>
</td>
<td>
['UNSPECIFIED', 'EXPORT', 'DIRECT_READ']
</td>
</tr><tr>
<td>
UNSPECIFIED<a id="UNSPECIFIED"></a>
</td>
<td>
<code>0</code>
</td>
</tr>
</table>
