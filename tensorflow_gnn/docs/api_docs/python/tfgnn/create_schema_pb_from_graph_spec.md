# tfgnn.create_schema_pb_from_graph_spec

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/schema_utils.py#L126-L213">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Converts scalar GraphTensorSpec to a graph schema proto message.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.create_schema_pb_from_graph_spec(
    graph: Union[<a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>, <a href="../tfgnn/GraphTensorSpec.md"><code>tfgnn.GraphTensorSpec</code></a>]
) -> <a href="../tfgnn/proto/GraphSchema.md"><code>tfgnn.proto.GraphSchema</code></a>
</code></pre>

<!-- Placeholder for "Used in" -->

The result graph schema containts entires for all graph pieces. The features
proto field contains type (`dtype`) and shape (`shape`) information for all
features. For edge sets their `source` and `target` fields are populated. All
other fields are left unset. (Callers can set them separately before writing out
the schema.)

It is guaranteed that the input graph is compatible with the output graph schema
(as
<a href="../tfgnn/check_compatible_with_schema_pb.md"><code>tfgnn.check_compatible_with_schema_pb()</code></a>.)

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>graph</code><a id="graph"></a>
</td>
<td>
The scalar graph tensor or its spec with single graph component.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An instance of the graph schema proto message.
</td>
</tr>

</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
<code>ValueError</code><a id="ValueError"></a>
</td>
<td>
if graph has multiple graph components or rank > 0.
</td>
</tr><tr>
<td>
<code>ValueError</code><a id="ValueError"></a>
</td>
<td>
if adjacency types is not an instance of <code>fgnn.Adjacency</code>.
</td>
</tr><tr>
<td>
<code>ValueError</code><a id="ValueError"></a>
</td>
<td>
if graph tensor features have types that are not supported
by the graph schema.
</td>
</tr>
</table>
