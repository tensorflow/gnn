# tfgnn.homogeneous

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L1790-L1869">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Constructs a homogeneous `GraphTensor` with node features and one edge_set.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.homogeneous(
    source: tf.Tensor,
    target: tf.Tensor,
    *,
    node_features: Optional[FieldOrFields] = None,
    edge_features: Optional[FieldOrFields] = None,
    context_features: Optional[FieldOrFields] = None,
    node_set_name: Optional[FieldName] = const.NODES,
    edge_set_name: Optional[FieldName] = const.EDGES,
    node_set_sizes: Optional[Field] = None,
    edge_set_sizes: Optional[Field] = None
) -> GraphTensor
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>source</code><a id="source"></a>
</td>
<td>
A dense Tensor with the source indices for edges
</td>
</tr><tr>
<td>
<code>target</code><a id="target"></a>
</td>
<td>
A dense Tensor with the target indices for edges
</td>
</tr><tr>
<td>
<code>node_features</code><a id="node_features"></a>
</td>
<td>
A Tensor or mapping from feature name to Tensor of features
corresponding to graph nodes.
</td>
</tr><tr>
<td>
<code>edge_features</code><a id="edge_features"></a>
</td>
<td>
Optional Tensor or mapping from feature name to Tensor of
features corresponding to graph edges.
</td>
</tr><tr>
<td>
<code>context_features</code><a id="context_features"></a>
</td>
<td>
Optional Tensor or mapping from name to Tensor for the
context (entire graph)
</td>
</tr><tr>
<td>
<code>node_set_name</code><a id="node_set_name"></a>
</td>
<td>
Optional name for the node set
</td>
</tr><tr>
<td>
<code>edge_set_name</code><a id="edge_set_name"></a>
</td>
<td>
Optional name for the edge set
</td>
</tr><tr>
<td>
<code>node_set_sizes</code><a id="node_set_sizes"></a>
</td>
<td>
Optional Tensor with the number of nodes per component. If
this is provided, edge_set_sizes should also be passed and it should be
the same length.
</td>
</tr><tr>
<td>
<code>edge_set_sizes</code><a id="edge_set_sizes"></a>
</td>
<td>
Optional Tensor with the number of edges per component. If
this is provided, node_set_sizes should also be passed and it should be
the same length.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A scalar <code>GraphTensor</code> with a single node set and edge set.
</td>
</tr>

</table>
