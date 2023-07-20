# tfgnn.pool

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/pool_ops.py#L207-L327">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Pools values from edges to nodes, or from nodes or edges to context.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.pool(
    graph_tensor: GraphTensor,
    to_tag: IncidentNodeOrContextTag,
    *,
    edge_set_name: Union[Sequence[EdgeSetName], EdgeSetName, None] = None,
    node_set_name: Union[Sequence[NodeSetName], NodeSetName, None] = None,
    reduce_type: str,
    feature_value: Union[Sequence[Field], Field, None] = None,
    feature_name: Optional[FieldName] = None
) -> Field
</code></pre>

<!-- Placeholder for "Used in" -->

This function pools to context if `to_tag=tfgnn.CONTEXT` and pools from edges to
incident nodes if `to_tag` is an ordinary node tag like
<a href="../tfgnn.md#SOURCE"><code>tfgnn.SOURCE</code></a> or
<a href="../tfgnn.md#TARGET"><code>tfgnn.TARGET</code></a>.

The `edge_set_name` (or `node_set_name`, when pooling to context) can be set to
a single name, or to a non-empty list of names. Pooling is done as if all named
edge sets (or node sets) were concatenated into a single edge set (or node set).
For example, `pool(reduce_type="mean", edge_sets=["a", "b"], ...)` will form the
sum over all edges in "a" and "b" and divide by their total number, giving equal
weight to each.

The following choices of `reduce_type` are supported:

`reduce_type`  | Description
-------------- | ----------------------------------------------------------
`"sum"`        | element-wise sum of input tensors
`"prod"`       | element-wise product of input tensors (beware of overflow)
`"mean"`       | element-wise mean (average), or zero for no inputs
`"max"`        | element-wise maximum, or `-inf` for no inputs
`"max_no_inf"` | element-wise maximum, or zero for no inputs
`"min"`        | element-wise minimum, or `-inf` for no inputs
`"min_no_inf"` | element-wise minimum, or zero for no inputs

The helper function
<a href="../tfgnn/get_registered_reduce_operation_names.md"><code>tfgnn.get_registered_reduce_operation_names()</code></a>
returns a list of these values.

Moreover, `reduce_type` can be set to a `|`-separated list of reduce types, such
as `reduce_type="mean|sum"`, which will return the concatenation of their
individual results along the innermost axis in the order of appearance.

support RaggedTensors.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`graph_tensor`<a id="graph_tensor"></a>
</td>
<td>
A scalar GraphTensor.
</td>
</tr><tr>
<td>
`to_tag`<a id="to_tag"></a>
</td>
<td>
Values are pooled to context if this is <a href="../tfgnn.md#CONTEXT"><code>tfgnn.CONTEXT</code></a> or to the
incident node on each edge with this tag.
</td>
</tr><tr>
<td>
`edge_set_name`<a id="edge_set_name"></a>
</td>
<td>
The name of the edge set from which values are pooled, or
a non-empty sequence of such names. Unless `to_tag=tfgnn.CONTEXT`,
all named edge sets must have the same incident node set at the given tag.
</td>
</tr><tr>
<td>
`node_set_name`<a id="node_set_name"></a>
</td>
<td>
The name of the node set from which values are pooled,
or a non-empty sequence of such names. Can only be set with
`to_tag=tfgnn.CONTEXT`. Exactly one of edge_set_name or node_set_name
must be set.
</td>
</tr><tr>
<td>
`reduce_type`<a id="reduce_type"></a>
</td>
<td>
A string with the name of a pooling operation, or multiple ones
separated by `|`. See the table above for the known names.
</td>
</tr><tr>
<td>
`feature_value`<a id="feature_value"></a>
</td>
<td>
A tensor or list of tensors, parallel to the node_set_names
or edge_set_names, to supply the input values of pooling. Each tensor
has shape `[num_items, *feature_shape]`, where `num_items` is the number
of edges in the given edge set or nodes in the given node set, and
`*feature_shape` is the same across all inputs. The `*feature_shape` may
contain ragged dimensions. All the ragged values that are reduced onto
any one item of the graph must have the same ragged index structure,
so that a result can be computed from them.
</td>
</tr><tr>
<td>
`feature_name`<a id="feature_name"></a>
</td>
<td>
The name of a feature stored on each graph piece from which
pooling is done, for use instead of an explicity passed feature_value.
Exactly one of feature_name or feature_value must be set.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tensor with the result of pooling from the conceptual concatenation of the
named edge set(s) or node set(s) to the destination selected by `to_tag`.
Its shape is `[num_items, *feature_shape]`, where `num_items` is the number
of destination nodes (or graph components if `to_tag=tfgnn.CONTEXT`)
and `*feature_shape` is as for all the inputs.
</td>
</tr>

</table>
