# tfgnn.keras.layers.Pool

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/graph_ops.py#L477-L593">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Pools a GraphTensor feature.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.Pool(
    tag: Optional[const.IncidentNodeOrContextTag] = None,
    reduce_type: Optional[str] = None,
    *,
    edge_set_name: Optional[gt.EdgeSetName] = None,
    node_set_name: Optional[gt.NodeSetName] = None,
    feature_name: Optional[gt.FieldName] = None,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This layer accepts a complete GraphTensor and returns a tensor with the
result of pooling some feature.

There are two kinds of pooling that this layer can be used for:

  * From an edge set to a node set. This is selected by specifying the
    origin as `edge_set_name=...` and the receiver with tag `tgnn.SOURCE`
    or <a href="../../../tfgnn.md#TARGET"><code>tfgnn.TARGET</code></a>; the corresponding node set name is implied.
    The result is a tensor shaped like a node feature in which each node
    has the aggregated feature values from the edges of the edge set that
    have it as their SOURCE or TARGET, resp.; that is, the outgoing or
    incoming edges of the node.
  * From a node set or edge set to the context. This is selected by specifying
    the origin as either a `node_set_name=...` or an `edge_set_name=...` and
    the receiver with tag <a href="../../../tfgnn.md#CONTEXT"><code>tfgnn.CONTEXT</code></a>. The result is a tensor shaped
    like a context feature in which each graph component has the aggregated
    feature values from those nodes/edges in the selected node or edge set
    that belong to the component.
    (For more on components, see GraphTensor.merge_batch_to_components().)

Feature values are aggregated into a single value by a reduction function
from <a href="../../../tfgnn/get_registered_reduce_operation_names.md"><code>tfgnn.get_registered_reduce_operation_names()</code></a>, see also
<a href="../../../tfgnn/register_reduce_operation.md"><code>tfgnn.register_reduce_operation()</code></a>. The pre-configured choices include
"sum", "mean", "max" and "min".

Both the initialization of and the call to this layer accept arguments for
the receiver tag, the node/edge_set_name, the reduce_type and the
feature_name. The call arguments take effect for that call only and can
supply missing values, but they are not allowed to contradict initialization
arguments.
The feature name can be left unset to select tfgnn.HIDDEN_STATE.

#### Init args:


* <b>`tag`</b>: Can be set to one of tfgnn.SOURCE, tfgnn.TARGET or tfgnn.CONTEXT
  to select the receiver/
* <b>`reduce_type`</b>: Can be set to any name from
  tfgnn.get_registered_reduce_operation_names().
* <b>`edge_set_name`</b>: If set, the feature will be pooled from this edge set
  to the given receiver `tag`. Mutually exclusive with node_set_name.
* <b>`node_set_name`</b>: If set, the feature will be pooled from this node set.
  The `tag` must be CONTEXT. Mutually exclusive with edge_set_name.
* <b>`feature_name`</b>: The name of the feature to read. If unset (also in call),
  the default state feature will be read.


#### Call args:


* <b>`graph`</b>: The scalar GraphTensor to read from.
* <b>`reduce_type`</b>: Same meaning as for init. Must be passed to init, or to call,
  or to both (with the same value).
* <b>`tag`</b>: Same meaning as for init. Must be passed to init, or to call,
  or to both (with the same value).
edge_set_name, node_set_name: Same meaning as for init. One of them must
  be passed to init, or to call, or to both (with the same value).
* <b>`feature_name`</b>: Same meaning as for init. If passed to both, the value must
  be the same. If passed to neither, tfgnn.HIDDEN_STATE is used.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tensor with the pooled feature value.
</td>
</tr>

</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`feature_name`
</td>
<td>
Returns the feature_name argument to init, or None if unset.
</td>
</tr><tr>
<td>
`location`
</td>
<td>
Returns dict of kwarg to init with the node or edge set name.
</td>
</tr><tr>
<td>
`reduce_type`
</td>
<td>
Returns the reduce_type argument to init, or None if unset.
</td>
</tr><tr>
<td>
`tag`
</td>
<td>
Returns the tag argument to init, or None if unset.
</td>
</tr>
</table>



