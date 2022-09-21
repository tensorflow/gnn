# tfgnn.keras.layers.Broadcast

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/graph_ops.py#L427-L510">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Broadcasts a GraphTensor feature.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.Broadcast(
    tag: Optional[const.IncidentNodeOrContextTag] = None,
    *,
    edge_set_name: Optional[gt.EdgeSetName] = None,
    node_set_name: Optional[gt.NodeSetName] = None,
    feature_name: Optional[gt.FieldName] = None,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This layer accepts a complete GraphTensor and returns a tensor with the
broadcast feature value.

There are two kinds of broadcast that this layer can be used for:

  * From a node set to an edge set. This is selected by specifying
    the origin by tag `tgnn.SOURCE` or <a href="../../../tfgnn.md#TARGET"><code>tfgnn.TARGET</code></a> and the receiver
    as `edge_set_name=...`; the node set name is implied.
    The result is a tensor shaped like an edge feature in which each edge
    has a copy of the feature that is present at its SOURCE or TARGET node.
    From a node's point of view, SOURCE means broadcast to outgoing edges,
    and TARGET means broadcast to incoming edges.
  * From the context to a node set or edge set. This is selected by
    specifying the origin by tag <a href="../../../tfgnn.md#CONTEXT"><code>tfgnn.CONTEXT</code></a> and the receiver as either
    a `node_set_name=...` or an `edge_set_name=...`.
    The result is a tensor shaped like a node/edge feature in which each
    node/edge has a copy of the context feature in its graph component.
    (For more on components, see GraphTensor.merge_batch_to_components().)

Both the initialization of and the call to this layer accept arguments to
set the tag, node/edge_set_name, and the feature_name. The call
arguments take effect for that call only and can supply missing values,
but they are not allowed to contradict initialization arguments.
The feature name can be left unset to select tfgnn.HIDDEN_STATE.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Init args</h2></th></tr>

<tr>
<td>
`tag`<a id="tag"></a>
</td>
<td>
Can be set to one of tfgnn.SOURCE, tfgnn.TARGET or tfgnn.CONTEXT.
</td>
</tr><tr>
<td>
`edge_set_name`<a id="edge_set_name"></a>
</td>
<td>
If set, the feature will be broadcast to this edge set
from the given origin. Mutually exclusive with node_set_name.
</td>
</tr><tr>
<td>
`node_set_name`<a id="node_set_name"></a>
</td>
<td>
If set, the feature will be broadcast to this node set.
Origin must be CONTEXT. Mutually exclusive with edge_set_name.
</td>
</tr><tr>
<td>
`feature_name`<a id="feature_name"></a>
</td>
<td>
The name of the feature to read. If unset (also in call),
the default state feature will be read.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Call args</h2></th></tr>

<tr>
<td>
`graph`<a id="graph"></a>
</td>
<td>
The scalar GraphTensor to read from.
</td>
</tr><tr>
<td>
`tag`<a id="tag"></a>
</td>
<td>
Same meaning as for init. Must be passed to init, or to call,
  or to both (with the same value).
edge_set_name, node_set_name: Same meaning as for init. One of them must
  be passed to init, or to call, or to both (with the same value).
</td>
</tr><tr>
<td>
`feature_name`<a id="feature_name"></a>
</td>
<td>
Same meaning as for init. If passed to both, the value must
be the same. If passed to neither, tfgnn.HIDDEN_STATE is used.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tensor with the feature value broadcast to the target.
</td>
</tr>

</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`feature_name`<a id="feature_name"></a>
</td>
<td>
Returns the feature_name argument to init, or None if unset.
</td>
</tr><tr>
<td>
`location`<a id="location"></a>
</td>
<td>
Returns dict of kwarg to init with the node or edge set name.
</td>
</tr><tr>
<td>
`tag`<a id="tag"></a>
</td>
<td>
Returns the tag argument to init, or None if unset.
</td>
</tr>
</table>
