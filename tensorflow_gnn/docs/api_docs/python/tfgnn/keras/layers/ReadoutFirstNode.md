# tfgnn.keras.layers.ReadoutFirstNode

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/graph_ops.py#L206-L289">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Reads a feature from the first node of each graph conponent.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.ReadoutFirstNode(
    *,
    node_set_name: Optional[gt.NodeSetName] = None,
    feature_name: Optional[gt.FieldName] = None,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

Given a particular node set (identified by `node_set_name`), this layer
will gather the given feature from the first node of each graph component.

This is often used for rooted graphs created by sampling around the
neighborhoods of seed nodes in a large graph: by convention, each seed node is
the first node of its component in the respective node set, and this layer
reads out the information it has accumulated there. (In other node sets, the
first node may be arbitrary -- or nonexistant, in which case this operation
must not be used and may raise an error at runtime.)

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Init args</h2></th></tr>

<tr>
<td>
`node_set_name`<a id="node_set_name"></a>
</td>
<td>
If set, the feature will be read from this node set.
</td>
</tr><tr>
<td>
`feature_name`<a id="feature_name"></a>
</td>
<td>
The name of the feature to read. If unset (also in call),
tfgnn.HIDDEN_STATE will be read.
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
`node_set_name`<a id="node_set_name"></a>
</td>
<td>
Same meaning as for init. Must be passed to init, or to call,
or to both (with the same value).
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
A tensor of gathered feature values, one for each graph component, like a
context feature.
</td>
</tr>

</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr> <td> `feature_name`<a id="feature_name"></a> </td> <td> Returns the
feature_name argument to init, or None if unset. </td> </tr><tr> <td>
`location`<a id="location"></a> </td> <td> Returns a dict with the kwarg to init
that selected the feature location.

</td>
</tr>
</table>



