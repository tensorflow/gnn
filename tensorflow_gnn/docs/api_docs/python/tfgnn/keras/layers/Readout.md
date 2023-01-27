# tfgnn.keras.layers.Readout

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/graph_ops.py#L26-L155">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Reads a feature out of a GraphTensor.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.Readout(
    *,
    edge_set_name: Optional[gt.EdgeSetName] = None,
    node_set_name: Optional[gt.NodeSetName] = None,
    from_context: bool = False,
    feature_name: Optional[gt.FieldName] = None,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

The Readout layer is a convenience wrapper for accessing a feature from
one of the edge_sets, node_sets, or the context of a GraphTensor, intended
for use in places such as tf.keras.Sequential that require a Keras layer
and do not allow for direct subscripting of the GraphTensor.

A location in the graph is selected by setting exactly one of the keyword
arguments `edge_set_name=...`, `node_set_name=...` or `from_context=True`.
From there, the keyword argument `feature_name=...` selects the feature.

Both the initialization of and the call to this layer accept arguments to
select the feature location and the feature name. The call arguments take
effect for that call only and can supply missing values, but they are not
allowed to contradict initialization arguments. The feature name can be left
unset to select tfgnn.HIDDEN_STATE.

#### For example:

```python
readout = tfgnn.keras.layers.Readout(feature_name="value")
value = readout(graph_tensor, edge_set_name="edges")
assert value == graph_tensor.edge_sets["edge"]["value"]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Init args</h2></th></tr>

<tr>
<td>
`edge_set_name`<a id="edge_set_name"></a>
</td>
<td>
If set, the feature will be read from this edge set.
Mutually exclusive with node_set_name and from_context.
</td>
</tr><tr>
<td>
`node_set_name`<a id="node_set_name"></a>
</td>
<td>
If set, the feature will be read from this node set.
Mutually exclusive with edge_set_name and from_context.
</td>
</tr><tr>
<td>
`from_context`<a id="from_context"></a>
</td>
<td>
If true, the feature will be read from the context.
Mutually exclusive with edge_set_name and node_set_name.
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
The GraphTensor to read from.
edge_set_name, node_set_name, from_context: Same meaning as for init. One of
  them must be passed to init, or to call, or to both (with the same value).
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
The tensor with the selected feature.
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

The result contains the keyword argument and value passed to `__init__()`
that selects the location from which the layer's output feature is read,
 that is, one of `edge_set_name=...`, `node_set_name=...` or
`from_context=True`. If none of these has been set, the result is
empty, and one of them must be set at call time.
</td>
</tr>
</table>



