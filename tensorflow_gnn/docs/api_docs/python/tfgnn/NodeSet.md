# tfgnn.NodeSet

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L418-L483">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

A composite tensor for node set features plus size information.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.NodeSet(
    data: Data, spec: 'GraphPieceSpecBase', validate: bool = False
)
</code></pre>



<!-- Placeholder for "Used in" -->

The items of the node set are subset of graph nodes.

All nodes in a node set have the same features, identified by a string key.
Each feature is stored as one tensor and has shape `[*graph_shape, num_nodes,
*feature_shape]`. The `num_nodes` is the number of nodes in a graph (could be
ragged). The `feature_shape` is the shape of the feature value for each node.
NodeSet supports both fixed-size and variable-size features. The fixed-size
features must have fully defined feature_shape. They are stored as `tf.Tensor`
if `num_nodes` is fixed-size or `graph_shape.rank = 0`. Variable-size node
features are always stored as `tf.RaggedTensor`.

Note that node set features are indexed without regard to graph components.
The information which node belong to which graph component is contained in
the `.sizes` tensor which defines the number of nodes in each graph component.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`data`<a id="data"></a>
</td>
<td>
Nest of Field or subclasses of GraphPieceBase.
</td>
</tr><tr>
<td>
`spec`<a id="spec"></a>
</td>
<td>
A subclass of GraphPieceSpecBase with a `_data_spec` that matches
`data`.
</td>
</tr><tr>
<td>
`validate`<a id="validate"></a>
</td>
<td>
if set, checks that data and spec are aligned, compatible and
supported.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr> <td> `features`<a id="features"></a> </td> <td> A read-only mapping of
feature name to feature specs. </td> </tr><tr> <td>
`indices_dtype`<a id="indices_dtype"></a> </td> <td> The integer type to
represent ragged splits. </td> </tr><tr> <td>
`num_components`<a id="num_components"></a> </td> <td> The number of graph
components for each graph. </td> </tr><tr> <td> `rank`<a id="rank"></a> </td>
<td> The rank of this Tensor. Guaranteed not to be `None`. </td> </tr><tr> <td>
`shape`<a id="shape"></a> </td> <td> A possibly-partial shape specification for
this Tensor.

The returned `TensorShape` is guaranteed to have a known rank, but the
individual dimension sizes may be unknown.
</td>
</tr><tr>
<td>
`sizes`<a id="sizes"></a>
</td>
<td>
The number of items in each graph component.
</td>
</tr><tr>
<td>
`spec`<a id="spec"></a>
</td>
<td>
The public type specification of this tensor.
</td>
</tr><tr>
<td>
`total_num_components`<a id="total_num_components"></a>
</td>
<td>
The total number of graph components.
</td>
</tr><tr>
<td>
`total_size`<a id="total_size"></a>
</td>
<td>
The total number of items.
</td>
</tr>
</table>

## Methods

<h3 id="from_fields"><code>from_fields</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L437-L474">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_fields(
    *,
    features: Optional[<a href="../tfgnn/Fields.md"><code>tfgnn.Fields</code></a>] = None,
    sizes: <a href="../tfgnn/Field.md"><code>tfgnn.Field</code></a>
) -> 'NodeSet'
</code></pre>

Constructs a new instance from node set fields.


#### Example:



```python
tfgnn.NodeSet.from_fields(
    sizes=tf.constant([3]),
    features={
        "tokenized_title": tf.ragged.constant(
            [["Anisotropic", "approximation"],
             ["Better", "bipartite", "bijection", "bounds"],
             ["Convolutional", "convergence", "criteria"]]),
        "embedding": tf.zeros([3, 128]),
        "year": tf.constant([2018, 2019, 2020]),
    })
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`features`
</td>
<td>
A mapping from feature name to feature Tensors or RaggedTensors.
All feature tensors must have shape `[*graph_shape, num_nodes,
*feature_shape]`, where `num_nodes` is the number of nodes in the node
set (could be ragged) and feature_shape is a shape of the feature value
for each node.
</td>
</tr><tr>
<td>
`sizes`
</td>
<td>
A number of nodes in each graph component. Has shape
`[*graph_shape, num_components]`, where `num_components` is the number
of graph components (could be ragged).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `NodeSet` composite tensor.
</td>
</tr>

</table>



<h3 id="get_features_dict"><code>get_features_dict</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L157-L159">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_features_dict() -> Dict[FieldName, Field]
</code></pre>

Returns features copy as a dictionary.


<h3 id="replace_features"><code>replace_features</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L404-L410">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>replace_features(
    features: <a href="../tfgnn/Fields.md"><code>tfgnn.Fields</code></a>
) -> '_NodeOrEdgeSet'
</code></pre>

Returns a new instance with a new set of features.


<h3 id="set_shape"><code>set_shape</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_piece.py#L304-L310">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_shape(
    new_shape: ShapeLike
) -> 'GraphPieceSpecBase'
</code></pre>

Enforce the common prefix shape on all the contained features.


<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L54-L56">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    feature_name: FieldName
) -> <a href="../tfgnn/Field.md"><code>tfgnn.Field</code></a>
</code></pre>

Indexing operator `[]` to access feature values by their name.




