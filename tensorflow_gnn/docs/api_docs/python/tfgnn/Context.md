<!-- lint-g3mark -->

# tfgnn.Context

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L310-L432">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

A composite tensor for graph context features.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.Context(
    data: Data, spec: 'GraphPieceSpecBase'
)
</code></pre>

<!-- Placeholder for "Used in" -->

The items of the context are the graph components (just like the items of a node
set are the nodes and the items of an edge set are the edges). The `Context` is
a composite tensor. It stores features that belong to a graph component as a
whole, not any particular node or edge. Each context feature has a shape
`[*graph_shape, num_components, ...]`, where `num_components` is the number of
graph components in a graph (could be ragged).

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
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`features`<a id="features"></a>
</td>
<td>
A read-only mapping of feature name to feature specs.
</td>
</tr><tr>
<td>
`indices_dtype`<a id="indices_dtype"></a>
</td>
<td>
The dtype for graph items indexing. One of `tf.int32` or `tf.int64`.
</td>
</tr><tr>
<td>
`num_components`<a id="num_components"></a>
</td>
<td>
The number of graph components for each graph.
</td>
</tr><tr>
<td>
`rank`<a id="rank"></a>
</td>
<td>
The rank of this Tensor. Guaranteed not to be `None`.
</td>
</tr><tr>
<td>
`row_splits_dtype`<a id="row_splits_dtype"></a>
</td>
<td>
The dtype for ragged row partions. One of `tf.int32` or `tf.int64`.
</td>
</tr><tr>
<td>
`shape`<a id="shape"></a>
</td>
<td>
A possibly-partial shape specification for this Tensor.

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

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L322-L412">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_fields(
    *_,
    features: Optional[Fields] = None,
    sizes: Optional[Field] = None,
    shape: Optional[ShapeLike] = None,
    indices_dtype: Optional[tf.dtypes.DType] = None,
    validate: bool = True
) -> 'Context'
</code></pre>

Constructs a new instance from context fields.

#### Example:

``` python
tfgnn.Context.from_fields(features={'country_code': ['CH']})
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
A mapping from feature name to feature Tensor or RaggedTensor.
All feature tensors must have shape `[*graph_shape, num_components,
*feature_shape]`, where `num_components` is the number of graph
components (could be ragged); `feature_shape` are feature-specific inner
dimensions.
</td>
</tr><tr>
<td>
`sizes`
</td>
<td>
A Tensor of 1's with shape `[*graph_shape, num_components]`, where
`num_components` is the number of graph components (could be ragged).
For symmetry with `sizes` in NodeSet and EdgeSet, this counts the items
per graph component, but since the items of Context are the components
themselves, each value is 1. Must be compatible with `shape`, if that is
specified.
</td>
</tr><tr>
<td>
`shape`
</td>
<td>
The shape of this tensor and a GraphTensor containing it, also
known as the `graph_shape`. If not specified, the shape is inferred from
`sizes` or set to `[]` if the `sizes` is not specified.
</td>
</tr><tr>
<td>
`indices_dtype`
</td>
<td>
An `indices_dtype` of a GraphTensor containing this object,
used as `row_splits_dtype` when batching potentially ragged fields. If
`sizes` are specified they are casted to that type.
</td>
</tr><tr>
<td>
`validate`
</td>
<td>
If true, use tf.assert ops to inspect the shapes of each field
and check at runtime that they form a valid Context.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Context` composite tensor.
</td>
</tr>

</table>

<h3 id="get_features_dict"><code>get_features_dict</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L184-L186">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_features_dict() -> Dict[FieldName, Field]
</code></pre>

Returns features copy as a dictionary.

<h3 id="replace_features"><code>replace_features</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L414-L421">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>replace_features(
    features: Fields
) -> 'Context'
</code></pre>

Returns a new instance with a new set of features.

<h3 id="set_shape"><code>set_shape</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_piece.py#L277-L279">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_shape(
    new_shape: ShapeLike
) -> 'GraphPieceBase'
</code></pre>

Deprecated. Use `with_shape()`.

<h3 id="with_indices_dtype"><code>with_indices_dtype</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_piece.py#L308-L321">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_indices_dtype(
    dtype: tf.dtypes.DType
) -> 'GraphPieceBase'
</code></pre>

Returns a copy of this piece with the given indices dtype.

<h3 id="with_row_splits_dtype"><code>with_row_splits_dtype</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_piece.py#L347-L360">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_row_splits_dtype(
    dtype: tf.dtypes.DType
) -> 'GraphPieceBase'
</code></pre>

Returns a copy of this piece with the given row splits dtype.

<h3 id="with_shape"><code>with_shape</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_piece.py#L281-L295">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_shape(
    new_shape: ShapeLike
) -> 'GraphPieceBase'
</code></pre>

Enforce the common prefix shape on all the contained features.

<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L53-L55">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    feature_name: FieldName
) -> Field
</code></pre>

Indexing operator `[]` to access feature values by their name.
