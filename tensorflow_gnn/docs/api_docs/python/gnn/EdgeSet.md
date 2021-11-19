description: A container for the features of a single edge set.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gnn.EdgeSet" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_fields"/>
<meta itemprop="property" content="get_features_dict"/>
<meta itemprop="property" content="replace_features"/>
<meta itemprop="property" content="set_shape"/>
</div>

# gnn.EdgeSet

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L342-L387">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A container for the features of a single edge set.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gnn.EdgeSet(
    data: Data,
    spec: "GraphPieceSpecBase",
    validate: bool = False
)
</code></pre>



<!-- Placeholder for "Used in" -->

This class is a container for the shapes of the features associated with a
graph's edge set from a `GraphTensor` instance. This graph piece stores
features that belong to an edge set, a `sizes` tensor with the number of edges
in each graph component and an `adjacency` `GraphPiece` tensor describing how
this edge set connects node sets (see adjacency.py).

(This graph piece does not use any metadata fields.)

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`data`
</td>
<td>
Nest of Field or subclasses of GraphPieceBase.
</td>
</tr><tr>
<td>
`spec`
</td>
<td>
A subclass of GraphPieceSpecBase with a `_data_spec` that matches
`data`.
</td>
</tr><tr>
<td>
`validate`
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

<tr>
<td>
`adjacency`
</td>
<td>

</td>
</tr><tr>
<td>
`features`
</td>
<td>
Read-only view for features.
</td>
</tr><tr>
<td>
`indices_dtype`
</td>
<td>
The integer type to represent ragged splits.
</td>
</tr><tr>
<td>
`rank`
</td>
<td>
The rank of this Tensor. Guaranteed not to be `None`.
</td>
</tr><tr>
<td>
`shape`
</td>
<td>
A possibly-partial shape specification for this Tensor.

The returned `TensorShape` is guaranteed to have a known rank, but the
individual dimension sizes may be unknown.
</td>
</tr><tr>
<td>
`sizes`
</td>
<td>
Tensor with a number of elements in each graph component.
</td>
</tr><tr>
<td>
`spec`
</td>
<td>
The public type specification of this tensor.
</td>
</tr><tr>
<td>
`total_size`
</td>
<td>
Returns the total number of elements across dimensions.
</td>
</tr>
</table>



## Methods

<h3 id="from_fields"><code>from_fields</code></h3>

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L356-L379">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_fields(
    *,
    features: Optional[<a href="../gnn/Fields.md"><code>gnn.Fields</code></a>] = None,
    sizes: <a href="../gnn/Field.md"><code>gnn.Field</code></a>,
    adjacency: Adjacency
) -> "EdgeSet"
</code></pre>

Constructs a new instance from edge set fields.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`features`
</td>
<td>
mapping from feature names to feature Tensors or RaggedTensors.
All feature tensors must have shape = graph_shape + [num_edges] +
feature_shape, where num_edges is the number of edges in the edge set
(could be ragged) and feature_shape are feature-specific inner
dimensions.
</td>
</tr><tr>
<td>
`sizes`
</td>
<td>
the number of edges in each graph component. Has shape =
graph_shape + [num_components], where num_components is the number of
graph components (could be ragged).
</td>
</tr><tr>
<td>
`adjacency`
</td>
<td>
one of supported adjacency types (see adjacency.py).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `EdgeSet` tensor.
</td>
</tr>

</table>



<h3 id="get_features_dict"><code>get_features_dict</code></h3>

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L45-L47">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_features_dict() -> Dict[FieldName, Field]
</code></pre>

Returns features copy as a dictionary.


<h3 id="replace_features"><code>replace_features</code></h3>

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L206-L212">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>replace_features(
    features: <a href="../gnn/Fields.md"><code>gnn.Fields</code></a>
) -> "_NodeOrEdgeSet"
</code></pre>

Returns a new instance with a new set of features.


<h3 id="set_shape"><code>set_shape</code></h3>

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_piece.py#L295-L301">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_shape(
    new_shape: ShapeLike
) -> "GraphPieceSpecBase"
</code></pre>

Enforce the common prefix shape on all the contained features.


<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L36-L38">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    feature_name: FieldName
) -> <a href="../gnn/Field.md"><code>gnn.Field</code></a>
</code></pre>

Indexing operator `[]` to access feature values by their name.




