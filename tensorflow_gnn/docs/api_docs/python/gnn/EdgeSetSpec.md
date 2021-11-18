description: A type spec for the features of a single edge set.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gnn.EdgeSetSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="from_field_specs"/>
<meta itemprop="property" content="from_value"/>
<meta itemprop="property" content="is_compatible_with"/>
<meta itemprop="property" content="most_specific_compatible_type"/>
</div>

# gnn.EdgeSetSpec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L391-L429">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A type spec for the features of a single edge set.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gnn.EdgeSetSpec(
    data_spec: DataSpec,
    shape: tf.TensorShape,
    indices_dtype: tf.dtypes.DType,
    metadata: Metadata = None,
    validate: bool = False
)
</code></pre>



<!-- Placeholder for "Used in" -->

This class is a type descriptor for the shapes of the features associated with
a graph's edge set from a `GraphTensor` instance. This graph piece stores
features that belong to an edge set, a `sizes` tensor with the number of edges
in each graph component and an `adjacency` `GraphPiece` tensor describing how
this edge set connects node sets (see adjacency.py).

(This graph piece does not use any metadata fields.)



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`adjacency_spec`
</td>
<td>
A type spec for the adjacency indices container.
</td>
</tr><tr>
<td>
`features_spec`
</td>
<td>
A mapping of feature name to feature specs.
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
The rank of the GraphPiece. Guaranteed not to be `None`.
</td>
</tr><tr>
<td>
`shape`
</td>
<td>
A possibly-partial shape specification of the GraphPiece.

The returned `TensorShape` is guaranteed to have a known rank, but the
individual dimension sizes may be unknown.
</td>
</tr><tr>
<td>
`sizes_spec`
</td>
<td>
A type spec for the sizes that provides num. elements per component.
</td>
</tr><tr>
<td>
`total_num_components`
</td>
<td>
The total number of graph components across dimensions if known.
</td>
</tr><tr>
<td>
`total_size`
</td>
<td>
Returns the total number of edges across dimensions if known.
</td>
</tr><tr>
<td>
`value_type`
</td>
<td>
The Python type for values that are compatible with this TypeSpec.

In particular, all values that are compatible with this TypeSpec must be an
instance of this type.
</td>
</tr>
</table>



## Methods

<h3 id="from_field_specs"><code>from_field_specs</code></h3>

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L403-L415">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_field_specs(
    *,
    features_spec: Optional[<a href="../gnn/FieldsSpec.md"><code>gnn.FieldsSpec</code></a>] = None,
    sizes_spec: <a href="../gnn/FieldSpec.md"><code>gnn.FieldSpec</code></a>,
    adjacency_spec: AdjacencySpec
) -> "EdgeSetSpec"
</code></pre>

Counterpart of <a href="../gnn/EdgeSet.md#from_fields"><code>EdgeSet.from_fields()</code></a> for values type specs.


<h3 id="from_value"><code>from_value</code></h3>

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_piece.py#L481-L484">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_value(
    value: GraphPieceBase
)
</code></pre>

Extension Types API: Factory method.


<h3 id="is_compatible_with"><code>is_compatible_with</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_compatible_with(
    spec_or_value
)
</code></pre>

Returns true if `spec_or_value` is compatible with this TypeSpec.


<h3 id="most_specific_compatible_type"><code>most_specific_compatible_type</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>most_specific_compatible_type(
    other: "TypeSpec"
) -> "TypeSpec"
</code></pre>

Returns the most specific TypeSpec compatible with `self` and `other`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
A `TypeSpec`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If there is no TypeSpec that is compatible with both `self`
and `other`.
</td>
</tr>
</table>



<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
) -> bool
</code></pre>

Return self==value.


<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L58-L59">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    feature_name: FieldName
) -> <a href="../gnn/FieldSpec.md"><code>gnn.FieldSpec</code></a>
</code></pre>




<h3 id="__ne__"><code>__ne__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
) -> bool
</code></pre>

Return self!=value.




