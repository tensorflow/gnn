description: A type spec for <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfgnn.GraphTensorSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="from_piece_specs"/>
<meta itemprop="property" content="from_value"/>
<meta itemprop="property" content="is_compatible_with"/>
<meta itemprop="property" content="is_subtype_of"/>
<meta itemprop="property" content="most_specific_common_supertype"/>
<meta itemprop="property" content="most_specific_compatible_type"/>
</div>

# tfgnn.GraphTensorSpec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L1096-L1161">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A type spec for <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.GraphTensorSpec(
    data_spec: DataSpec,
    shape: tf.TensorShape,
    indices_dtype: tf.dtypes.DType,
    metadata: Metadata = None,
    validate: bool = False
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`context_spec`
</td>
<td>
The graph context type spec.
</td>
</tr><tr>
<td>
`edge_sets_spec`
</td>
<td>
A read-only mapping form edge set name to the edge set type spec.
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
`node_sets_spec`
</td>
<td>
A read-only mapping form node set name to the node set type spec.
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
`total_num_components`
</td>
<td>
The total number of graph components if known.
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

<h3 id="from_piece_specs"><code>from_piece_specs</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L1100-L1130">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_piece_specs(
    context_spec: Optional[<a href="../tfgnn/ContextSpec.md"><code>tfgnn.ContextSpec</code></a>] = None,
    node_sets_spec: Optional[Mapping[NodeSetName, NodeSetSpec]] = None,
    edge_sets_spec: Optional[Mapping[EdgeSetName, EdgeSetSpec]] = None
) -> 'GraphTensorSpec'
</code></pre>

The counterpart of <a href="../tfgnn/GraphTensor.md#from_pieces"><code>GraphTensor.from_pieces</code></a> for pieces type specs.


<h3 id="from_value"><code>from_value</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_piece.py#L481-L484">View source</a>

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

Prefer using "is_subtype_of" and "most_specific_common_supertype" wherever
possible.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`spec_or_value`
</td>
<td>
A TypeSpec or TypeSpec associated value to compare against.
</td>
</tr>
</table>



<h3 id="is_subtype_of"><code>is_subtype_of</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_subtype_of(
    other: trace.TraceType
) -> bool
</code></pre>

Returns True if `self` is a subtype of `other`.

Implements the tf.types.experimental.func.TraceType interface.

If not overridden by a subclass, the default behavior is to assume the
TypeSpec is covariant upon attributes that implement TraceType and
invariant upon rest of the attributes as well as the structure and type
of the TypeSpec.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
A TraceType object.
</td>
</tr>
</table>



<h3 id="most_specific_common_supertype"><code>most_specific_common_supertype</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>most_specific_common_supertype(
    others: Sequence[trace.TraceType]
) -> Optional['TypeSpec']
</code></pre>

Returns the most specific supertype TypeSpec  of `self` and `others`.

Implements the tf.types.experimental.func.TraceType interface.

If not overridden by a subclass, the default behavior is to assume the
TypeSpec is covariant upon attributes that implement TraceType and
invariant upon rest of the attributes as well as the structure and type
of the TypeSpec.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`others`
</td>
<td>
A sequence of TraceTypes.
</td>
</tr>
</table>



<h3 id="most_specific_compatible_type"><code>most_specific_compatible_type</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>most_specific_compatible_type(
    other: 'TypeSpec'
) -> 'TypeSpec'
</code></pre>

Returns the most specific TypeSpec compatible with `self` and `other`. (deprecated)

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use most_specific_common_supertype instead.

Deprecated. Please use `most_specific_common_supertype` instead.
Do not override this function.

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


<h3 id="__ne__"><code>__ne__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
) -> bool
</code></pre>

Return self!=value.




