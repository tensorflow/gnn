# tfgnn.ContextSpec

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L331-L398">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

A type spec for <a href="../tfgnn/Context.md"><code>tfgnn.Context</code></a>.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.ContextSpec(
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

<tr> <td> `features_spec`<a id="features_spec"></a> </td> <td> A read-only
mapping of feature name to feature spec. </td> </tr><tr> <td>
`indices_dtype`<a id="indices_dtype"></a> </td> <td> The integer type to
represent ragged splits. </td> </tr><tr> <td> `rank`<a id="rank"></a> </td> <td>
The rank of the GraphPiece. Guaranteed not to be `None`. </td> </tr><tr> <td>
`shape`<a id="shape"></a> </td> <td> A possibly-partial shape specification of
the GraphPiece.

The returned `TensorShape` is guaranteed to have a known rank, but the
individual dimension sizes may be unknown. </td> </tr><tr> <td>
`sizes_spec`<a id="sizes_spec"></a> </td> <td> The type spec for the sizes that
provides num. elements per component. </td> </tr><tr> <td>
`total_num_components`<a id="total_num_components"></a> </td> <td> The total
number of graph components if known. </td> </tr><tr> <td>
`total_size`<a id="total_size"></a> </td> <td> The total number of graph items
if known. </td> </tr><tr> <td> `value_type`<a id="value_type"></a> </td> <td>
The Python type for values that are compatible with this TypeSpec.

In particular, all values that are compatible with this TypeSpec must be an
instance of this type.
</td>
</tr>
</table>



## Methods

<h3 id="experimental_as_proto"><code>experimental_as_proto</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>experimental_as_proto() -> struct_pb2.TypeSpecProto
</code></pre>

Returns a proto representation of the TypeSpec instance.

Do NOT override for custom non-TF types.

<h3 id="experimental_from_proto"><code>experimental_from_proto</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>experimental_from_proto(
    proto: struct_pb2.TypeSpecProto
) -> 'TypeSpec'
</code></pre>

Returns a TypeSpec instance based on the serialized proto.

Do NOT override for custom non-TF types.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`proto`
</td>
<td>
Proto generated using 'experimental_as_proto'.
</td>
</tr>
</table>

<h3 id="experimental_type_proto"><code>experimental_type_proto</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>experimental_type_proto() -> Type[struct_pb2.TypeSpecProto]
</code></pre>

Returns the type of proto associated with TypeSpec serialization.

Do NOT override for custom non-TF types.

<h3 id="from_field_specs"><code>from_field_specs</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L335-L371">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_field_specs(
    *,
    features_spec: Optional[<a href="../tfgnn/FieldsSpec.md"><code>tfgnn.FieldsSpec</code></a>] = None,
    sizes_spec: Optional[FieldSpec] = None,
    shape: ShapeLike = tf.TensorShape([]),
    indices_dtype: tf.dtypes.DType = const.default_indices_dtype
) -> 'ContextSpec'
</code></pre>

The counterpart of <a href="../tfgnn/Context.md#from_fields"><code>Context.from_fields()</code></a> for field type specs.


<h3 id="from_value"><code>from_value</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_piece.py#L495-L498">View
source</a>

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

<h3 id="relax"><code>relax</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L377-L398">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>relax(
    *, num_components: bool = False
) -> 'ContextSpec'
</code></pre>

Allows variable number of graph components.

Calling with all default parameters keeps the spec unchanged.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`num_components`
</td>
<td>
if True, allows variable number of graph components by
setting the outermost sizes dimension to `None`.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Relaxed compatible context spec.
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
if contex is not scalar (rank > 0).
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

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L165-L166">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    feature_name: FieldName
) -> <a href="../tfgnn/FieldSpec.md"><code>tfgnn.FieldSpec</code></a>
</code></pre>




<h3 id="__ne__"><code>__ne__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
) -> bool
</code></pre>

Return self!=value.




