# tfgnn.HyperAdjacencySpec

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L248-L375">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

A type spec for
<a href="../tfgnn/HyperAdjacency.md"><code>tfgnn.HyperAdjacency</code></a>.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.HyperAdjacencySpec(
    data_spec: DataSpec,
    shape: tf.TensorShape,
    indices_dtype: tf.dtypes.DType,
    row_splits_dtype: tf.dtypes.DType,
    metadata: Metadata = None,
    check_consistent_indices_dtype: bool = False,
    check_consistent_row_splits_dtype: bool = False
)
</code></pre>

<!-- Placeholder for "Used in" -->
<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr> <td> <code>indices_dtype</code><a id="indices_dtype"></a> </td> <td> The
dtype for graph items indexing. One of <code>tf.int32</code> or
<code>tf.int64</code>. </td> </tr><tr> <td> <code>rank</code><a id="rank"></a>
</td> <td> The rank of the GraphPiece. Guaranteed not to be <code>None</code>.
</td> </tr><tr> <td> <code>row_splits_dtype</code><a id="row_splits_dtype"></a>
</td> <td> The dtype for ragged row partitions. One of <code>tf.int32</code> or
<code>tf.int64</code>. </td> </tr><tr> <td> <code>shape</code><a id="shape"></a>
</td> <td> A possibly-partial shape specification of the GraphPiece.

The returned <code>TensorShape</code> is guaranteed to have a known rank, but
the individual dimension sizes may be unknown. </td> </tr><tr> <td>
<code>total_size</code><a id="total_size"></a> </td> <td> The total number of
edges if known. </td> </tr><tr> <td>
<code>value_type</code><a id="value_type"></a> </td> <td> The Python type for
values that are compatible with this TypeSpec.

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
<code>proto</code>
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

<h3 id="from_incident_node_sets"><code>from_incident_node_sets</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L252-L306">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_incident_node_sets(
    incident_node_sets: Mapping[IncidentNodeTag, NodeSetName],
    index_spec: <a href="../tfgnn/FieldSpec.md"><code>tfgnn.FieldSpec</code></a> = tf.TensorSpec((None,), const.default_indices_dtype)
) -> 'HyperAdjacencySpec'
</code></pre>

Constructs a new instance from the `incident_node_sets`.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<code>incident_node_sets</code>
</td>
<td>
A mapping from incident node tags to node set names.
</td>
</tr><tr>
<td>
<code>index_spec</code>
</td>
<td>
type spec for all index tensors of shape
<code>[*graph_shape, num_edges]</code>, where <code>num_edges</code> is the number of edges in
each graph. If <code>num_edges</code> is not <code>None</code> or <code>graph_shape.rank = 0</code> the
spec must be of <code>tf.TensorSpec</code> type and of <code>tf.RaggedTensorSpec</code> type
otherwise.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <code>HyperAdjacencySpec</code> TypeSpec.
</td>
</tr>

</table>

<h3 id="from_value"><code>from_value</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_piece.py#L672-L675">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_value(
    value: GraphPieceBase
)
</code></pre>

Extension Types API: Factory method.

<h3 id="get_index_specs_dict"><code>get_index_specs_dict</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L316-L323">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_index_specs_dict() -> Dict[IncidentNodeTag, Tuple[NodeSetName, FieldSpec]]
</code></pre>

Returns copy of indices type specs as a dictionary.

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
<code>spec_or_value</code>
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

If not overridden by a subclass, the default behavior is to assume the TypeSpec
is covariant upon attributes that implement TraceType and invariant upon rest of
the attributes as well as the structure and type of the TypeSpec.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<code>other</code>
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

Returns the most specific supertype TypeSpec of `self` and `others`.

Implements the tf.types.experimental.func.TraceType interface.

If not overridden by a subclass, the default behavior is to assume the TypeSpec
is covariant upon attributes that implement TraceType and invariant upon rest of
the attributes as well as the structure and type of the TypeSpec.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<code>others</code>
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

Returns the most specific TypeSpec compatible with `self` and `other`.
(deprecated)

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating: Use most_specific_common_supertype instead.

Deprecated. Please use `most_specific_common_supertype` instead. Do not override
this function.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<code>other</code>
</td>
<td>
A <code>TypeSpec</code>.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
<code>ValueError</code>
</td>
<td>
If there is no TypeSpec that is compatible with both <code>self</code>
and <code>other</code>.
</td>
</tr>
</table>

<h3 id="node_set_name"><code>node_set_name</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L325-L327">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>node_set_name(
    node_set_tag: IncidentNodeTag
) -> NodeSetName
</code></pre>

Returns a node set name for the given node set tag.

<h3 id="relax"><code>relax</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L336-L366">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>relax(
    *, num_edges: bool = False
) -> 'HyperAdjacencySpec'
</code></pre>

Allows variable number of graph edges.

Calling with all default parameters keeps the spec unchanged.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<code>num_edges</code>
</td>
<td>
if True, allows a variable number of edges in each edge set.
If False, returns spec unchanged.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Relaxed compatible spec.
</td>
</tr>

</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
<code>ValueError</code>
</td>
<td>
if adjacency is not scalar (rank > 0).
</td>
</tr>
</table>

<h3 id="with_indices_dtype"><code>with_indices_dtype</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_piece.py#L596-L608">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_indices_dtype(
    dtype: tf.dtypes.DType
) -> 'GraphPieceSpecBase'
</code></pre>

Returns a copy of this piece spec with the given indices dtype.

<h3 id="with_row_splits_dtype"><code>with_row_splits_dtype</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_piece.py#L637-L651">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_row_splits_dtype(
    dtype: tf.dtypes.DType
) -> 'GraphPieceSpecBase'
</code></pre>

Returns a copy of this piece spec with the given row splits dtype.

<h3 id="with_shape"><code>with_shape</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_piece.py#L570-L584">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_shape(
    new_shape: ShapeLike
) -> 'GraphPieceSpecBase'
</code></pre>

Enforce the common prefix shape on all the contained features.

<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
) -> bool
</code></pre>

Return self==value.

<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L312-L314">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    node_set_tag: IncidentNodeTag
) -> <a href="../tfgnn/FieldSpec.md"><code>tfgnn.FieldSpec</code></a>
</code></pre>

Returns an index tensor type spec for the given node set tag.

<h3 id="__ne__"><code>__ne__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
) -> bool
</code></pre>

Return self!=value.
