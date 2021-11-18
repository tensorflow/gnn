description: TypeSpec for HyperAdjacency.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gnn.HyperAdjacencySpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="from_incident_node_sets"/>
<meta itemprop="property" content="from_value"/>
<meta itemprop="property" content="get_index_specs_dict"/>
<meta itemprop="property" content="is_compatible_with"/>
<meta itemprop="property" content="most_specific_compatible_type"/>
<meta itemprop="property" content="node_set_name"/>
</div>

# gnn.HyperAdjacencySpec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L150-L217">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



TypeSpec for HyperAdjacency.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gnn.HyperAdjacencySpec(
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

<h3 id="from_incident_node_sets"><code>from_incident_node_sets</code></h3>

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L153-L189">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_incident_node_sets(
    incident_node_sets: Mapping[IncidentNodeTag, NodeSetName],
    index_spec: <a href="../gnn/FieldSpec.md"><code>gnn.FieldSpec</code></a> = tf.TensorSpec((None,), const.default_indices_dtype)
) -> "HyperAdjacencySpec"
</code></pre>

Constructs a new instance from the `incident_node_sets`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`incident_node_sets`
</td>
<td>
mapping from incident node tags to node set names.
</td>
</tr><tr>
<td>
`index_spec`
</td>
<td>
type spec for all index tensors. Its shape must be graph_shape
+ [num_edges], where num_edges is the number of edges in each graph. If
graph_shape.rank > 0 and num_edges has variable size, the spec should be
an instance of tf.RaggedTensorSpec.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `HyperAdjacencySpec` TypeSpec.
</td>
</tr>

</table>



<h3 id="from_value"><code>from_value</code></h3>

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_piece.py#L481-L484">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_value(
    value: GraphPieceBase
)
</code></pre>

Extension Types API: Factory method.


<h3 id="get_index_specs_dict"><code>get_index_specs_dict</code></h3>

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L199-L206">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_index_specs_dict() -> Dict[IncidentNodeTag, Tuple[NodeSetName, FieldSpec]]
</code></pre>

Returns copy of index type specs.


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



<h3 id="node_set_name"><code>node_set_name</code></h3>

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L208-L210">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>node_set_name(
    node_set_tag: IncidentNodeTag
) -> NodeSetName
</code></pre>

Returns node set name for a given node set tag.


<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
) -> bool
</code></pre>

Return self==value.


<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L195-L197">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    node_set_tag: IncidentNodeTag
) -> <a href="../gnn/FieldSpec.md"><code>gnn.FieldSpec</code></a>
</code></pre>

Returns index tensor type spec for a given node set tag.


<h3 id="__ne__"><code>__ne__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
) -> bool
</code></pre>

Return self!=value.




