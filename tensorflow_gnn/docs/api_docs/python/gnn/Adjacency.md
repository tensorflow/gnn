description: Stores simple binary edges with a source and target.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gnn.Adjacency" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_indices"/>
<meta itemprop="property" content="get_indices_dict"/>
<meta itemprop="property" content="node_set_name"/>
<meta itemprop="property" content="set_shape"/>
</div>

# gnn.Adjacency

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L220-L288">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Stores simple binary edges with a source and target.

Inherits From: [`HyperAdjacency`](../gnn/HyperAdjacency.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gnn.Adjacency(
    data: Data,
    spec: "GraphPieceSpecBase",
    validate: bool = False
)
</code></pre>



<!-- Placeholder for "Used in" -->

Node adjacency is represented as mapping of source and target edge endpoints
to pairs of (node set names, index tensors) into them. All index tensors must
agree in their type (`tf.Tensor` or `tf.RaggedTensor`), integer dtype, and
shape. Corresponding values are the indices of nodes that belong to the same
edge.

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
`source`
</td>
<td>

</td>
</tr><tr>
<td>
`source_name`
</td>
<td>
Returns the node set name for source nodes.
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
`target`
</td>
<td>

</td>
</tr><tr>
<td>
`target_name`
</td>
<td>
Returns the node set name for target nodes.
</td>
</tr>
</table>



## Methods

<h3 id="from_indices"><code>from_indices</code></h3>

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L230-L266">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_indices(
    source: <a href="../gnn/Field.md"><code>gnn.Field</code></a>,
    target: <a href="../gnn/Field.md"><code>gnn.Field</code></a>,
    *,
    validate: bool = True
) -> "Adjacency"
</code></pre>

Constructs a new instance from the `indices` tensors.

Example 1. Single graph (rank is 0). Connects pairs of nodes (a.0, b.2),
(a.1, b.1), (a.2, b.0) from node sets a and b:

    gnn.Adjacency.from_indices(('a', [0, 1, 2]),
                               ('b', [2, 1, 0]))

Example 2. Batch of two graphs (rank is 1). Connects pairs of nodes
graph 0: (a.0, b.2), (a.1, b.1); graph 1: (a.2, b.0):

    gnn.Adjacency.from_indices(('a', tf.ragged.constant([[0, 1], [2]])),
                               ('b', tf.ragged.constant([[2, 1], [0]])))

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`source`
</td>
<td>
Tuple of (node set name, integer Tensors or RaggedTensors with the
indices of nodes in the respective node set). Must have shape =
graph_shape + [num_edges], where num_edges is a number of edges in each
graph. If graph_shape.rank > 0 and num_edges has variable size, the
tensors are ragged.
</td>
</tr><tr>
<td>
`target`
</td>
<td>
Like `source` field, but for target edge endpoint.
</td>
</tr><tr>
<td>
`validate`
</td>
<td>
if set, checks that node indices have the same shapes.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An `Adjacency` tensor with a shape and an indices_dtype being inferred
from the `indices` values.
</td>
</tr>

</table>



<h3 id="get_indices_dict"><code>get_indices_dict</code></h3>

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L113-L120">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_indices_dict() -> Dict[IncidentNodeTag, Tuple[NodeSetName, Field]]
</code></pre>

Returns copy of indices tensor.


<h3 id="node_set_name"><code>node_set_name</code></h3>

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L109-L111">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>node_set_name(
    node_set_tag: IncidentNodeTag
) -> NodeSetName
</code></pre>

Returns node set name for a given node set tag.


<h3 id="set_shape"><code>set_shape</code></h3>

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_piece.py#L295-L301">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_shape(
    new_shape: ShapeLike
) -> "GraphPieceSpecBase"
</code></pre>

Enforce the common prefix shape on all the contained features.


<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L105-L107">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    node_set_tag: IncidentNodeTag
) -> <a href="../gnn/Field.md"><code>gnn.Field</code></a>
</code></pre>

Returns index tensor for a given node set tag.




