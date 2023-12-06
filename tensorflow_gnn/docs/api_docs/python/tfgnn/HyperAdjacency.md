# tfgnn.HyperAdjacency

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L34-L245">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Stores how (hyper-)edges connect tuples of nodes from incident node sets.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.HyperAdjacency(
    data: Data, spec: 'GraphPieceSpecBase'
)
</code></pre>

<!-- Placeholder for "Used in" -->

The incident node sets in the hyper-adjacency are referenced by a unique integer
identifier called the node set tag. (For a non-hyper graph, it is conventional
to use the integers <a href="../tfgnn.md#SOURCE"><code>tfgnn.SOURCE</code></a>
and <a href="../tfgnn.md#TARGET"><code>tfgnn.TARGET</code></a>.) This allows the
hyper-adjacency to connect nodes from the same or different node sets. Each
hyper-edge connects a fixed number of nodes, one node from each incident node
set. The adjacency information is stored as a mapping from the node set tags to
integer tensors containing indices of nodes in corresponding node sets. Those
tensors are indexed by edges. All index tensors have the same type spec and
shape of `[*graph_shape, num_edges]`, where `num_edges` is the number of edges
in the edge set (could be potentially ragged). The index tensors are of
`tf.Tensor` type if `num_edges` is not `None` or `graph_shape.rank = 0` and of
`tf.RaggedTensor` type otherwise.

The HyperAdjacency is a composite tensor.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>data</code><a id="data"></a>
</td>
<td>
Nest of Field or subclasses of GraphPieceBase.
</td>
</tr><tr>
<td>
<code>spec</code><a id="spec"></a>
</td>
<td>
A subclass of GraphPieceSpecBase with a <code>_data_spec</code> that matches
<code>data</code>.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr> <td> <code>indices_dtype</code><a id="indices_dtype"></a> </td> <td> The
dtype for graph items indexing. One of <code>tf.int32</code> or
<code>tf.int64</code>. </td> </tr><tr> <td> <code>rank</code><a id="rank"></a>
</td> <td> The rank of this Tensor. Guaranteed not to be <code>None</code>.
</td> </tr><tr> <td> <code>row_splits_dtype</code><a id="row_splits_dtype"></a>
</td> <td> The dtype for ragged row partitions. One of <code>tf.int32</code> or
<code>tf.int64</code>. </td> </tr><tr> <td> <code>shape</code><a id="shape"></a>
</td> <td> A possibly-partial shape specification for this Tensor.

The returned <code>TensorShape</code> is guaranteed to have a known rank, but the
individual dimension sizes may be unknown.
</td>
</tr><tr>
<td>
<code>spec</code><a id="spec"></a>
</td>
<td>
The public type specification of this tensor.
</td>
</tr>
</table>

## Methods

<h3 id="from_indices"><code>from_indices</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L54-L111">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_indices(
    indices: Indices, *_, validate: Optional[bool] = None
) -> 'HyperAdjacency'
</code></pre>

Constructs a new instance from the `indices` tensors.

#### Example 1:

```python
# Single graph (rank is 0). Connects pairs of nodes (a[0], b[2]),
# (a[1], b[1]), (a[2], b[0]) from node sets a and b.
tfgnn.HyperAdjacency.from_indices({
    tfgnn.SOURCE: ('a', [0, 1, 2]),
    tfgnn.TARGET: ('b', [2, 1, 0])
})
```

#### Example 2:

```python
# Single hypergraph (rank is 0). Connects triplets of nodes
# (a[0], b[2], c[1]), (a[1], b[1], c[0]) from the node sets a, b and c.
tfgnn.HyperAdjacency.from_indices({
    0: ('a', [0, 1]),
    1: ('b', [2, 1]),
    2: ('c', [1, 0]),
})
```

#### Example 3:

```python
# Batch of two graphs (rank is 1). Connects pairs of nodes in
# graph 0: (a[0], b[2]), (a[1], b[1]); graph 1: (a[2], b[0]).
tfgnn.HyperAdjacency.from_indices({
    tfgnn.SOURCE: ('a', tf.ragged.constant([[0, 1], [2]])),
    tfgnn.TARGET: ('b', tf.ragged.constant([[2, 1], [0]])),
})
```

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<code>indices</code>
</td>
<td>
A mapping from node tags to 2-tuples of node set name and node
index tensor. The index tensors must have the same type spec and shape
of <code>[*graph_shape, num_edges]</code>, where <code>num_edges</code> is the number of edges
in each graph (could be ragged). The index tensors are of <code>tf.Tensor</code>
type if <code>num_edges</code> is not <code>None</code> or <code>graph_shape.rank = 0</code> and of
<code>tf.RaggedTensor</code> type otherwise.
</td>
</tr><tr>
<td>
<code>validate</code>
</td>
<td>
If <code>True</code>, checks that node indices have the same type spec.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <code>HyperAdjacency</code> tensor with its <code>shape</code> and <code>indices_dtype</code> being
inferred from the passed <code>indices</code> values.
</td>
</tr>

</table>

<h3 id="get_indices_dict"><code>get_indices_dict</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L175-L182">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_indices_dict() -> Dict[IncidentNodeTag, Tuple[NodeSetName, Field]]
</code></pre>

Returns copy of indices as a dictionary.

<h3 id="node_set_name"><code>node_set_name</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L171-L173">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>node_set_name(
    node_set_tag: IncidentNodeTag
) -> NodeSetName
</code></pre>

Returns a node set name for the given node set tag.

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

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L167-L169">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    node_set_tag: IncidentNodeTag
) -> <a href="../tfgnn/Field.md"><code>tfgnn.Field</code></a>
</code></pre>

Returns an index tensor for the given node set tag.
