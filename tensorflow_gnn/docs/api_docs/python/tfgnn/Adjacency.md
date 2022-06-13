description: Stores how edges connect pairs of nodes from source and target node sets.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfgnn.Adjacency" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_indices"/>
<meta itemprop="property" content="get_indices_dict"/>
<meta itemprop="property" content="node_set_name"/>
<meta itemprop="property" content="set_shape"/>
</div>

# tfgnn.Adjacency

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L245-L337">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Stores how edges connect pairs of nodes from source and target node sets.

Inherits From: [`HyperAdjacency`](../tfgnn/HyperAdjacency.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.Adjacency(
    data: Data, spec: 'GraphPieceSpecBase', validate: bool = False
)
</code></pre>



<!-- Placeholder for "Used in" -->

Each hyper-edge connect one node from the source node set with one node from
the target node sets. The source and target node sets could be the same.
The adjacency information is a pair of integer tensors containing indices of
nodes in source and target node sets. Those tensors are indexed by
edges, have the same type spec and shape of `[*graph_shape, num_edges]`,
where `num_edges` is the number of edges in the edge set (could be potentially
ragged). The index tensors are of `tf.Tensor` type if `num_edges` is not
`None` or `graph_shape.rank = 0` and of`tf.RaggedTensor` type otherwise.

The Adjacency is a composite tensor and a special case of tfgnn.HyperAdjacency
class with <a href="../tfgnn.md#SOURCE"><code>tfgnn.SOURCE</code></a> and <a href="../tfgnn.md#TARGET"><code>tfgnn.TARGET</code></a> node tags used for the source and
target nodes correspondingly.

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
The indices of source nodes.
</td>
</tr><tr>
<td>
`source_name`
</td>
<td>
The node set name of source nodes.
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
The indices of target nodes.
</td>
</tr><tr>
<td>
`target_name`
</td>
<td>
The node set name of target nodes.
</td>
</tr>
</table>



## Methods

<h3 id="from_indices"><code>from_indices</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L263-L307">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_indices(
    source: <a href="../tfgnn/Field.md"><code>tfgnn.Field</code></a>,
    target: <a href="../tfgnn/Field.md"><code>tfgnn.Field</code></a>,
    *_,
    validate: bool = True
) -> 'Adjacency'
</code></pre>

Constructs a new instance from the `source` and `target` node indices.


#### Example 1:



```python
# Single graph (rank is 0). Connects pairs of nodes (a[0], b[2]),
# (a[1], b[1]), (a[2], b[0]) from node sets a and b.
tfgnn.Adjacency.from_indices(('a', [0, 1, 2]),
                             ('b', [2, 1, 0]))
```

#### Example 2:



```python
# Batch of two graphs (rank is 1). Connects pairs of nodes in
# graph 0: (a[0], b[2]), (a[1], b[1]); graph 1: (a[2], b[0]).
tfgnn.Adjacency.from_indices(('a', tf.ragged.constant([[0, 1], [2]])),
                             ('b', tf.ragged.constant([[2, 1], [0]])))
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`source`
</td>
<td>
The tuple of node set name and nodes index integer tensor. The
index must have shape of `[*graph_shape, num_edges]`, where `num_edges`
is the number of edges in each graph (could be ragged). It has
`tf.Tensor` type if `num_edges` is not `None` or `graph_shape.rank = 0`
and `tf.RaggedTensor` type otherwise.
</td>
</tr><tr>
<td>
`target`
</td>
<td>
Like `source` field, but for target edge endpoint. Index tensor
must have the same type spec as for the `source`.
</td>
</tr><tr>
<td>
`validate`
</td>
<td>
If `True`, checks that source and target indices have the same
type spec.
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

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L134-L141">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_indices_dict() -> Dict[IncidentNodeTag, Tuple[NodeSetName, Field]]
</code></pre>

Returns copy of indices as a dictionary.


<h3 id="node_set_name"><code>node_set_name</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L130-L132">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>node_set_name(
    node_set_tag: IncidentNodeTag
) -> NodeSetName
</code></pre>

Returns a node set name for the given node set tag.


<h3 id="set_shape"><code>set_shape</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_piece.py#L290-L296">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_shape(
    new_shape: ShapeLike
) -> 'GraphPieceSpecBase'
</code></pre>

Enforce the common prefix shape on all the contained features.


<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/adjacency.py#L126-L128">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    node_set_tag: IncidentNodeTag
) -> <a href="../tfgnn/Field.md"><code>tfgnn.Field</code></a>
</code></pre>

Returns an index tensor for the given node set tag.




