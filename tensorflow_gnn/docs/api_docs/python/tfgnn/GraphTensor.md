# tfgnn.GraphTensor

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L885-L1498">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

A composite tensor for heterogeneous directed graphs with features.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.GraphTensor(
    data: Data, spec: 'GraphPieceSpecBase'
)
</code></pre>

<!-- Placeholder for "Used in" -->

A GraphTensor is an immutable container (as any composite tensor) to represent
one or more heterogeneous directed graphs, as defined in the GraphTensor guide,
or even hypergraphs. A GraphTensor consists of NodeSets, EdgeSets and a Context
(collectively known as graph pieces), which are also composite tensors. The
graph pieces consist of fields, which are `tf.Tensor`s and/or `tf.RaggedTensor`s
that store the graph structure (esp. the edges between nodes) and user-defined
features.

In the same way as `tf.Tensor` has numbers as its elements, the elements of
the GraphTensor are graphs. Its `shape` of `[]` describes a scalar (single)
graph, a shape of `[d0]` describes a `d0`-vector of graphs, a shape of
`[d0, d1]` a `d0` x `d1` matrix of graphs, and so on.

RULE: In the shape of a GraphTensor, no dimension except the outermost is
allowed to be `None` (that is, of unknown size).

Each of those graphs in a GraphTensor consists of 0 or more disjoint
(sub-)graphs called graph components. The number of components could vary from
graph to graph or be fixed to a value known statically. On a batched
GraphTensor, one can call the method merge_batch_to_components() to merge all
graphs of the batch into one contiguously indexed graph containing the same
components as the original graph tensor. See the GraphTensor guide for the
typical usage that has motivated this design (going from input graphs with one
component each to a batch of input graphs and on to one merged graph with
multiple components for use in GNN model).

#### Example 1:

```python
# A homogeneous scalar graph tensor with 1 graph component, 10 nodes and 3
# edges. Edges connect nodes 0 and 3, 5 and 7, 9 and 1. There are no features.
tfgnn.GraphTensor.from_pieces(
    node_sets = {
        'node': tfgnn.NodeSet.from_fields(sizes=[10], features={})},
    edge_sets = {
        'edge': tfgnn.EdgeSet.from_fields(
            sizes=[3],
            features={},
            adjacency=tfgnn.Adjacency.from_indices(
                source=('node', [0, 5, 9]),
                target=('node', [3, 7, 1])))})
```

All graph pieces provide a mapping interface to access their features by name
as `graph_piece[feature_name]`. Each graph piece feature has the shape
`[*graph_shape, num_items, *feature_shape]`, where `graph_shape` is the shape
of the GraphTensor, `num_items` is the number of items in a piece (number of
graph components, number of nodes in a node set or edges in an edge set). The
`feature_shape` is the shape of the feature value for each item.

Naturally, the first <a href="../tfgnn/Adjacency.md#rank"><code>GraphTensor.rank</code></a> dimensions of all graph tensor fields
must index the same graphs, the item dimension must correspond to the same
item (graph component, node or edge) within the same graph piece (context,
node set or edge set).

RULE: 'None' always denotes ragged or outermost field dimension. Uniform
dimensions must have a fixed size that is given in the dimension.

In particular this rule implies that if a feature has `tf.Tensor` type its
`feature_shape` must by fully defined.

#### Example 2:

```python
# A scalar graph tensor with edges between authors, papers and their venues
# (journals or conferences). Each venue belongs to one graph component. The
# 1st venue (1980519) has 2 authors and 3 papers.
# The 2nd venue (9756463) has 2 authors and 1 paper.
# The paper 0 is written by authors 0 and 2; paper 1 - by authors 0 and 1;
# paper 2 - by author 2; paper 3 - by author 3.
venues = tfgnn.GraphTensor.from_pieces(
    context=tfgnn.Context.from_fields(
        features={'venue': [1980519, 9756463]}),
    node_sets={
        'author': tfgnn.NodeSet.from_fields(sizes=[2, 2], features={}),
        'paper': tfgnn.NodeSet.from_fields(
            sizes=[3, 1], features={'year': [2018, 2017, 2017, 2022]})},
    edge_sets={
        'is_written': tfgnn.EdgeSet.from_fields(
            sizes=[4, 2],
            features={},
            adjacency=tfgnn.Adjacency.from_indices(
                source=('paper', [0, 1, 1, 0, 2, 3]),
                target=('author', [0, 0, 1, 2, 3, 3])))})
```

The assignment of an item to its graph components is stored as the `sizes`
attribute of the graph piece. Its shape is `[*graph_shape, num_components]` (the
same for all graph pieces). The stored values are number of items in each graph
component.

#### Example 3:

```python
# The year of publication of the first article in each venue from the
# previous example.
papers = venues.node_sets['paper']
years_by_venue = tf.RaggedTensor.from_row_lengths(
    papers['year'], papers.sizes
)
first_paper_year = tf.reduce_min(years_by_venue, -1)  # [2017, 2022]
```

The GraphTensor, as a composite tensor, can be used directly in a
tf.data.Dataset, as an input or output of a Keras Layer or a tf.function,
and so on. As any other tensor, the GraphTensor has an associated type
specification object, a GraphTensorSpec, that holds the `tf.TensorSpec` and
`tf.RaggedTensorSpec` objects for all its fields, plus a bit of metadata such
as graph connectivity (see tfgnn.GraphTensorSpec).

The GraphTensor allows batching of graphs. Batching changes a GraphTensor
instance's shape to `[batch_size, *graph_shape]` and the GraphTensor's rank is
increased by 1. Unbatching removes dimension 0, as if truncating with
`shape[1:]`, and the GraphTensor's rank is decreased by 1. This works naturally
with the batch and unbatch methods of tf.data.Datatset.

RULE: Batching followed by unbatching results in a dataset with equal
GraphTensors as before, except for the last incomplete batch (if batching
used `drop_remainder=True`).

GraphTensor requires that GraphTensor.shape does not contain `None`, except
maybe as the outermost dimension. That means repeated calls to `.batch()` must
set `drop_remainder=True` in all but the last one.

All pieces and its fields are batched together with their GraphTensor so that
shapes of a graph tensor, its pieces and features are all in sync.

RULE: Batching fields with an outermost dimension of `None` turns it into a
ragged dimension of a RaggedTensor. (Note this is only allowed for the items
dimension, not a graph dimension.) In all other cases, the type of the field
(Tensor or RaggedTensor) is preserved.

Graph tensor allows `int64` or `int32` types to index graph items. There are two
types of indices: `indices_dtype` and `row_splits_dtype`. The `indices_dtype` is
used to index itemes within graph pieces (`sizes`) and as a `dtype` of adjacency
indices. The `row_splits_dtype` is a dtype for all ragged row partitions of all
GraphTensor fields of type `tf.RaggedTensor`.

RULE: `indices_dtype` and `row_splits_dtype` are consistent for all graph pieces
within the graph tensor.

IMPORTANT: This behaviour is disabled when loading legacy SavedModels created
before this requirement was introduced. It is strongly recommented to align
indices for all graph tensors generated by those legacy models using the methods
`.with_indices_dtype()` and `.with_row_splits_dtype()`.

The `indices_dtype` is `int32` by default, the default integer type in
Tensorflow The `indices_dtype` for graph tensor and its pieces can be changed
using `.with_indices_dtype()` method.

The `row_splits_dtype` is `int64` by default, the same as for `RaggedTensor`s.
They can be changed using `.with_row_splits_dtype()` method.

NOTE: graph tensors can be constructed from pieces with inconsistent
`indices_dtype` and `row_splits_dtype`. The indices types of the result
`GraphTensor` are resolved towards the integer types with the maximum capacity
and all pieces are casted towards those types. For example, if *any* graph piece
used in `.from_pieces()` has `int64` `indices_dtype` the result graph tensor
(and all its pieces) would have `int64` `indices_dtype`.

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

<tr> <td> <code>context</code><a id="context"></a> </td> <td> The graph context.
</td> </tr><tr> <td> <code>edge_sets</code><a id="edge_sets"></a> </td> <td> A
read-only mapping from node set name to the node set. </td> </tr><tr> <td>
<code>indices_dtype</code><a id="indices_dtype"></a> </td> <td> The dtype for
graph items indexing. One of <code>tf.int32</code> or <code>tf.int64</code>.
</td> </tr><tr> <td> <code>node_sets</code><a id="node_sets"></a> </td> <td> A
read-only mapping from node set name to the node set. </td> </tr><tr> <td>
<code>num_components</code><a id="num_components"></a> </td> <td> The number of
graph components for each graph. </td> </tr><tr> <td>
<code>rank</code><a id="rank"></a> </td> <td> The rank of this Tensor.
Guaranteed not to be <code>None</code>. </td> </tr><tr> <td>
<code>row_splits_dtype</code><a id="row_splits_dtype"></a> </td> <td> The dtype
for ragged row partitions. One of <code>tf.int32</code> or
<code>tf.int64</code>. </td> </tr><tr> <td> <code>shape</code><a id="shape"></a>
</td> <td> A possibly-partial shape specification for this Tensor.

The returned <code>tf.TensorShape</code> is guaranteed to have a known rank and no
unknown dimensions except possibly the outermost.
</td>
</tr><tr>
<td>
<code>spec</code><a id="spec"></a>
</td>
<td>
The public type specification of this tensor.
</td>
</tr><tr>
<td>
<code>total_num_components</code><a id="total_num_components"></a>
</td>
<td>
The total number of graph components.
</td>
</tr>
</table>

## Methods

<h3 id="from_pieces"><code>from_pieces</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L1052-L1163">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_pieces(
    context: Optional[Context] = None,
    node_sets: Optional[Mapping[NodeSetName, NodeSet]] = None,
    edge_sets: Optional[Mapping[EdgeSetName, EdgeSet]] = None,
    validate: Optional[bool] = None
) -> 'GraphTensor'
</code></pre>

Constructs a new `GraphTensor` from context, node sets and edge sets.


<h3 id="merge_batch_to_components"><code>merge_batch_to_components</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L1165-L1249">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>merge_batch_to_components() -> 'GraphTensor'
</code></pre>

Merges all contained graphs into one contiguously indexed graph.

On a batched GraphTensor, one can call this method to merge all graphs of the
batch into one contiguously indexed graph. The resulting GraphTensor has shape
`[]` (i.e., is scalar) and its features have the shape `[total_num_items,
*feature_shape]` where `total_num_items` is the sum of the previous `num_items`
per batch element. The adjacency indices have values from the range `[0,
total_num_nodes)` with respect to the incident node set. Most TF-GNN models
expect scalar GraphTensors. Currently, there is no function to reverse this
method.

Example: Flattening of

```python
tfgnn.GraphTensor.from_pieces(
    node_sets={
        'node': tfgnn.NodeSet.from_fields(
            # Three graphs:
            #   - 1st graph has two components with 2 and 1 nodes;
            #   - 2nd graph has 1 component with 1 node;
            #   - 3rd graph has 1 component with 1 node;
            sizes=tf.ragged.constant([[2, 1], [1], [1]]),
            features={
               'id': tf.ragged.constant([['a11', 'a12', 'a21'],
                                         ['b11'],
                                         ['c11']])})},
    edge_sets={
        'edge': tfgnn.EdgeSet.from_fields(
            sizes=tf.ragged.constant([[3, 1], [1], [1]]),
            features={},
            adjacency=tfgnn.Adjacency.from_indices(
                source=('node', tf.ragged.constant([[0, 1, 1, 2],
                                                    [0],
                                                    [0]])),
                target=('node', tf.ragged.constant([[0, 0, 1, 2],
                                                    [0],
                                                    [0]]))))})
```

results in the equivalent graph of

```python
tfgnn.GraphTensor.from_pieces(
    node_sets={
        'node': tfgnn.NodeSet.from_fields(
            # One graph with 4 components with 2, 1, 1, 1 nodes.
            sizes=[2, 1, 1, 1],
            features={'id': ['a11', 'a12', 'a21', 'b11', 'c11']})},
    edge_sets={
        'edge': tfgnn.EdgeSet.from_fields(
            sizes=[3, 2, 1, 1],
            features={},
            # Note how node indices have changes to reference nodes
            # within the same graph ignoring its components.
            adjacency=tfgnn.Adjacency.from_indices(
                source=('node', [0, 1, 1, 2, 3 + 0, 3 + 1 + 0]),
                target=('node', [0, 0, 1, 2, 3 + 0, 3 + 1 + 0])))})
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A scalar (rank 0) graph tensor.
</td>
</tr>

</table>



<h3 id="remove_features"><code>remove_features</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L1393-L1484">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>remove_features(
    context: Optional[Sequence[FieldName]] = None,
    node_sets: Optional[Mapping[NodeSetName, Sequence[FieldName]]] = None,
    edge_sets: Optional[Mapping[NodeSetName, Sequence[FieldName]]] = None
) -> 'GraphTensor'
</code></pre>

Returns a new GraphTensor with some features removed.

The graph topology and the other features remain unchanged.

Example 1. Removes the id feature from node set 'node.a'.

```python
graph = tfgnn.GraphTensor.from_pieces(
    node_sets={
        'node.a': tfgnn.NodeSet.from_fields(
            features={'id': ['a1', 'a3']},
            sizes=[2]),
        'node.b': tfgnn.NodeSet.from_fields(
            features={'id': ['b4', 'b1']},
            sizes=[2])})
result = graph.remove_features(node_sets={'node.a': ['id']})
```

#### Result:



```python
tfgnn.GraphTensor.from_pieces(
    node_sets={
        'node.a': tfgnn.NodeSet.from_fields(
            features={},
            sizes=[2]),
        'node.b': tfgnn.NodeSet.from_fields(
            features={'id': ['b4', 'b1']},
            sizes=[2])})
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<code>context</code>
</td>
<td>
A list of feature names to remove from the context, or <code>None</code>.
</td>
</tr><tr>
<td>
<code>node_sets</code>
</td>
<td>
A mapping from node set names to lists of feature names to be
removed from the respective node sets.
</td>
</tr><tr>
<td>
<code>edge_sets</code>
</td>
<td>
A mapping from edge set names to lists of feature names to be
removed from the respective edge sets.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <code>GraphTensor</code> with the same graph topology as the input and a subset
of its features. Each feature of the input either was named as a feature
to be removed or is still present in the output.
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
if some feature names in the arguments were not present in the
input graph tensor.
</td>
</tr>
</table>

<h3 id="replace_features"><code>replace_features</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L1291-L1391">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>replace_features(
    context: Optional[Fields] = None,
    node_sets: Optional[Mapping[NodeSetName, Fields]] = None,
    edge_sets: Optional[Mapping[EdgeSetName, Fields]] = None
) -> 'GraphTensor'
</code></pre>

Returns a new instance with a new set of features for the same topology.

Example 1. Replaces all features for node set 'node.a' but not 'node.b'.

```python
graph = tfgnn.GraphTensor.from_pieces(
    context=tfgnn.Context.from_fields(
        features={'label': tf.ragged.constant([['A'], ['B']])}),
    node_sets={
        'node.a': tfgnn.NodeSet.from_fields(
            features={'id': ['a1', 'a3']},
            sizes=[2]),
        'node.b': tfgnn.NodeSet.from_fields(
            features={'id': ['b4', 'b1']},
            sizes=[2])})
result = graph.replace_features(
    node_sets={
        'node.a': {
            'h': tf.ragged.constant([[1., 0.], [3., 0.]])
         }
    })
```

#### Result:



```python
tfgnn.GraphTensor.from_pieces(
    context=tfgnn.Context.from_fields(
        features={'label': tf.ragged.constant([['A'], ['B']])}),
    node_sets={
        'node.a': tfgnn.NodeSet.from_fields(
            features={
                'h': tf.ragged.constant([[1., 0.], [3., 0.]])
            },
            sizes=[2]),
        'node.b': tfgnn.NodeSet.from_fields(
            features={
                'id': ['b4', 'b1']
            },
            sizes=[2])
    })
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<code>context</code>
</td>
<td>
A substitute for the context features, or <code>None</code> (which keeps the
prior features). Their tensor shapes must match the graph shape and the
number of existing components, which remain unchanged.
</td>
</tr><tr>
<td>
<code>node_sets</code>
</td>
<td>
Substitutes for the features of the specified node sets. Their
tensor shapes must match the graph shape and the existing number of
nodes, which remain unchanged. Node sets not included in this argument
remain unchanged.
</td>
</tr><tr>
<td>
<code>edge_sets</code>
</td>
<td>
Substitutes for the features of the specified edge sets. Their
tensor shapes must match the graph shape and the existing number of
edges. The number of edges and their incident nodes are unchanged.
Edge sets not included in this argument remain unchanged.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <code>GraphTensor</code> instance with some feature maps replaced according to the
arguments.
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
if some node sets or edge sets are not present in the graph
tensor.
</td>
</tr>
</table>

<h3 id="set_shape"><code>set_shape</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_piece.py#L279-L281">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_shape(
    new_shape: ShapeLike
) -> 'GraphPieceBase'
</code></pre>

Deprecated. Use `with_shape()`.

<h3 id="with_indices_dtype"><code>with_indices_dtype</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_piece.py#L310-L323">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_indices_dtype(
    dtype: tf.dtypes.DType
) -> 'GraphPieceBase'
</code></pre>

Returns a copy of this piece with the given indices dtype.

<h3 id="with_row_splits_dtype"><code>with_row_splits_dtype</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_piece.py#L349-L362">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_row_splits_dtype(
    dtype: tf.dtypes.DType
) -> 'GraphPieceBase'
</code></pre>

Returns a copy of this piece with the given row splits dtype.

<h3 id="with_shape"><code>with_shape</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_piece.py#L283-L297">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_shape(
    new_shape: ShapeLike
) -> 'GraphPieceBase'
</code></pre>

Enforce the common prefix shape on all the contained features.




