# tfgnn.EdgeSet

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L698-L820">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

A composite tensor for edge set features, size and adjacency information.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.EdgeSet(
    data: Data, spec: 'GraphPieceSpecBase'
)
</code></pre>

<!-- Placeholder for "Used in" -->

Each edge set contains edges as its items that connect nodes from particular
node sets. The information which edges connect which nodes is encapsulated in
the <a href="../tfgnn/EdgeSet.md#adjacency"><code>EdgeSet.adjacency</code></a> composite tensor (see adjacency.py).

All edges in a edge set have the same features, identified by a string key.
Each feature is stored as one tensor and has shape `[*graph_shape, num_edges,
*feature_shape]`. The `num_edges` is a number of edges in a graph (could be
ragged). The `feature_shape` is a shape of the feature value for each edge.
EdgeSet supports both fixed-size and variable-size features. The fixed-size
features must have fully defined feature_shape. They are stored as `tf.Tensor`
if `num_edges` is fixed-size or `graph_shape.rank = 0`. Variable-size edge
features are always stored as `tf.RaggedTensor`.

Note that edge set features are indexed without regard to graph components.
The information which edge belong to which graph component is contained in
the `.sizes` tensor which defines the number of edges in each graph component.

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

<tr> <td> <code>adjacency</code><a id="adjacency"></a> </td> <td> The
information which edges connect which nodes (see tfgnn.Adjacency). </td>
</tr><tr> <td> <code>features</code><a id="features"></a> </td> <td> A read-only
mapping of feature name to feature specs. </td> </tr><tr> <td>
<code>indices_dtype</code><a id="indices_dtype"></a> </td> <td> The dtype for
graph items indexing. One of <code>tf.int32</code> or <code>tf.int64</code>.
</td> </tr><tr> <td> <code>num_components</code><a id="num_components"></a>
</td> <td> The number of graph components for each graph. </td> </tr><tr> <td>
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
<code>sizes</code><a id="sizes"></a>
</td>
<td>
The number of items in each graph component.
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
</tr><tr>
<td>
<code>total_size</code><a id="total_size"></a>
</td>
<td>
The total number of items.
</td>
</tr>
</table>

## Methods

<h3 id="from_fields"><code>from_fields</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L722-L805">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_fields(
    *_,
    features: Optional[Fields] = None,
    sizes: Field,
    adjacency: Adjacency,
    validate: Optional[bool] = None
) -> 'EdgeSet'
</code></pre>

Constructs a new instance from edge set fields.


#### Example 1:

```python
tfgnn.EdgeSet.from_fields(
    sizes=tf.constant([3]),
    adjacency=tfgnn.Adjacency.from_indices(
        source=("paper", [1, 2, 2]),
        target=("paper", [0, 0, 1])))
```

#### Example 2:

```python
tfgnn.EdgeSet.from_fields(
    sizes=tf.constant([4]),
    adjacency=tfgnn.Adjacency.from_indices(
        source=("paper", [1, 1, 1, 2]),
        target=("author", [0, 1, 1, 3])))
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<code>features</code>
</td>
<td>
A mapping from feature name to feature Tensor or RaggedTensor.
All feature tensors must have shape <code>[*graph_shape, num_edges,
*feature_shape]</code>, where num_edge is the number of edges in the edge set
(could be ragged) and feature_shape is a shape of the feature value for
each edge.
</td>
</tr><tr>
<td>
<code>sizes</code>
</td>
<td>
The number of edges in each graph component. Has shape
<code>[*graph_shape, num_components]</code>, where <code>num_components</code> is the number
of graph components (could be ragged).
</td>
</tr><tr>
<td>
<code>adjacency</code>
</td>
<td>
One of the supported adjacency types (see adjacency.py).
</td>
</tr><tr>
<td>
<code>validate</code>
</td>
<td>
If true, use tf.assert ops to inspect the shapes of each field
and check at runtime that they form a valid EdgeSet. The default
behavior is set by the <code>disable_graph_tensor_validation_at_runtime()</code>
and <code>enable_graph_tensor_validation_at_runtime()</code>.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An <code>EdgeSet</code> composite tensor.
</td>
</tr>

</table>



<h3 id="get_features_dict"><code>get_features_dict</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L222-L224">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_features_dict() -> Dict[FieldName, Field]
</code></pre>

Returns features copy as a dictionary.


<h3 id="replace_features"><code>replace_features</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L553-L559">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>replace_features(
    features: Mapping[FieldName, Field]
) -> '_NodeOrEdgeSet'
</code></pre>

Returns a new instance with a new set of features.


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


<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L53-L55">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    feature_name: FieldName
) -> Field
</code></pre>

Indexing operator `[]` to access feature values by their name.




