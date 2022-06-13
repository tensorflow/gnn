description: A composite tensor for edge set features, size and adjacency information.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfgnn.EdgeSet" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_fields"/>
<meta itemprop="property" content="get_features_dict"/>
<meta itemprop="property" content="replace_features"/>
<meta itemprop="property" content="set_shape"/>
</div>

# tfgnn.EdgeSet

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L465-L548">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A composite tensor for edge set features, size and adjacency information.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.EdgeSet(
    data: Data, spec: 'GraphPieceSpecBase', validate: bool = False
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
`adjacency`
</td>
<td>
The information which edges connect which nodes (see tfgnn.Adjacency).
</td>
</tr><tr>
<td>
`features`
</td>
<td>
A read-only mapping of feature name to feature specs.
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
`num_components`
</td>
<td>
The number of graph components for each graph.
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
`sizes`
</td>
<td>
The number of items in each graph component.
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
`total_num_components`
</td>
<td>
The total number of graph components.
</td>
</tr><tr>
<td>
`total_size`
</td>
<td>
The total number of items.
</td>
</tr>
</table>



## Methods

<h3 id="from_fields"><code>from_fields</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L488-L533">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_fields(
    *,
    features: Optional[<a href="../tfgnn/Fields.md"><code>tfgnn.Fields</code></a>] = None,
    sizes: <a href="../tfgnn/Field.md"><code>tfgnn.Field</code></a>,
    adjacency: Adjacency
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
`features`
</td>
<td>
A mapping from feature name to feature Tensor or RaggedTensor.
All feature tensors must have shape `[*graph_shape, num_edges,
*feature_shape]`, where num_edge is the number of edges in the edge
set (could be ragged) and feature_shape is a shape of the feature value
for each edge.
</td>
</tr><tr>
<td>
`sizes`
</td>
<td>
The number of edges in each graph component.
Has shape `[*graph_shape, num_components]`, where `num_components` is
the number of graph components (could be ragged).
</td>
</tr><tr>
<td>
`adjacency`
</td>
<td>
One of the supported adjacency types (see adjacency.py).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An `EdgeSet` composite tensor.
</td>
</tr>

</table>



<h3 id="get_features_dict"><code>get_features_dict</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L138-L140">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_features_dict() -> Dict[FieldName, Field]
</code></pre>

Returns features copy as a dictionary.


<h3 id="replace_features"><code>replace_features</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L362-L368">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>replace_features(
    features: <a href="../tfgnn/Fields.md"><code>tfgnn.Fields</code></a>
) -> '_NodeOrEdgeSet'
</code></pre>

Returns a new instance with a new set of features.


<h3 id="set_shape"><code>set_shape</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_piece.py#L290-L296">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_shape(
    new_shape: ShapeLike
) -> 'GraphPieceSpecBase'
</code></pre>

Enforce the common prefix shape on all the contained features.


<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L34-L36">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    feature_name: FieldName
) -> <a href="../tfgnn/Field.md"><code>tfgnn.Field</code></a>
</code></pre>

Indexing operator `[]` to access feature values by their name.




