description: Stores graphs, possibly heterogeneous (i.e., with multiple node sets).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gnn.GraphTensor" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_pieces"/>
<meta itemprop="property" content="merge_batch_to_components"/>
<meta itemprop="property" content="replace_features"/>
<meta itemprop="property" content="set_shape"/>
</div>

# gnn.GraphTensor

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L432-L811">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Stores graphs, possibly heterogeneous (i.e., with multiple node sets).

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gnn.GraphTensor(
    data: Data,
    spec: "GraphPieceSpecBase",
    validate: bool = False
)
</code></pre>



<!-- Placeholder for "Used in" -->

A `GraphTensor` consists of

* A `GraphTensorSpec` object that provides its type information. It defines
  the node and edge sets, how node sets are connected by edge sets, and the
  type and shape constraints of graph field values. The `GraphTensorSpec`s of
  two graphs are equal if they agree in the features and shapes, independent
  of the variable number of nodes and edges in an actual graph tensor.

* Graph data, or "fields", which can be instances of `Tensor`s or
  `RaggedTensor`s. Fields are stored on the `NodeSet`, `EdgeSet` and `Context`
  tensors that make up the `GraphTensor`. Each of those tensors have fields to
  represent user-defined data features. In addition, there are fields storing
  the graph topology:  NodeSets and EdgeSets have a special `size` field that
  provides a tensor of the number of nodes (or edges) of each graph component.
  Moreover, adjacency information is stored in the `adjacency` property of the
  EdgeSet.

A `GraphTensor` object is a tensor with graphs as its elements. Its `.shape`
attribute describes the shape of the graph tensor, where a shape of `[]`
describes a scalar (single) graph, a shape of `[d0]` describes a `d0`-vector
of graphs, a shape of `[d0, d1]` a `d0` x `d1` matrix of graphs, and so on.

Context, node set, and edge set features are accessed via the `context`,
`node_sets` and `edge_sets` properties, respectively. Note that
the node sets and edge sets are mappings of a set name (a string) to either a
`NodeSet` or `EdgeSet` object. These containers provide a mapping interface
(via `getitem`, i.e., `[]`) to access individual feature values by their name,
and a `features` property that provides an immutable mapping of feature names
to their values. These features are those you defined in your schema.

A "scalar" graph tensor describes a single graph with `C` disjoint components.
When utilized in building models, this usually represents `C` example graphs
bundled into a single graph with multiple disjoint graph components. This
allows you to build models that work on batches of graphs all at once, with
vectorized operations, as if they were a single graph with multiple
components. The shapes of the tensors have elided the prefix batch dimension,
but conceptually it is still present, and recoverable. The number of
components (`C`) could vary from graph to graph, or if necessary for custom
hardware, be fixed to a value known statically. In the simplest case of `C =
1`, this number is constrained only by the available RAM and example sizes.
The number of components in a graph corresponds to the concept of "batch-size"
in a regular neural network context.

Note that since context features store data for each graph, the first
dimension of all *context* features always index the graph component and has
size `C`.

Conceptually (but not in practice - see below), for scalar graphs, each
node/edge set feature could be described as a ragged tensor with a shape
`[c, n_c, f1..fk]` where `c` indexes the individual graph components, `n_c`
indexes the nodes or edges within each component `c`, and `f1..fk` are inner
dimensions of features, with `k` being the rank of the feature tensor.
Dimensions `f1..fk` are typically fully defined, but the `GraphTensor`
container also supports ragged features (of a variable size), in which case
instances of `tf.RaggedTensor`s are provided in those mappings. The actual
number of nodes in each graph is typically different for each `c` graph
(variable number of nodes), so this dimension is normally ragged.

Please note some limitations inherent in the usage of `tf.RaggedTensor`s to
represent features; it is not ideal, in that

  * `tf.RaggedTensor`s are not supported on XLA compilers, and when used on
    accelerators (e.g., TPUs), can cause error messages that are difficult to
    understand;

  * Slices of features for individual graph components are rarely needed in
    practice;

  * The ragged partitions (see docs on `tf.RaggedTensor`) are the same for all
    features within the same node/edge set, hence they would be redundant to
    represent as individual ragged tensor instances.

For these reasons, the `GraphTensor` extracts component partitions into a
special node set field called 'size'. For scalar `GraphTensor` instances this
is a rank-1 integer tensor containing the number of nodes/edges in each graph
component.

It is important to know that feature values are stored with their component
dimension flattened away, leading to shapes like `[n, f1..fk]`, where `n`
(instead of `c` and `n_c`) indexes a node within a graph over all of its
components. For the most common case of features with fully-defined shape of
dimensions `f1..fk`, this allows us to represent those features as simple
dense tensors. Finally, when all the dimensions including `n` are also
fully-defined, the `GraphTensor` is XLA compatible (and this provides
substantial performance opportunities). The same principle also applies to the
edge set features.

In general, for non-scalar graph tensors, the feature values can be a dense
tensor (an instance of `tf.Tensor`) or a ragged tensors (an instance of a
`tf.RaggedTensor`). This union is usually referred to as a "potentially ragged
tensor" (mainly due to the recursive nature of the definition of ragged
tensors). For our purposes, the leading dimensions of the shapes of a set of
feature tensors must match the shape of their containing graph tensor.

`GraphTensor` allows batching of graphs. Batching changes a `GraphTensor`
instance's shape to `[batch_size] + shape` and the graph tensor's rank is
increased by 1. Unbatching removes dimension-0, as if truncating with
`shape[1:]`, and the `GraphTensor`'s rank is decreased by 1. This works
naturally with the batch and unbatch methods of tf.data.Datatset.

Batching and unbatching are equivalent to the batching and unbatching of
individual fields. Dense fields with static shapes (that is, fully-defined
shapes known at compile time) are always batched to `(rank + 1)` dense
tensors. If a field has ragged dimensions, batching results in `(rank + 1)`
ragged tensors. In general, graph tensor operations always try to preserve
fully-defined field shapes and dense representations wherever possible, as
this makes it possible to leverage as XLA optimizations where possible.

A `GraphTensor` of any rank can be converted to a scalar graph using the
'merge_batch_to_components()' method. This method is a graph transformation
operation that merges the graph components of each graph into a single
disjoint graph. Typically, this happens after the input pipeline is done with
shuffling and batching the graphs from individual training examples and before
the actual model treats them as components of a single graph with contiguous
indexing.

Example 1: A homogeneous scalar graph with one component having 10 nodes and
20 edges and no values.

    gnn.GraphTensor.from_pieces(
      node_sets = {
        'node': gnn.NodeSet.from_fields(sizes=[10], features={})
      },
      edge_sets = {
        'edge': gnn.EdgeSet.from_fields(
           sizes=[10],
           features={},
           adjacency=gnn.Adjacency.from_indices(
             source=('node', [0, 5, 9]),
             target=('node', [19, 10, 0])))})

Example 2: A rank-1 graph tensor with three graphs. Each graph is a tree with
a single scalar label.

    rt = tf.ragged.constant

    gnn.GraphTensor.from_pieces(
      context=gnn.Context.from_fields(features={
        'label': rt([['GOOD'], ['BAD'], ['UGLY']])
      }),
      node_sets={
        'root': gnn.NodeSet.from_fields(
                  sizes=rt([[1], [1], [1]]),
                  features={}),
        'leaf': gnn.NodeSet.from_fields(
                  sizes=rt([[2], [3], [1]]),
                  features={'id': rt([['a', 'b'], ['c', 'a', 'd'], ['e']])})},
      edge_sets={
        'leaf->root': gnn.EdgeSet.from_fields(
           sizes=rt([[2], [3], [1]]),
           features={'weight': rt([[.5, .6], [.3, .4, .5], [.9]])},
           adjacency=gnn.Adjacency.from_indices(
             source=('leaf', rt([[0, 1], [0, 1, 2], [0]])),
             target=('root', rt([[0, 0], [0, 0, 0], [0]]))))})

Example 3: An application of `merge_batch_to_components()` to the previous
example. Please note how the source and target edge indices have changed to
reference nodes within a graph.

    gnn.GraphTensor.from_pieces(
      context=gnn.Context.from_fields(features={
        'label': ['GOOD', 'BAD', 'UGLY']
      }),
      node_sets={
        'root': gnn.NodeSet.from_fields(sizes=[1, 1, 1], features={}),
        'leaf': gnn.NodeSet.from_fields(
                  sizes=[2, 3, 1],
                  features={'id': ['a', 'b', 'c', 'a', 'd', 'e']}
                ),
      },
      edge_sets={
        'leaf->root': gnn.EdgeSet.from_fields(
                        sizes=[2, 3, 1],
                        features={'weight': [.5, .6, .3, .4, .5, .9]},
                        adjacency=gnn.Adjacency.from_indices(
                          source=('leaf', [0, 1, 0, 1, 2, 0]),
                          target=('root', [0, 0, 0, 0, 0, 0]),
                        ))})

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
`context`
</td>
<td>
The graph context feature container.
</td>
</tr><tr>
<td>
`edge_sets`
</td>
<td>
A read-only view for edge sets.
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
`node_sets`
</td>
<td>
A read-only view of node sets.
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
`spec`
</td>
<td>
The public type specification of this tensor.
</td>
</tr>
</table>



## Methods

<h3 id="from_pieces"><code>from_pieces</code></h3>

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L619-L640">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_pieces(
    context: Optional[<a href="../gnn/Context.md"><code>gnn.Context</code></a>] = None,
    node_sets: Optional[Mapping[NodeSetName, NodeSet]] = None,
    edge_sets: Optional[Mapping[EdgeSetName, EdgeSet]] = None
) -> "GraphTensor"
</code></pre>

Constructs a new `GraphTensor` from context, node sets and edge sets.


<h3 id="merge_batch_to_components"><code>merge_batch_to_components</code></h3>

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L642-L712">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>merge_batch_to_components() -> "GraphTensor"
</code></pre>

Merges the contained graphs into a single scalar `GraphTensor`.

For example, flattening of

    GraphTensor.from_pieces(
      node_sets={
        'node': NodeSet.from_fields(
          # Three graphs with
          #   - 1st graph having two components with 3 and 2 nodes;
          #   - 2nd graph having 1 component with 2 nodes;
          #   - 3rd graph having 1 component with 3 nodes;
          sizes=tf.ragged.constant([[3, 2], [2], [3]]),
          features={...},
        )
      }
      edge_sets={
        'edge': EdgeSet.from_fields(
          sizes=tf.ragged.constant([[6, 7], [8], [3]]),
          features={...},
          adjacency=...,
        )
      }
    )

would result in the equivalent graph of

    GraphTensor.from_pieces(
      node_sets={
        'node': NodeSet.from_fields(
          # One graph with 4 components with 3, 2, 2, 3 nodes each.
          sizes=[3, 2, 2, 3],
          features={...},
        )
      }
      edge_sets={
        'edge': EdgeSet.from_fields(
          sizes=[6, 7, 8, 3],
          features={...},
          adjacency=...,
        )
      }
    )

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



<h3 id="replace_features"><code>replace_features</code></h3>

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor.py#L729-L807">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>replace_features(
    context: Optional[<a href="../gnn/Fields.md"><code>gnn.Fields</code></a>] = None,
    node_sets: Optional[Mapping[NodeSetName, Fields]] = None,
    edge_sets: Optional[Mapping[EdgeSetName, Fields]] = None
) -> "GraphTensor"
</code></pre>

Returns a new instance with a new set of features for the same topology.

Example 1. Replaces all features for node set 'node.a' but not 'node.b'.

    graph = gnn.GraphTensor.from_pieces(
      context=gnn.Context.from_fields(features={
        'label': tf.ragged.constant([['A'], ['B']])
      }),
      node_sets={
          'node.a': gnn.NodeSet.from_fields(features={
            'id': ['a1', 'a3']
          }, sizes=[2]),
          'node.b': gnn.NodeSet.from_fields(features={
            'id': ['b4', 'b1']
          }, sizes=[2])
      }
    )
    result = graph.replace_features(
      node_sets={'node.a': {'h': tf.ragged.constant([[1., 0.], [3., 0.]])}}
    )

#### Result:


gnn.GraphTensor.from_pieces(
  context=gnn.Context.from_fields(features={
    'label': tf.ragged.constant([['A'], ['B']])
  }),
  node_sets={
      'node.a': gnn.NodeSet.from_fields(features={
        'h': tf.ragged.constant([[1., 0.], [3., 0.]])
      }, sizes=[2]),
      'node.b': gnn.NodeSet.from_fields(features={
        'id': ['b4', 'b1']
      }, sizes=[2])
  }
)



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`context`
</td>
<td>
A substitute for the context features, or None (which keeps the
prior features).
</td>
</tr><tr>
<td>
`node_sets`
</td>
<td>
A substitute for specified node set features. Node sets which
are not included remain unchanged.
</td>
</tr><tr>
<td>
`edge_sets`
</td>
<td>
A substitute for specified edge set features. Edge sets which
are not included remain unchanged.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `GraphTensor` instance with features overridden according to the
arguments.
</td>
</tr>

</table>



<h3 id="set_shape"><code>set_shape</code></h3>

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_piece.py#L295-L301">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_shape(
    new_shape: ShapeLike
) -> "GraphPieceSpecBase"
</code></pre>

Enforce the common prefix shape on all the contained features.




