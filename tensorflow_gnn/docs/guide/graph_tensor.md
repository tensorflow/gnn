# Introduction to GraphTensor

## Overview

The [`tfgnn.GraphTensor`](../api_docs/python/tfgnn/GraphTensor.md)
class is the cornerstone of the TF-GNN library.
It is a composite tensor (like RaggedTensor or SparseTensor), so it can be used
directly in a tf.data.Dataset, as an input or output of a Keras Layer or a
tf.function, and so on. The GraphTensor class represents heterogeneous directed
graphs with attached features as a collection of `tf.Tensor` and/or
`tf.RaggedTensor` objects.

A **heterogeneous directed graph** consists of several disjoint
node sets <i>V</i><sub>1</sub> ..., <i>V<sub>n</sub></i>
and edge sets <i>E</i><sub>1</sub>, ..., <i>E<sub>m</sub></i>
such that each edge set <i>E<sub>i</sub></i> connects a particular pair of node
sets: <i>E<sub>i</sub></i> ⊆ <i>V<sub>s(i)</sub></i> × <i>V<sub>t(i)</sub></i>.
This generalizes the usual notion of a homogeneous directed graph, which has
a single node set <i>V</i> connected by a single
edge set <i>E</i> ⊆ <i>V</i> × <i>V</i>.

Heterogeneous graphs come up naturally in practical applications of
Graph Neural Networks as soon as there is more than one type of object that
we want to model as a node. For example, academic papers and their citations
form a directed graph with papers as nodes and citations as edges.
All paper nodes allow essentially the same features (publication year,
extracted keywords, etc.).
However, there are more types of objects that capture significant relations
between papers: each paper's publication venue, its authors, the authors'
affiliation with a research institution, and so on. We want to add nodes and
edges to the graph to capture these relations, but they don't bring the same
features as the papers themselves, and they don't carry the same meaning as a
citation. Therefore, it is convenient for feature encoding and potentially
useful as an inductive bias for the GNN model to introduce separate node types
`"paper"`, `"author"`, `"institution"` etc. as well as separate edge types
`"cites"`, `"writes"`, `"affiliated_with"` and so on.

Let's see how GraphTensor does this for node sets `"paper"` and `"author"`
with edge sets `"cites"` and `"writes"`. (For bulk data input to training
pipelines, do not use this explicit construction of a single GraphTensor,
instead see the [input pipeline guide](input_pipeline.md).)

```python
# Encodes the following imaginary papers:
#   [0] K. Kernel, L. Limit: "Anisotropic approximation", 2018.
#   [1] K. Kernel, L. Limit, M. Minor: "Better bipartite bijection bounds", 2019.
#   [2] M. Minor, N. Normal: "Convolutional convergence criteria", 2020.
# where paper [1] cites [0] and paper [2] cites [0] and [1].
#
graph = tfgnn.GraphTensor.from_pieces(
   node_sets={
       "paper": tfgnn.NodeSet.from_fields(
           sizes=tf.constant([3]),
           features={
               "tokenized_title": tf.ragged.constant(
                   [["Anisotropic", "approximation"],
                    ["Better", "bipartite", "bijection", "bounds"],
                    ["Convolutional", "convergence", "criteria"]]),
               "embedding": tf.constant(  # One-hot encodes the first letter.
                   [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
               "year": tf.constant([2018, 2019, 2020]),
           }),
       "author": tfgnn.NodeSet.from_fields(
           sizes=tf.constant([4]),
           features={
               "name": tf.constant(
                   ["Kevin Kernel", "Leila Limit", "Max Minor", "Nora Normal"]),
           })},
   edge_sets={
       "cites": tfgnn.EdgeSet.from_fields(
           sizes=tf.constant([3]),
           adjacency=tfgnn.Adjacency.from_indices(
               source=("paper", tf.constant([1, 2, 2])),
               target=("paper", tf.constant([0, 0, 1])))),
       "writes": tfgnn.EdgeSet.from_fields(
           sizes=tf.constant([7]),
           adjacency=tfgnn.Adjacency.from_indices(
               source=("author", tf.constant([0, 0, 1, 1, 2, 2, 3])),
               target=("paper",  tf.constant([0, 1, 0, 1, 1, 2, 2]))))})
```

The node sets are simply containers of features, plus size information.
(We'll see later why the size is stored as a vector `[n]` and not a scalar `n`.)
Within each node set, the nodes are indexed 0,1, ..., <i>n</i>–1. All nodes
in a node set have the same features, identified by a string key, and each
feature is stored as one tensor for the whole node set with shape `[n, ...]`.
The first dimension is the item dimension, its size is the number <i>n</i>
of nodes. The remaining zero or more dimensions make up the feature shape,
that is, the shape of the feature value for each node. If the feature is stored
as a `tf.Tensor`, all feature values have the same shape. GraphTensor also
supports storing the feature as a `tf.RaggedTensor`, which allows for
variable-size feature values. This support for an arbitrary number of features,
ragged or dense, makes GraphTensor a good fit
for both input features (often more than one, possibly variable-length)
and hidden states in a GNN (stored as a single feature under the name
`tfgnn.HIDDEN_STATE = "hidden_state"`).

Each edge set contains edges that connect a particular pair of source and target
node sets. (There can be more than one edge set between the same pair of node
sets.) The edges in an edge set are indexed 0,1, ..., <i>m</i>–1.
Like a node set, an edge set stores size information and a map of features,
indexed by edge instead of node. (For brevity, the example above has an empty
feature map.)
Most importantly, though, each edge set has an `EdgeSet.adjacency` subobject
to represent how this edge set connects nodes from its incident node sets.
The basic `tfgnn.Adjacency` class stores two tensors `.source` and `.target`,
each of shape `[m]`, such that edge `i` connects nodes `adjacency.source[i]`
and `adjacency.target[i]` of the node sets `adjacency.source_name` and
`adjacency.target_name`. Code that treats the choice of incident node as a
parameter can also spell these properties as `adjacency[incident_node_tag]` and
`adjacency.node_set_name[incident_node_tag]` using the symbolic constants
`tfgnn.SOURCE` or `tfgnn.TARGET` as tags to distinguish the two incident nodes
of every edge.

The various NodeSets and EdgeSets of a GraphTensor (and also the graph context
introduced below) are called the **graph pieces**. Each graph piece consists of
tensors, which are called its **fields**; fields comprise the user-defined
dict of **features** as well as the structural information stored in size and
adjacency tensors.


## The GraphTensorSpec and static shapes

Recall how TensorFlow distinguishes between the actual numbers in a Tensor,
which may vary from one model input to the next, and the Tensor's dtype and
shape, which typically are the same for all inputs (maybe except for some
dimensions in the shape that are not statically known). This allows to translate
or "trace" the model's Python code into TensorFlow ops once, and then execute
them many times, say, for a training loop. As a composite tensor, the same is
true for each `tfgnn.GraphTensor`: Its associated
[`tfgnn.GraphTensorSpec`](../api_docs/python/tfgnn/GraphTensorSpec.md) object
holds the `tf.TensorSpec` and `tf.RaggedTensorSpec` objects for all its fields,
plus a bit of metadata, notably `adjacency.source_name` and `.target_name` for
each edge set. Inside a TensorFlow program, users can mostly just work with the
GraphTensor; its spec will be updated on the fly, and TensorFlow can get it
where needed.

For consistency between its fields, GraphTensor tightens TensorFlow's usual
rules for shapes as follows:

  * Each field must have a shape of known rank (number of dimension).
  * Each dimension has the meaning explained in the documentation. If the same
    dimension occurs in the shapes of multiple fields (e.g., the number *n* of
    nodes in a node set occurs in the shapes of all its features), it must have
    the same value in the static shape of each.
  * The value `None` is allowed only
      * for the outermost (i.e., first) dimension of a field,
      * for a ragged dimenson in a field that is a `tf.RaggedTensor`.

The last item can be summarized as: `None` means outermost or ragged.
It forbids the use of `None` for uniform dimensions of unknown size,
except the outermost. This comes naturally for most applications and greatly
simplifies code that deals with field values based on their shapes.

## The GraphSchema

The bulk input of GraphTensors from disk for training needs one GraphTensorSpec
for all incoming values before any of them can be read. This coincides with the
need of other tools that handle graph data in bulk to know about the node sets,
the edge sets, and their respective features.
To that end, the TF-GNN library defines the
[tfgnn.GraphSchema](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/proto/graph_schema.proto)
protocol message. We recommend that all GraphTensor datasets stored on disk
are accompanied by a `graph_schema.pbtxt` file that contains the applicable
GraphSchema in the format created by `protobuf.text_format.MessageToString()`.
In fact, it's easy to write that format by hand:

```
node_sets {
 key: "paper"
 value {
   description: "The research papers in this dataset."
   features {
     key: "tokenized_title"
     value: { dtype: DT_STRING  shape: { dim: { size: -1 } } }  # -1 means ragged.
   }
   features {
     key: "embedding"
     value: { dtype: DT_FLOAT  shape: { dim: { size: 3 } } }
   }
   features {
     key: "year"
     value: { dtype: DT_INT64  shape: { } }
   }
 }
}
node_sets {
 key: "author"
 value {
   description: "The authors of the papers in this dataset."
   features {
     key: "name"
     value: { dtype: DT_STRING  shape: { dim: { } } }
   }
 }
}
edge_sets {
 key: "cites"
 value {
   description: "Connects citing papers (source) to cited papers (target)."
   source: "paper"
   target: "paper"
 }
}
edge_sets {
 key: "writes"
 value {
   description: "Connects authors to all their papers."
   source: "author"
   target: "paper"
 }
}
```

The guide on [describing your graph](schema.md) introduces the GraphSchema
in greater detail.
The [input pipeline](input_pipeline.md) shows how to load a graph schema
and use it for parsing GraphTensors from files of tf.Example records.
Notice that a GraphTensor and hence a GraphTensorSpec can contain features of
arbitrary data type while the GraphSchema can only declare feature dtypes that
are directly supported by the tf.Example format: `DT_FLOAT32`, `DT_INT64` and
`DT_STRING`. Conversely, it has some extra information (like the description
fields) not present in the GraphTensorSpec.


## What to read next?

Congratulations, you have now understood the basic structure of a GraphTensor!

If you want to progress rapidly to training a first GNN model on your data,
you can stop here and continue with the introduction to TF-GNN's
[Runner](runner.md), which is an end-to-end training pipeline that comes with
a pre-configured GNN model.

If you want to define your own model and/or understand what happens under the
hood, you can read the following in parallel:

  * the end-to-end example (to be written),
  * the guides that cover its techniques in more depth:
      * the rest of this doc about GraphTensor,
      * the [modeling guide](gnn_modeling.md) for information about creating
        your own GNN models,
      * the [input pipeline guide](input_pipeline.md) about the details
        of consuming training data (although the Runner has you covered,
        even for custom models).


## Basic graph operations: broadcast and pool

GraphTensor supports two basic, complementary, low-level operations for sending
data across the graph: broadcasting from nodes onto edges, and pooling from
edges into nodes. Both take a GraphTensor, the name of an edge set, and an
incident node tag to select the edge set's source or target side. They operate
in bulk on the full node and edge sets, not on individual nodes. The node and
edge values are stored in tensors that are shaped like node and edge features,
resp., but need not actually be stored as such in the GraphTensor. Broadcasting
handles a one-to-many relationship (one node has zero or more incident edges)
and simply copies the node value onto each edge. Pooling handles a many-to-one
relationship and needs to aggregate a data-dependent number of edge values into
one node value; common choices are element-wise `"sum"`, `"mean"` or
`"max_no_inf"`. (They all produce 0 in case of zero edges, which blends well
with neural networks, although mathematically one might expect `nan` for the
empty average 0/0 or `-inf` for the maximum of an empty set.)

For illustration, consider the following toy example of computing the average
embeddings for the papers of each author.

```python
# Compute the mean paper embedding.
embedding_on_edge = tfgnn.broadcast_node_to_edges(
   graph, "writes", tfgnn.TARGET,
   feature_name="embedding")
mean_paper_embedding = tfgnn.pool_edges_to_node(
   graph, "writes", tfgnn.SOURCE, reduce_type="mean",
   feature_value=embedding_on_edge)

# Insert it into the GraphTensor.
author_features = graph.node_sets["author"].get_features_dict()
author_features["mean_paper_embedding"] = mean_paper_embedding
graph = graph.replace_features(node_sets={"author": author_features})

print(graph.node_sets["author"]["mean_paper_embedding"])
```

Notice that a GraphTensor (just like a Tensor) is immutable: changing part of
its value requires a new Python object. However, multiple GraphTensors can share
the underlying Tensors for those parts that do not change. Moreover, once
TensorFlow has traced the Python code into a graph of TF ops (like it does for
the training loop and other @tf.functions), only the Tensor operations remain,
so there is no performance penalty for Python-level abstractions like
GraphTensor.

Graph Neural Networks can be built by interspersing broadcast and pool
operations with trainable transformations of hidden states on nodes and
possibly edges. (Most users of TF-GNN will not need to write code at this level
themselves; please see the [modeling guide](gnn_modeling).) A tensor of values
for all nodes in a node set or all edges in an edge set is the basic unit of
work on which low-level TensorFlow operations are performed, much like a batch
of intermediate activations in a plain feed-forward network. That gives us a
rule of thumb when to put nodes into the same node set: They all have the same
data, and we want to transform all of them together in the same way (with the
same trained weights, if applicable). The same goes for edge sets, with the
additional constraint that all edges in an edge set must have the same
source/target node set, respectively.

## Components and context

Training a GNN on batches of *b* heterogeneous input graphs will benefit
from a way to represent such a batch as one merged graph with contiguously
indexed nodes and edges, like the example graphs shown above. This isn't hard:
it's just a matter of concatenating features and adjusting Adjacency objects
to the implied shift of node index values. Broadcast and pool operations along
edge sets can now treat the merged graph as a unit, but since no edges have
been added between input graphs, they do not interact.

To keep track of the boundaries between the original input graphs, GraphTensor
provides the notion of a graph component: Recall the size tensors of node sets
and edge sets. In the example above, the size tensors have been vectors of
length 1, i.e., with shape `[1]`. More generally, their shape is
`[num_components]` (same across all node/edge sets), and
`size[i] for i in range(num_components)` yields the number of items
(nodes or edges) in the i-th component. The size tensors are concatenated,
not added, when merging a batch of input graphs into one. This way,
the separation between components (usually one per input graph) is preserved.

Input graphs may come with features that belong to each input graph as a whole,
not any particular node or edge. GraphTensor supports the notion of graph
context to store such features. The `GraphTensor.context` object provides a map
of features, much like node sets and edge sets, except that the context features
have shape `[num_components, ...]`. That means, the items of the context are the
graph components (just like the items of a node set are the nodes and the items
of an edge set are the edges). For uniformity, the context also has a size
tensor: its shape is `[num_components]`, and its values are all 1s.

GraphTensor supports broadcast and pool operations between the context (that is,
values per component) and an edge set or a node set (that is, values per edge or
node). Broadcasting from context gives each node/edge the value from the graph
component it belongs to. Pooling to context gives each component an aggregate
(sum, mean, max, etc.) of its nodes/edges in the respective node/edge set. (Some
GNN models use this to read out one hidden state for each input graph in a
batch.)

Conceptually, broadcast from and pooling to context is fully analogous to
broadcast/pool between nodes and edges, with incidence to a node replaced by
containment in a component. In fact, the operations `tfgnn.broadcast()` and
`tfgnn.pool()` provide a unified API for broadcasting from or pooling to any of
`tfgnn.SOURCE`, `tfgnn.TARGET` or `tfgnn.CONTEXT`. (See their docstrings for
details.)


## Scalar vs.batched GraphTensors

The GraphTensors we've seen so far are scalar, that is to say, their shape is
`[]`, and they store a single graph. Most model-building code in the TF-GNN
library expects such scalar GraphTensors. The previous section has described
how one graph can contain multiple components as the result of merging multiple
inputs into one graph. However, the direct construction of the merged graph
from its pieces is impractical for bulk input for training, evaluation or
inference.

To integrate with `tf.data.Dataset`, GraphTensor supports a two-step approach to
combine multiple inputs: batching inputs, followed by merging each batch.

### Batching

Let's look at batching first. If a Dataset contains a GraphTensor of shape `[]`,
then `dataset = dataset.batch(batch_size, drop_remainder=True)` produces
batched GraphTensors that are no longer scalar but have shape `[batch_size]`.
If you like, you can think of them as vectors of graphs, all of the same length.
Technically, each graph in such a GraphTensor could contain multiple components,
but commonly it's just one.

With `drop_remainder=False` (the default), there might be a final batch that
contains fewer that `batch_size` graphs. To account for that, the static shape
of the GraphTensors from such a dataset is `[None]`.

Batching requires that all GraphTensors have the same fields and essentially
comes down to stacking them, which results in prepending the `batch_size`
or `None` as a new outermost dimension. If a field's previous outermost
dimension was `None`, GraphTensor treats it as a ragged dimension, in line
with the rule from above that `None` means outermost or ragged.

It is rarely useful, but GraphTensors can be batched more than once, yielding
GraphTensor shapes of rank 2 (a matrix of graphs), rank 3, rank 4, and so on.
Generally speaking, in a GraphTensor of shape `graph_shape`, all features have
the shape `[*graph_shape, num_items, *feature_shape]`.

For the common case of batching scalar GraphTensors once, this means all
features have the shape `[batch_size, num_items, *feature_shape]`.
If `num_items` is `None`, the field is a `tf.RaggedTensor`, and the
batching operation turned `num_items` from the outermost into a ragged
dimension. The dimensions in `feature_shape` stay unchanged.

For now, GraphTensor requires that `GraphTensor.shape` does not contain `None`,
except maybe as the outermost dimension. That means repeated calls to `.batch()`
must set `drop_remainder=True` in all but the last one. Future versions of
GraphTensor may lift that requirement.


### Merging a batch of graphs to components of one graph

On a batched GraphTensor, one can call the method
`graph = graph.merge_batch_to_components()` to merge all graphs of the batch
into one, contiguously indexed graph, as described above.
The resulting GraphTensor has shape `[]` (i.e., is scalar) and its features
have the shape `[total_num_items, *feature_shape]` where `total_num_items` is
the sum of the previous `num_items` per batch element. At that stage,
the GraphTensor is ready for use with the TF-GNN model-building code,
but it is no longer easy to split it up.

Please refer to the [input pipeline guide](input_pipeline.md) for more
information on how to build an input pipeline with a tf.data.Dataset
containing a GraphTensor.


## Use in Keras

Keras is the most-used and officially recommended library for building neural
networks on top of the distributed computation engine provided by TensorFlow.
It offers three different API levels, and GraphTensor lets you build GNNs with
each.

  * [tf.keras.Sequential](https://www.tensorflow.org/guide/keras/sequential_model)
    lets you express simple models as a chain of Keras Layer objects.
    A GNN layer is naturally expressed as a Keras Layer that accepts a
    GraphTensor as input and output, with a `"hidden_state"` feature on nodes
    (and possibly also edges or context). TF-GNN comes with pre-defined layers
    of that kind and a framework to build more.

  * Keras' [Functional API](https://www.tensorflow.org/guide/keras/functional)
    lets you compose layers in more advanced ways, by calling Keras Layer
    objects on symbolic tensors (that is, placeholders for future values).
    GraphTensor, its methods and its constituent pieces can be used directly
    in Keras' Functional API, much like Tensor or RaggedTensor.
    (In technical terms: they all provide the necessary specializations of the
    KerasTensor wrapper type.) You can even call GraphTensor methods and
    tensor-to-tensor functions from `tf.*`, which will be wrapped ad-hoc as
    layers. However (as of May 2022), you cannot call freestanding functions
    from `tfgnn.*`,due to a limitation in Keras.

  * Keras' [Subclassing API](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
    lets you define your own Keras Layers, or even a complete Keras Model
    in terms of raw TensorFlow code inside the overridden `call()` method.
    It can operate directly on `tfgnn.GraphTensor` objects (no wrapping)
    for maximum flexibility. However, all state creation (variables and other
    resources) must be factored out into the `__init__()` method, which makes
    it harder to follow for large models.

The [modeling guide](gnn_modeling.md) discusses model building with Keras
in greater detail. We recommend using the Functional API to compose models
of layers (custom ones as well as predefined ones) and the Subclassing API
to define custom Layer types.
