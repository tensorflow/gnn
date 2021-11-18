# Describing your Graph

## Introduction: Graphs as Training Data

Graph neural networks (GNNs) are neural networks which transform and aggregate
features across the edges of data graphs. The training examples of these models
are graphs themselves. It is not unusual for information in these models to flow
across multiple sets of edges and to be aggregated over multiple sets of nodes.
The possible graph topologies are varied and can be complex.

In order to train graph neural networks, model code must be aware of the
available sets of nodes and edges to run these convolutions on them. These
graphs must also be encoded in files in order to be streamed to workers as
training data. This data must then be decoded, parsed and represented in the
form of tensors. Since each graph may have a varying number of nodes and edges,
the data is shaped irregularly. The shape of these data is complex and varying
from example to example, but will follow a similar pattern over the topology of
a particular graph.

Models and parsers and the other tools involved in these pipelines need to
operate on a description of the graphs you’re going to feed them. The library
provides a way for you to describe the graph topology and the shapes and types
of its features. This “graph schema” is attached to the “graph tensor” object
containing all your graph data. This document describes how you provide this
information to the library.

### Terminology

Before we begin, we define some variables used throughout this document:

*   **Batch size.** The variable B will be used to refer to the batch size. A
    batch of training data contains multiple graphs in the same “graph tensor”
    container.

*   **Node set size.** The variable V will refer to the number of nodes in a
    particular node set. If a graph has multiple node sets, we index them as V0,
    V1, etc.

*   **Edge set size.** The variable E will refer to the number of edges in a
    particular edge set (and similarly with E0, E1, …, if a graph has multiple
    edge sets).

*   **Feature shape.** The shape of features will be represented by F.

Note that for a particular node set, V has a potentially different value for
each graph of a batch. For example, in a batch of 3 graphs where the graphs have
4, 5 and 6 nodes, respectively, V stands for that ragged 4, 5 or 6 dimension.
The same goes for edges. Because of this, these will be represented as ragged
dimensions in the tf.RaggedTensor objects that hold this data (more later on
this)

Also note that in TensorFlow some dimensions are partially-known; these are
dimensions which we know exist but whose size hasn't been determined yet. On the
other hand, ragged dimensions are dimensions which are known to have multiple
values. The way the TensorFlow libraries are built does not differentiate
between the two cases. These dimensions would show up as a None object in the
shape tensor, or rendered as “?” in symboling shapes shown in this document (and
as a “-1” integer in a protocol buffer shape list).

## The Graph Schema

The centerpiece and starting point of the TF-GNN library is called the "graph
schema." The graph schema is an object which provides a declaration of the
available node and edge sets and their features, including their shapes, data
types and some basic metadata (e.g., a description, some information about how
the features may relate to other features, and more). It also includes
declarations of the global features for the entire graph. The library includes
some example graph schemas. The full schema presented in this document can be
found here, and is a fictional example of handling an abuse case on a website
with UGC content.

The schema's own schema is defined in the graph_schema.proto file as a protocol
message type. If you prefer, you can also use a corresponding JSON file to
provide the schema. In either case, you create an instance of the schema in a
text file. This is the first step in preparing your training data. The schema is
then fed and used throughout the data preparation and modelling to inform and
configure the different components and tools.

The graph schema is not intended to provide information about how to build your
model, or to specify what the loss/objective is, or even how to combine the
particular features it provides and for what purpose; those concerns are
addressed at the level of the various model APIs, which should have their own
configuration. The schema merely represents which features are available to use
and guarantees some basic constraints about their shapes and indices.

A graph schema contains three types of objects:

*   **Node sets**, which define a type of node and their associated features.
    Generally speaking, if your graph has multiple types of nodes they are
    defined as separate node sets.

*   **Edge sets**, which contain source and target indices into each of the
    available node sets, and possibly some edge features.

*   **Context features**: Regular features that pertain to the entire graph
    (sometimes called “global” features).

Each of the features is described by providing:

*   A unique name.
*   A description (optional).
*   A data type (one of DT_STRING, DT_INT64, DT_FLOAT).
*   A tensor shape. The shape may be scalar (e.g., a single number, with shape
    unset), tensor (e.g., a vector of 64 floats, with shape `[64]`) or may
    itself include special ragged dimensions (e.g., sentences of words of
    variable length), which are indicated by the value -1.

Note that the shape excludes the prefix dimensions that are present in all
tensors of the same set; it is the shape of the feature for each node or edge.
For example, a feature in a node set will have one value for each node and for
each graph in the batch, and would be prefixed by the batch size (B) and number
of nodes in the graph (V), for example, if the shape of the feature is F, the
shape of the tensor would be `[B,V,F]`. Only the remaining dimensions are
specified here, i.e., `[F]`.

For example, the following declarations define that each node instance (e.g.,
"video") contains two features: a video_embedding embedding vector of 256 floats
and a video_title as a variable-length vector of word strings:

```
feature {
  key: "video_embedding"
  value {
    description: "Video embeddings computed by VEmbed team."
    dtype: DT_FLOAT
    shape { dim { size: 256 } }
  }
}

feature {
  key: "video_title"
  value {
    description: "Full description paragraph for a product."
    dtype: DT_STRING
    shape { dim { size: -1 } }
  }
}
```

A ragged feature dimension is indicated with the special value -1 in the schema,
as can be seen in that second feature.

The feature specifications are used to define a mapping of these variable-shaped
tensors to flat lists of features stored in a file, an associated decoder, and a
graph construction function. Ultimately, your model will simply be provided with
instances of tf.RaggedTensor with each feature’s data, which you will feed to
graph convolutions.

## Constraints on Shapes

The tensors of a graph are not shaped arbitrarily. They contain variable-shaped
tensors that are constrained to share some common dimensions over all their
features, within each set of nodes or edges. For example, each of the shapes of
the features of a batch of graphs will begin with a batch-size dimension, and
then a dimension for the number of nodes: `[B, V, ...]`; all the features on
that node set will share the prefix dimensions B and V. One node feature might
have a shape of `[B, V, 64]`, another one `[B, V, ?, 3]` and a third one simply
`[B, V]` (a scalar feature, e.g. just a single integer per node).

The common prefix dimensions do not have to get declared in the schema. For
example, for an embedding feature of shape [64] associated with each node, you
would simply specify `shape { dim { size: 64 } }`. With a batch size of 32 on a
training example with 172 nodes you would obtain a ragged tensor of shape `[32,
172, 64]`. If you print that shape within your code, it would show as `[?, ?,
64]` because the batch dimension is partially known and the number of nodes is
ragged, that is, it will vary from example to example.

The shapes will match each other as follows:

*   Node feature shapes. All the features on a node set will share the same
    batch-size and number-of-nodes prefix dimensions.
*   Edge feature shapes. All the features on an edge set will share the same
    batch-size and number-of-edges prefix dimensions.
*   Edge indices. Special “source” and “target” features defining the edge
    endpoints as indices into the node sets will also share the same prefix
    dimensions as the edge feature shapes.
*   Context feature shapes. All the features in common for the graph will share
    the same batch-size dimension and a prefix “1”.

## An Example

In the following sections we will introduce most of the graph schema’s features
by incrementally building a description for a moderately complex data set.

Here is an example of a purely hypothetical graph schema: let's assume we have a
public video content platform, where users can view videos that have been
aggregated through public channels. We would like to classify whether some of
these channels are hosting egregiously abusive content violating the platform's
policy rules. Let's build a graph with three types of nodes:

*   **user**: Represents an end user watching videos.
*   **video**: Represents content watched by users.
*   **channel**: Represents collections of videos.

Let us define three sets of edges:

*   **user->video**: Edges representing watches of videos by users.
*   **video->channel**: Edges representing a video belonging to a channel.
*   **user->user**: A co-watch graph of similarity between users based on common
    watched videos.

![takedown schema diagram](images/takedown_schema_diagram.svg)

The model will propagate information from users through the videos that they
watched aggregated to the channels that contain them, and leverage similarity
information between users watching similar videos, in order to build a
classification model on top of the channel embedding. Let's call our imaginary
project "Project Takedown."

### Defining Scalar Features (Defining User Nodes)

Let's first define the "user" nodes. These nodes will sport two features: the
account age, and the user's activity level:

```
node_sets {
  key: "user"
  value {
    description: "An end user who watches videos on our platform."

    features {
      key: "account_age"
      value: {
        description: "The number of days since this account was created."
        dtype: DT_INT64
      }
    }
    features {
      key: "daily_activity"
      value: {
        description: "Average daily number of videos watched by this user over the past month."
        dtype: DT_FLOAT
      }
    }
  }
}
```

Both the node set and the features have dedicated description fields. We
encourage you to fill in these fields, as these act as documentation and will
help other people reading your model code to understand the meaning of your
model's inputs.

Each of the features must provide their data type; at the moment we support
types `DT_INT64`, `DT_STRING` and `DT_FLOAT`and their shapes. In this example
node set, the shapes are left empty, and the default behavior is used to specify
scalar tensors (a single value per node). Therefore, the corresponding tensor
shapes will be `[B, V]`.

### Implicit Size Features[g]

Note that in the previously defined node set, and in all node and edge sets, a
special scalar tensor is implicitly maintained and tracks the number of
corresponding nodes in each graph. This tensor does not have to be declared.

For a graph with V nodes, the value of this tensor will be `[V]`. Furthermore,
in a batch of B graphs, the shape of the size feature will be `[B]` and contain
the number of nodes for each of the graphs in the batch, for that node set. For
example, in a batch of 3 graphs with 4, 5, and 2 nodes each (for this particular
node set), the size tensor's value would be `[4, 5, 2]`.

(In theory you could derive these shapes from any one of the ragged tensors in
the node set, but for node sets without features we still need to provide the
number of latent nodes (see section on latent sets below). In practice you will
probably not need to use this tensor often, but if you make manipulations on
your graph, it will come in handy.

### Feature Shapes (Video Nodes)

We define another node set in which more complex shapes are present. "Video"
nodes will contain features describing each of the videos in the graph:

```
node_sets {
  key: "video"
  value {
    description: "Unique video content."

    features {
      key: "title"
      value: {
        description: "A bag of words of the title of the video."
        dtype: DT_STRING
        shape { dim { size: -1 } }
      }
    }
    features {
      key: "days_since_upload"
      value: {
        description: "The number of days since this video was uploaded."
        dtype: DT_INT64
      }
    }
    features {
      key: "embedding"
      value: {
        description: "A precomputed embedding of each video."
        dtype: DT_FLOAT
        shape { dim { size: 64 } }
      }
    }
  }
}
```

For each video, we will provide the following features:

*   **Title.** The title of the video, converted to a bag of words, as a list of
    strings. Note that each video will have a title with a different number of
    words, so we must indicate that the dimension of those video features will
    be "ragged"; this is done by using -1 in the shape. The final shape of that
    tensor will be `[B, V, None]`, where the third dimension will vary.

*   **Upload Age.** The number of days since the video was uploaded, as a single
    feature. This is a scalar feature. A common minor mistake is to specify a
    shape of `shape { dim { size: 1 } }`. That would provide a ragged tensor of
    shape `[B, V, 1]` which still works fine, but has a redundant shape at the
    tail. It is preferable to instead leave the shape as a scalar to obtain a
    simpler (and equivalent) shape of `[B, V]`.

*   **A video embedding.** This feature contains a precomputed embedding of
    videos. This particular embedding is a vector of 64 floating-point numbers,
    so we specify a shape of `[64]`. The resulting ragged tensor shape will be
    `[B, V, 64]`.

### Latent Nodes (Channel Nodes)

Next, we introduce a latent node type. Latent nodes are nodes which do not carry
any explicit features. The embeddings for those nodes are computed purely from
the convolution of features on their incoming edges. You still need to declare
latent node sets because adjacent edges will refer to them.

Here we declare the channel node as a latent node set, and provide only a
description for it:

```
node_sets {
  key: "channel"
  value {
    description: "A channel aggregating multiple videos."
  }
}
```

Note that we do not need to provide shape information about the embedding
computed at the latent node set; this information belongs to the model
specification (specifying a GNN model involves other information that does not
get included in a graph schema). In the example setting, we would use the
activations computed at this node for classification.

### Defining Edge Sets (Videos to Channels)

Now that we've defined three sets of nodes, we can define the edges between
them. The first set of edges will be the edges from videos to channels:

```
edge_sets {
  key: "video_channel"
  value {
    description: "Membership of videos to a channel."
    source: "video"
    target: "channel"
  }
}
```

The “source” and “target” fields define which node set the edges will be linking
to. Note that the keys for “source” and “target” must match one of the names of
the node sets provided above. It is also valid for edges to be defined to and
from the same node set (both fields would be set to the same node set name).
This set of edges has no features.

### Implicit Source and Target Features

For each edge set, two additional feature tensors are always implicitly defined:

*   source: The indices on the source side of an edge (where it points from).
*   target: The indices on the target side of an edge (where it points to).

These are defined as scalar features of type tf.int64, and are used to index
into the features of the node sets they refer to, to gather their corresponding
features during convolutions. They do not need to be defined in the schema. If
you look at the encoding of the graphs on file, you may find those tensors with
special feature names like #source and #target.

This is how we build graph convolution operations: by using these indices to
gathering the node features on each side of an edge, transforming those features
over the edge, and then aggregating the results back to the nodes as an
embedding of the neighborhoods of each node. These indices are usable in the
various modeling APIs, and you could write these convolution operators by hand
if desired.

Note that there are no special provisions for specifying undirected edges.
Undirected edges differ from directional edges only in their treatment in your
model code. How you use and combine the tensors, for example, making the
features commutative with respect to the direction of the edges, is how we treat
the edges as undirected. For example, if you know that a set of edges is going
to be undirected, you could avoid duplicating an edge from A to B as an edge
from B to A in the input data and insert the relevant operations in your model
code.

### Edge Features (Users to Videos)

Let’s define a second set of edges, this time with some features attached to
them. We define the edges between users and the videos they watched:

```
edge_sets {
  key: "watches"
  value {
    description: "Watches of videos by users."
    source: "user"
    target: "video"

    features {
      key: "watch_fraction"
      value: {
        description: "The fraction of the video the user has watched, between 0 and 1."
        dtype: DT_FLOAT
      }
    }
  }
}
```

These edges are weighted with a single float-pointing value: the fraction of the
video the user has watched). The shape of this tensor will be `[B, E]`, where
`E` is the number of edges in that particular edge set. Because watch_fraction
is a scalar, we do not provide a shape.

### Homogeneous Graphs (Co-Watch Edges)

We then add a third and final set of edges, this time demonstrating how you
would define a graph over a single type of nodes. This set of edges consists of
a co-watch similarity graph between user nodes, i.e., users are similar if they
watched some of the same videos:

```
edge_sets {
  key: "co-watch"
  value {
    description: "Co-watch similarity graph between users."
    source: "user"
    target: "user"

    features {
      key: "similarities"
      value: {
        description: "Similarity scores between users (Jaccard and cosine)."
        dtype: DT_FLOAT
        shape { dim { size: 2 } }
      }
    }
  }
}
```

Note how the source and target sets of nodes are both defined to refer to the
same user nodes. If you had a graph with a single node type and a single set of
edges—a common scenario in practice—you would provide a graph schema with only
one node set and one edge set.

This particular set of edges is weighted by multiple scores. Each edge carries
two floats, for example one for the Jaccard similarity and one for a cosine
similarity between encoded features of the user nodes. The final shape of this
feature will be `[B, E, 2]`.

### Context Features (Adding a Label)

We also provide a feature that pertains to the entire graph. These are called
"context" features, or “global” features. Context features share the same
leading batch size (B) as the node and edge features, but are otherwise not
repeated (there's a single feature per graph).

This particular feature will be an optional string feature that provides the
label associated with each training example:

```
context {
  features {
    key: "abuse_class"
    value: {
      description: "A label classifying if the channel is abusive."
      dtype: DT_STRING
      shape { dim { size: -1 } }
    }
  }
}
```

Since in this example the feature is optional, we use a ragged dimension (-1)
for its shape: it can be present (one value) or not (zero values). This shape is
declared the same as it would for a variable-length feature.

#### About Labels

In the context of a graph classification problem, a label feature would
typically be the target class associated with the entire graph example provided
(the ground truth). You could imagine a collection of small graphs which we want
to classify into one of a fixed number of categories.

In the context of classifying nodes that are a part of a larger graph, the
training examples are usually sampled subgraphs around a particular “seed” or
“root” node. By convention the data preparation would make sure to place the
seed node at the front of the node list so that we identify it. In this
scenario, multiple convolutions propagate information from all the other nodes
one or multiple times, across all the nodes, in parallel, and learning kernels
along the way. We would then segment out the embedding of the seed node (the
first node in the tensor of features) and feed it to a loss term for
optimization and backpropagation.

In this example, we have a heterogeneous graph, and we're assuming that a
sampler will produce a single channel node per graph. All the other features
will bubble up to that root node across the different sets of edges, aggregating
through the intervening nodes, and this is where our model will insert a loss
function and a prediction.

It is possible to build models that diverge from some of these conventional
scenarios, e.g., models that train jointly against a loss computed over multiple
labels on multiple nodes. Ultimately you define the features for your model and
labels as just another data feature. It’s how you use them that makes them
labels or not. They are not identified explicitly nor treated differently in the
graph schema, and the schema does not specify how those features are intended to
be used in your model, only how the data is presented to the model.

### Global Feature References

We skipped a detail in the "channel" node set, the “context” field:

```
node_sets {
  key: "channel"
  value {
    description: "A channel aggregating multiple videos."
    context: "abuse_class"  <---------------------------- this field
  }
}
```

This is metadata hinting at the fact that one of the global features is
associated with this node set. The abuse_class feature is a label that applies
to the seed channel node around which we sampled to obtain this training graph.
This is validated by the schema---the name must refer to an existing global
feature---but it is otherwise not used anywhere else. It’s something you can use
in documenting your model, or to automate the generation of models.

## Validating your Schema

Now that we've got a full description of the graph features, we can validate our
graph schema for correctness. Do this by running a tool we've built specifically
for that purpose:

```
tfgnn_validate_graph_schema --graph_schema=examples/schemas/takedown.pbtxt
```

This tool will issue errors if anything in your schema breaks some of the
constraints, for example, if a set of edges refers to nodes that don't exist. It
will also print out the expected shapes of the ragged tensors you will
manipulate in symbolic form, like this:

```
{'context': {'abuse_class': ([B, None], string)},
 'edges': {'co-watch': {'#size': ([B], int64),
                        '#source': ([B, E0], int64),
                        '#target': ([B, E0], int64),
                        'similarity': ([B, E0], float32)},
           'video_channel': {'#size': ([B], int64),
                             '#source': ([B, E1], int64),
                             '#target': ([B, E1], int64)},
           'watches': {'#size': ([B], int64),
                       '#source': ([B, E2], int64),
                       '#target': ([B, E2], int64),
                       'watch_fraction': ([B, E2], float32)}},
 'nodes': {'channel': {'#size': ([B], int64)},
           'user': {'#size': ([B], int64),
                    'account_age': ([B, V1], int64),
                    'daily_activity': ([B, V1], float32)},
           'video': {'#size': ([B], int64),
                     'days_since_upload': ([B, V2], int64),
                     'embedding': ([B, V2, 64], float32),
                     'title': ([B, V2, None], string)}}}
```

Notice how each node set and edge set may have a different size, which is why we
write their sizes as V0, V1, etc. Also note how all the features within each set
have the same prefix shape, as explained in an earlier section of this document.

You can verify that the shapes of the tensors above match those you would expect
from your schema. The symbolic shapes map to those discussed in this document
and their particular numbers will vary from example to example. It will be
useful to know your shapes well or refer to this as you build your model code
because manipulating those tensors can be tricky. We recommend documenting at
least some of the expected shapes in code comments, as it makes model code much
easier to understand for collaborators.

