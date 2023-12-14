# Data Preparation and Sampling

## Introduction

The `tensorflow_gnn` library supports reading streams of `tf.train.Example`
proto messages with all the contents of a graph, or subgraph, encoded in them.
This document describes how to produce such a stream of encoded data using the
library helper functions, details of the encoding (if you’d like to write your
own graph data generator), and describes the steps required to prepare for
graph sampling.

## Writing Graph Tensors to Files

If you use Python, the most straightforward method to produce streams of
[`GraphTensor`](./graph_tensor.md) instances to files is to

1.  Create eager instances of `GraphTensor`
2.  Call `tensorflow_gnn.write_example()`
3.  You serialize the `tf.train.Example` message to a file.

Instances of `GraphTensor` can be arbitrarily created. For example, to write out
a stream of 1000 randomly generated `Example` protos to a `TFRecords` file, you
can use this code:

```
import tensorflow_gnn as tfgnn

schema = tfgnn.read_schema(schema_filename)
graph_spec = tfgnn.create_graph_spec_from_schema_pb(schema)
with tf.io.TFRecordWriter(record_file) as writer:
  for _ in range(1000):
    graph = tfgnn.random_graph_tensor(graph_spec)
    example = tfgnn.write_example(graph)
    writer.write(example.SerializeToString())
```

In order to scale this up to large graphs, you need to write some code that

*   Iterates through the subset of nodes around which you will train a model.
*   Samples a subset of nodes of interest in its neighborhood
*   Creates an instance of `GraphTensor` to write out (as above).

We recommend utilizing a framework based on the Map-Reduce conceptual framework
in order to run this at scale. Apache Beam provides a portable abstraction for
writing jobs that will run on a variety of platforms, clusters and clouds.

## Encoding Graph Tensors

There are a variety of graph data formats in existence and your application data
may even live on an in-house data warehousing server. This means that in many
cases you will end up having to write your own sampler and produce the sampled
data format directly instead of
using the sampling tools
and converters we provide. The
sampled data is encoded as a set of `tf.train.Feature` objects within a
`tf.train.Example` protocol buffer (protocol buffer messages are efficient
compact binary containers for your training data). For more information about
these, see the
[TensorFlow documentation](https://www.tensorflow.org/api_docs/python/tf/train/Example).

If the encoding is carried out in Python (e.g. from an Apache Beam job), you
should be able to create an instance of an eager GraphTensor in memory (e.g.
using the `GraphTensor.from_pieces()` method) and use the
[`tfgnn.write_example()`](../api_docs/python/tfgnn/write_example.md)
function to encode it to a `tf.train.Example`. However,
if this is not done in Python, you will have to write your own encoder.

Fortunately, this is not difficult. This section describes in detail the
encoding of graph tensors that you will need to carry out your data production,
matching the [graph schema](./schema.md) you defined for your application.
(Recall the schema might require auxiliary node sets and edge sets, e.g., for
readout.)

### Feature Names

A `GraphTensor` instance contains a hierarchy of tensors. Each of those are
named after their location in the `GraphTensor`, with the following
`tf.train.Feature` names in a `tf.train.Example` proto:

```
context/<feature_name>
nodes/<set_name>.<feature_name>
edges/<set_name>.<feature_name>
```

`<set_name>` is the name of the node set or edge set. These should correspond
directly to the features declared in the [`GraphSchema`](./schema.md) message.

### Feature dtypes

The [graph schema](./schema.md) supports a variety of TensorFlow dtypes while
`tf.train.Feature` supports only three
[protobuf types](https://protobuf.dev/programming-guides/proto3/#scalar):

  * `int64` for all integer types and the boolean type,
  * `float` for all floating-point types,
  * `bytes` for strings.

Values are automatically converted between the schema-defined dtype in the
`GraphTensor` and the matching dtype in `tf.train.Feature` using `tf.cast()`.

<!-- TODO(b/226098194): Address the following sentence. -->
WARNING: Beware of `DT_DOUBLE` in the `GraphSchema`. It gets stored as a
`DT_FLOAT`; better to declare it as `DT_FLOAT` then.

### Special Features

Three of the features are implicitly defined:

*   **Size features.** These features provide the number of nodes in a set.
    These are important and should always appear if a set is not empty, because
    it is possible for node sets to have no feature data but only aggregate
    features from incoming edges. We call these “latent nodes.” Size features
    are always of type `tf.int64` and are encoded as the following
    `tf.train.Feature` name:

    ```
    nodes/<set_name>.#size
    ```

    or

    ```
    edges/<set_name>.#size
    ```

*   **Source and Target Indices**. These features provide the node indices for
    each endpoint of a binary edge. The features are always of type `tf.int64`
    and are encoded as

    ```
    edges/<set_name>.#source
    ```

    and

    ```
    edges/<set_name>.#target
    ```

    Recall that the notion of "source" and "target" captures the direction
    of an edge in the graph (as a data structure), but does not prescribe
    which way a GNN model can send information along the edge.

Note: Serialization for hyper-adjacency — edges with more than two endpoints —
is not currently supported, and when support for it is added, some modifications
may occur on these names.

### Encoding Dense Tensors

For dense tensors of fixed shape, the list of values for an entire set are
flattened and concatenated. This means that if you have a feature of 4 x 4
blocks of floats, for a node set with 3 nodes, that feature would have to encode
3 x 4 x 4 = 48 floats. The parser takes care to automatically reshape the
feature based on the size of the graph and the known shape for each node.

### Scalar Features

Scalar features differ from tensors of shape `[1]` in the resulting container.
However, when encoded, since all we provide is a flattened list of all the data,
encodings for scalar features and features of rank 1 with one value are
identical. The parser takes care to reshape the features as expected upon
reading them.

### Encoding Ragged Tensors

Features with some dimensions of variable size require you to provide the shape
of each value for those dimensions. For example, for a feature of the following
declared shape:

```
shape { dim { size: -1 } }
dtype: DT_INT64
```

for a node set with 3 nodes, an example of this feature might have the following
value:

```
[[10, 15, 23],
 [89],
 [64, 53, 25, 29]]
```

This would be encoded like this:

```
features {
  feature {
    key: "nodes/students.#size"
    value {
      int64_list {
        value: 3
      }
    }
  }
  feature {
    key: "nodes/students.scores"
    value {
      int64_list {
        value: 10
        value: 15
        value: 23
        value: 89
        value: 64
        value: 53
        value: 25
        value: 29
      }
    }
  }
  feature {
    key: "nodes/students.scores.d1"
    value {
      int64_list {
        value: 3
        value: 1
        value: 4
      }
    }
  }
}
```

The values `[3, 1, 4]` refer to the number of items in each row. Notice how
those sizes—the ragged dimension—are provided as a feature with a special name:

```
edges/<set_name>.d<dim>
```

where `<dim>` is the index of the ragged dimension. In the example, the shape of
that unbatched tensor will be `[3, None]`, and the ragged dimension is at index
1 (starting from 0), so we store it as `<feature-name>.d1`. The feature values
are provided as a single flat list of the `flat_values` of the corresponding
ragged tensor. Note that if you have multiple dimensions, you would encode each
of the ragged dimensions as its own feature. And for context features, do take
into account the fact that the shape includes an initial `[1]` dimension.

Finally, note that our parser is a thin wrapper onto the regular TensorFlow
feature specs for parsing ragged tensors:
[tf.io.RaggedFeature](https://www.tensorflow.org/api_docs/python/tf/io/RaggedFeature).
If you’re familiar with those, you can generate the parsing spec using our
`tfgnn.get_io_spec()` utility function.

### Empty Features

Features that have no values can either be omitted or present but with empty
lists of values. The former would be encoded like this:

```
features {
  feature {
    key: "nodes/students.#size"
    value {
      int64_list {
        value: 3
      }
    }
  }
}
```

This is also a valid encoding for the same tensor of 3 empty rows:

```
features {
  feature {
    key: "nodes/students.#size"
    value {
      int64_list {
        value: 3
      }
    }
  }
  feature {
    key: "nodes/students.scores"
    value {
      bytes_list {
      }
    }
  }
  feature {
    key: "nodes/students.scores.d1"
    value {
      int64_list {
      }
    }
  }
}
```

### Empty Sets

Declared sets that happen to have no nodes can simply be omitted and will be
created as empty tensors. For example:

```
ex = tf.train.Example()
gt2 = tfgnn.parse_single_example(spec, tf.constant(ex.SerializeToString()))
print(gt2.node_sets["students"]["scores"])
print(gt2.node_sets["students"].sizes)
<tf.RaggedTensor []>
tf.Tensor([0], shape=(1,), dtype=int32)
```

Alternatively, you can explicitly encode a zero tensor for the sizes. For
example, this would also work:

```
ex = tf.train.Example()
ex.features.feature["nodes/students.#size"].int64_list.value.append(0)
gt2 = tfgnn.parse_single_example(spec, tf.constant(ex.SerializeToString()))
```

### Generating Example Encodings

In order to understand the encoding format, it may be additionally helpful to
poke around and print encodings for a particular spec or graph schema. You may
do this by inspecting randomly generated data fitting a schema as follows:

```
import tensorflow_gnn as tfgnn
schema = tfgnn.parse_schema("""
  node_sets {
    key: "students"
    value {
      features {
        key: "scores"
        value {
          shape { dim { size: 3 }}
          dtype: DT_INT32
        }
      }
    }
  }
""")
spec = tfgnn.create_graph_spec_from_schema_pb(schema)
gt = tfgnn.random_graph_tensor(spec)
print(tfgnn.write_example(gt))
```

You might get:

```
features {
  feature {
    key: "nodes/students.#size"
    value {
      int64_list {
        value: 4
      }
    }
  }
  feature {
    key: "nodes/students.scores"
    value {
      int64_list {
        value: 36
        value: 34
        value: 65
        value: 67
        value: 51
        value: 71
        value: 83
        value: 19
        value: 91
        value: 84
        value: 79
        value: 79
      }
    }
  }
}
```

And for a ragged feature:

```
schema = tfgnn.parse_schema("""
  node_sets {
    key: "students"
    value {
      features {
        key: "scores"
        value {
          shape { dim { size: -1 }}
          dtype: DT_STRING
        }
      }
    }
  }
""")
spec = tfgnn.create_graph_spec_from_schema_pb(schema)
gt = tfgnn.random_graph_tensor(spec)
print(tfgnn.write_example(gt))
```

You might get:

```
You might get:
features {
  feature {
    key: "nodes/students.#size"
    value {
      int64_list {
        value: 4
      }
    }
  }
  feature {
    key: "nodes/students.scores"
    value {
      bytes_list {
        value: "S"
        value: "L"
        value: "Z"
        value: "M"
        value: "A"
        value: "X"
        value: "Z"
        value: "W"
        value: "E"
        value: "I"
        value: "J"
        value: "K"
        value: "L"
        value: "K"
        value: "Z"
        value: "M"
      }
    }
  }
  feature {
    key: "nodes/students.scores.d1"
    value {
      int64_list {
        value: 6
        value: 5
        value: 2
        value: 3
      }
    }
  }
}
```

If you have a specific tensor value you’d like to test out, you can also create
a specific `GraphTensor` value and encode it like this:

```
rt = tf.ragged.constant(
  [[10, 15, 23],
   [89],
   [64, 53, 25, 29]])

gt = tfgnn.GraphTensor.from_pieces(
    node_sets={
        "students": tfgnn.NodeSet.from_fields(
          sizes=[3],
          features={"scores": rt}
    )})

tfgnn.write_example(gt)
```

## Readout

The library supports various kinds of prediction tasks on `GraphTensor`
datasets. Some of them deal with the graph as a whole, while many others refer
to specific nodes (or edges) in the graph. For the latter kind, GNN models
eventually need to read out the final hidden state only from a specific subset
of nodes, as the [modeling guide](gnn_modeling.md) describes in greater detail.

Usually, it falls on the data preparation stage of modeling to identify the
nodes for readout, more precisely, on the part of data preparation that
produces the `GraphTensor` values consumed by the GNN model. (If a graph
sampling tool is used, it usually falls on that tool.)

Recall the preceding example: graphs with a node set `"students"`. Consider
data preparation for a node regression task, in which some but not all nodes
have a training label. Let's call those the `"seed"` nodes (for reasons that
will become more clear in the next section). Suppose we have a `GraphTensor`
with student nodes indexed 0, 1, 2, and 3, of which only 1 and 3 have a training
label, so model training shall make a prediction only for those two seed nodes.

We use the flexibility of heterogeneous graphs to express this as a
*readout structure* that consists of the following auxiliary node sets and
edge sets:

  * The auxiliary node set `"_readout"` has 2 nodes, one for each prediction
    to make.
  * The auxiliary edge set `"_readout/seed"` connects node set `"students"`
    as its source to the node set `"_readout"` as its target. It has exactly one
    edge per prediction (that is, the edges `1->0` and `3->1` for this example),
    and the edges are ordered such that their target indices form a contiguous
    range 0, 1, ... up to the size of node set `"_readout"`.

Structured readout is not constrained to a fixed node set. Consider a refinement
of the schema that distinguishes `"undergrad_students"` and `"grad_students"`,
which may have different input features but feed into the same prediction task.
This is represented by a readout structure like the following:

  * The auxiliary node set `"_readout"` has one node for each prediction
    to make.
  * The auxiliary edge sets `"_readout/seed/1"` and `"_readout/seed/2"`
    connect the distinct source node sets (here: `"undergrad_students"` and
    `"grad_students"`) to the target node set `"_readout"`. (You don't need to
    number them, just use unique suffixes.)
    Every `"_readout"` node occurs as the target of exactly one edge.
    Within each of these edge sets, the edges are sorted by ascending order of
    target node ids.

Readout is not constrained to a single node (what we called `"seed"` above).
As a different example, consider a link prediction task: given a `"source"`
and a `"target"` node, predict whether or not there was an edge between them
in the ground-truth graph data (before conversion to the `GraphTensor`s seen in
the training data). To make it more interesting, let's say the source of the
predicted link is from node set `"users"` but the target can be from node set
`"users"` or `"items"`.

This is represented by a readout structure as follows:

  * The auxiliary node set `"_readout"` has one node for each link prediction
    to make.
  * The auxiliary edge set `"_readout/source"` connects node set `"users"` as
    its source to node set `"_readout"` as its target, subject to the rules on
    indices explained above.
  * The auxiliary edge sets `"_readout/target/1"` and `"_readout/target/2"`
    connect node sets `"users"` and `"items"` as their respective sources to
    node set `"_readout"` as their respective target, subject to the rules on
    indices explained above.

To summarize: A readout structure consists of an auxiliary node set,
conventionally called `"_readout"`, and auxiliary edge sets pointing into that
node set with matching names `"_readout/.*"`. Together, they define

  * a number of predictions to make: the size of node set `"_readout"`;
  * a set of keys for readout: these appear in the names of the edge sets with
    target `"_readout"` (`{"seed"}` in the first example, `{"source", "target"}`
    in the second);
  * for each prediction *i* and each key *k*, exactly one node from which
    to read out a value.

As far as `GraphTensor` and its serialization is concerned, these are node sets
and edge sets like any other. In particular, they need to be declared in the
[`GraphSchema`](./schema.md) so that they can be parsed from a `tf.Example`
proto.
The leading underscore `_` marks them as auxiliary, that is, distinct from
the arbitrarily-named node sets and edge sets which represent objects and
relations of the model's application domain.

Prior to TF-GNN 0.6 (released in 2023), there was no support for structured
readout. Instead, by convention, readout for a subset of nodes could only happen
for a single node in each input graph, which had to be from a fixed node set and
appear as the first node of that node set. The GraphTensor itself did not
encode which node set it is. This had to be conveyed separately.

### Storing labels

The primary purpose of the `"_readout"` node set is to be the target of the
readout edge sets, along which the GNN model will read out its final hidden
states in order to make predictions. We can think of its node set size as
`num_predictions`.

A convenient secondary purpose of the `"_readout"` node set is to store data
in its feature dict that naturally has a shape `[num_predictions, ...]`
and is not seen by the GNN as it operates on the non-auxiliary node sets.
That makes `"_readout"` a good fit for storing the label, especially for tasks
like link prediction in which there is no other single item in the `GraphTensor`
that directly corresponds to the (source, target) pair in question. (The edge
in question is usually removed in case it exists, as to not leak the correct
prediction.)


## Graph Sampling

As described in the [introduction](intro.md), many practically relevant graphs
are very large (e.g., a large social network may have billions of nodes) and may
not fit in memory, so this library resorts to sampling neighborhoods around
nodes which we want to train over (say, nodes with associated ground truth
labels). The library comes with a [Beam Sampler](beam_sampler.md) written in
Apache Beam that stores sampled subgraphs encoded as `tf.train.Example` proto
messages in sharded output files. This is the format we produce for training
datasets from data preparation jobs.

### Input Graph Format

The graph sampler accepts graphs in a simple data format we call “unigraph.”
This data format supports very large, homogeneous and heterogeneous graphs with
variable number of node sets and edge sets. In order to use the graph sampler
we provide, you need to convert your graph in the unigraph format.

A unigraph dataset is defined by a central text-formatted protocol buffer
message that describes the topology of the graph using the same `GraphSchema`
message used for graph tensors (but describing the full, unsampled graph).
For each context, node set and edge set, there is an associated “table” of ids
and features. Each table can be one of many supported formats, such as a CSV
file, sharded files of serialized `tf.train.Example` protos in a TFRecords
container, and more. The filename associated with each set’s table is provided
as metadata in the `filename` field of its metadata and can be a local name.
Typically, a schema and all the tables live under the same directory, which is
dedicated to that graph’s data.

Any sets of features can be defined on these tables; requirements on the table
files are minimal:

*   Node sets are required to provide a special **id** string column to identify
    the node row.
*   Edge sets are required to provide two special string columns: **source** and
    **target**, defining the origin and destination of the edge. Edge rows may
    also contain features, such as weight (or anything else).
*   Context sets have no special requirement, this is for any data applying to
    entire sampled subgraphs. Common uses include storing sampled subgraph
    labels or properties common to all sampled nodes (e.g., a gravitational
    constant in a physics simulation).

This format is kept as simple and flexible on purpose. See `unigraph.py` in the
source code for an Apache Beam reader library that can be used to read those
files and process them.

### Sampler Configuration

The sampler is configured by providing three files:

*   **A graph, in unigraph format.** This is specified by providing the path to
    the text-formatted
    [`GraphSchema` protocol buffer file](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/proto/graph_schema.proto),
    or the directory containing it and the graph’s data tables. This file has to
    include `filename` fields to existing files.
*   **A sampling specification.** This is a text-formatted
    [`sampling_spec.proto:SamplingSpec`](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/sampler/sampling_spec.proto)
    proto file that defines how sampling will be performed. For example, you can
    specify how many sampling steps to run and which sampling strategy to use at
    each hop. For full details on the sampling specification, see the proto file
    for instructions. There is also an example below.
*   **The seed node ids to sampler.** An (optional) input file with a list of
    nodes of interest at which to seed the sampling. This defines the points
    where the sampler will gather a neighborhood for training, testing and/or
    inference. This file can be in any of the supported `Universal Graph Format`
    table formats. If this is not provided, every node specified in the `seed_op`
    node set will be used as a sampling seed.

Upon completion, the sampler will output files with **serialized
[`GraphTensor`](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/graph/graph_tensor.py#:~:text=class%20GraphTensor)
instances as
[`tf.train.Example`](https://www.tensorflow.org/api_docs/python/tf/train/Example)
protos**. These can then be read using the `tfgnn.parse_example()` function
mapping over a stream of these protos provided by a
[`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset),
as is customary in TensorFlow.

Recall that sampling is meant to give the seed node a sufficiently large
neighborhood such that a GNN model can compute a useful hidden state.
Hence readout only makes sense for the seed node; other nodes (even if in the
same node set) need their own subgraphs sampled around them to compute an
equally useful hidden state. Conversely, a node whose hidden state is never
read out is not useful as a seed node for sampling.

Hence training with sampled subgraphs usually implies that readout happens
precisely from the seed node of each input graph.

As of this writing (July 2023), the Beam sampler does not yet create an
explicit `"_readout"` node set. Instead, it samples each subgraph from a single
seed node, always from the same node set, and stores the seed as the first node
of that node set. It falls on the model to know that special node set and read
out the hidden state of its first node with `tf.gather_first_node()`, or to add
the equivalent readout structure using `tfgnn.add_readout_from_first_node()`.

### End-to-End Heterogeneous Graph Sampling: OGBN-MAG

#### The OGBN-MAG Dataset

The following is an example of sampling over the
[OBGN-MAG](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag) dataset, a popular
citation network used for benchmarking a node prediction task; classifying a
label associated with certain nodes in the network. In the case of OGBN-MAG a
node prediction challenge would be to predict the venue (journal or conference)
that a paper in a held-out test set is published at. The node and edge sets for
the OBGN-MAG dataset are as follows:

##### Node Sets

*   `"paper"` contains 736,389 published academic papers, each with a
    128-dimensional word2vec feature vector computed by averaging the embeddings
    of the words in its title and abstract.
*   `"field_of_study"` contains 59,965 fields of study, with no associated
    features.
*   `"author"` contains the 1,134,649 distinct authors of the papers, with no
    associated features
*   `"institution"` contains 8740 institutions listed as affiliations of
    authors, with no associated features.

##### Edge Sets:

*   `"cites"` contains 5,416,217 edges from papers to the papers they cite.
*   `"has_topic"` contains 7,505,078 edges from papers to their zero or more
    fields of study.
*   `"writes"` contains 7,145,660 edges from authors to the papers that list
    them as authors.
*   `"affiliated_with"` contains 1,043,998 edges from authors to the zero or
    more institutions that have been listed as their affiliation(s) on any
    paper.

The graph schema file defines the topology of the full graph by enumerating the
node (entity) and edge (relationship) types, the features each entry contains
and the location of the tables that represent the graph:

```
node_sets {
  key: "author"
  value {
    metadata {
      filename: "nodes-author.tfrecords@15"
      cardinality: 1134649
    }
  }
}
node_sets {
  key: "field_of_study"
  value {
    metadata {
      filename: "nodes-field_of_study.tfrecords@2"
      cardinality: 59965
    }
  }
}
node_sets {
  key: "institution"
  value {
    metadata {
      filename: "nodes-institution.tfrecords"
      cardinality: 8740
    }
  }
}
node_sets {
  key: "paper"
  value {
    features {
      key: "feat"
      value {
        dtype: DT_FLOAT
        shape {
          dim {
            size: 128
          }
        }
      }
    }
    features {
      key: "labels"
      value {
        dtype: DT_INT64
        shape {
          dim {
            size: 1
          }
        }
      }
    }
    features {
      key: "year"
      value {
        dtype: DT_INT64
        shape {
          dim {
            size: 1
          }
        }
      }
    }
    metadata {
      filename: "nodes-paper.tfrecords@397"
      cardinality: 736389
    }
  }
}
edge_sets {
  key: "affiliated_with"
  value {
    source: "author"
    target: "institution"
    metadata {
      filename: "edges-affiliated_with.tfrecords@30"
      cardinality: 1043998
    }
  }
}
edge_sets {
  key: "cites"
  value {
    source: "paper"
    target: "paper"
    metadata {
      filename: "edges-cites.tfrecords@120"
      cardinality: 5416271
    }
  }
}
edge_sets {
  key: "has_topic"
  value {
    source: "paper"
    target: "field_of_study"
    metadata {
      filename: "edges-has_topic.tfrecords@226"
      cardinality: 7505078
    }
  }
}
edge_sets {
  key: "writes"
  value {
    source: "author"
    target: "paper"
    metadata {
      filename: "edges-writes.tfrecords@172"
      cardinality: 7145660
    }
  }
}
```

A TF-GNN modeler would craft a SamplingSpec proto configuration for a particular
task. For the OGBN-MAG venue prediction challenge,
[examples/mag/sampling_spec.pbtxt](https://github.com/tensorflow/gnn/blob/main/examples/mag/sampling_spec.pbtxt)
contains one such valid specification:

```
seed_op <
  op_name: "seed"
  node_set_name: "paper"
>
sampling_ops <
  op_name: "seed->paper"
  input_op_names: "seed"
  edge_set_name: "cites"
  sample_size: 32
  # Sample edges uniformly at random, because that works without any further
  # information. We could use TOP_K or RANDOM_WEIGHTED if we had put a
  # "#weight" column into the edge set's input table.
  strategy: RANDOM_UNIFORM
>
sampling_ops <
  op_name: "paper->author"
  input_op_names: "seed"
  input_op_names: "seed->paper"
  edge_set_name: "written"
  sample_size: 8
  strategy: RANDOM_UNIFORM
>
sampling_ops <
  op_name: "author->paper"
  input_op_names: "paper->author"
  edge_set_name: "writes"
  sample_size: 16
  strategy: RANDOM_UNIFORM
>
sampling_ops <
  op_name: "author->institution"
  input_op_names: "paper->author"
  edge_set_name: "affiliated_with"
  sample_size: 16
  strategy: RANDOM_UNIFORM
>
sampling_ops <
  op_name: "paper->field_of_study"
  input_op_names: "seed"
  input_op_names: "seed->paper"
  input_op_names: "author->paper"
  edge_set_name: "has_topic"
  sample_size: 16
  strategy: RANDOM_UNIFORM
>
```

The sampling schema may be better understood by visualizing it's operations and
graph exploration (traversal) in plate notation:

<p align=center>
  <img style="width:50%" src="images/ogbn_mag_sampling_spec.svg">
</p>

Which specifies the following sampling procedure:

1.   Select all nodes from `"paper"` node set as sampling seeds.
2.   Sample up to 32 `"cites"` edges outgoing from the seeds.
3.   Sample up to 8 `"written"` edges from each node in the seed `"paper"` and
     `"paper"` nodes drawn on the step 2.
4.   Sample up to 16 more `"writes"` edges for every `"author"` node from the
     step 3.
5.   For every `"author"` sample up to 16 `"affiliated_with"` edges.
6.   For every paper sample up to 16 `"has_topic"` edges.

NOTE: Graph Sampler samples **edges** from the graph. This allows
to better control complexity even for dense graphs with large-degree nodes.

##### Apache Beam and Google Cloud Dataflow

See [beam_sampler](beam_sampler.md) guide on how to use `tfgnn_sampler` tool for
distributed sampling using [Apache Beam](https://beam.apache.org/).

### Sampling Homogeneous Graphs

The example above supports any number of node sets and is broadly general over
heterogeneous graphs. Note that this maps well to database tables with cross
references: you would write a custom job to extract the data to be sampled from
your database to Unigraph format, or write a sampler that directly accesses them
yourself.

Homogeneous graphs in this context are simply a special case of heterogeneous
graphs with a single set of nodes and a single set of edges. There is nothing
special to do to support them.