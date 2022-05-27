# Data Preparation and Sampling

## Introduction

The `tensorflow_gnn` library supports reading streams of `tf.train.Example`
proto messages with all the contents of a graph encoded in them. This document
describes how to produce such a stream of encoded data using the library helper
functions, details of the encoding (if you’d like to write your own graph data
generator), and showcases a scalable sampling tool based on Apache Beam that we
provide to sample from very large graph. You can provide this tool the large
graph format it supports, further adapt it to process data in the formats you
own.

## Writing Graph Tensors to Files

If you use Python, the most straightforward method to produce streams of
`GraphTensor` instances to files is to

1.  Create eager instances of `GraphTensor`
2.  Call `tensorflow_gnn.write_example()`
3.  You serialize the `tf.train.Example` message to a file.

Instances of `GraphTensor` can be arbitrarily created. For example, to write out
a stream of 1000 randomly generated `Example` protos to a `TFRecords` file, you
can use this code:

```
import tensorflow_gnn as tfgnn

schema = tfgnn.read_schema(schema_filename)
with tf.io.TFRecordWriter(record_file) as writer:
  for _ in range(1000):
    graph = tfgnn.random_graph_tensor_from_schema(schema)
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
data format directly instead of using the sampling tools and converters we
provide. The sampled data is encoded as a set of `tf.train.Feature` objects
within a `tf.train.Example` protocol buffer (protocol buffer messages are
efficient compact binary containers for your training data). For more
information about these, see the
[TensorFlow documentation](https://www.tensorflow.org/api_docs/python/tf/train/Example)
.

If the encoding is carried out in Python (e.g. from an Apache Beam job), you
should be able to create an instance of an eager GraphTensor in memory (e.g.
using the `GraphTensor.from_pieces()` method) and use the
`tfgnn.write_example()` function to encode it to a `tf.train.Example`. However,
if this is not done in Python, you will have to write your own encoder.

Fortunately, this is not difficult. This section describes in detail the
encoding of graph tensors that you will need to carry out your data production.

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
directly to the features declared in the `GraphSchema` message.

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

*   **Source and Target Indices**. These features provide the edge indices for
    each endpoint of a binary edge. The features are always of type `tf.int64`
    and are encoded as

    ```
    edges/<set_name>.#source
    ```

    and

    ```
    edges/<set_name>.#target
    ```

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
shape { dim { size: -1 }
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
feature specs for parsing ragged tensors::
[tf.io.RaggedFeature](https://www.tensorflow.org/api_docs/python/tf/io/RaggedFeature)
. If you’re familiar with those, you can generate the parsing spec using our
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
print(gt2.node_sets['students']['scores'])
print(gt2.node_sets['students'].sizes)
<tf.RaggedTensor []>
tf.Tensor([0], shape=(1,), dtype=int32)
```

Alternatively, you can explicitly encode a zero tensor for the sizes. For
example, this would also work:

```
ex = tf.train.Example()
ex.features.feature['nodes/students.#size'].int64_list.value.append(0)
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
          dtype: DT_FLOAT
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
      float_list {
        value: 0.2121654748916626
        value: 0.5609347820281982
        value: 0.4877915382385254
        value: 0.1555272340774536
        value: 0.2997016906738281
        value: 0.8022844791412354
        value: 0.033922433853149414
        value: 0.3420950174331665
        value: 0.7682920694351196
        value: 0.49579453468322754
        value: 0.03295755386352539
        value: 0.4508802890777588
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
        'object': tfgnn.NodeSet.from_fields(
          sizes=[3],
          features={'counts': rt}
    )})

tfgnn.write_example(gt)
```

## Running the Built-in Sampler

The library comes with a basic sampler implementation that can produce such
streams of encoded `tf.train.Example` proto messages in sharded output files.
This is the format we produce for training datasets from data preparation jobs.

### Input Graph Format

The graph sampler accepts graphs in a simple data format we call “unigraph.”
This data format supports very large graphs, homogeneous and heterogeneous
graphs, with variable numbers of node sets and edge sets. In order to use the
graph sampler tool we provide, you need to convert your graph to unigraph
format.

A text-formatted protocol buffer message file describes the topology of a
unigraph formatted graph using the same `GraphSchema` message used for graph
tensors (but describing the full, unsampled graph), and for each context, node
set and edge set, there is an associated “table” of ids and features. Each table
can be one of many supported formats, such as a CSV file, sharded files of
serialized `tf.train.Example` protos in a TFRecords container, and more. The
filename associated with each set’s table is provided as metadata in the
`filename` field of its metadata and can be an absolute or local path
specification. Typically, a schema and all the tables live under the same
directory, which is dedicated to that graph’s data.

Any sets of features can be defined on these tables; requirements on the table
files are minimal:

*   Node sets are required to provide a special **id** string column to identify
    the node row.
*   Edge sets are required to provide two special string columns: **source** and
    **target**, defining the origin and destination of the edge. Edge rows may
    also contain features, such as weight, or anything else, really.
*   Context sets have no special requirement, this is for a table of data
    applying to the entire graph (and rarely used).

This format is kept as simple and flexible on purpose. See `unigraph.py` in the
source code for an Apache Beam reader library that can be used to read those
files and process them.

### Sampler Configuration

The sampler is configured by providing three files:

*   **A graph, in unigraph format.** This is specified by providing the path to
    the text-formatted `GraphSchema` protocol buffer file, or the directory
    containing it and the graph’s data tables. This file has to include
    `filename` fields to existing files.
*   **A sampling specification.** This is a text-formatted
    `sampling_spec.proto:SamplingSpec`proto file that defines how sampling will
    be performed. For example, you can specify how many sampling steps to run
    and which sampling strategy to use at each hop. For full details on the
    sampling specification, see the proto file for instructions. There is also
    an example below.
*   **The seed node ids to sampler.** An (optional) input file with a list of
    nodes of interest at which to seed the sampling. This defines the points
    where the sampler will gather a neighborhood for training, testing and/or
    inference. This file can be in any of the supported `Universal Graph Format`
    table formats. If this is not provided, very node specified in the `seed_op`
    node set will be used as a sampling seed.

Upon completion, the sampler will output a file with **serialized `GraphTensor`
instances as `tf.train.Example` protos**. These can then be read using the
`tfgnn.parse_example()` function mapping over a stream of these protos provided
by a `tf.data.Dataset`, as is customary in TensorFlow.

### Example Configuration

The following is an example of sampling over the OGBN-MAG dataset, a large,
popular heterogeneous citation network. The graph has four node sets (or types):

*   "paper" contains 736,389 published academic papers, each with a
    128-dimensional word2vec feature vector computed by averaging the embeddings
    of the words in its title and abstract.
*   "field_of_study" contains 59,965 fields of study, with no associated
    features.
*   "author" contains the 1,134,649 distinct authors of the papers, with no
    associated features.
*   "institution" contains 8740 institutions listed as affiliations of authors,
    with no associated features.

The graph has four directed edge sets (or types), with no associated features on
any of them.

*   "cites" contains 5,416,217 edges from papers to the papers they cite.
*   "has_topic" contains 7,505,078 edges from papers to their zero or more
    fields of study.
*   "writes" contains 7,145,660 edges from authors to the papers that list them
    as authors.
*   "affiliated_with" contains 1,043,998 edges from authors to the zero or more
    institutions that have been listed as their affiliation(s) on any paper.

The following GraphSchema message (pbtxt file) would define the graph topology
for the full graph, defining all the node sets and the edges sets relating them:

```
node_sets {
  key: "author"
  value {
    features {
      key: "#id"
      value {
        dtype: DT_STRING
      }
    }
    metadata {
      filename: "nodes-author.tfrecords@15"
      cardinality: 1134649
    }
  }
}
node_sets {
  key: "field_of_study"
  value {
    features {
      key: "#id"
      value {
        dtype: DT_STRING
      }
    }
    metadata {
      filename: "nodes-field_of_study.tfrecords@2"
      cardinality: 59965
    }
  }
}
node_sets {
  key: "institution"
  value {
    features {
      key: "#id"
      value {
        dtype: DT_STRING
      }
    }
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
     key: "#id"
      value {
        dtype: DT_STRING
      }
    }
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

One task is to **predict the venue** (journal or conference) at which a paper
from a test set has been published. Given the large number of papers in the
dataset, it may be possible to sample subgraphs from the full graph starting
from the "paper" node set. A plausible sampling specification for this task
could be:

```
seed_op {
  op_name: "seed"
  node_set_name: "paper"

}
sampling_ops {
  op_name: "seed->paper"
  input_op_names: "seed"
  edge_set_name: "cites"
  sample_size: 32
  # Sample edges uniformly at random, because that works without any further
  # information. We could use TOP_K or RANDOM_WEIGHTED if we had put a
  # "#weight" column into the edge set's input table.
  strategy: RANDOM_UNIFORM
}
sampling_ops {
  op_name: "paper->author"
  input_op_names: ["seed", "seed->paper"]
  edge_set_name: "written"
  sample_size: 8
  strategy: RANDOM_UNIFORM
}
sampling_ops {
  op_name: "author->paper"
  input_op_names: "paper->author"
  edge_set_name: "writes"
  sample_size: 16
  strategy: RANDOM_UNIFORM
}
sampling_ops {
  op_name: "author->institution"
  input_op_names: "paper->author"
  edge_set_name: "affiliated_with"
  sample_size: 16
  strategy: RANDOM_UNIFORM
}
sampling_ops {
  op_name: "paper->field_of_study"
  input_op_names: ["seed", "seed->paper", "author->paper"]
  edge_set_name: "has_topic"
  sample_size: 16
  strategy: RANDOM_UNIFORM
}
```

The sampling specification deines the following graph sampling traversal scheme:

1.  Use all entries in the "papers" node set as "seed" nodes (roots of the
    sampled subgraphs).
2.  Sample 16 more papers randomly starting from the "seed" nodes through the
    citation network. Call this sampled set "seed->paper".
3.  For both the "seed" and "seed->paper" sets, sample 8 authors using the
    "written" edge set. Name the resulting set of sampled authors
    "paper->author".
4.  For each author in the "paper->author" set, sample 16 institutions via the
    "affiliated_with" edge set.
5.  For each paper in the "seed", "seed->paper" and "author->paper" sample 16
    fields of study via the "has_topic" relation.

### Sampling Homogeneous Graphs

The example above supports any number of node sets and is broadly general over
heterogeneous graphs. Note that this maps well to database tables with cross
references: you would write a custom job to extract the data to be sampled from
your database to Unigraph format, or write a sampler that directly accesses them
yourself.

Homogeneous graphs in this context are simply degenerate graphs with a single
set of nodes and a single set of edges. There is nothing special to do to
support them.
