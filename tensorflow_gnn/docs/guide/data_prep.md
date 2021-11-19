# Data Preparation and Sampling

## Introduction

The `tensorflow_gnn` library supports reading streams of `tf.train.Example`
proto messages with all the contents of a graph encoded in them. This document
describes how to produce such a stream of encoded data using the library helper
functions, details of the encoding (if you’d like to write your own graph data
generator)<!-- copybara:strip_begin(google-internal) -->, and showcases a
scalable sampling tool based on Apache Beam that we provide to sample from very
large graphs<!-- copybara:strip_end -->. You can provide this tool the large
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
data format directly instead of
using<!-- copybara:strip_begin(google-internal) --> the sampling tools
and<!-- copybara:strip_end --> converters we provide. The
sampled data is encoded as a set of `tf.train.Feature` objects within a
`tf.train.Example` protocol buffer (protocol buffer messages are efficient
compact binary containers for your training data). For more information about
these, see the
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

<!-- copybara:strip_begin(graph sampler) -->

## Running the Built-in Sampler

The library comes with a basic sampler implementation that can produce such
streams of encoded `tf.train.Example` proto messages in sharded output files.
This is the format we produce for training datasets from data preparation jobs.

### Input Graph Format

The graph sampler accepts graphs in a simple data format we call “unigraph.”
This data format supports very large graphs, homogeneous and heterogeneous
graphs, and any number of node sets and edge sets. In order to use the graph
sampler tool we provide, you need to convert your graph in this flexible format.

The format can be described simply: a central text-formatted protocol buffer
message file describes the topology of the graph using the same `GraphSchema`
message used for graph tensors (but describing the full, unsampled graph), and
for each context, node set and edge set, there is an associated “table” of ids
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
*   **The seed node ids to sampler.** An input file with a list of nodes of
    interest at which to seed the sampling. This defines the points where the
    sampler will gather a neighborhood for training, testing and/or inference.
    This file can be in any of the supported `Universal Graph Format` table
    formats.

Upon completion, the sampler will output a file with **serialized `GraphTensor`
instances as `tf.train.Example` protos**. These can then be read using the
`tfgnn.parse_example()` function mapping over a stream of these protos provided
by a `tf.data.Dataset`, as is customary in TensorFlow.

### Example Configuration

The following is an example of sampling over the OGBN-MAG dataset. We start with
a single author node and sample 10 of their written papers in the first hop, and
then 2 topics from each of those 10 papers in the second hop.

The graph schema file would define the graph topology for the full graph,
defining all the node sets and the edges between them:

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

Then, a specification for how to sample these would be crafted, maybe like this:

```
insert_sample_ids: True
seed_op: {
   op_name: 'seed'
   node_set_name: 'author'
}
sampling_ops: {
   op_name: 'hop-1'
   input_op_names: [ 'seed' ]
   strategy: TOP_K
   sample_size: 10
   edge_set_name: 'affiliated_with'
}
sampling_ops: {
   op_name: 'hop-2'
   input_op_names: [ 'seed' ]
   strategy: TOP_K
   sample_size: 10
   edge_set_name: 'writes'
}
sampling_ops: {
   op_name: 'hop-3'
   input_op_names: [ 'hop-2' ]
   strategy: TOP_K
   sample_size: 2
   edge_set_name: 'has_topic'
}
```

This defines an initial set of seed node ids named “seed”, and three sampling
steps:

*   A sampling step around “seed” across the “affiliated_with” edges with a
    resulting set of nodes named “hop-1”;
*   Another sampling step around the original seed nodes across the “writes”
    edges with a result name called “hop-2”;
*   A second hop from the nodes in “hop-1” across the “has_topic” set of edges,
    named “hop-3”.

The resulting graph schema should be produced from this.

The seeds nodes would be provided as a table with a column “id”, e.g., a CSV
file like this:

```
#id
17
42
56
…
```

### Sampling Homogeneous Graphs

The example above supports any number of node sets and is broadly general over
heterogeneous graphs. Note that this maps well to database tables with cross
references: you would write a custom job to extract the data to be sampled from
your database to Unigraph format, or write a sampler that directly accesses them
yourself.

Homogeneous graphs in this context are simply degenerate graphs with a single
set of nodes and a single set of edges. There is nothing special to do to
support them.

<!-- copybara:strip_end -->
