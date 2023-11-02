# The TF-GNN Sampler

## Overview

*TensorFlow GNN (TF-GNN)* provides a graph sampling tool to facilitate local
neighborhood learning and convenient batching for graph datasets. Using Apache
Beam, it provides a scalable and distributed means to sample even the largest
publicly-available graph datasets.

<!--*
# Document freshness: For more information, see go/fresh-source.
freshness: { owner: 'mparadkar' reviewed: '2023-06-30' }
*-->

[TOC]

## Getting Started - Direct Runner

To successfully use the Graph Sampler, we need a few items set up. In
particular, we need a schema for the graph, a specification for the sampling
operations, and available data. As a motivating example, we can use a dataset of
fake credit card data in the `examples/sampler/creditcard` directory.

In particular, let's use the following graph schema, with a graph of customers,
credit cards, and ownership linking them. For any node sets with features, there
should also be a feature called `#id`, and edge sets should contain `#source`
and `#target` features that correspond to node `#id`s. These special features
should not be explicitly specified in the graph schema. For more information,
see [the data prep guide](./data_prep.md). Here, both node sets have features
besides simple `#id`s, so we need to specify the files that map the
`#id`s to features.

```
node_sets {
  key: "customer"
  value {
    features {
      key: "name"
      value: {
        description: "Name"
        dtype: DT_STRING
      }
    }
    features {
      key: "address"
      value: {
        description: "address"
        dtype: DT_STRING
      }
    }
    features {
      key: "zipcode"
      value: {
        description: "Zipcode"
        dtype: DT_INT64
      }
    }
    features {
      key: "score"
      value: {
        description: "Credit score"
        dtype: DT_FLOAT
      }
    }
    metadata {
      filename: "customer.csv"
    }
  }
}

node_sets {
  key: "creditcard"
  value {
    metadata {
      filename: "creditcard.csv"
    }
    features {
      key: "number"
      value: {
        description: "Credit card number"
        dtype: DT_INT64
      }
    }
    features {
      key: "issuer"
      value: {
        description: "Credit card issuer institution"
        dtype: DT_STRING
      }
    }
  }
}

edge_sets {
  key: "owns_card"
  value {
    description: "Owns and uses the credit card."
    source: "customer"
    target: "creditcard"
    metadata {
      filename: "owns_card.csv"
    }
  }
}
```

We also need to create a sampling spec to indicate how we want to create the
subgraphs. Here, we'll treat the customer as the seed node, and sample up to 3
associated credit cards at random.

```
seed_op <
  op_name: "seed"
  node_set_name: "customer"
>
sampling_ops <
  op_name: "seed->creditcard"
  input_op_names: "seed"
  edge_set_name: "owns_card"
  sample_size: 3
  strategy: RANDOM_UNIFORM
>
```

We can run the sampler on this data via the following command, using the Apache
Beam direct runner.

```
cd <path-to>/gnn/examples/sampler/creditcard
python3 -m tensorflow_gnn.experimental.sampler.beam.sampler \
  --graph_schema graph_schema.pbtxt \
  --sampling_spec sampling_spec.pbtxt \
  --output_samples outputs/examples.tfrecord \
  --runner DirectRunner
```

## Larger-scale Sampling with the Beam DataFlow runner

The Beam direct runner requires all data to fit in memory on a single machine.
The sampler also works with DataFlow on GCP to provide distributed sampling for
graphs with up to hundreds of millions of nodes and billions of edges.

The `examples/sampler/mag` directory contains some of the components needed to
run the sampler on OGBN-MAG. This example further assumes that the graph schema,
sampling spec, and data are in a directory on cloud storage. For this example,
the directory should contain the following (with data files in either TFRecord
or CSV format):

```
graph_schema.pbtxt
sampling_spec.pbtxt
nodes-paper.tfrecords
edges-affiliated-with.tfrecords
edges-cites.tfrecords
edges-has_topic.tfrecords
edges-writes.tfrecords
```

Graph Schema:

The graph schema should have filenames in the metadata for all edge sets and all
node sets with features. This is an example schema for OGBN-MAG.

```
node_sets {
  key: "author"
  value {}
}
node_sets {
  key: "field_of_study"
  value {}
}
node_sets {
  key: "institution"
  value {}
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
    }
  }
}
edge_sets {
  key: "written"
  value {
    source: "paper"
    target: "author"
    metadata {
      filename: "edges-writes.tfrecords@172"
      extra {
        key: "edge_type"
        value: "reversed"
      }
    }
  }
}
```

The sampler currently supports CSV files and TFRecord files corresponding to
each graph piece. For TFRecords, the filename should be a glob pattern that
identifies the relevant shards. The sampler also supports shorthand for a common
sharding pattern, where `<filename>.tfrecords@<shard-count>` is read as
`filename.tfrecords-?????-of-<5-digit shard-count>`.

Sampling Spec:

The sampling spec is similar to the previous example, just with some more
sampling operations.

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

Finally, we have a script `run_mag.sh` with the command to start the DataFlow
pipeline run. This refers to a `setup.py` file which the pipeline workers need
to install their dependencies.

## Performance and Cost Comparisons

The sampler has achieved the following performance and costs with GCP DataFlow
on the following datasets. The costs here reflect GCP pricing in June 2023, and
may change in the future.

Dataset                                                                                   | Generated Subgraphs | Input Data Size | Subgraph Data Size | Machine Type | Min/Max Workers | Threads per Worker | Execution Time | Estimated Cost
----------------------------------------------------------------------------------------- | ------------------- | --------------- | ------------------ | ------------ | --------------- | ------------------ | -------------- | --------------
OGBN-Arxiv                                                                                | 169k                | 1.4GB           | 1.2GB              | n1-highmem-2 | 5/15            | 2                  | 18min          | $0.18
OGBN-MAG                                                                                  | 736k                | 2.4GB           | 108GB              | n1-highmem-2 | 30/100          | 2                  | 45min          | $11
MAG-240M ([MAG240m](https://ogb.stanford.edu/docs/lsc/mag240m/) LSC contest requirements) | 1.4M                | 502GB           | 945GB              | n1-highmem-2 | 100/300         | 2                  | 47min          | $47

## Scalability Examples

To demonstrate the scalability of this distributed sampler, we have also
generated subgraphs for all 121M papers in the OGB LSC dataset.

Dataset                                           | Generated Subgraphs | Input Data Size | Subgraph Data Size | Machine Type | Min/Max Workers | Threads per Worker | Execution Time | Estimated Cost
------------------------------------------------- | ------------------- | --------------- | ------------------ | ------------ | --------------- | ------------------ | -------------- | --------------
MAG-240M (sampling subgraphs for all 121M papers) | 121M                | 502GB           | 67TB               | n1-highmem-4 | 300/1000        | 1                  | 4h             | $4100
