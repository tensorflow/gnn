# The TF-GNN Beam Sampler

## Overview

*TensorFlow GNN (TF-GNN)* provides the `tfgnn_sampler` tool to
facilitate local neighborhood learning and convenient batching for graph
datasets. It provides a scalable and distributed means to sample even the
largest publicly-available graph datasets.

The Graph Sampler takes a sampling configuration, graph data, and optionally a
list of seed nodes as its inputs and produces sampled subgraphs as its output.
The graph data comes as `tf.Example`s in sharded files for graph edges and node
features. The generated subgraphs are as serialized `tf.Example`s that can be
parsed as `tfgnn.GraphTensor` using `tfgnn.parse_example()`.

The Graph Sampler is written in [Apache Beam](https://beam.apache.org/), an
open-source SDK for expressing
[Dataflow-Model](https://research.google/pubs/pub43864/) data processing
pipelines with support for multiple infrastructure backends. A client writes an
Apache Beam Pipeline and, at runtime, specifies a Runner to define the compute
environment in which the pipeline will execute.

The two main abstractions defined
by Apache Beam of concern are:

-   [Pipelines](https://beam.apache.org/documentation/programming-guide/#creating-a-pipeline):
    computational steps expressed as a DAG (Directed Acyclic Graph)
-   [Runners](https://beam.apache.org/documentation/runners/capability-matrix/):
    Environments for running Beam Pipelines


The simplest Beam runner is the
[DirectRunner](https://beam.apache.org/documentation/runners/direct/) which
allows to test Beam pipelines on local hardware. It requires all data to fit in
memory on a single machine and runs user code with extra debug checks enabled.
It is slow and should be used only for small-scale testing or prototyping.

[DataflowRunner](https://beam.apache.org/documentation/runners/dataflow/) that
enables clients to connect to a
[Google Cloud Platform (GCP)](https://cloud.google.com/) and execute a Beam
pipeline on GCP hardware through the
[Dataflow](https://cloud.google.com/dataflow) service. It enables
[horizontal scaling](https://cloud.google.com/dataflow/docs/horizontal-autoscaling)
which allows to sample graphs even with billions of edges.

## Getting Started - Direct Runner

NOTE: Only use the DirectRunner for small-scale testing.


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
tfgnn_sampler \
  --data_path="." \
  --graph_schema graph_schema.pbtxt \
  --sampling_spec sampling_spec.pbtxt \
  --output_samples outputs/examples.tfrecords \
  --runner DirectRunner
```

## Larger-scale Sampling with the Beam DataFlow runner

The `examples/sampler/mag` directory contains some of the components needed to
run the sampler for [OGBN-MAG](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag)
described in detail in [the data prep guide](./data_prep.md)

The directory should contain the following:

*   The `graph_schema.pbtxt` is a Graph Schema with filenames in the metadata
    for all edge sets and all node sets with features.

*   The `sampling_spec.pbtxt` is a sampling specification.

*   The `run_mag.sh` script has the command to start the DataFlow pipeline run.
    This script configures location, machine types to use, sets desired
    parallelism as minimum/maximum number of workers and the number of threads
    for each worker.

*   The `setup.py` file is used by the pipeline workers to install their
    [dependencies](https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/),
    e.g., the `apache-beam[gcp]`, `tensorflow` and `tensorflow_gnn` libraries.

This example further assumes that OGB data is converted to the unigraph format,
e.g. using `tfgnn_convert_ogb_dataset`, and stored cloud storage as sharded
files for edges and node features:

```
nodes-paper.tfrecords-?????-of-?????
edges-affiliated-with.tfrecords-?????-of-?????
edges-cites.tfrecords-?????-of-?????
edges-has_topic.tfrecords-?????-of-?????
edges-writes.tfrecords-?????-of-?????
```



The sampler currently supports CSV files and TFRecord files corresponding to
each graph piece. For TFRecords, the filename should be a glob pattern that
identifies the relevant shards. The sampler also supports shorthand for a common
sharding pattern, where `<filename>.tfrecords@<shard-count>` is read as
`filename.tfrecords-?????-of-<5-digit shard-count>`.


Before running `run_mag.sh`, users must edit the `GOOGLE_CLOUD_PROJECT` and
`DATA_PATH` variables in the script.


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
