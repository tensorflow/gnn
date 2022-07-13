# TF-GNN: TensorFlow Graph Neural Networks

The TensorFlow GNN library makes it easy to build Graph Neural Networks, that
is, neural networks on graph data (nodes and edges with arbitrary features).
It provides TensorFlow code for building GNN models as well as tools for
preparing their input data and running the training.

Throughout, TF-GNN supports *heterogeneous* graphs, that is, graphs consisting
of multiple sets of nodes and multiple sets of edges, each with their own set of
features. These come up naturally when modeling different types of objects
(nodes) and their different types of relations (edges).

## User Documentation

Start with our introductory guides:

  * [Introduction to Graph Neural Networks](intro.md). This page introduces the
    concept of graph neural networks with a focus on their application at scale.

  * [The GraphTensor type](graph_tensor.md). This page introduces the
    `tfgnn.GraphTensor` class, which defines our representation of graph data
    in TensorFlow. We recommend that every user of our library understands its
    basic data model.

  * [Describing your graph](schema.md). This page explains how to declare the
    node sets and edge sets of your graph, including their respective features,
    with the `GraphSchema`
    [protocol message](https://developers.google.com/protocol-buffers).
    This defines the interface between data preparation (which creates such
    graphs) and the GNN model written in TensorFlow (which consumes these
    graphs as training data).

  * [Data preparation and sampling](data_prep.md). Training data for GNN
    models are graphs. When operating on very large graphs, we produce local
    subgraph samples from the full dataset which are serialized along with their
    features and streamed for training. This document describes the sampler tool
    as well as the data format we support for storing large graphs.

  * [The TF-GNN Runner](runner.md) lets you train GNN models on the
    prepared input data for a variety of tasks (e.g., node prediction).
    We recommend using the Runner to get started quickly with a first model
    for the data at hand, and then customize it as needed.

The following docs go deeper into particular topics.

  * The [Input pipeline](input_pipeline.md) guide explains how to set up
    a `tf.data.Dataset` for bulk input of the training and validation datasets
    produced by the [data preparation](data_prep.md) step. The TF-GNN Runner
    already takes care of this for its users.

  * [TF-GNN modeling](gnn_modeling.md) explains how to build a Graph Neural
    Network with TensorFlow and Keras, using the GraphTensor data from the
    previous steps. The TF-GNN library provides both a collection of standard
    models and a toolbox for writing your own. Users of the TF-GNN Runner
    are encouraged to consult this page to define custom models in the Runner.

## Colab Tutorials

These Colab notebooks run complete examples of building and training a TF-GNN
model on a Google server from within your browser.

  * [Molecular Graph
    Classification](https://colab.research.google.com/github/tensorflow/gnn/blob/master/examples/notebooks/intro_mutag_example.ipynb)
    trains a model for the MUTAG dataset (from the
    [TUDatasets](https://chrsmrrs.github.io/datasets/) collection) that consists
    of 188 small, homogeneous graphs representing molecules. This is a good
    first read to get acquainted with GNNs.
  * [Solving OGBN-MAG
    end-to-end](https://colab.research.google.com/github/tensorflow/gnn/blob/master/examples/notebooks/ogbn_mag_e2e.ipynb#scrollTo=udvGTpefWRE_)
    trains a model on heterogeneous sampled subgraphs from the
    [OGBN-MAG](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag) dataset (from
    Stanford's Open Graph Benchmark) that contains 1 million research papers,
    their authors, and other relations. This is a more advanced example that
    introduces node classification from sampled subgraphs as well as the nuts
    and bolts of training in parallel on multiple accelerators (GPU, TPU).
  * [Learning shortest paths with
    GraphNetworks](https://colab.research.google.com/github/tensorflow/gnn/blob/master/examples/notebooks/graph_network_shortest_path.ipynb)
    demonstrates an Encoder/Process/Decoder architecture for predicting the
    edges of a shortest path, using an Graph Network with edge states.
    Take a look if you are interested in advanced modeling.

## API Reference

TF-GNN has reference documentation for the API, extracted from the source code.

  * See [here](../api_docs/python/tfgnn.md) for the core library
    (`import tensorflow_gnn`) .
  * See
    [this list](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/models/README.md)
    for the bundled models (`from tensorflow_gnn.models import foo`).

## Developer Documentation

How to contribute to the TF-GNN library.

  * [CONTRIBUTING.md](https://github.com/tensorflow/gnn/blob/main/CONTRIBUTING.md)
    describes the process for open-source contributions.
  * The [Developer](developer.md) guide describes how to clone our github repo
    and install the tools and libraries required to build and run TF-GNN code.

## Talks

You can find related talks at our NeurIPS 2020 Expo workshop:
https://gm-neurips-2020.github.io/
