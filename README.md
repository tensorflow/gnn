# TensorFlow GNN

## Summary

TensorFlow GNN is a library to build
[Graph Neural Networks](tensorflow_gnn/docs/guide/intro.md) on the TensorFlow platform.
It provides...

  * a [`tfgnn.GraphTensor`](tensorflow_gnn/docs/guide/graph_tensor.md) type to represent
    graphs with a [heterogeneous schema](tensorflow_gnn/docs/guide/schema.md), that is,
    multiple types of nodes and edges;
  * tools for [data preparation](tensorflow_gnn/docs/guide/data_prep.md),
    notably a [graph sampler](tensorflow_gnn/docs/guide/beam_sampler.md)
    to convert a huge database into a stream of reasonably-sized subgraphs for
    training and inference;
  * a collection of [ready-to-use models](tensorflow_gnn/models/README.md)
    and Keras layers to do your own [GNN modeling](tensorflow_gnn/docs/guide/gnn_modeling.md);
  * a high-level API for training [orchestration](tensorflow_gnn/docs/guide/runner.md).

This library is an OSS port of a Google-internal library used in a broad variety
of contexts, on homogeneous and heterogeneous graphs, and in conjunction with
other scalable graph mining tools.

For background and discussion, please see O. Ferludin et al.:
[TF-GNN: Graph Neural Networks in TensorFlow](https://arxiv.org/abs/2207.03522),
2023 (full citation below).

## Quickstart

Google Colab lets you run TF-GNN demos from your browser, no installation
required:

  * [Molecular Graph
    Classification](https://colab.research.google.com/github/tensorflow/gnn/blob/master/examples/notebooks/intro_mutag_example.ipynb)
     with the MUTAG dataset.
  * [Solving OGBN-MAG
    end-to-end](https://colab.research.google.com/github/tensorflow/gnn/blob/master/examples/notebooks/ogbn_mag_e2e.ipynb)
    trains a model on heterogeneous sampled subgraphs from the popular
    [OGBN-MAG](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag) benchmark.
  * [Learning shortest paths with
    GraphNetworks](https://colab.research.google.com/github/tensorflow/gnn/blob/master/examples/notebooks/graph_network_shortest_path.ipynb)
    demonstrates an advanced Encoder/Process/Decoder architecture for predicting
    the edges of a shortest path.

For all colabs and user guides, please see the
[Documentation overview](tensorflow_gnn/docs/guide/overview.md)
page, which also links to the
[API docs](tensorflow_gnn/docs/api_docs/README.md).

## Installation Instructions

The latest stable release of TensorFlow GNN is available from

```
pip install tensorflow_gnn
```

For installation from source, see our [Developer
Guide](tensorflow_gnn/docs/guide/developer.md).

Key platform requirements:

  * TensorFlow 2.12, 2.13, 2.14 or 2.15, and any GPU drivers it needs
    [[instructions](https://www.tensorflow.org/install)].
  * Keras v2, as traditionally included with TensorFlow 2.x.
    (TF-GNN does not work with the new multi-backend Keras v3.)
  * Apache Beam for distributed graph sampling.

TF-GNN is developed and tested on Linux. Running on other platforms supported
by TensorFlow may be possible.

## Citation

When referencing this library in a paper, please cite the
[TF-GNN paper](https://arxiv.org/abs/2207.03522):

```
@article{tfgnn,
  author  = {Oleksandr Ferludin and Arno Eigenwillig and Martin Blais and
             Dustin Zelle and Jan Pfeifer and Alvaro Sanchez{-}Gonzalez and
             Wai Lok Sibon Li and Sami Abu{-}El{-}Haija and Peter Battaglia and
             Neslihan Bulut and Jonathan Halcrow and
             Filipe Miguel Gon{\c{c}}alves de Almeida and Pedro Gonnet and
             Liangze Jiang and Parth Kothari and Silvio Lattanzi and 
             Andr{\'{e}} Linhares and Brandon Mayer and Vahab Mirrokni and
             John Palowitch and Mihir Paradkar and Jennifer She and
             Anton Tsitsulin and Kevin Villela and Lisa Wang and David Wong and
             Bryan Perozzi},
  title   = {{TF-GNN:} Graph Neural Networks in TensorFlow},
  journal = {CoRR},
  volume  = {abs/2207.03522},
  year    = {2023},
  url     = {http://arxiv.org/abs/2207.03522},
}
```
