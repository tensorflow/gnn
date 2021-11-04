# TensorFlow GNN (EXPERIMENTAL)

**This is an early (alpha) release to get community feedback.** It's under
active development and **we may break API compatibility in the future**.


TensorFlow GNN is a library to build Graph Neural Networks on the TensorFlow
platform. It contains the following components:

* A high-level Keras-style API to create GNN models that can easily be composed
  with other types of models. GNNs are often used in combination with ranking,
  deep-retrieval (dual-encoders) or mixed with other types of models
  (image, text, etc.)

* GNN API for heterogeneous graphs. Many of the graph problems we approach at
  Google and in the real world contain different types of nodes and edges.
  Hence the emphasis in heterogeneous models.

* A well-defined schema to declare the topology of a graph, and tools to
  validate it. It describes the shape of its training data and serves to guide
  other tools.

* A GraphTensor composite tensor type which holds graph data, can be batched,
  and has efficient graph manipulation functionality available.

* A library of operations on the GraphTensor structure:

  * Various efficient broadcast and pooling operations on nodes and edges, and
    related tools.

  * A library of standard baked convolutions, that can be easily extended by
    ML engineers/researchers.

  * A high-level API for product engineers to quickly build GNN models without
    necessarily worrying about its details.

* A set of tools used to convert graph datasets and sample from large
  graphs.

* An encoding of graph-shaped training data on file, as well as a library used
  to parse this data into a data structure your model can extract the various
  features.

This library is an OSS port of a Google internal library used in a broad variety
of contexts, on homogeneous and heterogeneous graphs, and in conjunction with
other scalable graph mining tools.
