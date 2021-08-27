# TensorFlow GNN

TensorFlow GNN is a library to build Graph Neural Networks on the TensorFlow
platform. It contains the following components:

* A well-defined schema to declare the topology of a graph, and tools to
  validate it. It describes the shape of its training data and serves to guide
  other tools.

* A set of tools used to convert graph datasets and sample from very large
  graphs. Some tools focus on small graphs that fit in a single computer's
  memory, and other tools handle cases of very large graphs that do not.

* An encoding of graph-shaped training data on disk, as well as a library used
  to parse this data into a data structure your model can extract the various
  features from.

* A `GraphTensor` composite tensor type which holds graph data, can be batched,
  and has graph manipulation routines available.

* A library of operations on this data structure which you can use to perform
  various operations, such as extracting the adjacency matrix, masking nodes
  and/or edges, and basic blocks to run graph convolutions.

* A high-level Keras-style API to compose GNN models.

This library is used at Google in a broad variety of contexts, on homogeneous
and heterogeneous graphs, and in conjunction with other scalable graph mining
tools.
