# TensorFlow GNN

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

For more details, please see our [documentation](tensorflow_gnn/docs/guide/overview.md).
For background and discussion, please see O. Ferludin et al.:
[TF-GNN: Graph Neural Networks in TensorFlow](https://arxiv.org/abs/2207.03522),
2022 (full citation below).

## Installation Instructions

##### Latest available pip wheel.

`pip install tensorflow_gnn`

##### Installation from source.

A virtual environment is highly recommended.

1.  **Clone tensorflow_gnn**

    > `$> git clone https://github.com/tensorflow/gnn.git tensorflow_gnn`

2.  **Install TensorFlow**

    TF-GNN currently uses
    [tf.ExtensionTypes](https://www.tensorflow.org/api_docs/python/tf/experimental/ExtensionType),
    which is a feature of TensorFlow 2.7. As such, you will need to install
    TensorFlow build, following the instructions here:
    https://www.tensorflow.org/install/pip.

    > `$> pip install tensorflow`

3.  **Install Bazel**

    Bazel is required to build the source of this package. Follow the
    instructions here to install Bazel for your OS:
    https://docs.bazel.build/versions/main/install.html

4.  **Install tensorflow_gnn**

    > `$> cd tensorflow_gnn && python3 -m pip install .`


## Citation

When referencing this library in a paper, please cite the
[TF-GNN paper](https://arxiv.org/abs/2207.03522):

```
@article{tfgnn,
  author  = {Oleksandr Ferludin and Arno Eigenwillig and Martin Blais and
             Dustin Zelle and Jan Pfeifer and Alvaro Sanchez{-}Gonzalez and
             Sibon Li and Sami Abu{-}El{-}Haija and Peter Battaglia and
             Neslihan Bulut and Jonathan Halcrow and
             Filipe Miguel Gon{\c{c}}alves de Almeida and Silvio Lattanzi and
             Andr{\'{e}} Linhares and Brandon Mayer and Vahab Mirrokni and
             John Palowitch and Mihir Paradkar and Jennifer She and
             Anton Tsitsulin and Kevin Villela and Lisa Wang and David Wong and
             Bryan Perozzi},
  title   = {{TF-GNN:} Graph Neural Networks in TensorFlow},
  journal = {CoRR},
  volume  = {abs/2207.03522},
  year    = {2022},
  url     = {http://arxiv.org/abs/2207.03522},
}
```

