# TensorFlow Graph Neural Networks

The TensorFlow GNN library makes it easy to build neural networks on graph data
(nodes and edges with arbitrary features). The library provides input data
formats graph datasets, and an encoding for sampled subgraphs as training data.

The library provides a GraphTensor class, a container of all the graph data as
tensors (or that of a batch of graphs) and an API that makes it easy to build
graph neural network models by convolving features over any edges, learning
neural network kernels at any stage. Highlights:

*   Heterogeneous graphs (both heterogeneous node and edge types).
*   Flexible arbitrary kernels for updates and convolutions.
*   Easy broadcasting and pooling operations between nodes, edges and graph
    features.

Start with our documentation:

*   [Introduction to Graph Neural Networks](intro.md). This page introduces the
    concept of graph neural networks with a focus on their application at scale.
    It touches on various scenarios where GNNs are well-suited to the problem
    and provides you with enough context to start thinking about model design.

*   [Scaling GNNs](scaling.md). This library adopts a method for batch sampling
    for scaling up the training of GNN models in parallel. This page introduces
    the various solutions that have been used to train scalable models and the
    advantages and disadvantages of our approach.

*   [Describing your graph](schema.md). The first step to designing your GNN
    model is to describe the topology of your graph data, and which features are
    available on its nodes and edges. This page outlines the schema that is
    required to configure and inform the various components of the library of
    your graphs’ characteristics.

*   [Data preparation](data_prep.md) Training data for GNN
    models are graphs. When operating on very large graphs, we produce local
    subgraph samples from the full dataset which are serialized along with their
    features and streamed for training. This document describes the data format
    we support for storing large graphs.

*   **Feature engineering** (doc incomplete). Before your graph data enters the
    model, the individual features attached to each node or edge set need to be
    encoded, normalized and concatenated to a tensor of floating-point numbers
    which is fed to the neural networks & convolutions for that set. This
    document explains how to do that, and how to read and print your encoded
    data for confirmation that it is usable and as expected for training.

*   **Building models** (doc incomplete; see
    [this colab](https://colab.sandbox.google.com/drive/1sDIHr_-XvhKGXcFEQf1F4NiRDa9x8gTH)
    in the meantime). With data now generated and ingested into Tensorflow, we
    describe how to use the features from the `GraphTensor` object to put
    together GNN models. This page also describes various operations on your
    graph that you can use to regularize and/or normalize your graph to enhance
    the model performance.

*   **Tutorials** (doc incomplete). Some step-by-step tutorials with the
    end-to-end process of building a model using Tensorflow GNN library.

*   **Using GNNs in Inference** (doc incomplete). Inference over graphs involves
    building and providing graph data that must be distributionally similar to
    data used in training. We describe the application of a learned GNN model
    over an existing graph (in batch) and how to use them dynamically (in a live
    system).

## Talks

You can find related talks at our NeurIPS 2020 Expo workshop:
https://gm-neurips-2020.github.io/
