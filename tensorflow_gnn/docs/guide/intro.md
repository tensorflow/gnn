# Introduction to Graph Neural Networks

This page provides a very brief, high-level introduction to what graph neural
networks are, from the perspective of their application in practice.

## What is a Graph?

You can model very many domains of the world as entities and their
relationships. "Nodes" are used to represent various types of entities,
and "edges" represent their relationships as links between them.
These structures are called "graphs" (as mathematical objects) or "networks"
(as real-world phenomena). Their properties are studied in
[Graph Theory](https://en.wikipedia.org/wiki/Graph_theory) and
[Network Science](https://en.wikipedia.org/wiki/Network_science).

For example, a log of historical transactions may involve nodes of type
"customer" and "merchant", and edges of type "purchase".
Each of the entities can be associated with features. A customer could have a
location, a credit score, and a credit card, which itself has a history of
payments. The edges also can have feature information associated with them as
well, for instance, the amount, date, and time of day of the transaction.

## Homogeneous Graphs

Many graphs have a single type of entity and edges between them. A classic
example of this is a social network: nodes are persons and edges are their
friendship relations.

<img style="width:30%" src="images/homogeneous.svg">

Homogeneous graphs are often used to learn summary representations that
represent the local information near and around each node. These representations
are then used as inputs to regular deep learning models.

## Heterogeneous Graph and Relational Tables

Relational models such as those found in database tables can be mapped to
graphs. Each of the tables defines a type of node, and references across the
tables can be instantiated as edges. Our previous example of, a database with a
table of customers and a table of merchants with transactions between them would
define two sets of nodes and a set of directed edges from customers to
merchants:

<img style="width:25%" src="images/heterogeneous.svg">

Heterogeneous models are general and flexible and can be applied nearly
anywhere, with a pretty straightforward mapping of real-world entities to model
structure.

## Neural Networks on Graphs: Convolutions

Graphs have been used in learning tasks for a long time. Algorithms that create
clusters of nodes on existing graphs are well-established and label propagation
algorithms that trickle information between nodes have been around for decades.
But how do we use graphs with neural networks?

The essence of the method is the graph convolution. The input features of each
node are transformed into a single vector of floating-point numbers.
The specifics of this initial transformation can vary a lot, ranging from a
simple normalization and encoding (as required for any neural network) up to
running a sophisticated, trained encoder (e.g., to embed words or images).
In any case, this provides each node with its initial *hidden state*.
The hidden states then go through rounds of updates using information from their
neighbors. One such update is called a "graph convolution". In its simplest
form, it consists of four stages:

 1. broadcast: the node states are copied onto edges (that is, outgoing edges,
    possibly also incoming edges);
 2. transform: the value(s) on each edge are transformed by a function,
    possibly itself a neural network;
 3. pooling: the results of these edge transformations are aggregated back
    onto nodes (e.g., by averaging over incoming edges);
 4. state update: the pooled results from the edges are used to produce an
    updated state for each node (e.g., by running them through another small
    neural network).

This happens over all the nodes and edges, at the same time. This process can
be repeated for multiple rounds and allows for the propagation of features,
hopping over graph edges. Every time a feature is propagated, it can be
transformed by a neural network or a fixed function. Finally, classification
and/or regression over nodes, edges or the entire graph is possible by reading
out the relevant state(s) and sending them through a final neural
network, the prediction head.

The neural networks involved have trainable weights. These are shared across
the different locations in the graph within one round (like a convnet uses the
same convolution kernel at each pixel), potentially even across rounds.
As usual, backpropagation is used to compute gradients over the entire network
(all rounds) to minimize some loss function of the predictions.

There is a rapidly growing body of research literature, and practical experience
is improving on Graph Neural Networks (GNN).

## The Problem with Irregularity

Standard deep neural networks are served well by a `Tensor` datatype and the
kind of data-parallel, large-scale matrix multiplications offered on it by the
usual software-hardware stack for deep learning.

In contrast, graph neural networks encounter irregular shapes and data flow
in a number of places.

  * Each training example of a GNN is a graph with a different number of nodes
    or edges, leading to ragged feature shapes when training examples are
    batched.
  * Even within one example, input features themselves can have irregular shapes
    from one node (or edge) to the next. This is similar to the challenges in
    sequence or NLP data, but here it happens in addition to the variable number
    of nodes/edges.
  * The incidence of edges to nodes and hence the dataflow between them is
    not fixed, but defined differently by the input data of each example.

The `tfgnn.GraphTensor` type helps to address all these. It is the unified
container type for graph data for all stages of a TensorFlow program: from
the results of the data preparation tools included with this library,
along the entire TensorFlow input pipeline that processes them, and then
through the GNN model itself. It uses the `tf.RaggedTensor` class to represent
irregular inputs and supports their incremental transformation to uniformly
shaped `tf.Tensor` representations inside the GNN model, including padding to
fixed sizes if needed (usually for Cloud TPUs). It provides broadcast and pool
operations along the graph using the endpoints it stores for each edge.

Two strengths of GraphTensor are often ignored in GNN research: the need to
encode and represent multiple features for each node and/or edge (which is
necessary for categorical features and computed embeddings), and first-class
support for heterogeneous models (which extends the reach of GNNs
substantially).

## Scaling Up

In practice, many graphs are very large (e.g., a large social network may have
billions of nodes) and may not fit in memory. The approach this library uses
in the case of large graphs is to sample neighborhoods around nodes which we
want to train over (say, nodes with associated ground truth labels), and stream
these sampled subgraphs from the filesystem into the TensorFlow training code.
The node of interest is the root node of the sampled subgraph.

## About edge direction

A `tfgnn.GraphTensor` (whether in memory or serialized) stores *directed* edges,
connecting a *source* to a *target* node. By convention, GraphTensors that
represent sampled subgraphs have their edges directed away from the root.
That is to say, the edge's source endpoint was found before the edge itself,
and then the sampling has proceeded from source to target.

However, **a GNN model can use edges in either direction**: the user can select
either the source or target node as the *receiver node* of a convolution
(that is, the endpoint whose state gets updated with a message sent along the
edge); the other endpoint becomes the *sender node*. In a sampled subgraph,
convolutions often have the source nodes of edges (those closer to the root)
as their receivers.

## More Information and Research

In the field of research on deep learning, graph neural networks are currently
enjoying an unprecedented amount of attention and popularity. Their generality
promises a broad level of application and new techniques and variants are
appearing fast. This library intends to bridge the gap between research and
application by providing a flexible platform which also supports aspects only
required from practitioners (e.g., categorical features on nodes that allow one
to also train embeddings). Please consult the various workshops on GNNs for more
details.

Here are a few papers surveying the development of the field and the various
methods that have been published:

  * I. Chami, S. Abu-El-Haija, B. Perozzi, C. Ré, K. Murphy:
    [Machine Learning on Graphs: A Model and Comprehensive
    Taxonomy](https://arxiv.org/abs/2005.03675), 2020.
  * P.W. Battaglia et al.: [Relational inductive biases, deep learning, and
    graph networks](https://arxiv.org/abs/1806.01261), 2018.
  * J. Zhou, G. Cui, S. Hu, Z. Zhang, C. Yang, Z. Liu, L. Wang, C. Li, M. Sun:
    [Graph neural networks: A review of methods
    and applications](https://arxiv.org/abs/1812.08434), 2018.
  * X. Wang, D. Bo, C. Shi, S. Fan, Y. Ye, P.S. Yu:
    [A Survey on Heterogeneous Graph Embedding: Methods, Techniques,
    Applications and Sources](https://arxiv.org/abs/2011.14867), 2020.
  * Z. Wu, S. Pan, F. Chen, G. Long, C. Zhang, P.S. Yu:
    [A Comprehensive Survey
    on Graph Neural Networks](https://arxiv.org/abs/1901.00596), 2019.

For more comprehensive introduction to the field at large, see

  * W.L. Hamilton:
    [Graph Representation Learning](https://doi.org/10.2200/S00980ED1V01Y202001AIM045).
    *Synthesis Lectures on AI and ML* **14** (2020), pp.1-159.
    [Preprint](https://www.cs.mcgill.ca/~wlh/grl_book/) available from
    the author.
