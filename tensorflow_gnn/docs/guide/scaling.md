# Scaling Graph Neural Networks

This document describes various approaches to applying graph neural neural
networks and the approach we support in TF-GNN.

## Motivation

Graphs are a common representation for a large class of dataset, and in
particular, relational data can be mapped to graphs naturally (with
heterogeneous graphs). We want to be able to train models on very large
datasets. Real world graphs can consist of billions of nodes and trillions of
edges. These datasets often cannot fit in a single computer’s memory.

Unfortunately, CNNs, the image analogue of GNNs, does not provide a solution
either. Convolutional Neural Networks (CNNs) have the benefit of regularity and
an implicit partitioning in that convolved kernels overlap only between each of
the adjacent regions of a single image. In contrast, neural network kernels
trained over Graph Neural Networks potentially overlap over the entire surface
of the graph, are irregularly shaped, and much more sparse (the diameter of
those graphs is smaller).

## Other Approaches

**Single Machine with Interpolation.** The GraphSage model, one of the earliest
descriptions of a GNN on a larger real-world graph, used an in-memory graph with
sampling during its training on a single host with a very large amount of memory
(500GB as per 2018). Moreover, it was trained on a 20% sample of the full graph
(see section 4.1 in [Ying 2018]). The ability to train on a small sample rests
on an inductive assumption about the nature of the graph, that is, that
distributional properties of the features on the unseen 80% of the graph are
similar to the sampled portion. This method is still limited by the amount of
data available in memory and the inductive assumption may not be applicable on
all datasets.

**Training over the PPR.** One method to avoid having to feed irregular graph
structure as tensors is to pre-compute in batch the Personalized PageRank
[Bojchevski 2020] and train a model with the PPR neighbors as a single
convolution. The advantage for scalability is that one could use the largest
neighbors by edge weight to account for the most important relationships, cap
and fix the number used and run the model with a fixed data structure size
(padding where the size of the PPR vector is smaller than the fixed size).
Another line of work removes non-linearities between intermediate layers of a
GNN (without explicitly using PPR).

**Precomputing aggregations.** Another method that can be used to reduce the
structural complexity of the input is to pre-calculate the product of the
adjacency matrix and the training dataset features (expand to edges and pool
from edges operations) in a scalable batch process and to turn the training
portion from a GNN to a DNN with regular structure, by using this matrix to
train the weights [Chen 2018], [Frasca 2020].

The last two methods are approximations of GNN models which trade off the
ability to leverage the full edge structure and resulting embeddings you can
obtain by running convolutions over multiple hops.

## Our Approach: Graph Subsampling

The approach supported by TF-GNN does not compromise, and allows you to
naturally represent irregular graph structure over multiple types of nodes and
sets of edges between them, without cutting corners or making approximations.
The benefit is flexibility in the expression of models and the ability to take
advantage of the full amount of structural connectivity of the graph. The cost
is more complexity in its the representation, which we mitigate by providing a
convenient API for building models using irregular graph structure.

In particular, our representation naturally supports the common case of models
with different types of nodes with dedicated sets of edges between them.
Unfortunately this type of model is not discussed frequently in the literature
but has a very large number of instantiations in its application: any database
with cross-references between tables can be seen as a graph with one node type
for each table (each row is an instance of a node), references as edges, and
data in its rows and columns as features.

## Ragged Features & Feature Encoding

Another key aspect of the TF-GNN library is that it handles the original storage
of categorical features on nodes and edges so you can train embeddings on them
as part of the GNN.

The vast majority of the research assumes input features that are already
encoded as vectors of floats. In practical application, categorical features,
such as words, will incur an embedding stage which most often has to be trained
as part of the model. This is an aspect of practical application of such models
that is most often ignored in the literature (because it has little impact on
modeling).

This means that input features must be represented in their original format in
training examples. Multiple features must be supported, since it could be
required to embed them separately. A mix of categorical, numerical and sparse
features are supported, associated with each node set and edge set.

Sampling tools ensure that the features for each node type are suitably
concatenated and that for variable-length features, a vector representing the
row splits are included in the encoding.

## Trading Off Disk Space for Runtime

Sampling and storing all the features of a local subgraph in the data
preparation offers the advantage of having all the data bundled and sharded over
multiple files. These files contain complete training examples, and can be read
concurrently by multiple workers operating in parallel, calculating gradients,
and updating a set of shared model parameters either synchronously or
asynchronously.

The trade-off we’re making is assuming that disk space is cheap and plentiful.
This is often the case in a cloud environment. However, it can be quite
expensive with very large graphs. If your full original graph has 1M nodes and
your sampling program generates 1000 nodes per graph (on average), this requires
storage for 1B nodes (each of the features is on average repeated 1000 times).

If this is a problem, there are few alternative ideas to alleviate this
requirement while still avoiding having to load all the data in memory.

* **Join features late.** Instead of storing all the features on disk, only the
  local subgraph topology and edge relationships can be stored and instead of
  storing the features themselves we can store a unique identifier for each
  node. By storing the features in a distributed hash table (any key/value store
  which allows looking up data based on a string key), it is possible to write a
  custom TensorFlow op can be written to pull in and join the node features
  corresponding to an array of node ids for each subgraph. The graph topology
  itself is generally not a large amount of data (the features consist in the
  bulk of it).

* **Dynamic sampling.** One could go further write a custom TensorFlow op that
  implements a sampling algorithm by querying a node-indexed storage at runtime.
  The drawback of this approach is that fetching the node features in this way
  would typically require multiple round-trips to the distributed hash table,
  and it would be very slow. The previous approach strikes a better compromise
  between storage requirement and computation.

While we do not currently provide implementations of TensorFlow ops for open
source backends, it would be straightforward to build one in order to access
your backend database of choice.
