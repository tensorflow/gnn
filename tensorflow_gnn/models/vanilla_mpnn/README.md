# Vanilla MPNN (TF-GNN's basic model flavor)

## Overview

Message Passing Neural Networks (MPNNs) are a general formulation of
Graph Neural Networks, originally published for the case of homogeneous graphs
by

  * J. Gilmer, S.S. Schoenholz, P.F. Riley, O. Vinyals, G.E. Dahl:
    ["Neural message passing for Quantum
    chemistry"](https://proceedings.mlr.press/v70/gilmer17a), ICML 2017.

TF-GNN supports all sorts of Message Passing Neural Networks (MPNNs) on
heterogeneous graphs through its `tfgnn.keras.layers.GraphUpdate` and
`tfgnn.keras.layers.NodeUpdate` classes. For those, users need to specify a
convolution for each edge set (to compute messages on edges and their
aggregation to nodes) as well as a next-state layer for each node set.
(The [TF-GNN Modeling
Guide](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/gnn_modeling.md)
has the details.)

This code here provides a ready-to-use, straightforward realization of MPNNs on
heterogeneous graphs in which the messages on an edge set and the next states
of a node set are each computed from a single-layer neural network on the
concatenation of all relevant inputs.

Reduced to the homogeneous case, this recovers Interaction Networks, originally
published by

  * P. Battaglia, R. Pascanu, M. Lai, D. Jimenez Rezende, K. Kavukcuoglu:
    ["Interaction Networks for Learning about Objects, Relations and
    Physics"](  https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html),
    NIPS 2016.

Gilmer&al. (loc.cit.) discuss them as "pair message" MPNNs, when using both
endpoint states and the edge feature for the message.
The authors of this code found them to be a quite powerful baseline.

## Usage

TensorFlow programs can import and use this model as described in its
[API docs](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/api_docs/python/models/vanilla_mpnn.md).


## Maintenance and stability

This code is experimental for now, with no promises of maintenance or stability,
and no assigned maintainer. Use at your own risk.
