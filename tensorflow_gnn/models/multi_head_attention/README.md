# Transformer-style multi-head attention

## Overview

This code implements transformer-style (dot-product) multi-head attention,
with different variants and optional attention score leaks.

Some publications in the GNN context that either use this multi-head attention as
a component ([1]&[2]) or a baseline ([3]) of their method:

*   [1] Vijay Prakash Dwivedi, Xavier Bresson: ["A Generalization of Transformer
    Networks to Graphs"](https://arxiv.org/abs/2012.09699), 2021.
    (We only implement their attention, not their position encoding.)
*   [2] Dongkwan Kim, Alice Oh: ["How to Find Your Friendly Neighborhood: Graph
    Attention Design with Self-Supervision"](https://arxiv.org/abs/2204.04879)
    , 2022. (They call it "DP" attention.)
*   [3] Shaked Brody, Uri Alon, Eran Yahav: ["How Attentive are Graph Attention
    Networks?"](https://arxiv.org/abs/2105.14491), 2021.
    (They discuss "DPGAT" as a baseline in the appendix, citing further uses.
    Their main contribution "GATv2" is implemented elsewhere in TF-GNN.)

TensorFlow programs can import it as

```python
from tensorflow_gnn.models import multi_head_attention
```

which provides the following components:

*   `MultiHeadAttentionConv` and `MultiHeadAttentionEdgePool` for use with the
    node or context updates of a `tfgnn.keras.layers.GraphUpdate`.
*   `MultiHeadAttentionHomGraphUpdate` for one round of MultiHeadAttention on a
    `GraphTensor` that stores a homogeneous graph.
*   `MultiHeadAttentionMPNNGraphUpdate` for one round of message passing between
    the nodes of a heterogeneous `GraphTensor`.

## Maintenance and stability

This code is experimental for now, with no promises of maintenance or stability,
and no assigned maintainer. Use at your own risk.