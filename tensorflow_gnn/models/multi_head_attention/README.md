# Transformer-style multi-head attention

## Overview

This code implements transformer-style (dot-product) multi-head attention,
with different variants and optional attention score leaks.

Some publications in the GNN context that either use this multi-head attention
as a component ([1]&[2]) or a baseline ([3]) of their method:

*   [1] Vijay Prakash Dwivedi, Xavier Bresson: ["A Generalization of Transformer
    Networks to Graphs"](https://arxiv.org/abs/2012.09699), 2021.
    (We only implement their attention, not their position encoding.)
*   [2] Dongkwan Kim, Alice Oh: ["How to Find Your Friendly Neighborhood: Graph
    Attention Design with Self-Supervision"](https://arxiv.org/abs/2204.04879)
    , 2022. (They call it "DP" attention.)
*   [3] Shaked Brody, Uri Alon, Eran Yahav: ["How Attentive are Graph Attention
    Networks?"](https://arxiv.org/abs/2105.14491), 2021.
    (They discuss "DPGAT" as a baseline in the appendix, citing further uses.
    Their main contribution "GATv2" is implemented [elsewhere](../gat_v2)
    in TF-GNN.)

## Usage

TensorFlow programs can import and use this model as described in its
[API docs](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/api_docs/python/models/multi_head_attention.md).

## API stability

The API of this model may change between OSS library versions.

TF-GNN's [Model Template "Albis"](../mt_albis/README.md) offers a stable and
simplified API for a subset of this model's configuration options.

<!-- PLACEHOLDER FOR README GOOGLE EXTRAS -->