# Contrastive Losses

## Overview

This code implements and collections various contrastive losses for
self-supervised learning. This code is under *active development*. An overview
of the included:

### Deep Graph Infomax

Deep Graph Infomax [1] attempts to learn a bilinear
layer capable of discriminating between positive examples (any input
`GraphTensor`) and negative examples (the input `GraphTensor` but with perturbed
features: this implementation, as in the original paper, shuffles features
across batch, that is, the components the merged `GraphTensor`).

Deep Graph Infomax is particularly useful in unsupervised tasks that wish to
learn latent representations informed primarily by a node's neighborhood
attributes (vs. its structure).

*   [1] Petar Veličković, William Fedus, William L. Hamilton, Pietro Liò,
    Yoshua Bengio, R Devon Hjelm:
    ["Deep Graph Infomax"](https://arxiv.org/abs/1809.10341), 2018.

## Usage

TensorFlow programs can import and use this model as described in its
[API docs](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/api_docs/python/models/contrastive_losses.md).

## API stability

The API of this model may change between OSS library versions.

<!-- PLACEHOLDER FOR README GOOGLE EXTRAS -->
