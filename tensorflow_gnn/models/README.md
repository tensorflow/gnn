# TF-GNN Models

## Introduction

This directory contains a collection of GNN models implemented with the
TF-GNN library. Some of them offer reusable pieces that can be imported
_next to_ the core TF-GNN library, which effectively makes them little
libraries of their own.

### Usage

If, for example, the hypothetical FancyNet model offered a convolution
layer compatible with the standard NodeSetUpdate, its use would look like

```python
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import fancynet

_ = tfgnn.keras.layers.NodeSetUpdate(
    {"edges": fancynet.FancyConv(units=42, fanciness=0.99, ...)}, ...)
```

...and require a separate dependency for `fancynet` in a BUILD file.

### Maintenance and stability

Each model comes with a README file that lists its maintainers and the intended
level of stability and maintenance; please check before depending on it for
anything beyond one-off experimentation. In particular, the API stability
promises of TF-GNN releases do **not** extend to particular models, unless they
say so in their README files.

## List of Models

<!-- Sorted alphabetically by title. -->

  * [Contrastive Losses](contrastive_losses/README.md): Contrastive losses for
    self-supervised learning.
  * [GATv2](gat_v2/README.md): Graph Attention Networks v2
    (Brody&al, 2021).
  * [GCN](gcn/README.md): Graph Convolutional Networks
    (Kipf&Welling, 2016), for homogeneous graphs only.
  * [GraphSAGE](graph_sage/README.md) (Hamilton&al., 2017).
  * [MtAlbis](mt_albis/README.md): a model template for easy configuration
    of selected GNN architectures.
  * [MultiHeadAttention](multi_head_attention/README.md): Transformer-style
    multi-head attention on graph (Dwivedi&Bresson, 2021).
  * [VanillaMPNN](vanilla_mpnn/README.md): TF-GNN's frequently used baseline
    model, based on (Gilmer&al., 2016).
