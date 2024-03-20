# TF-GNN Models

## Introduction

This directory contains a collection of GNN models implemented with the
TF-GNN library. Some of them offer reusable pieces that can be imported
_next to_ the core TF-GNN library, which effectively makes them little
libraries of their own.

### Usage

If, for example, the hypothetical FancyNet model offered a graph update layer,
its use would look like

```python
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import fancynet

graph = fancynet.FancyGraphUpdate(units=42, fanciness=0.99, ...)(graph)
```

...and require a separate dependency for `fancynet` in a BUILD file.

### API stability

Each model comes with a README file that describes its intended level of
API stability. Not all models are covered by the [semantic
versioning](https://semver.org/spec/v2.0.0.html) of the TF-GNN package.

## List of Models

<!-- Sorted alphabetically by title. -->

  * [Contrastive Losses](contrastive_losses/README.md): Contrastive losses for
    self-supervised learning.
  * [GATv2](gat_v2/README.md): Graph Attention Networks v2
    (Brody&al, 2021).
  * [GCN](gcn/README.md): Graph Convolutional Networks
    (Kipf&Welling, 2016), for homogeneous graphs only.
  * [GraphSAGE](graph_sage/README.md) (Hamilton&al., 2017).
  * [MtAlbis](mt_albis/README.md): Model Template "Albis" for easy configuration
    of a few field-tested GNN architectures, generalizing VanillaMPNN.
  * [MultiHeadAttention](multi_head_attention/README.md): Transformer-style
    multi-head attention on graph (Dwivedi&Bresson, 2021).
  * [VanillaMPNN](vanilla_mpnn/README.md): TF-GNN's classic baseline model,
    based on (Gilmer&al., 2016).

Unsure? For generic node prediction tasks on relational data, we recommend
to start with MtAlbis.