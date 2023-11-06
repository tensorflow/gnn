<!-- lint-g3mark -->

# Module: gcn

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/gcn/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Graph Convolutional Networks.

Users of TF-GNN can use this model by importing it next to the core library as

``` python
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import gcn
```

## Classes

[`class GCNConv`](./gcn/GCNConv.md): Implements the Graph Convolutional Network
by Kipf\&Welling (2016).

## Functions

[`GCNHomGraphUpdate(...)`](./gcn/GCNHomGraphUpdate.md): Returns a graph update
layer for GCN convolution.
