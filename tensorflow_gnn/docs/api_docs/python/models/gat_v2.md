description: Graph Attention Networks v2.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gat_v2" />
<meta itemprop="path" content="Stable" />
</div>

# Module: gat_v2

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/gat_v2/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Graph Attention Networks v2.

Users of TF-GNN can use this model by importing it next to the core library as

```python
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import gat_v2
```

## Classes

[`class GATv2Conv`](./gat_v2/GATv2Conv.md): The multi-head attention from Graph
Attention Networks v2 (GATv2).

## Functions

[`GATv2EdgePool(...)`](./gat_v2/GATv2EdgePool.md): Returns a layer for pooling
edges with GATv2-style attention.

[`GATv2GraphUpdate(...)`](./gat_v2/GATv2GraphUpdate.md): Returns a GraphUpdater
layer with a Graph Attention Network V2 (GATv2).
