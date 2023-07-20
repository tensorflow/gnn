# Module: gat_v2

[TOC]

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

[`GATv2HomGraphUpdate(...)`](./gat_v2/GATv2HomGraphUpdate.md): Returns a
GraphUpdate layer with a Graph Attention Network V2 (GATv2).

[`GATv2MPNNGraphUpdate(...)`](./gat_v2/GATv2MPNNGraphUpdate.md): Returns a
GraphUpdate layer for message passing with GATv2 pooling.

[`graph_update_from_config_dict(...)`](./gat_v2/graph_update_from_config_dict.md):
Returns a GATv2MPNNGraphUpdate initialized from `cfg`.

[`graph_update_get_config_dict(...)`](./gat_v2/graph_update_get_config_dict.md):
Returns ConfigDict for graph_update_from_config_dict() with defaults.
