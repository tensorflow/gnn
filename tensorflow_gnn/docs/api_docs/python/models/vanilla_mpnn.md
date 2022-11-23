# Module: vanilla_mpnn

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/vanilla_mpnn/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

TF-GNN's "Vanilla MPNN" model.

Users of TF-GNN can use this model by importing it next to the core library as

```python
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import vanilla_gnn
```

This model ties together some simple convolutions from the TF-GNN core library,
so it does not define any Conv class by itself.

## Functions

[`VanillaMPNNGraphUpdate(...)`](./vanilla_mpnn/VanillaMPNNGraphUpdate.md):
Returns a GraphUpdate layer for a Vanilla MPNN.

[`graph_update_from_config_dict(...)`](./vanilla_mpnn/graph_update_from_config_dict.md):
Returns a VanillaMPNNGraphUpdate initialized from `cfg`.

[`graph_update_get_config_dict(...)`](./vanilla_mpnn/graph_update_get_config_dict.md):
Returns ConfigDict for graph_update_from_config_dict() with defaults.
