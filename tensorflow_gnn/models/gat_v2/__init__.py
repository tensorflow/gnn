"""Graph Attention Networks v2.

Users of TF-GNN can use this model by importing it next to the core library as

```python
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import gat_v2
```
"""

from tensorflow_gnn.models.gat_v2 import layers

GATv2Conv = layers.GATv2Conv
GATv2EdgePool = layers.GATv2EdgePool
GATv2GraphUpdate = layers.GATv2GraphUpdate  # Deprecated.
GATv2HomGraphUpdate = layers.GATv2HomGraphUpdate
GATv2MPNNGraphUpdate = layers.GATv2MPNNGraphUpdate

# Prune imported module symbols so they're not accessible implicitly.
del layers
