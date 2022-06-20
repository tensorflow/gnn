"""GraphSAGE.

Users of TF-GNN can use this model by importing it next to the core library as

```python
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import graph_sage
```
"""

from tensorflow_gnn.models.graph_sage import layers

GCNGraphSAGENodeSetUpdate = layers.GCNGraphSAGENodeSetUpdate
GraphSAGEAggregatorConv = layers.GraphSAGEAggregatorConv
GraphSAGEPoolingConv = layers.GraphSAGEPoolingConv
GraphSAGENextState = layers.GraphSAGENextState
GraphSAGEGraphUpdate = layers.GraphSAGEGraphUpdate

# Prune imported module symbols so they're not accessible implicitly.
del layers
