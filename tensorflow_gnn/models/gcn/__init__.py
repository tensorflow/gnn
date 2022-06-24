"""Graph Convolutional Networks.

Users of TF-GNN can use this model by importing it next to the core library as

```python
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import gcn
```
"""
from tensorflow_gnn.models.gcn import gcn_conv

GCNConv = gcn_conv.GCNConv
GCNHomGraphUpdate = gcn_conv.GCNHomGraphUpdate

del gcn_conv
