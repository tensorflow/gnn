"""Transformer-style multi-head attention.

Users of TF-GNN can use this model by importing it next to the core library as

```python
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import multi_head_attention
```
"""

from tensorflow_gnn.models.multi_head_attention import layers

MultiHeadAttentionConv = layers.MultiHeadAttentionConv
MultiHeadAttentionEdgePool = layers.MultiHeadAttentionEdgePool
MultiHeadAttentionHomGraphUpdate = layers.MultiHeadAttentionHomGraphUpdate
MultiHeadAttentionMPNNGraphUpdate = layers.MultiHeadAttentionMPNNGraphUpdate

# Prune imported module symbols so they're not accessible implicitly.
del layers
