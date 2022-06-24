"""TF-GNN's "Vanilla MPNN" model.

Users of TF-GNN can use this model by importing it next to the core library as

```python
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import vanilla_gnn
```

This model ties together some simple convolutions from the TF-GNN core library,
so it does not define any Conv class by itself.
"""

from tensorflow_gnn.models.vanilla_mpnn import layers

VanillaMPNNGraphUpdate = layers.VanillaMPNNGraphUpdate

# Prune imported module symbols so they're not accessible implicitly.
del layers
