# TF-GNN Models

This directory contains a selection of GNN models implemented with the
TF-GNN library. Some of them offer reusable pieces that can be imported
_next to_ the core TF-GNN library, which effectively makes them little
libraries of their own. Indeed, each model comes with a README file that
lists its maintainers and the intended level of stability and maintenance;
please check before depending on it or anything beyond one-off experimentation.

For example, if the hypothetical FancyNet model offered a convolution
layer compatible with the standard NodeSetUpdate, its use would look like

```python
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import fancynet

_ = tfgnn.keras.layers.NodeSetUpdate(
    {"edges": fancynet.FancyConv(units=42, fanciness=0.99, ...)}, ...)
```

...and require a separate dependency for `fancynet` in a BUILD file.

