# Graph Convolutional Network

## Overview

This code implements Graph Convolutional Networks, originally published by

  * Thomas N. Kipf & Max Welling (2016):
    ["Semi-Supervised Classification with Graph Convolutional
    Networks"](https://arxiv.org/abs/1609.02907), 2016.

TensorFlow programs can import it as

```python
from tensorflow_gnn.models import gcn
```

to reuse the following components:

  * `GCNConv` for use with the node
    updates of a `tfgnn.keras.layers.GraphUpdate`.

## Maintenance and stability

This code is experimental for now, with no promises of maintenance or stability,
and no assigned maintainer. Use at your own risk.
