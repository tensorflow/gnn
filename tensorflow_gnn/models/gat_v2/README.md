# Graph Attention Networks v2

## Overview

This code implements Graph Attention Networks v2, originally published by

  * Shaked Brody, Uri Alon, Eran Yahav:
    ["How Attentive are Graph Attention
    Networks?"](https://arxiv.org/abs/2105.14491), 2021.

TensorFlow programs can import it as

```python
from tensorflow_gnn.models import gat_v2
```

to reuse the following components:

  * `GATv2Conv` and `GATv2EdgePool` for use with the node or context
    updates of a `tfgnn.keras.layers.GraphUpdate`.
  * `GATv2GraphUpdate` for one round of GATv2 on a `GraphTensor` with one
    node set and edge set.

## Maintenance and stability

This code is experimental for now, with no promises of maintenance or stability,
and no assigned maintainer. Use at your own risk.
