# Module: graph_sage

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/graph_sage/__init__.py">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

GraphSAGE.

Users of TF-GNN can use this model by importing it next to the core library as

```python
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import graph_sage
```

## Classes

[`class GCNGraphSAGENodeSetUpdate`](./graph_sage/GCNGraphSAGENodeSetUpdate.md):
GCNGraphSAGENodeSetUpdate is an extension of the mean aggregator operator.

[`class GraphSAGEAggregatorConv`](./graph_sage/GraphSAGEAggregatorConv.md):
GraphSAGE: element-wise aggregation of neighbors and their linear
transformation.

[`class GraphSAGENextState`](./graph_sage/GraphSAGENextState.md):
GraphSAGENextState: compute new node states with GraphSAGE algorithm.

[`class GraphSAGEPoolingConv`](./graph_sage/GraphSAGEPoolingConv.md): GraphSAGE:
pooling aggregator transform of neighbors followed by linear transformation.

## Functions

[`GraphSAGEGraphUpdate(...)`](./graph_sage/GraphSAGEGraphUpdate.md): Returns a
GraphSAGE GraphUpdater layer for nodes in node_set_names.
