"""The tensorflow_gnn.models.graph_sage package."""

from tensorflow_gnn.models.graph_sage import layers

GraphSAGEPoolingConv = layers.GraphSAGEPoolingConv
GraphSAGENextState = layers.GraphSAGENextState
GraphSAGEGraphUpdate = layers.GraphSAGEGraphUpdate

# Prune imported module symbols so they're not accessible implicitly.
del layers
