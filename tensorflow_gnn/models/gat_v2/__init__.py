"""The tensorflow_gnn.models.gat_v2 package."""

from tensorflow_gnn.models.gat_v2 import layers

GATv2Conv = layers.GATv2Conv
GATv2EdgePool = layers.GATv2EdgePool
GATv2GraphUpdate = layers.GATv2GraphUpdate

# Prune imported module symbols so they're not accessible implicitly.
del layers
