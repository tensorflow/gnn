"""The tfgnn.keras.layers package."""

from tensorflow_gnn.graph.keras.layers import graph_ops
from tensorflow_gnn.graph.keras.layers import graph_update
from tensorflow_gnn.graph.keras.layers import graph_update_options
from tensorflow_gnn.graph.keras.layers.gat import gatv2

Broadcast = graph_ops.Broadcast
Pool = graph_ops.Pool
Readout = graph_ops.Readout

EdgeSetUpdate = graph_update.EdgeSetUpdate
NodeSetUpdate = graph_update.NodeSetUpdate
ContextUpdate = graph_update.ContextUpdate

GraphUpdateOptions = graph_update_options.GraphUpdateOptions
GraphUpdateEdgeSetOptions = graph_update_options.GraphUpdateEdgeSetOptions
GraphUpdateNodeSetOptions = graph_update_options.GraphUpdateNodeSetOptions
GraphUpdateContextOptions = graph_update_options.GraphUpdateContextOptions

# GATv2 model.
GATv2 = gatv2.GATv2

# Prune imported module symbols so they're not accessible implicitly.
del gatv2
del graph_ops
del graph_update
del graph_update_options
