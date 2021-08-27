"""The tfgnn.keras.layers package."""

from tensorflow_gnn.graph.keras.layers import graph_ops
from tensorflow_gnn.graph.keras.layers import graph_update
from tensorflow_gnn.graph.keras.layers import graph_update_options

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

# Prune imported module symbols so they're not accessible implicitly.
del graph_ops
del graph_update
del graph_update_options
