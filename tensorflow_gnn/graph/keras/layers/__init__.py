"""The tfgnn.keras.layers package."""

from tensorflow_gnn.graph.keras.layers import convolutions
from tensorflow_gnn.graph.keras.layers import graph_ops
from tensorflow_gnn.graph.keras.layers import graph_update
from tensorflow_gnn.graph.keras.layers import next_state
from tensorflow_gnn.graph.keras.layers.gat import gatv2

Broadcast = graph_ops.Broadcast
Pool = graph_ops.Pool
Readout = graph_ops.Readout
ReadoutFirstNode = graph_ops.ReadoutFirstNode

ConvolutionFromEdgeSetUpdate = convolutions.ConvolutionFromEdgeSetUpdate
SimpleConvolution = convolutions.SimpleConvolution

NextStateFromConcat = next_state.NextStateFromConcat
ResidualNextState = next_state.ResidualNextState

EdgeSetUpdate = graph_update.EdgeSetUpdate
NodeSetUpdate = graph_update.NodeSetUpdate
ContextUpdate = graph_update.ContextUpdate
GraphUpdate = graph_update.GraphUpdate

# GATv2 model.
GATv2 = gatv2.GATv2

# Prune imported module symbols so they're not accessible implicitly.
del convolutions
del graph_ops
del graph_update
del next_state
del gatv2
