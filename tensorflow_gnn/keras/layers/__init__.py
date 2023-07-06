"""The tfgnn.keras.layers package."""

from tensorflow_gnn.keras.layers import convolution_base
from tensorflow_gnn.keras.layers import convolutions
from tensorflow_gnn.keras.layers import graph_ops
from tensorflow_gnn.keras.layers import graph_update
from tensorflow_gnn.keras.layers import item_dropout
from tensorflow_gnn.keras.layers import map_features
from tensorflow_gnn.keras.layers import next_state
from tensorflow_gnn.keras.layers import padding_ops
from tensorflow_gnn.keras.layers import parse_example

ParseExample = parse_example.ParseExample
ParseSingleExample = parse_example.ParseSingleExample

MapFeatures = map_features.MapFeatures
MakeEmptyFeature = map_features.MakeEmptyFeature

PadToTotalSizes = padding_ops.PadToTotalSizes

AddSelfLoops = graph_ops.AddSelfLoops
Broadcast = graph_ops.Broadcast
Pool = graph_ops.Pool
Readout = graph_ops.Readout
ReadoutFirstNode = graph_ops.ReadoutFirstNode
AddReadoutFromFirstNode = graph_ops.AddReadoutFromFirstNode
StructuredReadout = graph_ops.StructuredReadout
StructuredReadoutIntoFeature = graph_ops.StructuredReadoutIntoFeature
# DO NOT USE the obsolete aliases `ReadoutNamed*`.
ReadoutNamed = graph_ops.StructuredReadout
ReadoutNamedIntoFeature = graph_ops.StructuredReadoutIntoFeature

AnyToAnyConvolutionBase = convolution_base.AnyToAnyConvolutionBase
SimpleConv = convolutions.SimpleConv

ItemDropout = item_dropout.ItemDropout

NextStateFromConcat = next_state.NextStateFromConcat
ResidualNextState = next_state.ResidualNextState
SingleInputNextState = next_state.SingleInputNextState

EdgeSetUpdate = graph_update.EdgeSetUpdate
NodeSetUpdate = graph_update.NodeSetUpdate
ContextUpdate = graph_update.ContextUpdate
GraphUpdate = graph_update.GraphUpdate

# Prune imported module symbols so they're not accessible implicitly.
# Please use the same order as for the import statements at the top.
del convolution_base
del convolutions
del graph_ops
del graph_update
del item_dropout
del map_features
del next_state
del padding_ops
del parse_example
