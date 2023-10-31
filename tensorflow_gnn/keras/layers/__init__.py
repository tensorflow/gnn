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
from tensorflow_gnn.utils import api_utils

# NOTE: This package is covered by tensorflow_gnn/api_def/api_symbols_test.py.
# Please see there for instructions how to reflect API changes.
# LINT.IfChange

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

# Remove all names added by module imports, unless explicitly allowed here.
api_utils.remove_submodules_except(__name__, [])
# LINT.ThenChange(../../api_def/tfgnn-symbols.txt)
