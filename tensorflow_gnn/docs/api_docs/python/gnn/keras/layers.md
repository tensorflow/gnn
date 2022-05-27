description: The tfgnn.keras.layers package.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gnn.keras.layers" />
<meta itemprop="path" content="Stable" />
</div>

# Module: gnn.keras.layers

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

The tfgnn.keras.layers package.

## Classes

[`class AnyToAnyConvolutionBase`](../../gnn/keras/layers/AnyToAnyConvolutionBase.md):
Convenience base class for convolutions to nodes or to context.

[`class Broadcast`](../../gnn/keras/layers/Broadcast.md): Broadcasts a GraphTensor feature.

[`class ContextUpdate`](../../gnn/keras/layers/ContextUpdate.md): A context update with input from node sets and/or edge sets.

[`class EdgeSetUpdate`](../../gnn/keras/layers/EdgeSetUpdate.md): Computes the new state of an EdgeSet from select input features.

[`class GraphUpdate`](../../gnn/keras/layers/GraphUpdate.md): Applies one round of updates to EdgeSets, NodeSets and Context.

[`class MakeEmptyFeature`](../../gnn/keras/layers/MakeEmptyFeature.md): Returns
an empty feature with a shape that fits the input graph piece.

[`class MapFeatures`](../../gnn/keras/layers/MapFeatures.md): Transforms
features on a GraphTensor by user-defined callbacks.

[`class NextStateFromConcat`](../../gnn/keras/layers/NextStateFromConcat.md): Computes a new state by concatenating inputs and applying a Keras Layer.

[`class NodeSetUpdate`](../../gnn/keras/layers/NodeSetUpdate.md): A node state update with input from convolutions or other edge set inputs.

[`class PadToTotalSizes`](../../gnn/keras/layers/PadToTotalSizes.md): Applies
tfgnn.pad_to_total_sizes() to a GraphTensor.

[`class ParseExample`](../../gnn/keras/layers/ParseExample.md): Applies
tfgnn.parse_example(graph_tensor_spec, _) to a batch of strings.

[`class ParseSingleExample`](../../gnn/keras/layers/ParseSingleExample.md):
Applies tfgnn.parse_single_example(graph_tensor_spec, _).

[`class Pool`](../../gnn/keras/layers/Pool.md): Pools a GraphTensor feature.

[`class Readout`](../../gnn/keras/layers/Readout.md): Reads a feature out of a GraphTensor.

[`class ReadoutFirstNode`](../../gnn/keras/layers/ReadoutFirstNode.md): Reads a feature from the first node of each graph conponent.

[`class ResidualNextState`](../../gnn/keras/layers/ResidualNextState.md): Updates a state with a residual block.

[`class SimpleConvolution`](../../gnn/keras/layers/SimpleConvolution.md): A
convolution layer that applies a passed-in message_fn.

[`class TotalSize`](../../gnn/keras/layers/TotalSize.md): Returns the
.total_size of a graph piece.
