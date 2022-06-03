description: The tfgnn.keras.layers package.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfgnn.keras.layers" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfgnn.keras.layers

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

[`class AnyToAnyConvolutionBase`](../../tfgnn/keras/layers/AnyToAnyConvolutionBase.md): Convenience base class for convolutions to nodes or to context.

[`class Broadcast`](../../tfgnn/keras/layers/Broadcast.md): Broadcasts a GraphTensor feature.

[`class ContextUpdate`](../../tfgnn/keras/layers/ContextUpdate.md): A context update with input from node sets and/or edge sets.

[`class EdgeSetUpdate`](../../tfgnn/keras/layers/EdgeSetUpdate.md): Computes the new state of an EdgeSet from select input features.

[`class GraphUpdate`](../../tfgnn/keras/layers/GraphUpdate.md): Applies one round of updates to EdgeSets, NodeSets and Context.

[`class MakeEmptyFeature`](../../tfgnn/keras/layers/MakeEmptyFeature.md): Returns an empty feature with a shape that fits the input graph piece.

[`class MapFeatures`](../../tfgnn/keras/layers/MapFeatures.md): Transforms features on a GraphTensor by user-defined callbacks.

[`class NextStateFromConcat`](../../tfgnn/keras/layers/NextStateFromConcat.md): Computes a new state by concatenating inputs and applying a Keras Layer.

[`class NodeSetUpdate`](../../tfgnn/keras/layers/NodeSetUpdate.md): A node state update with input from convolutions or other edge set inputs.

[`class PadToTotalSizes`](../../tfgnn/keras/layers/PadToTotalSizes.md): Applies tfgnn.pad_to_total_sizes() to a GraphTensor.

[`class ParseExample`](../../tfgnn/keras/layers/ParseExample.md): Applies tfgnn.parse_example(graph_tensor_spec, _) to a batch of strings.

[`class ParseSingleExample`](../../tfgnn/keras/layers/ParseSingleExample.md): Applies tfgnn.parse_single_example(graph_tensor_spec, _).

[`class Pool`](../../tfgnn/keras/layers/Pool.md): Pools a GraphTensor feature.

[`class Readout`](../../tfgnn/keras/layers/Readout.md): Reads a feature out of a GraphTensor.

[`class ReadoutFirstNode`](../../tfgnn/keras/layers/ReadoutFirstNode.md): Reads a feature from the first node of each graph conponent.

[`class ResidualNextState`](../../tfgnn/keras/layers/ResidualNextState.md): Updates a state with a residual block.

[`class SimpleConvolution`](../../tfgnn/keras/layers/SimpleConvolution.md): A convolution layer that applies a passed-in message_fn.

[`class TotalSize`](../../tfgnn/keras/layers/TotalSize.md): Returns the .total_size of a graph piece.

