# Module: tfgnn.keras.layers

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/__init__.py">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

The tfgnn.keras.layers package.



## Classes

[`class AddReadoutFromFirstNode`](../../tfgnn/keras/layers/AddReadoutFromFirstNode.md):
Adds readout node set equivalent to
<a href="../../tfgnn/keras/layers/ReadoutFirstNode.md"><code>tfgnn.keras.layers.ReadoutFirstNode</code></a>.

[`class AddSelfLoops`](../../tfgnn/keras/layers/AddSelfLoops.md): Adds
self-loops to scalar graphs.

[`class AnyToAnyConvolutionBase`](../../tfgnn/keras/layers/AnyToAnyConvolutionBase.md): Convenience base class for convolutions to nodes or to context.

[`class Broadcast`](../../tfgnn/keras/layers/Broadcast.md): Broadcasts a GraphTensor feature.

[`class ContextUpdate`](../../tfgnn/keras/layers/ContextUpdate.md): A context update with input from node sets and/or edge sets.

[`class EdgeSetUpdate`](../../tfgnn/keras/layers/EdgeSetUpdate.md): Computes the new state of an EdgeSet from select input features.

[`class GraphUpdate`](../../tfgnn/keras/layers/GraphUpdate.md): Applies one round of updates to EdgeSets, NodeSets and Context.

[`class ItemDropout`](../../tfgnn/keras/layers/ItemDropout.md): Dropout of
feature values for entire edges, nodes or components.

[`class MakeEmptyFeature`](../../tfgnn/keras/layers/MakeEmptyFeature.md): Returns an empty feature with a shape that fits the input graph piece.

[`class MapFeatures`](../../tfgnn/keras/layers/MapFeatures.md): Transforms features on a GraphTensor by user-defined callbacks.

[`class NextStateFromConcat`](../../tfgnn/keras/layers/NextStateFromConcat.md): Computes a new state by concatenating inputs and applying a Keras Layer.

[`class NodeSetUpdate`](../../tfgnn/keras/layers/NodeSetUpdate.md): A node state update with input from convolutions or other edge set inputs.

[`class PadToTotalSizes`](../../tfgnn/keras/layers/PadToTotalSizes.md): Applies tfgnn.pad_to_total_sizes() to a GraphTensor.

[`class ParseExample`](../../tfgnn/keras/layers/ParseExample.md): Applies tfgnn.parse_example(graph_tensor_spec, _) to a batch of strings.

[`class ParseSingleExample`](../../tfgnn/keras/layers/ParseSingleExample.md): Applies tfgnn.parse_single_example(graph_tensor_spec, _).

[`class Pool`](../../tfgnn/keras/layers/Pool.md): Pools a GraphTensor feature.

[`class Readout`](../../tfgnn/keras/layers/Readout.md): Reads a feature out of a GraphTensor.

[`class ReadoutFirstNode`](../../tfgnn/keras/layers/ReadoutFirstNode.md): Reads
a feature from the first node of each graph component.

[`class ReadoutNamedIntoFeature`](../../tfgnn/keras/layers/ReadoutNamedIntoFeature.md):
Reads out a feature value from select nodes (or edges) in a graph.

[`class ResidualNextState`](../../tfgnn/keras/layers/ResidualNextState.md): Updates a state with a residual block.

[`class SimpleConv`](../../tfgnn/keras/layers/SimpleConv.md): A convolution
layer that applies a passed-in message_fn.

[`class SingleInputNextState`](../../tfgnn/keras/layers/SingleInputNextState.md):
Replaces a state from a single input.

[`class StructuredReadout`](../../tfgnn/keras/layers/StructuredReadout.md):
Reads out a feature value from select nodes (or edges) in a graph.

[`class StructuredReadoutIntoFeature`](../../tfgnn/keras/layers/ReadoutNamedIntoFeature.md):
Reads out a feature value from select nodes (or edges) in a graph.
