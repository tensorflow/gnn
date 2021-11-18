description: The tfgnn.keras.layers package.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gnn.keras.layers" />
<meta itemprop="path" content="Stable" />
</div>

# Module: gnn.keras.layers

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/keras/layers/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



The tfgnn.keras.layers package.



## Modules

[`gat`](../../gnn/keras/layers/gat.md) module

## Classes

[`class Broadcast`](../../gnn/keras/layers/Broadcast.md): Broadcasts a GraphTensor feature.

[`class ContextUpdate`](../../gnn/keras/layers/ContextUpdate.md): A context update with input from node sets and/or edge sets.

[`class ConvolutionFromEdgeSetUpdate`](../../gnn/keras/layers/ConvolutionFromEdgeSetUpdate.md): Wraps an EdgeSetUpdate as a Convolution.

[`class EdgeSetUpdate`](../../gnn/keras/layers/EdgeSetUpdate.md): Computes the new state of an EdgeSet from select input features.

[`class GATv2`](../../gnn/keras/layers/GATv2.md): Simple Graph Attention Network V2 (GATv2).

[`class GraphUpdate`](../../gnn/keras/layers/GraphUpdate.md): Applies one round of updates to EdgeSets, NodeSets and Context.

[`class NextStateFromConcat`](../../gnn/keras/layers/NextStateFromConcat.md): Computes a new state by concatenating inputs and applying a Keras Layer.

[`class NodeSetUpdate`](../../gnn/keras/layers/NodeSetUpdate.md): A node state update with input from convolutions or other edge set inputs.

[`class Pool`](../../gnn/keras/layers/Pool.md): Pools a GraphTensor feature.

[`class Readout`](../../gnn/keras/layers/Readout.md): Reads a feature out of a GraphTensor.

[`class ReadoutFirstNode`](../../gnn/keras/layers/ReadoutFirstNode.md): Reads a feature from the first node of each graph conponent.

[`class ResidualNextState`](../../gnn/keras/layers/ResidualNextState.md): Updates a state with a residual block.

[`class SimpleConvolution`](../../gnn/keras/layers/SimpleConvolution.md): A convolution layer that applies message_fn on each edge.

