# gcn.GCNConv

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/gcn/gcn_conv.py#L14-L174">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Implements the Graph Convolutional Network by Kipf&Welling (2016).

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gcn.GCNConv(
    units: int,
    *,
    receiver_tag: tfgnn.IncidentNodeTag = tfgnn.TARGET,
    activation=&#x27;relu&#x27;,
    use_bias: bool = True,
    add_self_loops: bool = False,
    normalize: bool = True,
    kernel_initializer: bool = None,
    node_feature: Optional[str] = tfgnn.HIDDEN_STATE,
    kernel_regularizer: Optional[_RegularizerType] = None,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

This class implements a Graph Convolutional Network from
https://arxiv.org/abs/1609.02907 as a Keras layer that can be used as a
convolution on an edge set in a tfgnn.keras.layers.NodeSetUpdate.

#### Init arguments:

*   <b>`units`</b>: The number of output features (input features are inferred).
*   <b>`receiver_tag`</b>: This layer's result is obtained by pooling the
    per-edge results at this endpoint of each edge. The default is
    `tfgnn.TARGET`, but it is perfectly reasonable to do a convolution towards
    the `tfgnn.SOURCE` instead. (Source and target are conventional names for
    the incident nodes of a directed edge, data flow in a GNN may happen in
    either direction.)
*   <b>`activation`</b>: Keras activation to apply to the result, defaults to
    'relu'.
*   <b>`use_bias`</b>: Whether to add bias in the final transformation. The
    original paper doesn't use a bias, but this defaults to True to be
    consistent with Keras and other implementations.
*   <b>`add_self_loops`</b>: Whether to compute the result as if a loop from
    each node to itself had been added to the edge set.
*   <b>`normalize`</b>: Whether to normalize the node features by in-degree.
*   <b>`kernel_initializer`</b>: initializer of type tf.keras.initializers .
*   <b>`node_feature`</b>: Name of the node feature to transform.
*   <b>`**kwargs`</b>: additional arguments for the Layer.

#### Call arguments:

*   <b>`graph`</b>: The GraphTensor on which to apply the layer.
*   <b>`edge_set_name`</b>: Edge set of `graph` over which to apply the layer.

#### Example:

This example shows how to apply GCNConv to a graph with 2 discrete components.
This graph has one edge set, named `tfgnn.EDGES`. This returns a tf.Tensor of
shape (4,3). In order to return a GraphTensor this should be wrapped in
NodeSetUpdate/ EdgeSetUpdate.

```python
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models.gcn import gcn_conv
graph = tfgnn.GraphTensor.from_pieces(
   node_sets={
       tfgnn.NODES: tfgnn.NodeSet.from_fields(
           sizes=[2, 2],
           features={tfgnn.HIDDEN_STATE: tf.constant(
                         [[1., 0, 0], [0, 1, 0]]*2)})},
   edge_sets={
       tfgnn.EDGES: tfgnn.EdgeSet.from_fields(
           sizes=[2, 2],
           adjacency=tfgnn.Adjacency.from_indices(
               source=(tfgnn.NODES, tf.constant([0, 1, 2, 3],
                                                dtype=tf.int64)),
               target=(tfgnn.NODES, tf.constant([1, 0, 3, 2],
                                                dtype=tf.int64))))})
gcnconv = gcn_conv.GCNConv(3)
gcnconv(graph, edge_set_name=tfgnn.EDGES)   # Has shape=(4, 3).
```
