# gcn.GCNConv

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/gcn/gcn_conv.py#L26-L282">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Implements the Graph Convolutional Network by Kipf&Welling (2016).

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gcn.GCNConv(
    units: int,
    *,
    receiver_tag: tfgnn.IncidentNodeTag = tfgnn.TARGET,
    activation=&#x27;relu&#x27;,
    use_bias: bool = True,
    add_self_loops: bool = False,
    kernel_initializer: Any = None,
    node_feature: Optional[str] = tfgnn.HIDDEN_STATE,
    kernel_regularizer: Any = None,
    edge_weight_feature_name: Optional[tfgnn.FieldName] = None,
    degree_normalization: str = &#x27;in_out&#x27;,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

This class implements a Graph Convolutional Network from
https://arxiv.org/abs/1609.02907 as a Keras layer that can be used as a
convolution on an edge set in a tfgnn.keras.layers.NodeSetUpdate. The original
algorithm proposed in the Graph Convolutional Network paper expects a symmetric
graph as input. That is, if there is an edge from node i to node j, there is
also an edge from node j to node i. This implementation, however, is able to
take asymmetric graphs as input.

Let $w_{ij}$ be the weight of the edge from sender i to receiver j. Let
$\deg^{in}_i$ be the number of incoming edges to i (in the direction of message
flow, see `receiver_tag`), and $\deg^{out}_i$ the number of outgoing edges from
i. In a symmetric graphs, both are equal.

In this implementation, we provide multiple approaches for normalizing an edge
weight $w_{ij}$ in $v_{ij}$, namely `"none"`, `"in"`, `"out"`, `"in_out"`, and
`"in_in"`. Setting normalization to `"none"` will end up in set $v_{ij} =
w_{ij}$. The `"in"` normalization normalizes edge weights using the in-degree of
the receiver node, that is:

$$v_{ij} = w_{ij} / \deg^{in}_j.$$

The `"out"` normalization normalizes edges using the out-degree of sender nodes
that is:

$$v_{ij} = w_{ij} / \deg^{out}_i.$$

The `"in_out"` normalization normalizes edges as follows:

$$v_{ij} = w_{ij} / (\sqrt{\deg^{out}_i} \sqrt{\deg^{in}_j}).$$

The `"in_in"` normalization normalizes the edge weights as:

$$v_{ij} = w_{ij} / (\sqrt{\deg^{in}_i} \sqrt{\deg^{in}_j}).$$

For symmetric graphs (as in the original GCN paper), `"in_out"` and `"in_in"`
are equal, but the latter needs to compute degrees just once.

This layer can be restored from config by `tf.keras.models.load_model()` when
saved as part of a Keras model using `save_format="tf"`.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Init arguments</h2></th></tr>

<tr>
<td>
<code>units</code><a id="units"></a>
</td>
<td>
Number of output units for this transformation applied to sender
node features.
</td>
</tr><tr>
<td>
<code>receiver_tag</code><a id="receiver_tag"></a>
</td>
<td>
This layer's result is obtained by pooling the per-edge
results at this endpoint of each edge. The default is <code>tfgnn.TARGET</code>,
but it is perfectly reasonable to do a convolution towards the
<code>tfgnn.SOURCE</code> instead. (Source and target are conventional names for
the incident nodes of a directed edge, data flow in a GNN may happen
in either direction.)
</td>
</tr><tr>
<td>
<code>activation</code><a id="activation"></a>
</td>
<td>
Keras activation to apply to the result, defaults to 'relu'.
</td>
</tr><tr>
<td>
<code>use_bias</code><a id="use_bias"></a>
</td>
<td>
Whether to add bias in the final transformation. The original
paper doesn't use a bias, but this defaults to True to be consistent
with Keras and other implementations.
</td>
</tr><tr>
<td>
<code>add_self_loops</code><a id="add_self_loops"></a>
</td>
<td>
Whether to compute the result as if a loop from each node
to itself had been added to the edge set. The self-loop edges are added
with an edge weight of one.
</td>
</tr><tr>
<td>
<code>kernel_initializer</code><a id="kernel_initializer"></a>
</td>
<td>
Can be set to a <code>kernel_initializer</code> as understood
by <code>tf.keras.layers.Dense</code> etc.
An <code>Initializer</code> object gets cloned before use to ensure a fresh seed,
if not set explicitly. For more, see <code>tfgnn.keras.clone_initializer()</code>.
</td>
</tr><tr>
<td>
<code>node_feature</code><a id="node_feature"></a>
</td>
<td>
Name of the node feature to transform.
</td>
</tr><tr>
<td>
<code>edge_weight_feature_name</code><a id="edge_weight_feature_name"></a>
</td>
<td>
Can be set to the name of a feature on the edge
set that supplies a scalar weight for each edge. The GCN computation uses
it as the edge's entry in the adjacency matrix, instead of the default 1.
</td>
</tr><tr>
<td>
<code>degree_normalization</code><a id="degree_normalization"></a>
</td>
<td>
Can be set to <code>"none"</code>, <code>"in"</code>, <code>"out"</code>, <code>"in_out"</code>,
or <code>"in_in"</code>, as explained above.
</td>
</tr><tr>
<td>
<code>**kwargs</code><a id="**kwargs"></a>
</td>
<td>
additional arguments for the Layer.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Call arguments</h2></th></tr>

<tr>
<td>
<code>graph</code><a id="graph"></a>
</td>
<td>
The GraphTensor on which to apply the layer.
</td>
</tr><tr>
<td>
<code>edge_set_name</code><a id="edge_set_name"></a>
</td>
<td>
Edge set of <code>graph</code> over which to apply the layer.
</td>
</tr>
</table>

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
