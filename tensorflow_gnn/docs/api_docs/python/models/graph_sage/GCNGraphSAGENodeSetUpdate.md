<!-- lint-g3mark -->

# graph_sage.GCNGraphSAGENodeSetUpdate

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/graph_sage/layers.py#L256-L472">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

GCNGraphSAGENodeSetUpdate is an extension of the mean aggregator operator.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>graph_sage.GCNGraphSAGENodeSetUpdate(
    *,
    edge_set_names: Set[str],
    receiver_tag: tfgnn.IncidentNodeTag,
    reduce_type: str = &#x27;mean&#x27;,
    self_node_feature: str = tfgnn.HIDDEN_STATE,
    sender_node_feature: str = tfgnn.HIDDEN_STATE,
    units: int,
    dropout_rate: float = 0.0,
    activation: Union[str, Callable[..., Any]] = &#x27;relu&#x27;,
    use_bias: bool = False,
    share_weights: bool = False,
    add_self_loop: bool = True,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

For a complete GraphSAGE update on a node set, use this layer in a `GraphUpdate`
call as a `NodeSetUpdate` layer. An example update would look as below:

``` python
import tensorflow_gnn as tfgnn
graph = tfgnn.keras.layers.GraphUpdate(
    node_sets={
        "paper":
            graph_sage.GCNGraphSAGENodeSetUpdate(
                edge_set_names=["cites", "writes"],
                receiver_tag=tfgnn.TARGET,
                units=32)
    })(graph)
```

This class extends Eq. (2) from [Hamilton et
al., 2017](https://arxiv.org/abs/1706.02216) to multiple edge sets. For each
node state pooled from the configured edge list and the self node states there's
a separate weight vector learned which is mapping each to the same output
dimensions. Also if specified a random dropout operation with given probability
will be applied to all the node states. If share_weights is enabled, then it'll
learn the same weights for self and sender node states, this is the
implementation for homogeneous graphs from the paper. Note that enabling this
requires both sender and receiver node states to have the same dimension. Below
is the simplified summary of the applied transformations to generate new node
states:

    h_v = activation(
              reduce(  {W_E * D_p[h_{N(v)}] for all edge sets E}
                     U {W_self * D_p[h_v]})
              + b)

N(v) denotes the neighbors of node v, D_p denotes dropout with probability p
which is applied independenly to self and sender node states, W_E and W_self
denote the edge and self node transformation weight vectors and b is the bias.
If add_self_loop is disabled then self node states won't be used during the
reduce operation, instead only the sender node states will be accumulated based
on the reduce_type specified. If share_weights is set to True, then single
weight matrix will be used in place of W_E and W_self.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`edge_set_names`<a id="edge_set_names"></a>
</td>
<td>
A list of edge set names to broadcast sender node states.
</td>
</tr><tr>
<td>
`receiver_tag`<a id="receiver_tag"></a>
</td>
<td>
Either one of `tfgnn.SOURCE` or `tfgnn.TARGET`. The results
of GraphSAGE convolution are aggregated for this graph piece. If set to
`tfgnn.SOURCE` or `tfgnn.TARGET`, the layer will be called for each edge
set and will aggregate results at the specified endpoint of the edges.
This should point at the node_set_name for each of the specified edge
set name in the edge_set_name_dict.
</td>
</tr><tr>
<td>
`reduce_type`<a id="reduce_type"></a>
</td>
<td>
An aggregation operation name. Supported list of aggregation
operators are sum or mean.
</td>
</tr><tr>
<td>
`self_node_feature`<a id="self_node_feature"></a>
</td>
<td>
Feature name for the self node sets to be aggregated
with the broadcasted sender node states. Default is
`tfgnn.HIDDEN_STATE`.
</td>
</tr><tr>
<td>
`sender_node_feature`<a id="sender_node_feature"></a>
</td>
<td>
Feature name for the sender node sets. Default is
`tfgnn.HIDDEN_STATE`.
</td>
</tr><tr>
<td>
`units`<a id="units"></a>
</td>
<td>
Number of output units for the linear transformation applied to
sender node and self node features.
</td>
</tr><tr>
<td>
`dropout_rate`<a id="dropout_rate"></a>
</td>
<td>
Can be set to a dropout rate that will be applied to both
self node and the sender node states.
</td>
</tr><tr>
<td>
`activation`<a id="activation"></a>
</td>
<td>
The nonlinearity applied to the update node states. This can
be specified as a Keras layer, a tf.keras.activations.* function, or a
string understood by tf.keras.layers.Activation(). Defaults to relu.
</td>
</tr><tr>
<td>
`use_bias`<a id="use_bias"></a>
</td>
<td>
If true a bias term will be added to mean aggregated feature
vectors before applying non-linear activation.
</td>
</tr><tr>
<td>
`share_weights`<a id="share_weights"></a>
</td>
<td>
If left unset, separate weights are used to transform the
inputs along each edge set and the input of previous node states (unless
disabled by add_self_loop=False). If enabled, a single weight matrix is
applied to all inputs.
</td>
</tr><tr>
<td>
`add_self_loop`<a id="add_self_loop"></a>
</td>
<td>
If left at True (the default), each node state update takes
  the node's old state as an explicit input next to all the inputs along
  edge sets. Typically, this is done when the graph does not have loops.
  If set to False, each node state update uses only the inputs along the
  requested edge sets. Typically, this is done when loops are already
  contained among the edges.
**kwargs:
</td>
</tr>
</table>
