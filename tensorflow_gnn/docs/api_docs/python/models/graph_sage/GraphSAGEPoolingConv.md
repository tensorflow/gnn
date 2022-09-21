# graph_sage.GraphSAGEPoolingConv

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/graph_sage/layers.py#L125-L250">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

GraphSAGE: pooling aggregator transform of neighbors followed by linear
transformation.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>graph_sage.GraphSAGEPoolingConv(
    *,
    receiver_tag: tfgnn.IncidentNodeTag,
    sender_node_feature: Optional[tfgnn.FieldName] = tfgnn.HIDDEN_STATE,
    units: int,
    hidden_units: int,
    reduce_type: str = &#x27;max_no_inf&#x27;,
    use_bias: bool = True,
    dropout_rate: float = 0.0,
    activation: Union[str, Callable[..., Any]] = &#x27;relu&#x27;,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

For a complete GraphSAGE update on a node set, use a this class in a
NodeSetUpdate together with the
<a href="../graph_sage/GraphSAGENextState.md"><code>graph_sage.GraphSAGENextState</code></a>
layer to update the final node state (see there for details).

GraphSAGE and the pooling aggregation are from Hamilton et al.:
["Inductive Representation Learning on Large Graphs"](https://arxiv.org/abs/1706.02216),
2017. Similar to
<a href="../graph_sage/GraphSAGEAggregatorConv.md"><code>graph_sage.GraphSAGEAggregatorConv</code></a>,
dropout is applied to the inputs of neighbor nodes (separately for each
node-neighbor pair). Then, they are passed through a fully connected layer and
aggregated by an element-wise maximum (or whichever reduce_type is specified),
see Eq. (3) in paper. Finally, the result is multiplied with the final weights
mapping it to output space of units dimension.

The name of this class reflects the terminology of the paper, where "pooling"
involves the aforementioned hidden layer. For element-wise aggregation (as in
`tfgnn.pool_edges_to_node()`), see
<a href="../graph_sage/GraphSAGEAggregatorConv.md"><code>graph_sage.GraphSAGEAggregatorConv</code></a>.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`receiver_tag`<a id="receiver_tag"></a>
</td>
<td>
Either one of `tfgnn.SOURCE` or `tfgnn.TARGET`. The results
of GraphSAGE are aggregated for this graph piece. If set to
`tfgnn.SOURCE` or `tfgnn.TARGET`, the layer will be called for an edge
set and will aggregate results at the specified endpoint of the edges.
</td>
</tr><tr>
<td>
`sender_node_feature`<a id="sender_node_feature"></a>
</td>
<td>
Can be set to specify the feature name for use as the
input feature from sender nodes to GraphSAGE aggregation, defaults to
`tfgnn.HIDDEN_STATE`.
</td>
</tr><tr>
<td>
`units`<a id="units"></a>
</td>
<td>
Number of output units for the final dimensionality of the output
from the layer.
</td>
</tr><tr>
<td>
`hidden_units`<a id="hidden_units"></a>
</td>
<td>
Number of output units for the linear transformation applied
to the sender node features.This specifies the output dimensions of the
W_pool from Eq. (3) in
[Hamilton et al., 2017](https://arxiv.org/abs/1706.02216).
</td>
</tr><tr>
<td>
`reduce_type`<a id="reduce_type"></a>
</td>
<td>
An aggregation operation name. Supported list of aggregation
operators can be found at
`tfgnn.get_registered_reduce_operation_names()`.
</td>
</tr><tr>
<td>
`use_bias`<a id="use_bias"></a>
</td>
<td>
If true a bias term will be added to the linear transformations
for the sender node features.
</td>
</tr><tr>
<td>
`dropout_rate`<a id="dropout_rate"></a>
</td>
<td>
Can be set to a dropout rate that will be applied to sender
node features (independently on each edge).
</td>
</tr><tr>
<td>
`activation`<a id="activation"></a>
</td>
<td>
The nonlinearity applied to the concatenated or added node
state and aggregated sender node features. This can be specified as a
Keras layer, a tf.keras.activations.* function, or a string understood
by `tf.keras.layers.Activation()`. Defaults to relu.
</td>
</tr><tr>
<td>
`**kwargs`<a id="**kwargs"></a>
</td>
<td>
Additional arguments for the Layer.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`takes_receiver_input`<a id="takes_receiver_input"></a>
</td>
<td>
If `False`, all calls to convolve() will get `receiver_input=None`.
</td>
</tr><tr>
<td>
`takes_sender_edge_input`<a id="takes_sender_edge_input"></a>
</td>
<td>
If `False`, all calls to convolve() will get `sender_edge_input=None`.
</td>
</tr><tr>
<td>
`takes_sender_node_input`<a id="takes_sender_node_input"></a>
</td>
<td>
If `False`, all calls to convolve() will get `sender_node_input=None`.
</td>
</tr>
</table>

## Methods

<h3 id="convolve"><code>convolve</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/graph_sage/layers.py#L235-L250">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>convolve(
    *,
    sender_node_input: Optional[tf.Tensor],
    sender_edge_input: Optional[tf.Tensor],
    receiver_input: Optional[tf.Tensor],
    broadcast_from_sender_node: Callable[[tf.Tensor], tf.Tensor],
    broadcast_from_receiver: Callable[[tf.Tensor], tf.Tensor],
    pool_to_receiver: Callable[..., tf.Tensor],
    training: bool
) -> tf.Tensor
</code></pre>

Overridden internal method of the base class.
