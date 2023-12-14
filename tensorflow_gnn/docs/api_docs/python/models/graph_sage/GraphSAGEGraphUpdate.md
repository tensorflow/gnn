# graph_sage.GraphSAGEGraphUpdate

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/graph_sage/layers.py#L484-L597">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Returns a GraphSAGE GraphUpdater layer for nodes in node_set_names.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>graph_sage.GraphSAGEGraphUpdate(
    *,
    units: int,
    hidden_units: Optional[int] = None,
    receiver_tag: tfgnn.IncidentNodeTag,
    node_set_names: Optional[Collection[tfgnn.NodeSetName]] = None,
    reduce_type: str = &#x27;mean&#x27;,
    use_pooling: bool = True,
    use_bias: bool = True,
    dropout_rate: float = 0.0,
    l2_normalize: bool = True,
    combine_type: str = &#x27;sum&#x27;,
    activation: Union[str, Callable[..., Any]] = &#x27;relu&#x27;,
    feature_name: str = tfgnn.HIDDEN_STATE,
    name: str = &#x27;graph_sage&#x27;,
    **kwargs
) -> tf.keras.layers.Layer
</code></pre>

<!-- Placeholder for "Used in" -->

For more information on GraphSAGE algorithm please refer to
[Hamilton et al., 2017](https://arxiv.org/abs/1706.02216). Returned layer
applies only one step of GraphSAGE convolution over the incident nodes of the
edge_set_name_list for the specified node_set_name node.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>units</code><a id="units"></a>
</td>
<td>
Number of output units of the linear transformation applied to both
final aggregated sender node features as well as the self node feature.
</td>
</tr><tr>
<td>
<code>hidden_units</code><a id="hidden_units"></a>
</td>
<td>
Number of output units to be configure for GraphSAGE pooling
type convolution only.
</td>
</tr><tr>
<td>
<code>receiver_tag</code><a id="receiver_tag"></a>
</td>
<td>
Either one of <code>tfgnn.SOURCE</code> or <code>tfgnn.TARGET</code>. The results of
GraphSAGE are aggregated for this graph piece. When set to <code>tfgnn.SOURCE</code>
or <code>tfgnn.TARGET</code>, the layer is called for an edge set and will aggregate
results at the specified endpoint of the edges.
</td>
</tr><tr>
<td>
<code>node_set_names</code><a id="node_set_names"></a>
</td>
<td>
By default, this layer updates all node sets that receive
from at least one edge set. Optionally, this argument can specify a subset
of those node sets. It is not allowed to include node sets that do not
receive messages from any edge set. It is also not allowed to include
auxiliary node sets.
</td>
</tr><tr>
<td>
<code>reduce_type</code><a id="reduce_type"></a>
</td>
<td>
An aggregation operation name. Supported list of aggregation
operators can be found at <code>tfgnn.get_registered_reduce_operation_names()</code>.
</td>
</tr><tr>
<td>
<code>use_pooling</code><a id="use_pooling"></a>
</td>
<td>
If enabled, <a href="../graph_sage/GraphSAGEPoolingConv.md"><code>graph_sage.GraphSAGEPoolingConv</code></a> will be used,
otherwise <a href="../graph_sage/GraphSAGEAggregatorConv.md"><code>graph_sage.GraphSAGEAggregatorConv</code></a> will be executed for the
provided edges.
</td>
</tr><tr>
<td>
<code>use_bias</code><a id="use_bias"></a>
</td>
<td>
If true a bias term will be added to the linear transformations
for the incident node features as well as for the self node feature.
</td>
</tr><tr>
<td>
<code>dropout_rate</code><a id="dropout_rate"></a>
</td>
<td>
Can be set to a dropout rate that will be applied to both
incident node features as well as the self node feature.
</td>
</tr><tr>
<td>
<code>l2_normalize</code><a id="l2_normalize"></a>
</td>
<td>
If enabled l2 normalization will be applied to final node
states.
</td>
</tr><tr>
<td>
<code>combine_type</code><a id="combine_type"></a>
</td>
<td>
Can be set to "sum" or "concat". If it's specified as concat
node state will be concatenated with the sender node features, otherwise
node state will be added with the sender node features.
</td>
</tr><tr>
<td>
<code>activation</code><a id="activation"></a>
</td>
<td>
The nonlinearity applied to the concatenated or added node state
and aggregated sender node features. This can be specified as a Keras
layer, a tf.keras.activations.* function, or a string understood by
<code>tf.keras.layers.Activation()</code>. Defaults to relu.
</td>
</tr><tr>
<td>
<code>feature_name</code><a id="feature_name"></a>
</td>
<td>
The feature name of node states; defaults to
<code>tfgnn.HIDDEN_STATE</code>.
</td>
</tr><tr>
<td>
<code>name</code><a id="name"></a>
</td>
<td>
Optionally, a name for the layer returned.
</td>
</tr><tr>
<td>
<code>**kwargs</code><a id="**kwargs"></a>
</td>
<td>
Any optional arguments to <a href="../graph_sage/GraphSAGEPoolingConv.md"><code>graph_sage.GraphSAGEPoolingConv</code></a>,
<a href="../graph_sage/GraphSAGEAggregatorConv.md"><code>graph_sage.GraphSAGEAggregatorConv</code></a> or <a href="../graph_sage/GraphSAGENextState.md"><code>graph_sage.GraphSAGENextState</code></a>,
see there.
</td>
</tr>
</table>
