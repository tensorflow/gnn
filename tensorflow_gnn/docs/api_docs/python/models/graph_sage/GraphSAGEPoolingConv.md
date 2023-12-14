# graph_sage.GraphSAGEPoolingConv

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/graph_sage/layers.py#L129-L259">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

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

This layer can be restored from config by `tf.keras.models.load_model()` when
saved as part of a Keras model using `save_format="tf"`.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>receiver_tag</code><a id="receiver_tag"></a>
</td>
<td>
Either one of <code>tfgnn.SOURCE</code> or <code>tfgnn.TARGET</code>. The results
of GraphSAGE are aggregated for this graph piece. If set to
<code>tfgnn.SOURCE</code> or <code>tfgnn.TARGET</code>, the layer will be called for an edge
set and will aggregate results at the specified endpoint of the edges.
</td>
</tr><tr>
<td>
<code>sender_node_feature</code><a id="sender_node_feature"></a>
</td>
<td>
Can be set to specify the feature name for use as the
input feature from sender nodes to GraphSAGE aggregation, defaults to
<code>tfgnn.HIDDEN_STATE</code>.
</td>
</tr><tr>
<td>
<code>units</code><a id="units"></a>
</td>
<td>
Number of output units for the final dimensionality of the output
from the layer.
</td>
</tr><tr>
<td>
<code>hidden_units</code><a id="hidden_units"></a>
</td>
<td>
Number of output units for the linear transformation applied
to the sender node features.This specifies the output dimensions of the
W_pool from Eq. (3) in
[Hamilton et al., 2017](https://arxiv.org/abs/1706.02216).
</td>
</tr><tr>
<td>
<code>reduce_type</code><a id="reduce_type"></a>
</td>
<td>
An aggregation operation name. Supported list of aggregation
operators can be found at
<code>tfgnn.get_registered_reduce_operation_names()</code>.
</td>
</tr><tr>
<td>
<code>use_bias</code><a id="use_bias"></a>
</td>
<td>
If true a bias term will be added to the linear transformations
for the sender node features.
</td>
</tr><tr>
<td>
<code>dropout_rate</code><a id="dropout_rate"></a>
</td>
<td>
Can be set to a dropout rate that will be applied to sender
node features (independently on each edge).
</td>
</tr><tr>
<td>
<code>activation</code><a id="activation"></a>
</td>
<td>
The nonlinearity applied to the concatenated or added node
state and aggregated sender node features. This can be specified as a
Keras layer, a tf.keras.activations.* function, or a string understood
by <code>tf.keras.layers.Activation()</code>. Defaults to relu.
</td>
</tr><tr>
<td>
<code>**kwargs</code><a id="**kwargs"></a>
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
<code>takes_receiver_input</code><a id="takes_receiver_input"></a>
</td>
<td>
If <code>False</code>, all calls to convolve() will get <code>receiver_input=None</code>.
</td>
</tr><tr>
<td>
<code>takes_sender_edge_input</code><a id="takes_sender_edge_input"></a>
</td>
<td>
If <code>False</code>, all calls to convolve() will get <code>sender_edge_input=None</code>.
</td>
</tr><tr>
<td>
<code>takes_sender_node_input</code><a id="takes_sender_node_input"></a>
</td>
<td>
If <code>False</code>, all calls to convolve() will get <code>sender_node_input=None</code>.
</td>
</tr>
</table>

## Methods

<h3 id="convolve"><code>convolve</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/graph_sage/layers.py#L242-L259">View
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
    extra_receiver_ops: Any = None,
    training: bool
) -> tf.Tensor
</code></pre>

Overridden internal method of the base class.
