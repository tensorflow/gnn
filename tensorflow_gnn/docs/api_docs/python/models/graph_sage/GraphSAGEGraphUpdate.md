description: Returns a GraphSAGE GraphUpdater layer for nodes in node_set_names.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="graph_sage.GraphSAGEGraphUpdate" />
<meta itemprop="path" content="Stable" />
</div>

# graph_sage.GraphSAGEGraphUpdate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/graph_sage/layers.py#L458-L580">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Returns a GraphSAGE GraphUpdater layer for nodes in node_set_names.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>graph_sage.GraphSAGEGraphUpdate(
    *,
    node_set_names: Set[str],
    receiver_tag: tfgnn.IncidentNodeTag,
    reduce_type: str = &#x27;mean&#x27;,
    use_pooling: bool = True,
    use_bias: bool = True,
    dropout_rate: float = 0.0,
    units: int,
    hidden_units: Optional[int] = None,
    l2_normalize: bool = True,
    combine_type: str = &#x27;sum&#x27;,
    activation: Union[str, Callable[..., Any]] = &#x27;relu&#x27;,
    feature_name: str = tfgnn.HIDDEN_STATE,
    name: str = &#x27;graph_sage&#x27;,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

For more information on GraphSAGE algorithm please refer to
[Hamilton et al., 2017](https://arxiv.org/abs/1706.02216). Returned layer
applies only one step of GraphSAGE convolution over the incident nodes of the
edge_set_name_list for the specified node_set_name node.

Example: GraphSAGE aggregation on heterogenous incoming edges would look as
below:

```python
graph = tfgnn.keras.layers.GraphUpdate(
    node_sets={"paper": tfgnn.keras.layers.NodeSetUpdate(
        {"cites": graph_sage.GraphSAGEPoolingConv(
             receiver_tag=tfgnn.TARGET, units=32),
          "writes": graph_sage.GraphSAGEPoolingConv(
             receiver_tag=tfgnn.TARGET, units=32, hidden_units=16)},
        graph_sage.GraphSAGENextState(units=32, dropout_rate=0.05))}
)(graph)
```

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`node_set_names`
</td>
<td>
A set of node_set_names for which GraphSAGE graph update
happens over each of their incident edges, where node_set_name is
configured as the receiver_tag end.
</td>
</tr><tr>
<td>
`receiver_tag`
</td>
<td>
Either one of `tfgnn.SOURCE` or `tfgnn.TARGET`. The results of
GraphSAGE are aggregated for this graph piece. When set to `tfgnn.SOURCE`
or `tfgnn.TARGET`, the layer is called for an edge set and will aggregate
results at the specified endpoint of the edges.
</td>
</tr><tr>
<td>
`reduce_type`
</td>
<td>
An aggregation operation name. Supported list of aggregation
operators can be found at `tfgnn.get_registered_reduce_operation_names()`.
</td>
</tr><tr>
<td>
`use_pooling`
</td>
<td>
If enabled, <a href="../graph_sage/GraphSAGEPoolingConv.md"><code>graph_sage.GraphSAGEPoolingConv</code></a> will be used,
otherwise <a href="../graph_sage/GraphSAGEAggregatorConv.md"><code>graph_sage.GraphSAGEAggregatorConv</code></a> will be executed for the
provided edges.
</td>
</tr><tr>
<td>
`use_bias`
</td>
<td>
If true a bias term will be added to the linear transformations
for the incident node features as well as for the self node feature.
</td>
</tr><tr>
<td>
`dropout_rate`
</td>
<td>
Can be set to a dropout rate that will be applied to both
incident node features as well as the self node feature.
</td>
</tr><tr>
<td>
`units`
</td>
<td>
Number of output units of the linear transformation applied to both
final aggregated sender node features as well as the self node feature.
</td>
</tr><tr>
<td>
`hidden_units`
</td>
<td>
Number of output units to be configure for GraphSAGE pooling
type convolution only.
</td>
</tr><tr>
<td>
`l2_normalize`
</td>
<td>
If enabled l2 normalization will be applied to final node
states.
</td>
</tr><tr>
<td>
`combine_type`
</td>
<td>
Can be set to "sum" or "concat". If it's specified as concat
node state will be concatenated with the sender node features, otherwise
node state will be added with the sender node features.
</td>
</tr><tr>
<td>
`activation`
</td>
<td>
The nonlinearity applied to the concatenated or added node state
and aggregated sender node features. This can be specified as a Keras
layer, a tf.keras.activations.* function, or a string understood by
`tf.keras.layers.Activation()`. Defaults to relu.
</td>
</tr><tr>
<td>
`feature_name`
</td>
<td>
The feature name of node states; defaults to
`tfgnn.HIDDEN_STATE`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Optionally, a name for the layer returned.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Any optional arguments to <a href="../graph_sage/GraphSAGEPoolingConv.md"><code>graph_sage.GraphSAGEPoolingConv</code></a>,
<a href="../graph_sage/GraphSAGEAggregatorConv.md"><code>graph_sage.GraphSAGEAggregatorConv</code></a> or <a href="../graph_sage/GraphSAGENextState.md"><code>graph_sage.GraphSAGENextState</code></a>,
see there.
</td>
</tr>
</table>
