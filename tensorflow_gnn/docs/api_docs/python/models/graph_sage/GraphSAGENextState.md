# graph_sage.GraphSAGENextState

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/graph_sage/layers.py#L600-L751">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

GraphSAGENextState: compute new node states with GraphSAGE algorithm.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>graph_sage.GraphSAGENextState(
    *,
    units: int,
    use_bias: bool = True,
    dropout_rate: float = 0.0,
    feature_name: str = tfgnn.HIDDEN_STATE,
    l2_normalize: bool = True,
    combine_type: str = &#x27;sum&#x27;,
    activation: Union[str, Callable[..., Any]] = &#x27;relu&#x27;,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

This layer lets you compute a GraphSAGE update of node states from the outputs
of a
<a href="../graph_sage/GraphSAGEAggregatorConv.md"><code>graph_sage.GraphSAGEAggregatorConv</code></a>
and/or a
<a href="../graph_sage/GraphSAGEPoolingConv.md"><code>graph_sage.GraphSAGEPoolingConv</code></a>
on each of the specified end-point of edge sets.

Usage example (with strangely mixed aggregations for demonstration):

```python
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import graph_sage
graph = tfgnn.keras.layers.GraphUpdate(node_sets={
    "papers": tfgnn.keras.layers.NodeSetUpdate(
        {"citations": graph_sage.GraphSAGEAggregatorConv(
             units=32, receiver_tag=tfgnn.TARGET),
         "affiliations": graph_sage.GraphSAGEPoolingConv(
             units=32, hidden_units=16, receiver_tag=tfgnn.SOURCE)},
         graph_sage.GraphSAGENextState(units=32)),
    "...": ...,
})(graph)
```

The `units=...` parameter of the next-state layer and all convolutions must be
equal, unless `combine_type="concat"`is set.

GraphSAGE is Algorithm 1 in Hamilton et al.:
["Inductive Representation Learning on Large Graphs"](https://arxiv.org/abs/1706.02216),
2017. It computes the new hidden state h_v for each node v from a concatenation
of the previous hidden state with an aggregation of the neighbor states as

$$h_v = \sigma(W \text{ concat}(h_v, h_{N(v)}))$$

...followed by L2 normalization. This implementation uses the mathematically
equivalent formulation

$$h_v = \sigma(W_{\text{self}} h_v + W_{\text{neigh}} h_{N(v)}),$$

which transforms both inputs separately and then combines them by summation
(assuming the default `combine_type="sum"`; see there for more).

The GraphSAGE*Conv classes are in charge of computing the right-hand term
W_{neigh} h_{N(v)} (for one edge set each, typically with separate weights).
This class is in charge of computing the left-hand term W_{self} h_v from the
old node state h_v, combining it with the results for each edge set and
computing the new node state h_v from it.

Beyond the original GraphSAGE, this class supports:

*   dropout, applied to the input h_v, analogous to the dropout provided by
    GraphSAGE*Conv for their inputs;
*   a bias term added just before the final nonlinearity;
*   a configurable combine_type (originally "sum");
*   additional options to influence normalization, activation, etc.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>units</code><a id="units"></a>
</td>
<td>
Number of output units for the linear transformation applied to the
node feature.
</td>
</tr><tr>
<td>
<code>use_bias</code><a id="use_bias"></a>
</td>
<td>
If true a bias term will be added to the linear transformations
for the self node feature.
</td>
</tr><tr>
<td>
<code>dropout_rate</code><a id="dropout_rate"></a>
</td>
<td>
Can be set to a dropout rate that will be applied to the
node feature.
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
<code>l2_normalize</code><a id="l2_normalize"></a>
</td>
<td>
If enabled l2 normalization will be applied to node state
vectors.
</td>
</tr><tr>
<td>
<code>combine_type</code><a id="combine_type"></a>
</td>
<td>
Can be set to "sum" or "concat". The default "sum" recovers
the original behavior of GraphSAGE (as reformulated above): the results
of transforming the old state and the neighborhood state are added.
Setting this to "concat" concatenates the results of the transformations
(not described in the paper).
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
Forwarded to the base class tf.keras.layers.Layer.
</td>
</tr>
</table>
