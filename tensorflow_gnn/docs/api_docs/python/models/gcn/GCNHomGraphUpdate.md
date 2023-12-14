# gcn.GCNHomGraphUpdate

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/gcn/gcn_conv.py#L285-L345">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Returns a graph update layer for GCN convolution.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gcn.GCNHomGraphUpdate(
    *,
    units: int,
    receiver_tag: tfgnn.IncidentNodeTag = tfgnn.TARGET,
    add_self_loops: bool = False,
    feature_name: str = tfgnn.HIDDEN_STATE,
    name: str = &#x27;gcn&#x27;,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

The returned layer performs one update step of a Graph Convolutional Network
(GCN) from https://arxiv.org/abs/1609.02907 on a GraphTensor that stores a
homogeneous graph. For heterogeneous graphs with multiple edge sets connecting a
single node set, users are advised to consider a GraphUpdate with one or more
GCNConv objects instead.

> IMPORTANT: By default, the graph convolution computed by this class takes
> inputs only along edges that are explicitly stored in the input GraphTensor.
> Including the old node state in the inputs for computing the new node state
> requires having an explicit loop in the edge set, or setting
> `add_self_loops=True`.

Thie layer returned by this function can be restored from config by
`tf.keras.models.load_model()` when saved as part of a Keras model using
`save_format="tf"`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>units</code><a id="units"></a>
</td>
<td>
The dimension of output hidden states for each node.
</td>
</tr><tr>
<td>
<code>receiver_tag</code><a id="receiver_tag"></a>
</td>
<td>
The default is <code>tfgnn.TARGET</code>,
but it is perfectly reasonable to do a convolution towards the
<code>tfgnn.SOURCE</code> instead. (Source and target are conventional names for
the incident nodes of a directed edge, data flow in a GNN may happen
in either direction.)
</td>
</tr><tr>
<td>
<code>add_self_loops</code><a id="add_self_loops"></a>
</td>
<td>
Whether to compute the result as if a loop from each node
to itself had been added to the edge set.
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
Any optional arguments to GCNConv, see there.
</td>
</tr>
</table>
