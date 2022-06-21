description: Returns a graph update layer for GCN convolution.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gcn.GCNConvGraphUpdate" />
<meta itemprop="path" content="Stable" />
</div>

# gcn.GCNConvGraphUpdate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/gcn/gcn_conv.py#L176-L229">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Returns a graph update layer for GCN convolution.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gcn.GCNConvGraphUpdate(
    *,
    units: int,
    edge_set_name: str,
    receiver_tag: tfgnn.IncidentNodeTag = tfgnn.TARGET,
    feature_name: str = tfgnn.HIDDEN_STATE,
    name: str = &#x27;gcn&#x27;,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

The returned layer performs one update step of a Graph Convolutional Network
(GCN) from https://arxiv.org/abs/1609.02907 on an edge set of a GraphTensor. It
is best suited for graphs that have just that one edge set, which connects one
node set to itself. For heterogeneous graphs with multiple edge sets connecting
a single node set, users are advised to consider a GraphUpdate with one or more
GCNConv objects instead.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`units`
</td>
<td>
The desired number of output node features.
</td>
</tr><tr>
<td>
`edge_set_name`
</td>
<td>
A GCNConv update happens on this edge set and its incident
node set of the input GraphTensor.
</td>
</tr><tr>
<td>
`receiver_tag`
</td>
<td>
The default is `tfgnn.TARGET`,
but it is perfectly reasonable to do a convolution towards the
`tfgnn.SOURCE` instead. (Source and target are conventional names for
the incident nodes of a directed edge, data flow in a GNN may happen
in either direction.)
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
Any optional arguments to GCNConv, see there.
</td>
</tr>
</table>
