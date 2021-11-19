description: Broadcasts a context value to the node_set nodes.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gnn.broadcast_context_to_nodes" />
<meta itemprop="path" content="Stable" />
</div>

# gnn.broadcast_context_to_nodes

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_ops.py#L139-L171">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Broadcasts a context value to the `node_set` nodes.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gnn.broadcast_context_to_nodes(
    graph_tensor: <a href="../gnn/GraphTensor.md"><code>gnn.GraphTensor</code></a>,
    node_set_name: NodeSetName,
    *,
    feature_value: Optional[Field] = None,
    feature_name: Optional[FieldName] = None
) -> <a href="../gnn/Field.md"><code>gnn.Field</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

Given a particular node set (as identified by `node_set_name`), this operation
collects context features from the corresponding graphs to each node. See the
corresponding `pool_nodes_to_context()` mirror operation).

The context feature to fetch values from is provided either by name (using
`feature_name`) and found in the graph tensor itself, or provided explicitly
(using `feature_value`) in which case its shape has to be compatible with the
shape prefix of the node set being gathered from. One of `feature_value` or
`feature_name` must be specified.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`graph_tensor`
</td>
<td>
A scalar GraphTensor.
</td>
</tr><tr>
<td>
`node_set_name`
</td>
<td>
A node set name.
</td>
</tr><tr>
<td>
`feature_value`
</td>
<td>
A ragged or dense graph context feature value.
</td>
</tr><tr>
<td>
`feature_name`
</td>
<td>
A context feature name.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Graph context value broadcast to the `node_set` nodes. The first dimension
size equals to the number of nodes, the higher dimensions do not change.
</td>
</tr>

</table>

