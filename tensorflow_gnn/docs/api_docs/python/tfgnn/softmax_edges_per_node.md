description: Returns softmax() of edge values per common node_tag node.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfgnn.softmax_edges_per_node" />
<meta itemprop="path" content="Stable" />
</div>

# tfgnn.softmax_edges_per_node

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/normalization_ops.py#L80-L89">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns softmax() of edge values per common `node_tag` node.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.softmax_edges_per_node(
    graph_tensor: <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>,
    edge_set_name: gt.EdgeSetName,
    node_tag: const.IncidentNodeTag,
    *,
    feature_value: Optional[gt.Field] = None,
    feature_name: Optional[gt.FieldName] = None
) -> <a href="../tfgnn/Field.md"><code>tfgnn.Field</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->
