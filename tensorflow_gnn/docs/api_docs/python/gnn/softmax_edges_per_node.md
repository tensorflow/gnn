description: Returns softmax() of edge values per common SOURCE or TARGET node.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gnn.softmax_edges_per_node" />
<meta itemprop="path" content="Stable" />
</div>

# gnn.softmax_edges_per_node

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/normalization_ops.py#L78-L87">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Returns softmax() of edge values per common SOURCE or TARGET node.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gnn.softmax_edges_per_node(
    graph_tensor: <a href="../gnn/GraphTensor.md"><code>gnn.GraphTensor</code></a>,
    edge_set_name: gt.EdgeSetName,
    node_tag: const.IncidentNodeTag,
    *,
    feature_value: Optional[gt.Field] = None,
    feature_name: Optional[gt.FieldName] = None
) -> <a href="../gnn/Field.md"><code>gnn.Field</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->
