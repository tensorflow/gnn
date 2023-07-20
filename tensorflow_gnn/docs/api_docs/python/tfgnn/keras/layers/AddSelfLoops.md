# tfgnn.keras.layers.AddSelfLoops

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/graph_ops.py#L204-L223">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Adds self-loops to scalar graphs.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.AddSelfLoops(
    edge_set_name
)
</code></pre>

<!-- Placeholder for "Used in" -->

The edge_set_name is expected to be a homogeneous edge (connects a node pair of
the node set). NOTE: Self-connections will always be added, regardless if if
self-connections already exist or not.
