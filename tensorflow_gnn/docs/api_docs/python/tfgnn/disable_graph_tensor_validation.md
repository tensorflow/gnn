# tfgnn.disable_graph_tensor_validation

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_constants.py#L125-L135">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Disables both static and runtime checks of graph tensors.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.disable_graph_tensor_validation()
</code></pre>

<!-- Placeholder for "Used in" -->

IMPORTANT: This is temporary workaround for the legacy code (before TF-GNN 1.0
release) that may rely on the inconsistent number of graph tensor items and
allowed edges with adjaceny indices for non-existing nodes. **DO NOT USE**.
