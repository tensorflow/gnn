<!-- lint-g3mark -->

# runner.RootNodeLabelFn

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/utils/label_fns.py#L43-L72">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Reads out a `tfgnn.Field` from the `GraphTensor` root (i.e. first) node.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>runner.RootNodeLabelFn(
    node_set_name: tfgnn.NodeSetName,
    *,
    feature_name: tfgnn.FieldName = tfgnn.HIDDEN_STATE,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->
