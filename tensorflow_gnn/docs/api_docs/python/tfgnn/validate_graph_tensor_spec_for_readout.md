<!-- lint-g3mark -->

# tfgnn.validate_graph_tensor_spec_for_readout

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/readout.py#L30-L65">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Checks `graph_spec` supports `structured_readout()` from `required_keys`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.validate_graph_tensor_spec_for_readout(
    graph_spec: <a href="../tfgnn/GraphTensorSpec.md"><code>tfgnn.GraphTensorSpec</code></a>,
    required_keys: Optional[Sequence[str]] = None,
    *,
    readout_node_set: const.NodeSetName = &#x27;_readout&#x27;
) -> None
</code></pre>

<!-- Placeholder for "Used in" -->

This function checks that the `graph.spec` of a
<a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a> contains
correctly connected auxiliary graph pieces (edge sets and node sets) such that
subsequent calls to
<a href="../tfgnn/structured_readout.md"><code>tfgnn.structured_readout(graph,
key)</code></a> can work for all readout keys encoded in the spec. The argument
`required_keys` can be set to check that these particular values of `key` are
present.

This function only considers the
<a href="../tfgnn/GraphTensorSpec.md"><code>tfgnn.GraphTensorSpec</code></a>,
which means it can be handled completely at the time of tracing a `tf.function`
for non-eager execution. To also check the index values found in actual
<a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a> values,
call
<a href="../tfgnn/validate_graph_tensor_for_readout.md"><code>tfgnn.validate_graph_tensor_for_readout()</code></a>;
preferably during dataset preprocessing, as it incurs a runtime cost for every
input.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`graph_spec`<a id="graph_spec"></a>
</td>
<td>
The graph tensor spec to check. Must be scalar, that is, have
shape [].
</td>
</tr><tr>
<td>
`required_keys`<a id="required_keys"></a>
</td>
<td>
Can be set to a list of readout keys that are required to be
provided by the spec.
</td>
</tr><tr>
<td>
`readout_node_set`<a id="readout_node_set"></a>
</td>
<td>
The name of the auxiliary node set for readout, which is
"_readout" by default. This name is also used as a prefix for the
auxiliary edge sets connected to it.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
if the auxiliary graph pieces for readout are malformed.
</td>
</tr><tr>
<td>
`KeyError`<a id="KeyError"></a>
</td>
<td>
if any of the `required_keys` is missing.
</td>
</tr>
</table>
