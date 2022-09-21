# tfgnn.write_example

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_encode.py#L32-L59">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Encode an eager `GraphTensor` to a tf.train.Example proto.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.write_example(
    graph: <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>,
    prefix: Optional[str] = None
) -> tf.train.Example
</code></pre>



<!-- Placeholder for "Used in" -->

This routine can be used to create a stream of training data for GNNs from a
Python job. Create instances of scalar `GraphTensor` with a single graph
component and call this to write them out. It is recommended to always accompany
serialized graph tensor tensorflow examples by their graph schema file (see
`tfgnn.create_schema_pb_from_graph_spec()`). TF-GNN library provides
`tfgnn.check_compatible_with_schema_pb()` to check that graph tensor instances
(or their specs) are compatible with the graph schema. The graph tensors
materialized in this way will be parseable by
<a href="../tfgnn/parse_example.md"><code>tfgnn.parse_example()</code></a>
(using the spec deserialized from the schema) and have the same contents (up to
the choice of indices_dtype).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`graph`<a id="graph"></a>
</td>
<td>
An eager instance of `GraphTensor` to write out.
</td>
</tr><tr>
<td>
`prefix`<a id="prefix"></a>
</td>
<td>
An optional prefix string over all the features. You may use
this if you are encoding other data in the same protocol buffer.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A reference to `result`, if provided, or a to a freshly created instance
of `tf.train.Example`.
</td>
</tr>

</table>

