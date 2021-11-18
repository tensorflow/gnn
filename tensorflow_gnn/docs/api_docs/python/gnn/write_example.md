description: Encode an eager GraphTensor to a tf.train.Example proto.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gnn.write_example" />
<meta itemprop="path" content="Stable" />
</div>

# gnn.write_example

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_encode.py#L18-L38">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Encode an eager `GraphTensor` to a tf.train.Example proto.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gnn.write_example(
    graph: <a href="../gnn/GraphTensor.md"><code>gnn.GraphTensor</code></a>,
    prefix: Optional[str] = None
) -> tf.train.Example
</code></pre>



<!-- Placeholder for "Used in" -->

This routine can be used to create a stream of training data for GNNs from a
Python job. Create instances of `GraphTensor` and call this to write them
out in a format that will be parseable by `tensorflow_gnn.parse_example()`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`graph`
</td>
<td>
An eager instance of `GraphTensor` to write out.
</td>
</tr><tr>
<td>
`prefix`
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

