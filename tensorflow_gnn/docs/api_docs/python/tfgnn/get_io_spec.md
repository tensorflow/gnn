# tfgnn.get_io_spec

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_io.py#L118-L213">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns tf.io parsing features for `GraphTensorSpec` type spec.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.get_io_spec(
    spec: <a href="../tfgnn/GraphTensorSpec.md"><code>tfgnn.GraphTensorSpec</code></a>,
    prefix: Optional[str] = None,
    validate: bool = False
) -> Dict[str, IOFeature]
</code></pre>



<!-- Placeholder for "Used in" -->

This function returns a mapping of `tf.train.Feature` names to configuration
objects that can be used to parse instances of `tf.train.Example` (see
https://www.tensorflow.org/api_docs/python/tf/io). The resulting mapping can
be used with `tf.io.parse_example()` for reading the individual fields of a
`GraphTensor` instance. This essentially forms our encoding of a `GraphTensor`
to a `tf.train.Example` proto.

(This is an internal function. You are not likely to be using this if you're
decoding graph tensors. Instead, you should use the <a href="../tfgnn/parse_example.md"><code>tfgnn.parse_example()</code></a>
routine directly, which handles this process for you.)

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`spec`
</td>
<td>
A graph tensor type specification.
</td>
</tr><tr>
<td>
`prefix`
</td>
<td>
An optional prefix string over all the features. You may use
this if you are encoding other data in the same protocol buffer.
</td>
</tr><tr>
<td>
`validate`
</td>
<td>
A boolean indicating whether or not to validate that the input
fields form a valid `GraphTensor`. Defaults to `True`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A dict of `tf.train.Feature` name to feature configuration object, to be
used in `tf.io.parse_example()`.
</td>
</tr>

</table>

