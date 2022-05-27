description: Register a new reduction operation for pooling.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gnn.register_reduce_operation" />
<meta itemprop="path" content="Stable" />
</div>

# gnn.register_reduce_operation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_ops.py#L604-L631">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Register a new reduction operation for pooling.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gnn.register_reduce_operation(
    reduce_type: str,
    *,
    unsorted_reduce_op: UnsortedReduceOp,
    allow_override: bool = False
) -> None
</code></pre>

<!-- Placeholder for "Used in" -->

This function can be used to insert a reduction operation in the supported
list of `reduce_type` aggregations for all the pooling operations defined in
this module.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`reduce_type`
</td>
<td>
A pooling operation name. This name must not conflict with the
existing pooling operations in the registry, except if `allow_override` is
set. For the full list of supported values use
`get_registered_reduce_operation_names()`.
</td>
</tr><tr>
<td>
`unsorted_reduce_op`
</td>
<td>
The TensorFlow op for reduction. This op does not rely
on sorting over edges.
</td>
</tr><tr>
<td>
`allow_override`
</td>
<td>
A boolean flag to allow overwriting the existing registry of
operations. Use this with care.
</td>
</tr>
</table>

