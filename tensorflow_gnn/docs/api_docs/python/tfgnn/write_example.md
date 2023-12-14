# tfgnn.write_example

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_encode.py#L34-L72">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

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
<a href="../tfgnn/create_schema_pb_from_graph_spec.md"><code>tfgnn.create_schema_pb_from_graph_spec()</code></a>).
TF-GNN library provides
<a href="../tfgnn/check_compatible_with_schema_pb.md"><code>tfgnn.check_compatible_with_schema_pb()</code></a>
to check that graph tensor instances (or their specs) are compatible with the
graph schema. The graph tensors materialized in this way will be parseable by
<a href="../tfgnn/parse_example.md"><code>tfgnn.parse_example()</code></a>
(using the spec deserialized from the schema) and have the same contents (up to
the choice of indices_dtype).

All features stored on the `graph` must have dtypes that are supported by the
graph schema. For the following dtypes, special caveats apply to their
representation in `tf.train.Example`:

*   `tf.float64` features are serialized as `tf.float32`, which truncates them
    (perhaps fatally to 0 or +/- inf, if exceeding the exponent range).
*   `tf.uint64` features are serialized as `tf.int64` values with the same bit
    pattern. Deserializing from `tf.train.Example` recovers the original value.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>graph</code><a id="graph"></a>
</td>
<td>
An eager instance of <code>GraphTensor</code> to write out.
</td>
</tr><tr>
<td>
<code>prefix</code><a id="prefix"></a>
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
A <code>tf.train.Example</code> with the serialized <code>graph</code>.
</td>
</tr>

</table>

