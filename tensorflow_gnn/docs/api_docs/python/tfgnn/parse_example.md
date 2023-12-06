# tfgnn.parse_example

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_io.py#L54-L100">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Parses a batch of serialized Example protos into a single `GraphTensor`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.parse_example(
    spec: <a href="../tfgnn/GraphTensorSpec.md"><code>tfgnn.GraphTensorSpec</code></a>,
    serialized: tf.Tensor,
    prefix: Optional[str] = None,
    validate: bool = True
) -> <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>
</code></pre>

<!-- Placeholder for "Used in" -->

We expect `serialized` to be a string tensor batched with `batch_size` many
entries of individual `tf.train.Example` protos. Each example contains
serialized graph tensors with the `spec` graph tensor type specification.

See `tf.io.parse_example()`. In contrast to the regular tensor parsing routine
which operates directly from `tf.io` feature configuration objects, this
function accepts a type spec for a graph tensor and implements an encoding for
all container tensors, including ragged tensors, from a batched sequence of
`tf.train.Example` protocol buffer messages.

Parsed feature values are converted to their dtype declared in the `spec` using
`tf.cast()`. Note that `tf.float64` features are converted from a `tf.float32`
representation in `tf.train.Example`.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>spec</code><a id="spec"></a>
</td>
<td>
A graph tensor type specification of a single serialized graph tensor
value.
</td>
</tr><tr>
<td>
<code>serialized</code><a id="serialized"></a>
</td>
<td>
A rank-1 dense tensor of strings with serialized Example protos,
where each example is a graph tensor object with type corresponding <code>spec</code>
type spec.
</td>
</tr><tr>
<td>
<code>prefix</code><a id="prefix"></a>
</td>
<td>
An optional prefix string over all the features. You may use
this if you are encoding other data in the same protocol buffer.
</td>
</tr><tr>
<td>
<code>validate</code><a id="validate"></a>
</td>
<td>
A boolean indicating whether or not to validate that the input
values form a valid GraphTensor. Defaults to <code>True</code>.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A graph tensor object with <code>spec.batch(serialized.shape[0])</code> type spec.
</td>
</tr>

</table>
