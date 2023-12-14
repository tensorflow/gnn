# tfgnn.iter_features

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/schema_utils.py#L330-L359">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Utility function to iterate over the features of a graph schema.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.iter_features(
    schema: Union[<a href="../tfgnn/proto/GraphSchema.md"><code>tfgnn.proto.GraphSchema</code></a>, <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>]
) -> Iterator[Tuple[Text, Text, Text, Union[schema_pb2.Feature, gt.Field]]]
</code></pre>

<!-- Placeholder for "Used in" -->

This function iterates over all the feature values of each of the context set,
each of the node sets, and each of the edge sets.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>schema</code><a id="schema"></a>
</td>
<td>
An instance of a <code>GraphSchema</code> proto message.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Yields</h2></th></tr>
<tr class="alt">
<td colspan="2">
Triplets of (set-type, set-name, feature-name, feature-value) where

*   set-type: A type of set, which is either of "context", "nodes" or "edges".
*   set-name: A string, the name of the set.
*   feature-name: A string, the name of the feature in the set.
*   feature-value: A potentially ragged tensor (either a <code>tf.Tensor</code>
    or a <code>tf.RaggedTensor</code>). </td> </tr>

</table>
