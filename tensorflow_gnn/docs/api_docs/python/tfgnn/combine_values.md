# tfgnn.combine_values

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_ops.py#L344-L378">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Combines a list of tensors into one (by concatenation or otherwise).

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.combine_values(
    inputs: List[Field], combine_type: str
) -> Field
</code></pre>

<!-- Placeholder for "Used in" -->

This is a convenience wrapper around standard TensorFlow operations, to provide
standard names for common types of combining.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr> <td> <code>inputs</code><a id="inputs"></a> </td> <td> a list of Tensors or
RaggedTensors, with shapes and types that are compatible for the selected
combine_type. </td> </tr><tr> <td>
<code>combine_type</code><a id="combine_type"></a> </td> <td> one of the
following string values, to select the method for combining the inputs:

*   "sum": The input tensors are added. Their dtypes and shapes must match.
*   "concat": The input tensors are concatenated along the last axis.
    Their dtypes and shapes must match, except for the number of elements
    along the last axis.
    </td>
    </tr>
    </table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tensor with the combined value of the inputs.
</td>
</tr>

</table>
