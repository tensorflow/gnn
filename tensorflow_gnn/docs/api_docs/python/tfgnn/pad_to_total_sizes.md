# tfgnn.pad_to_total_sizes

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/padding_ops.py#L31-L185">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Pads graph tensor to the total sizes by inserting fake graph components.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.pad_to_total_sizes(
    graph_tensor: <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>,
    size_constraints: <a href="../tfgnn/SizeConstraints.md"><code>tfgnn.SizeConstraints</code></a>,
    *,
    padding_values: Optional[<a href="../tfgnn/FeatureDefaultValues.md"><code>tfgnn.FeatureDefaultValues</code></a>] = None,
    validate: bool = True
) -> Tuple[<a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>, tf.Tensor]
</code></pre>



<!-- Placeholder for "Used in" -->

Padding is done by inserting "fake" graph components at the end of the input
graph tensor until target total sizes are exactly matched. If that is not
possible (e.g. input already has more nodes than allowed by the constraints)
function raises `tf.errors.InvalidArgumentError`.

If size_constraints.min_nodes_per_component is specified for a node set,
the inserted graph components satisfy that constraint (e.g., such that there
is a node for tf.gather_first_node()). Components in the input graph tensor
must satisfy that constraint already, or tf.errors.InvalidArgumentError will
be raised. (This function cannot add padding within existing components.)

Context, node or edge features of the appended fake components are filled with
user-provided scalar values or with zeros if the latter are not specified.
Fake edges are created such that each fake node has an approximately uniform
number of incident edges (this behavior is not guaranteed and may change in
the future).

NOTE(b/275338236): This operation is not available in TFLite (last checked for
TF 2.12).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>graph_tensor</code><a id="graph_tensor"></a>
</td>
<td>
scalar graph tensor (rank=0) to pad.
</td>
</tr><tr>
<td>
<code>size_constraints</code><a id="size_constraints"></a>
</td>
<td>
target total sizes for each graph piece. Must define the
target number of graph components (<code>.total_num_components</code>), target total
number of items for each node set (<code>.total_num_nodes[node_set_name]</code>) and
likewise for each edge set (<code>.total_num_edges[edge_set_name]</code>).
If <code>min_nodes_per_component</code> is set, the inserted graph components satisfy
that constraint and graph components of the input graph tensor are checked
against this constraint.
</td>
</tr><tr>
<td>
<code>padding_values</code><a id="padding_values"></a>
</td>
<td>
optional mapping from a context, node set or edge set
feature name to a scalar tensor to use for padding. If no value is
specified for some feature, its type 'zero' is used (as in <code>tf.zeros()</code>).
</td>
</tr><tr>
<td>
<code>validate</code><a id="validate"></a>
</td>
<td>
If true, then use assertions to check that the input graph tensor
could be padded. NOTE: while these assertions provide more readable error
messages, they incur a runtime cost, since assertions must be checked for
each input value.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Tuple of padded graph tensor and padding mask. The mask is a rank-1 dense
boolean tensor wth size equal to the number of graph compoents is the result
containing <code>True</code> for real graph components and <code>False</code> - for fake one used
for padding.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
<code>ValueError</code><a id="ValueError"></a>
</td>
<td>
if input parameters are invalid.
</td>
</tr><tr>
<td>
<code>tf.errors.InvalidArgumentError</code><a id="tf.errors.InvalidArgumentError"></a>
</td>
<td>
if input graph tensor could not be padded to
the <code>size_constraints</code> or has less nodes in a component than allowed by
the <code>min_nodes_per_component</code>
</td>
</tr>
</table>
