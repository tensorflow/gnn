# tfgnn.find_tight_size_constraints

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/batching_utils.py#L190-L254">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Returns smallest possible size constraints that allow dataset padding.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.find_tight_size_constraints(
    dataset: tf.data.Dataset,
    *,
    min_nodes_per_component: Optional[Mapping[const.NodeSetName, int]] = None,
    target_batch_size: Optional[Union[int, tf.Tensor]] = None
) -> <a href="../tfgnn/SizeConstraints.md"><code>tfgnn.SizeConstraints</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

Evaluated constraints are intended to be used when it is required that all
elements of the `dataset` can be padded, e.g., when evaluating models.

Typically, this function is used on a dataset of individual examples (that is,
not batched), and the `target_batch_size` is passed as an argument. The
returned constraints will work for all possible batches up to that size drawn
from the dataset.

Alternatively, this function can be used on a dataset that is already batched,
passing `target_batch_size=None`. The returned constraints will work for the
batches exactly as seen in the dataset. However, note that many performance-
optimized ways of building a Dataset (like parallel .map() and .interleave()
calls before .batch()) introduce nondeterminism and may not deliver the exact
same batches again.

Note that this function iterates over all elements of the input dataset, so
its execution time is proportional to the dataset's cardinality.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>dataset</code><a id="dataset"></a>
</td>
<td>
finite dataset of graph tensors of any rank.
</td>
</tr><tr>
<td>
<code>min_nodes_per_component</code><a id="min_nodes_per_component"></a>
</td>
<td>
mapping from a node set name to a minimum number of
nodes in each graph component. Defaults to 0.
</td>
</tr><tr>
<td>
<code>target_batch_size</code><a id="target_batch_size"></a>
</td>
<td>
if not <code>None</code>, an integer for multiplying the sizes
measured from dataset before making room for padding.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Smalles possible size constraints that allows padding of all graph tensors
in the input dataset.
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
if dataset elements are not GraphTensors or its cardinality
is <code>tf.data.INFINITE_CARDINALITY</code>.
</td>
</tr>
</table>
