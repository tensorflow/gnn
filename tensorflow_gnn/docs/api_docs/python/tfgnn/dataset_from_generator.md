# tfgnn.dataset_from_generator

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/batching_utils.py#L786-L879">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Creates dataset from generator of any nest of scalar graph pieces.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.dataset_from_generator(
    generator
) -> tf.data.Dataset
</code></pre>

<!-- Placeholder for "Used in" -->

Similar to `tf.data.Dataset.from_generator()`, but requires the generator to
yield at least one element and sets the result's `.element_spec` from it. In
subsequent elements, graph pieces must have the same features (incl. their
shapes and dtypes), and graphs must have the same edge sets and node sets, but
the numbers of nodes and edges may vary between elements.

NOTE: Compared to `tf.data.from_generator()` the generator is first called
during the dataset construction. If generator is shared between two datasets
this could lead to some obscure behaviour, like:

```
my_generator = [pieceA, pieceB, pieceC, pieceD]
dataset1 = tfgnn.dataset_from_generator(my_generator).take(2)
dataset2 = tfgnn.dataset_from_generator(my_generator).take(2)
print([dataset2])  # prints: pieceB, pieceC, while expected pieceA, pieceB.
print([dataset1])  # prints: pieceA, pieceD.
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>generator</code><a id="generator"></a>
</td>
<td>
a callable object that returns an object that supports the iter()
protocol. Could consist of any nest of tensors and scalar graph pieces
(e.g. <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>, <a href="../tfgnn/Context.md"><code>tfgnn.Context</code></a>, <a href="../tfgnn/NodeSet.md"><code>tfgnn.NodeSet</code></a>,
<a href="../tfgnn/EdgeSet.md"><code>tfgnn.EdgeSet</code></a>, <a href="../tfgnn/Adjacency.md"><code>tfgnn.Adjacency</code></a>, etc.)
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A <code>tf.data.Dataset</code>.
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
if any contained graph piece is not scalar or has not compatible
number of graph components.
</td>
</tr>
</table>
