# tfgnn.learn_fit_or_skip_size_constraints

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/batching_utils.py#L257-L621">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Learns the optimal size constraints for the fixed size batching with retry.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.learn_fit_or_skip_size_constraints(
    dataset: tf.data.Dataset,
    batch_size: Union[int, Iterable[int]],
    *,
    min_nodes_per_component: Optional[Mapping[const.NodeSetName, int]] = None,
    success_ratio: Union[float, Iterable[float]] = 1.0,
    sample_size: int = 100000,
    num_thresholds: int = 1000
) -> Union[<a href="../tfgnn/SizeConstraints.md"><code>tfgnn.SizeConstraints</code></a>, List[Any]]
</code></pre>



<!-- Placeholder for "Used in" -->

The function estimates the smallest possible size constraints so that a random
sample of `batch_size` graph tensors meets those constraints with probability
no less than `success_ratio`. The success ratio is treated as a hard
constraint, up to sampling error. The constraints can be used for graph tensor
padding to the fully defined shapes required by XLA.

#### Example:



```python
# Learn size constraints for a given dataset of graph tensors and the target
# batch size(s). The constraints could be learned once and then reused.
constraints = tfgnn.learn_fit_or_skip_size_constraints(dataset, batch_size)

# Batch merge contained graphs into scalar graph tensors.
if training:
  # Randomize and repeat dataset for training. Note that the fit-or-skip
  # technique is only applicable for randomizer infinite datasets. It is
  # incorrect to apply it during models evaluation because some input
  # examples may be filtered out.
  dataset = dataset.shuffle(shuffle_size).repeat()
dataset = dataset.batch(batch_size)
dataset = dataset.map(lambda graph: graph.merge_batch_to_components())

if training:
  # Remove all batches that do not satisfy the learned constraints.
  dataset = dataset.filter(
      functools.partial(
          tfgnn.satisfies_size_constraints,
          total_sizes=constraints))

  # Pad graph to the learned size constraints.
  dataset = dataset.map(
      functools.partial(
          tfgnn.pad_to_total_sizes,
          size_constraints=constraints,
          validate=False))
```

The learned constraints are intend to be used only with randomized repeated
dataset. This dataset are first batched using `tf.data.Dataset.batch()`, the
batches that are too large to fit the learned constraints are filtered using
<a href="../tfgnn/satisfies_size_constraints.md"><code>tfgnn.satisfies_size_constraints()</code></a>
and then padded
<a href="../tfgnn/pad_to_total_sizes.md"><code>tfgnn.pad_to_total_sizes()</code></a>.

This approach, if applicable, is more efficient compared to padding to the
maximum possible sizes. It is also simpler and faster compared to the dynamic
batching, especially for the large batch sizes (>10).  To illustrate the main
point, consider graphs containing only 0 or 1 nodes. A random batch of 1000 of
those graphs could contain 1000 nodes in the worst case. If this maximum limit
is used to reseve space for random 1000 graphs, the space of 425 nodes is used
only in 1:1000_000 cases. It is >40% more efficient to reserve space only for
575 nodes and resample batches in the rare cases when they do not fit.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>dataset</code><a id="dataset"></a>
</td>
<td>
dataset of graph tensors that is intended to be batched.
</td>
</tr><tr>
<td>
<code>batch_size</code><a id="batch_size"></a>
</td>
<td>
the target batch size(s). Could be a single positive integer
value or any iterable. For the latter case the result is reported for each
requested value.
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
<code>success_ratio</code><a id="success_ratio"></a>
</td>
<td>
the target probability(s) that a random batch of graph tensor
satisfies the learned constraints. Could be a single float value between 0
and 1 or any iterable. For the latter case the result is reported for
each requested value. NOTE: setting success_ratio to 1 only guarantees
that all sampled graphs are satisfy the learned constraints. This does not
in general apply to an arbitrary sample. When <code>sample_size</code> tends to
infinity, the 1 ratio corresponds to the "almost surely satisfies" event.
</td>
</tr><tr>
<td>
<code>sample_size</code><a id="sample_size"></a>
</td>
<td>
the number of the first dataset examples to use for inference.
</td>
</tr><tr>
<td>
<code>num_thresholds</code><a id="num_thresholds"></a>
</td>
<td>
the number of quantiles to use to approximate probability
distributions.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Learned size constraints. If both <code>batch_size</code> and <code>success_ratio</code> are
iterables, the result is returned as a nested lists, were <code>result[b][r]</code>
is a size constraints for <code>batch_size[b]</code> and <code>success_ratio[r]</code>. If any of
<code>batch_size</code> or/and <code>success_ratio</code> are scalars the corresponding dimension
is squeezed in the output.
</td>
</tr>

</table>

