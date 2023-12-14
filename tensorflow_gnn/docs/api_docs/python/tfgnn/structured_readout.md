# tfgnn.structured_readout

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/readout.py#L141-L274">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Reads out a feature value from select nodes (or edges) in a graph.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.structured_readout(
    graph: <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>,
    key: str,
    *,
    feature_name: str,
    readout_node_set: const.NodeSetName = &#x27;_readout&#x27;,
    validate: bool = True
) -> <a href="../tfgnn/Field.md"><code>tfgnn.Field</code></a>
</code></pre>

<!-- Placeholder for "Used in" -->

This function implements "structured readout", that is, the readout of final
hidden states of a GNN computation from a distinguished subset of nodes (or
edges) in a <a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>
into a freestanding `tf.Tensor`, by moving them along one (or more) auxiliary
edge sets and collecting them in one auxiliary node set stored in the
<a href="../tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>.
Collectively, that auxiliary node set and the edge sets into it are called a
"readout structure". It is the responsibility of the graph data creation code to
create this structure.

A typical usage of structured readout looks as follows:

```python
graph = ...  # Run your GNN here.
seed_node_states = tfgnn.structured_readout(graph, "seed",
                                            feature_name=tfgnn.HIDDEN_STATE)
```

...where `"seed"` is a key defined by the readout structure. There can be
multiple of those. For example, a link prediction model could read out
`"source"` and `"target"` node states from the graph. It is on the dataset to
document which keys it provides.

Suppose all `"seed"` nodes come from node set `"users"`. Then this example
requires an auxiliary edge set called `"_readout/seed"` with source node set
`"users"` and target node set `"_readout"`, such that the target node indices
form a sorted(!) sequence `[0, 1, ..., n-1]` up to the size `n` of node set
`"_readout"`. The `seed_node_states` returned are the result of passing the
<a href="../tfgnn.md#HIDDEN_STATE"><code>tfgnn.HIDDEN_STATE</code></a> features
along the edges from `"users"` nodes to distinct `"_readout"` nodes. The number
of readout results per graph component is given by the sizes field of the
`"_readout"` node set, but can vary between graphs.

Advanced users may need to vary the source node set of readout. That is possible
by adding multiple auxiliary edge sets with names `"_readout/seed/" +
unique_suffix`. In each node set, the target node ids must be sorted, and
together they reach each `"_readout"` node exactly once.

Very advanced users may need to read out features from edge sets instead of node
sets. To read out from edge set `"links"`, create an auxiliary node set
`"_shadow/links"` with the same sizes field as the edge set but no features of
its own. When `"_shadow/links"` occurs as the source node set of an auxiliary
node set like `"_readout/seed"`, features are taken from edge set `"links"`
instead.

Note that this function returns a tensor shaped like a feature of the
`"_readout"` node set, not a modified GraphTensor. See
<a href="../tfgnn/structured_readout_into_feature.md"><code>tfgnn.structured_readout_into_feature()</code></a>
for a function that returns a GraphTensor with the readout result stored as a
feature on the `"_readout"` node set. To retrieve a feature `"ft"` that is
stored on the `"_readout"` node set, do not use either of these, but access it
as usual with `GraphTensor.node_sets["_readout"]["ft"]`.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>graph</code><a id="graph"></a>
</td>
<td>
A scalar GraphTensor with a readout structure composed of auxiliary
graph pieces as described above.
</td>
</tr><tr>
<td>
<code>key</code><a id="key"></a>
</td>
<td>
A string key to select between possibly multiple named readouts
(such as <code>"source"</code> and <code>"target"</code> for link prediction).
</td>
</tr><tr>
<td>
<code>feature_name</code><a id="feature_name"></a>
</td>
<td>
The name of a feature that is present on the node set(s)
(or edge set(s)) referenced by the auxiliary edge sets. The feature
must have shape <code>[num_items, *feature_dims]</code> with the same <code>feature_dims</code>
on all graph pieces, and the same dtype.
</td>
</tr><tr>
<td>
<code>readout_node_set</code><a id="readout_node_set"></a>
</td>
<td>
The name for the readout node set and the name prefix for
its edge sets. Permissible values are <code>"_readout"</code> (the default) and
<code>f"_readout:{tag}"</code> where <code>tag</code> matches <code>[a-zA-Z0-9_]+</code>.
Setting this to a different value allows to select between multiple
independent readout structures in the same graph.
</td>
</tr><tr>
<td>
<code>validate</code><a id="validate"></a>
</td>
<td>
Setting this to false disables the validity checks for the
auxiliary edge sets. This is stronlgy discouraged, unless great care is
taken to run <a href="../tfgnn/validate_graph_tensor_for_readout.md"><code>tfgnn.validate_graph_tensor_for_readout()</code></a> earlier on
structurally unchanged GraphTensors.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tensor of shape <code>[readout_size, *feature_dims]</code> with the read-out feature
values.
</td>
</tr>

</table>
