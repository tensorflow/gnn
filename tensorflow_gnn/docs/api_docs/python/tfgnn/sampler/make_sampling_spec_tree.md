<!-- lint-g3mark -->

# tfgnn.sampler.make_sampling_spec_tree

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/sampler/sampling_spec_builder.py#L143-L194">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Automatically creates `SamplingSpec` by starting from seed node set.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.sampler.make_sampling_spec_tree(
    graph_schema: <a href="../../tfgnn/GraphSchema.md"><code>tfgnn.GraphSchema</code></a>,
    seed_nodeset: NodeSetName,
    *,
    sample_sizes: List[int],
    sampling_strategy=SamplingStrategy.RANDOM_UNIFORM
) -> <a href="../../tfgnn/sampler/SamplingSpec.md"><code>tfgnn.sampler.SamplingSpec</code></a>
</code></pre>

<!-- Placeholder for "Used in" -->

From seed node set, `sample_sizes[0]` are sampled from *every* edge set `E` that
originates from seed node set. Subsequently, from sampled edge `e` in `E` the
created `SamplingSpec` instructs sampling up to `sample_sizes[1]` edges for
`e`'s target node, and so on, until depth of `len(sample_sizes)`.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`graph_schema`<a id="graph_schema"></a>
</td>
<td>
contains node-sets & edge-sets.
</td>
</tr><tr>
<td>
`seed_nodeset`<a id="seed_nodeset"></a>
</td>
<td>
name of node-set that the sampler will be instructed to use as
seed nodes.
</td>
</tr><tr>
<td>
`sample_sizes`<a id="sample_sizes"></a>
</td>
<td>
list of number of nodes to sample. E.g. if `sample_sizes` are
`[5, 2, 2]`, then for every sampled node, up-to `5` of its neighbors will
be sampled, and for each, up to `2` of its neighbors will be sampled, etc,
totalling sampled nodes up to `5 * 2 * 2 = 20` for each seed node.
</td>
</tr><tr>
<td>
`sampling_strategy`<a id="sampling_strategy"></a>
</td>
<td>
one of the supported sampling strategies, the same for
each depth.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
`SamplingSpec` that instructs the sampler to sample according to the
`sampling_strategy` and `sample_sizes`.
</td>
</tr>

</table>
