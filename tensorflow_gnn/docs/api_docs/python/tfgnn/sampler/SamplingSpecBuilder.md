<!-- lint-g3mark -->

# tfgnn.sampler.SamplingSpecBuilder

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/sampler/sampling_spec_builder.py#L197-L324">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Mimics builder pattern that eases creation of `tfgnn.SamplingSpec`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.sampler.SamplingSpecBuilder(
    graph_schema: <a href="../../tfgnn/GraphSchema.md"><code>tfgnn.GraphSchema</code></a>,
    default_strategy: SamplingStrategy = SamplingStrategy.TOP_K
)
</code></pre>

<!-- Placeholder for "Used in" -->

Example: Homogeneous Graphs.

If your graph is *homogeneous* and your node set is named "nodes" and edge set
is named "edges", then you can create the sampling spec proto as:

NOTE: This should come from the outside, e.g., `graph_tensor.schema`.

``` python
schema = schema_pb2.GraphSchema()
schema.edge_sets['edges'].source = s.edge_sets['edges'].target = 'nodes'

proto = (SamplingSpecBuilder(schema)
         .seed('nodes').sample('edges', 10).sample('edges', 5)
         .build())
```

The above spec is instructing to start at:

  - Nodes of type set name "nodes", then,
  - for each seed node, sample 10 of its neighbors (from edge set "edges").
  - for each of those neighbors, sample 5 neighbors (from same edge set).

Example: Heterogeneous Graphs.

E.g., if you consider citation datasets, you can make a SamplingSpec proto as:

``` python
proto = (SamplingSpecBuilder(schema)
         .seed('author').sample('writes', 10).sample('cited_by', 5)
         .build())
```

This samples, starting from author node, 10 papers written by author, and for
each paper, 10 papers citing it.

Example: DAG Sampling.

Finally, your sampling might consist of a DAG. For this, you need to cache some
returns of `.sample()` calls.

## Methods

<h3 id="build"><code>build</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/sampler/sampling_spec_builder.py#L285-L324">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build() -> <a href="../../tfgnn/sampler/SamplingSpec.md"><code>tfgnn.sampler.SamplingSpec</code></a>
</code></pre>

Creates new SamplingSpec that is built at this moment.

<h3 id="join"><code>join</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/sampler/sampling_spec_builder.py#L249-L251">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>join(
    steps
)
</code></pre>

<h3 id="seed"><code>seed</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/sampler/sampling_spec_builder.py#L253-L279">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>seed(
    node_set_name: Optional[str] = None
)
</code></pre>

Initializes sampling by seeding on node with `node_set_name`.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`node_set_name`
</td>
<td>
Becomes the `node_set_name` of built `spec.sampling_op`. If
not given, the graph schema must be homogeneous (with one `node_set`).
If given, it must correspond to some node set name in `graph_schema`
given to constructor.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Object which support builder pattern, upon which, you may repeatedly call
`.sample()`, per header comments.
</td>
</tr>

</table>

<h3 id="to_sampling_spec"><code>to_sampling_spec</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/sampler/sampling_spec_builder.py#L281-L283">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_sampling_spec() -> <a href="../../tfgnn/sampler/SamplingSpec.md"><code>tfgnn.sampler.SamplingSpec</code></a>
</code></pre>

DEPRECATED: use `build` instead.
