# runner.TightPadding

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/utils/padding.py#L94-L109">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Calculates tight `SizeConstraints` for `GraphTensor` padding.

Inherits From: [`GraphTensorPadding`](../runner/GraphTensorPadding.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>runner.TightPadding(
    gtspec: tfgnn.GraphTensorSpec,
    dataset_provider: <a href="../runner/DatasetProvider.md"><code>runner.DatasetProvider</code></a>,
    min_nodes_per_component: Optional[Mapping[str, int]] = None
)
</code></pre>

<!-- Placeholder for "Used in" -->

See: `tfgnn.find_tight_size_constraints.`

## Methods

<h3 id="get_filter_fn"><code>get_filter_fn</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/utils/padding.py#L100-L102">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_filter_fn(
    size_constraints: SizeConstraints
) -> Callable[..., bool]
</code></pre>

<h3 id="get_size_constraints"><code>get_size_constraints</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/utils/padding.py#L104-L109">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_size_constraints(
    target_batch_size: int
) -> SizeConstraints
</code></pre>
