# runner.FitOrSkipPadding

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/utils/padding.py#L60-L91">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Calculates fit or skip `SizeConstraints` for `GraphTensor` padding.

Inherits From: [`GraphTensorPadding`](../runner/GraphTensorPadding.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>runner.FitOrSkipPadding(
    gtspec: tfgnn.GraphTensorSpec,
    dataset_provider: <a href="../runner/DatasetProvider.md"><code>runner.DatasetProvider</code></a>,
    min_nodes_per_component: Optional[Mapping[str, int]] = None,
    fit_or_skip_sample_sample_size: int = 10000,
    fit_or_skip_success_ratio: float = 0.99
)
</code></pre>

<!-- Placeholder for "Used in" -->

See: `tfgnn.learn_fit_or_skip_size_constraints.`

## Methods

<h3 id="get_filter_fn"><code>get_filter_fn</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/utils/padding.py#L78-L82">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_filter_fn(
    size_constraints: SizeConstraints
) -> Callable[..., bool]
</code></pre>

<h3 id="get_size_constraints"><code>get_size_constraints</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/utils/padding.py#L84-L91">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_size_constraints(
    target_batch_size: int
) -> SizeConstraints
</code></pre>
