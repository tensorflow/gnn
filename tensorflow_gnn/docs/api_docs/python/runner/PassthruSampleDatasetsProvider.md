# runner.PassthruSampleDatasetsProvider

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/input/datasets.py#L227-L271">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Builds a sampled `tf.data.Dataset` from multiple pass thru datasets.

Inherits From: [`DatasetProvider`](../runner/DatasetProvider.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>runner.PassthruSampleDatasetsProvider(
    principal_dataset: tf.data.Dataset,
    extra_datasets: Sequence[tf.data.Dataset],
    principal_weight: Optional[float] = None,
    extra_weights: Optional[Sequence[float]] = None,
    *,
    principal_cardinality: Optional[int] = None,
    fixed_cardinality: bool = False,
    shuffle_dataset: bool = False,
    examples_shuffle_size: Optional[int] = None
)
</code></pre>

<!-- Placeholder for "Used in" -->

Passes any `principal_dataset` and `extra_datasets` thru: omitting any sharding.
For detailed documentation, see the filename dataset provider complement:
`SimpleSampleDatasetsProvider.`

## Methods

<h3 id="get_dataset"><code>get_dataset</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/input/datasets.py#L258-L271">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_dataset(
    _: tf.distribute.InputContext
) -> tf.data.Dataset
</code></pre>

Gets a sampled `tf.data.Dataset` omitting any input context.
