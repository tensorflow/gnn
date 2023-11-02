<!-- lint-g3mark -->

# runner.PassthruDatasetProvider

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/input/datasets.py#L59-L83">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Builds a `tf.data.Dataset` from a pass thru dataset.

Inherits From: [`DatasetProvider`](../runner/DatasetProvider.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>runner.PassthruDatasetProvider(
    dataset: tf.data.Dataset,
    *,
    shuffle_datasets: bool = False,
    examples_shuffle_size: Optional[int] = None
)
</code></pre>

<!-- Placeholder for "Used in" -->

Passes any `dataset` thru: omitting any sharding. For detailed documentation,
see the filename dataset provider complement: `SimpleDatasetsProvider.`

## Methods

<h3 id="get_dataset"><code>get_dataset</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/input/datasets.py#L75-L83">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_dataset(
    _: tf.distribute.InputContext
) -> tf.data.Dataset
</code></pre>

Gets a `tf.data.Dataset` omitting any input context.
