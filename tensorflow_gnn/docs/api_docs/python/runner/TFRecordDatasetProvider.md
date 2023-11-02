<!-- lint-g3mark -->

# runner.TFRecordDatasetProvider

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/input/datasets.py#L440-L446">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Builds a `tf.data.Dataset` from a list of files.

Inherits From: [`SimpleDatasetProvider`](../runner/SimpleDatasetProvider.md),
[`DatasetProvider`](../runner/DatasetProvider.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>runner.TFRecordDatasetProvider(
    *args, **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

This `SimpleDatasetProvider` builds a `tf.data.Dataset` as follows:

  - The object is initialized with a list of filenames. For convenience, a file
    pattern can be specified instead, which will be expanded to a sorted list.
  - The filenames are sharded between replicas according to the `InputContext`
    (order matters).
  - Filenames are shuffled per replica (if requested).
  - The files in each shard are interleaved after being read by the
    `interleave_fn`.
  - Examples are shuffled (if requested), auto-prefetched, and returned for use
    in one replica of the trainer.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`file_pattern`<a id="file_pattern"></a>
</td>
<td>
A file pattern, to be expanded by `tf.io.gfile.glob`
and sorted into the list of all `filenames`.
</td>
</tr><tr>
<td>
`filenames`<a id="filenames"></a>
</td>
<td>
A list of all filenames, specified explicitly.
This argument is mutually exclusive with `file_pattern`.
</td>
</tr><tr>
<td>
`shuffle_filenames`<a id="shuffle_filenames"></a>
</td>
<td>
If enabled, filenames will be shuffled after sharding
between replicas, before any file reads. Through interleaving, some
files may be read in parallel: the details are auto-tuned for
throughput.
</td>
</tr><tr>
<td>
`interleave_fn`<a id="interleave_fn"></a>
</td>
<td>
A callback that receives a single filename and returns
a `tf.data.Dataset` with the `tf.Example` values from that file.
</td>
</tr><tr>
<td>
`examples_shuffle_size`<a id="examples_shuffle_size"></a>
</td>
<td>
An optional buffer size for example shuffling.
</td>
</tr>
</table>

## Methods

<h3 id="get_dataset"><code>get_dataset</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/input/datasets.py#L134-L146">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_dataset(
    context: tf.distribute.InputContext
) -> tf.data.Dataset
</code></pre>

Gets a `tf.data.Dataset` by `context` per replica.
