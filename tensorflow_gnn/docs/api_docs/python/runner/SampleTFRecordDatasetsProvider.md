<!-- lint-g3mark -->

# runner.SampleTFRecordDatasetsProvider

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/input/datasets.py#L449-L455">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Builds a sampling `tf.data.Dataset` from multiple filenames.

Inherits From:
[`SimpleSampleDatasetsProvider`](../runner/SimpleSampleDatasetsProvider.md),
[`DatasetProvider`](../runner/DatasetProvider.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>runner.SampleTFRecordDatasetsProvider(
    *args, **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

For complete explanations regarding sampling see `_process_sampled_dataset()`.

This `SimpleSampleDatasetsProvider` builds a `tf.data.Dataset` as follows:

  - The object is initialized with a list of filenames specified by
    `principle_filenames` and `extra_filenames` argument. For convenience, the
    corresponding file pattern `principal_file_pattern` and
    `extra_file_patterns` can be specified instead, which will be expanded to a
    sorted list.
  - The filenames are sharded between replicas according to the `InputContext`
    (order matters).
  - Filenames are shuffled per replica (if requested).
  - Examples from all file patterns are sampled according to `principal_weight`
    and `extra_weights.`
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
`principal_file_pattern`<a id="principal_file_pattern"></a>
</td>
<td>
A principal file pattern for sampling, to be
expanded by `tf.io.gfile.glob` and sorted into the list of
`principal_filenames`.
</td>
</tr><tr>
<td>
`extra_file_patterns`<a id="extra_file_patterns"></a>
</td>
<td>
File patterns, to be expanded by `tf.io.gfile.glob`
and sorted into the list of `extra_filenames`.
</td>
</tr><tr>
<td>
`principal_weight`<a id="principal_weight"></a>
</td>
<td>
An optional weight for the dataset corresponding to
`principal_file_pattern.` Required iff `extra_weights` are also
provided.
</td>
</tr><tr>
<td>
`extra_weights`<a id="extra_weights"></a>
</td>
<td>
Optional weights corresponding to `file_patterns` for
sampling. Required iff `principal_weight` is also provided.
</td>
</tr><tr>
<td>
`principal_filenames`<a id="principal_filenames"></a>
</td>
<td>
A list of principal filenames, specified explicitly.
This argument is mutually exclusive with `principal_file_pattern`.
</td>
</tr><tr>
<td>
`extra_filenames`<a id="extra_filenames"></a>
</td>
<td>
A list of extra filenames, specified explicitly.
This argument is mutually exclusive with `extra_file_patterns`.
</td>
</tr><tr>
<td>
`principal_cardinality`<a id="principal_cardinality"></a>
</td>
<td>
Iff `fixed_cardinality`=True, the size of the
returned dataset is computed as `principal_cardinality` /
`principal_weight` (with a default of uniform weights).
</td>
</tr><tr>
<td>
`fixed_cardinality`<a id="fixed_cardinality"></a>
</td>
<td>
Whether to take a fixed number of elements.
</td>
</tr><tr>
<td>
`shuffle_filenames`<a id="shuffle_filenames"></a>
</td>
<td>
If enabled, filenames will be shuffled after sharding
 between replicas, before any file reads. Through interleaving, some
files may be read in parallel: the details are auto-tuned for throughput.
</td>
</tr><tr>
<td>
`interleave_fn`<a id="interleave_fn"></a>
</td>
<td>
A fn applied with `tf.data.Dataset.interleave.`
</td>
</tr><tr>
<td>
`examples_shuffle_size`<a id="examples_shuffle_size"></a>
</td>
<td>
An optional buffer size for example shuffling. If
specified, the size is adjusted to `shuffle_size //
(len(file_patterns) + 1).`
</td>
</tr>
</table>

## Methods

<h3 id="get_dataset"><code>get_dataset</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/input/datasets.py#L360-L437">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_dataset(
    context: tf.distribute.InputContext
) -> tf.data.Dataset
</code></pre>

Creates a `tf.data.Dataset` by sampling.

The contents of the resulting `tf.data.Dataset` are sampled from several
sources, each stored as a sharded dataset:

  - one principal input, whose size determines the size of the resulting
    `tf.data.Dataset`;
  - zero or more side inputs, which are repeated if necessary to preserve the
    requested samping weights.

Each input dataset is shared before interleaving. The result of interleaving is
only shuffled if a `examples_shuffle_size` is provided.

Datasets are sampled from with `tf.data.Dataset.sample_from_datasets.` For
sampling details, please refer to the TensorFlow documentation at:
<https://www.tensorflow.org/api_docs/python/tf/data/Dataset#sample_from_datasets>.

Two methods are supported to determine the end of the resulting
`tf.data.Dataset`:

fixed_cardinality=True) Returns a dataset with a fixed cardinality, set at
`principal_cardinality` // `principal_weight.` `principal_dataset` and
`principal_cardinality` are required for this method. `principal_weight` is
required iff `extra_weights` are also provided.

fixed_cardinality=False) Returns a dataset that ends after the principal input
has been exhausted, subject to the random selection of samples.
`principal_dataset` is required for this method. `principal_weight` is required
iff `extra_weights` are also provided.

The choice of `principal_dataset` is important and should, in most cases, be
chosen as the largest underlying dataset as compared to `extra_datasets.`
`positives` and `negatives` where `len(negatives)` \>\> `len(positives)` and
with `positives` corresponding to `principal_dataset,` the desired behavior of
epochs determined by the exhaustion of `positives` and the continued mixing of
unique elements from `negatives` may not occur: On sampled dataset reiteration
`positives` will again be exhausted but elements from `negatives` may be those
same seen in the previous epoch (as they occur at the beginning of the same,
reiterated underlying `negatives` dataset). In this case, the recommendations
are to:

1)  Reformulate the sampling in terms of the larger dataset (`negatives`),
    where, with `fixed_cardinality=False`, if the exhaustion of `negatives` is
    desired, or, with `fixed_cardinality=True`, when `principal_cardinality` can
    be used to specify the desired number of elements from `negatives.`
2)  Ensure that the underlying `principal_dataset` of `negatives` are
    well-sharded. In this way, the nondeterminism of interleaving will randomly
    access elements of `negatives` on reiteration.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`context`
</td>
<td>
An `tf.distribute.InputContext` for sharding.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `tf.data.Dataset.`
</td>
</tr>

</table>
