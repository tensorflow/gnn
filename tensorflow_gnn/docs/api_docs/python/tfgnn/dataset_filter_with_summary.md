# tfgnn.dataset_filter_with_summary

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/preprocessing_common.py#L103-L188">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Dataset filter with a summary for the fraction of dataset elements removed.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.dataset_filter_with_summary(
    dataset: tf.data.Dataset,
    predicate: Callable[[Any], tf.Tensor],
    *,
    summary_name: str = &#x27;dataset_removed_fraction&#x27;,
    summary_steps: int = 1000,
    summary_decay: Optional[float] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The fraction of removed elements is computed using exponential moving average.
See https://en.wikipedia.org/wiki/Moving_average.

The summary is reported each `summary_steps` elements in the input dataset
before filtering. Statistics are reported using `tf.summary.scalar()` with
`step` set to the element index in the result (filtered) dataset, see
https://tensorflow.org/tensorboard/scalars_and_keras#logging_custom_scalars for
how to write and retrieve them.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>dataset</code><a id="dataset"></a>
</td>
<td>
An input dataset.
</td>
</tr><tr>
<td>
<code>predicate</code><a id="predicate"></a>
</td>
<td>
A function mapping a dataset element to a boolean.
</td>
</tr><tr>
<td>
<code>summary_name</code><a id="summary_name"></a>
</td>
<td>
A name for this summary.
</td>
</tr><tr>
<td>
<code>summary_steps</code><a id="summary_steps"></a>
</td>
<td>
Report summary for this number of elements in the input
dataset before filtering.
</td>
</tr><tr>
<td>
<code>summary_decay</code><a id="summary_decay"></a>
</td>
<td>
An exponential moving average decay factor. If not set,
defaults to the <code>exp(- 1 / summary_steps)</code>.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Thed dataset containing the elements of this dataset for which predicate is
<code>True</code>.
</td>
</tr>

</table>

