<!-- lint-g3mark -->

# contrastive_losses.self_clustering

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/metrics.py#L24-L61">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Self-clustering metric implementation.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@tf.function</code>
<code>contrastive_losses.self_clustering(
    representations: tf.Tensor, *, subtract_mean: bool = False, **_
) -> tf.Tensor
</code></pre>

<!-- Placeholder for "Used in" -->

Computes a metric that measures how well distributed representations are, if
projected on the unit sphere. If `subtract_mean` is True, we additionally remove
the mean from representations. The metric has a range of (-0.5, 1\]. It achieves
its maximum of 1 if representations collapse to a single point, and it is
approximately 0 if representations are distributed randomly on the sphere. In
theory, it can achieve negative values if the points are maximally equiangular,
although this is very rare in practice. Refer to
<https://arxiv.org/abs/2305.16562> for more details.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`representations`<a id="representations"></a>
</td>
<td>
Input representations.
</td>
</tr><tr>
<td>
`subtract_mean`<a id="subtract_mean"></a>
</td>
<td>
Whether to subtract the mean from representations.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Metric value as scalar `tf.Tensor`.
</td>
</tr>

</table>
