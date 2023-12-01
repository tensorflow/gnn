<!-- lint-g3mark -->

# contrastive_losses.coherence

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/metrics.py#L184-L213">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Coherence metric implementation.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@tf.function</code>
<code>contrastive_losses.coherence(
    representations: tf.Tensor,
    *,
    sigma: Optional[tf.Tensor] = None,
    u: Optional[tf.Tensor] = None
) -> tf.Tensor
</code></pre>

<!-- Placeholder for "Used in" -->

Coherence measures how easy it is to construct a linear classifier on top of
data without knowing downstream labels. Refer to
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
Input representations, a rank-2 tensor.
</td>
</tr><tr>
<td>
`sigma`<a id="sigma"></a>
</td>
<td>
Unused.
</td>
</tr><tr>
<td>
`u`<a id="u"></a>
</td>
<td>
An optional tensor with left singular vectors of representations. If not
present, computes a SVD of representations.
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
