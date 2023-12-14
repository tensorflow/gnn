# contrastive_losses.rankme

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/metrics.py#L127-L156">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

RankMe metric implementation.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@tf.function</code>
<code>contrastive_losses.rankme(
    representations: tf.Tensor,
    *,
    sigma: Optional[tf.Tensor] = None,
    u: Optional[tf.Tensor] = None,
    epsilon: float = 1e-12,
    **_
) -> tf.Tensor
</code></pre>

<!-- Placeholder for "Used in" -->

Computes a metric that measures the decay rate of the singular values. For the
paper, see https://arxiv.org/abs/2210.02885.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>representations</code><a id="representations"></a>
</td>
<td>
Input representations as rank-2 tensor.
</td>
</tr><tr>
<td>
<code>sigma</code><a id="sigma"></a>
</td>
<td>
An optional tensor with singular values of representations. If not
present, computes SVD (singular values only) of representations.
</td>
</tr><tr>
<td>
<code>u</code><a id="u"></a>
</td>
<td>
Unused.
</td>
</tr><tr>
<td>
<code>epsilon</code><a id="epsilon"></a>
</td>
<td>
Epsilon for numerican stability.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Metric value as scalar <code>tf.Tensor</code>.
</td>
</tr>

</table>
