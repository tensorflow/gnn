# contrastive_losses.numerical_rank

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/metrics.py#L94-L124">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Numerical rank implementation.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@tf.function</code>
<code>contrastive_losses.numerical_rank(
    representations: tf.Tensor,
    *,
    sigma: Optional[tf.Tensor] = None,
    u: Optional[tf.Tensor] = None
) -> tf.Tensor
</code></pre>

<!-- Placeholder for "Used in" -->

Computes a metric that estimates the numerical column rank of a matrix. Rank is
estimated as a matrix trace divided by the largest eigenvalue. When our matrix
is a covariance matrix, we can compute both the trace and the largest eigenvalue
efficiently via SVD as the largest singular value squared.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>representations</code><a id="representations"></a>
</td>
<td>
Input representations. We expect rank 2 input.
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
