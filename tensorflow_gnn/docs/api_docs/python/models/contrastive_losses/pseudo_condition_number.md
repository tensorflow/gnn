<!-- lint-g3mark -->

# contrastive_losses.pseudo_condition_number

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/metrics.py#L64-L91">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Pseudo-condition number metric implementation.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@tf.function</code>
<code>contrastive_losses.pseudo_condition_number(
    representations: tf.Tensor,
    *,
    sigma: Optional[tf.Tensor] = None,
    u: Optional[tf.Tensor] = None
) -> tf.Tensor
</code></pre>

<!-- Placeholder for "Used in" -->

Computes a metric that measures the decay rate of the singular values. NOTE: Can
be unstable in practice, when using small batch sizes, leading to numerical
instabilities.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`representations`<a id="representations"></a>
</td>
<td>
Input representations. We expect rank 2 input.
</td>
</tr><tr>
<td>
`sigma`<a id="sigma"></a>
</td>
<td>
An optional tensor with singular values of representations. If not
present, computes SVD (singular values only) of representations.
</td>
</tr><tr>
<td>
`u`<a id="u"></a>
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
Metric value as scalar `tf.Tensor`.
</td>
</tr>

</table>
