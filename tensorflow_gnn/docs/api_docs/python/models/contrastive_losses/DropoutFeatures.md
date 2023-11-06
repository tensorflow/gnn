<!-- lint-g3mark -->

# contrastive_losses.DropoutFeatures

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/layers.py#L185-L194">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Base class for graph corruptor.

Inherits From: [`Corruptor`](../contrastive_losses/Corruptor.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>contrastive_losses.DropoutFeatures(
    *args, seed: Optional[float] = None, **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`corruption_spec`<a id="corruption_spec"></a>
</td>
<td>
A spec for corruption application.
</td>
</tr><tr>
<td>
`corruption_fn`<a id="corruption_fn"></a>
</td>
<td>
Corruption function.
</td>
</tr><tr>
<td>
`default`<a id="default"></a>
</td>
<td>
Global application default of the corruptor. This is only used
when `corruption_spec` is None.
</td>
</tr><tr>
<td>
`**kwargs`<a id="**kwargs"></a>
</td>
<td>
Additional keyword arguments.
</td>
</tr>
</table>
