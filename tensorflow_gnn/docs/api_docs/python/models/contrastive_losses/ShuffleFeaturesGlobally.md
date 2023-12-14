# contrastive_losses.ShuffleFeaturesGlobally

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/layers.py#L155-L165">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

A corruptor that shuffles features.

Inherits From: [`Corruptor`](../contrastive_losses/Corruptor.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>contrastive_losses.ShuffleFeaturesGlobally(
    *args, seed: Optional[float] = None, **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

NOTE: this function does not currently support TPUs. Consider using other
corruptor functions if executing on TPUs. See b/269249455 for reference.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>corruption_spec</code><a id="corruption_spec"></a>
</td>
<td>
A spec for corruption application.
</td>
</tr><tr>
<td>
<code>corruption_fn</code><a id="corruption_fn"></a>
</td>
<td>
Corruption function.
</td>
</tr><tr>
<td>
<code>default</code><a id="default"></a>
</td>
<td>
Global application default of the corruptor. This is only used
when <code>corruption_spec</code> is None.
</td>
</tr><tr>
<td>
<code>**kwargs</code><a id="**kwargs"></a>
</td>
<td>
Additional keyword arguments.
</td>
</tr>
</table>
