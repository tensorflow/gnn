# contrastive_losses.Corruptor

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/layers.py#L84-L142">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Base class for graph corruptor.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>contrastive_losses.Corruptor(
    corruption_spec: Optional[CorruptionSpec[T]] = None,
    *,
    corruption_fn: Callable[[tfgnn.Field, T], tfgnn.Field],
    default: Optional[T] = None,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->
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
