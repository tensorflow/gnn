# mt_albis.graph_update_from_config_dict

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/mt_albis/config_dict.py#L50-L77">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Constructs a MtAlbisGraphUpdate from a ConfigDict.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mt_albis.graph_update_from_config_dict(
    cfg: config_dict.ConfigDict,
    node_set_names: Optional[Collection[tfgnn.NodeSetName]] = None
) -> tf.keras.layers.Layer
</code></pre>

<!-- Placeholder for "Used in" -->
<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>cfg</code><a id="cfg"></a>
</td>
<td>
A <code>ConfigDict</code> with the fields defined by
<code>graph_update_get_config_dict()</code>. All fields with non-<code>None</code> values are
used as keyword arguments for initializing and returning a
<code>MtAlbisGraphUpdate</code> object. For the required arguments of
<code>MtAlbisGraphUpdate.__init__</code>, users must set a value in <code>cfg</code> before
passing it here.
</td>
</tr><tr>
<td>
<code>node_set_names</code><a id="node_set_names"></a>
</td>
<td>
Optionally, the names of NodeSets to update; forwarded to
<code>MtAlbisGraphUpdate.__init__</code>.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A new Layer object as returned by <code>MtAlbisGraphUpdate()</code>.
</td>
</tr>

</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
<code>TypeError</code><a id="TypeError"></a>
</td>
<td>
if <code>cfg</code> fails to supply a required argument for
<code>MtAlbisGraphUpdate()</code>.
</td>
</tr>
</table>
