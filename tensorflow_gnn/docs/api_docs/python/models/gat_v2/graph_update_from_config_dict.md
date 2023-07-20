# gat_v2.graph_update_from_config_dict

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/gat_v2/config_dict.py#L42-L62">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Returns a GATv2MPNNGraphUpdate initialized from `cfg`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gat_v2.graph_update_from_config_dict(
    cfg: config_dict.ConfigDict
) -> tf.keras.layers.Layer
</code></pre>

<!-- Placeholder for "Used in" -->
<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`cfg`<a id="cfg"></a>
</td>
<td>
A `ConfigDict` with the fields defined by
`graph_update_get_config_dict()`. All fields with non-`None` values are
used as keyword arguments for initializing and returning a
`GATv2MPNNGraphUpdate` object. For the required arguments of
`GATv2MPNNGraphUpdate.__init__`, users must set a value in
`cfg` before passing it here.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A new `GATv2MPNNGraphUpdate` object.
</td>
</tr>

</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`<a id="TypeError"></a>
</td>
<td>
if `cfg` fails to supply a required argument for
`GATv2MPNNGraphUpdate.__init__`.
</td>
</tr>
</table>
