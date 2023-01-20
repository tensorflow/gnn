# tfgnn.keras.layers.EdgeSetUpdate

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/graph_update.py#L261-L349">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Computes the new state of an EdgeSet from select input features.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.EdgeSetUpdate(
    next_state: next_state_lib.NextStateForEdgeSet,
    *,
    edge_input_feature: Optional[const.FieldNameOrNames] = const.HIDDEN_STATE,
    node_input_tags: Sequence[const.IncidentNodeTag] = (const.SOURCE, const.TARGET),
    node_input_feature: Optional[const.FieldName] = const.HIDDEN_STATE,
    context_input_feature: Optional[const.FieldNameOrNames] = None,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Init args</h2></th></tr>

<tr>
<td>
`next_state`<a id="next_state"></a>
</td>
<td>
The NextState layer to apply.
</td>
</tr><tr>
<td>
`edge_input_feature`<a id="edge_input_feature"></a>
</td>
<td>
The feature name(s) of inputs from the edge set to
`next_state`, defaults to <a href="../../../tfgnn.md#HIDDEN_STATE"><code>tfgnn.HIDDEN_STATE</code></a>.
If set to a single feature name, a single tensor is passed.
If set to `None` or an empty sequence, an empty dict is passed.
Otherwise, a dict of tensors keyed by feature names is passed.
</td>
</tr><tr>
<td>
`node_input_tags`<a id="node_input_tags"></a>
</td>
<td>
The incident nodes of each edge whose states are used
as an input, specified as IncidentNodeTags (tfgnn.SOURCE and tfgnn.TARGET
by default).
</td>
</tr><tr>
<td>
`node_input_feature`<a id="node_input_feature"></a>
</td>
<td>
The feature name of the input from node sets to
`next_state`, defaults to <a href="../../../tfgnn.md#HIDDEN_STATE"><code>tfgnn.HIDDEN_STATE</code></a>.
Setting this to `None` passes an empty dict of node inputs.
This class supports only a single input feature from nodes. For more
complex settings, you need to write your own, or start a design discussion
about a node_input_map from tags to the respective features for each.
</td>
</tr><tr>
<td>
`context_input_feature`<a id="context_input_feature"></a>
</td>
<td>
The feature name(s) of inputs from the context to
`next_state`. Defaults to `None`, which passes an empty dict.
If set to a single feature name, a single tensor is passed.
Otherwise, a dict of tensors keyed by feature names is passed.
To pass the default state tensor of the context, set this to
<a href="../../../tfgnn.md#HIDDEN_STATE"><code>tfgnn.HIDDEN_STATE</code></a>.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Call returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The result of next_state called on the configured inputs.
</td>
</tr>

</table>
