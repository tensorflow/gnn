# tfgnn.keras.layers.NodeSetUpdate

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/graph_update.py#L353-L443">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

A node state update with input from convolutions or other edge set inputs.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.NodeSetUpdate(
    edge_set_inputs: Mapping[const.EdgeSetName, EdgesToNodePoolingLayer],
    next_state: next_state_lib.NextStateForNodeSet,
    *,
    node_input_feature: Optional[const.FieldNameOrNames] = const.HIDDEN_STATE,
    context_input_feature: Optional[const.FieldNameOrNames] = None,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Init args</h2></th></tr>

<tr> <td> `edge_set_inputs`<a id="edge_set_inputs"></a> </td> <td> A dict
`{edge_set_name: edge_set_input, ...}` of Keras layers (such as convolutions)
that return values shaped like node features with information aggregated from
the given edge set. They are run in parallel on the input graph tensor as
`edge_set_input(graph, edge_set_name=edge_set_name)`. </td> </tr><tr> <td>
`next_state`<a id="next_state"></a> </td> <td> A Keras layer to compute the new
node state from a tuple of inputs that contains, in this order:

-   the `node_input_feature` (see there),
-   a dict `{edge_set_name: input}` with the results of `edge_set_inputs`, in
    which each result is a tensor or dict of tensors,
-   if context_input_feature is not `None`, those feature(s).
    </td>
    </tr><tr>
    <td>
    `node_input_feature`<a id="node_input_feature"></a>
    </td>
    <td>
    The feature name(s) of inputs from the node set to
    `next_state`, defaults to <a href="../../../tfgnn.md#HIDDEN_STATE"><code>tfgnn.HIDDEN_STATE</code></a>.
    If set to a single feature name, a single tensor is passed.
    If set to `None` or an empty sequence, an empty dict is passed.
    Otherwise, a dict of tensors keyed by feature names is passed.
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
<tr><th colspan="2"><h2 class="add-link">Call result</h2></th></tr>
<tr class="alt">
<td colspan="2">
The tensor or dict of tensors with the new node state, as returned by
next_state.
</td>
</tr>

</table>
