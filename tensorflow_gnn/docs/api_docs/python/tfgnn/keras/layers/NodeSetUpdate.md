# tfgnn.keras.layers.NodeSetUpdate

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/graph_update.py#L359-L452">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

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

This layer can be restored from config by `tf.keras.models.load_model()` when
saved as part of a Keras model using `save_format="tf"`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Init args</h2></th></tr>

<tr> <td> <code>edge_set_inputs</code><a id="edge_set_inputs"></a> </td> <td> A
dict <code>{edge_set_name: edge_set_input, ...}</code> of Keras layers (such as
convolutions) that return values shaped like node features with information
aggregated from the given edge set. They are run in parallel on the input graph
tensor as <code>edge_set_input(graph, edge_set_name=edge_set_name)</code>. </td>
</tr><tr> <td> <code>next_state</code><a id="next_state"></a> </td> <td> A Keras
layer to compute the new node state from a tuple of inputs that contains, in
this order:

-   the <code>node_input_feature</code> (see there),
-   a dict <code>{edge_set_name: input}</code> with the results of
    <code>edge_set_inputs</code>, in which each result is a tensor or dict of
    tensors,
-   if context_input_feature is not <code>None</code>, those feature(s).
    </td>
    </tr><tr>
    <td>
    <code>node_input_feature</code><a id="node_input_feature"></a>
    </td>
    <td>
    The feature name(s) of inputs from the node set to
    <code>next_state</code>, defaults to <a href="../../../tfgnn.md#HIDDEN_STATE"><code>tfgnn.HIDDEN_STATE</code></a>.
    If set to a single feature name, a single tensor is passed.
    If set to <code>None</code> or an empty sequence, an empty dict is passed.
    Otherwise, a dict of tensors keyed by feature names is passed.
    </td>
    </tr><tr>
    <td>
    <code>context_input_feature</code><a id="context_input_feature"></a>
    </td>
    <td>
    The feature name(s) of inputs from the context to
    <code>next_state</code>. Defaults to <code>None</code>, which passes an empty dict.
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
