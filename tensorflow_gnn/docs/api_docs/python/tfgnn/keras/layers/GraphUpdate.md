# tfgnn.keras.layers.GraphUpdate

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/graph_update.py#L125-L258">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Applies one round of updates to EdgeSets, NodeSets and Context.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.GraphUpdate(
    *,
    edge_sets: Optional[Mapping[const.EdgeSetName, EdgeSetUpdateLayer]] = None,
    node_sets: Optional[Mapping[const.NodeSetName, NodeSetUpdateLayer]] = None,
    context: Optional[ContextUpdateLayer] = None,
    deferred_init_callback: Optional[Callable[[gt.GraphTensorSpec], Mapping[str, Any]]] = None,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

The updates of EdgeSets, NodeSets and Context can either be passed as
init arguments, or constructed later by passing a deferred_init_callback,
which allows advanced users to adjust the updates to the GraphTensorSpec
of the input (which EdgeSets and NodeSets even exist).

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Init args</h2></th></tr>

<tr>
<td>
`edge_sets`<a id="edge_sets"></a>
</td>
<td>
A dict `{edge_set_name: edge_set_update, ...}` of EdgeSetUpdate
layers (or custom reimplementations). They are run on the input graph
tensor as `edge_set_update(graph, edge_set_name=edge_set_name)`.
Their results are merged into the feature map of the respective edge set.
This argument can be omitted, which is common in models with node set
updates that use convolutions (i.e., read from adjacent nodes without
computing explicit edge states).
</td>
</tr><tr>
<td>
`node_sets`<a id="node_sets"></a>
</td>
<td>
A dict `{node_set_name: node_set_update, ...}` of NodeSetUpdate
layers (or custom reimplementations). They are run on the graph tensor
with edge set updates (if any) as
`node_set_update(graph, node_set_name=node_set_name)`,
Their results are merged into the feature map of the respective node set.
This argument can be omitted (but that is uncommon).
</td>
</tr><tr>
<td>
`context`<a id="context"></a>
</td>
<td>
A ContextUpdate that is run on the graph tensor with edge set and
node set updates (if any). Its results are merged back into the context
feature map. This argument can be omitted, which is common in models
without a context state.
</td>
</tr><tr>
<td>
`deferred_init_callback`<a id="deferred_init_callback"></a>
</td>
<td>
Can be set to a function that accepts a
GraphTensorSpec and returns a dictionary with the kwargs
edge_sets=..., node_sets=... and context=... that would otherwise be
passed directly at initialization time. If this argument is set,
edge_sets, node_sets and context must all be unset.
The object is initialized upon its first call from the results of
the callback on the spec of the input. Before that, the object cannot
be saved.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Call result</h2></th></tr>
<tr class="alt">
<td colspan="2">
A graph tensor with feature maps that have all configured updates merged in:
If an update returns a str-keyed dict, it gets merged into respective
feature map with the given names. If an update returns a single tensor,
the name tfgnn.HIDDEN_STATE is used.
</td>
</tr>

</table>
