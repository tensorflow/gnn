description: A context update with input from node sets and/or edge sets.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfgnn.keras.layers.ContextUpdate" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# tfgnn.keras.layers.ContextUpdate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/graph_update.py#L431-L528">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A context update with input from node sets and/or edge sets.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.layers.ContextUpdate(
    node_set_inputs: Mapping[const.NodeSetName, NodesToContextPoolingLayer],
    next_state: next_state_lib.NextStateForContext,
    *,
    edge_set_inputs: Optional[Mapping[const.EdgeSetName, EdgesToContextPoolingLayer]] = None,
    context_input_feature: Optional[const.FieldNameOrNames] = const.HIDDEN_STATE,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Init args:


* <b>`node_set_inputs`</b>: A dict `{node_set_name: node_set_input, ...}` of Keras
  layers that return values shaped like context features with information
  aggregated from the given edge set. They are run on the input graph tensor
  as `node_set_input(graph, node_set_name=node_set_name)`.
* <b>`edge_set_inputs`</b>: A dict `{edge_set_name: edge_set_input, ...}` of Keras
  layers that return values shaped like context features with information
  aggregated from the given edge set. They are run on the input graph tensor
  as `edge_set_input(graph, edge_set_name=edge_set_name)`.
* <b>`next_state`</b>: A Keras layer to compute the new node state from a tuple of
  inputs that contains, in this order:

    - the `context_input_feature` (see there),
    - a dict `{node_set_name: input}` with the results of `node_set_inputs`,
      in which each result is a tensor or dict of tensors,
    - a dict `{edge_set_name: input}` with the results of `edge_set_inputs`,
      in which each result is a tensor or dict of tensors, if there are any.
* <b>`context_input_feature`</b>: The feature name(s) of inputs from the context to
  `next_state`, defaults to <a href="../../../tfgnn.md#HIDDEN_STATE"><code>tfgnn.HIDDEN_STATE</code></a>.
  If set to a single feature name, a single tensor is passed.
  If set to `None` or an empty sequence, an empty dict is passed.
  Otherwise, a dict of tensors keyed by feature names is passed.


#### Call result:

The tensor or dict of tensors with the new node state, as returned by
next_state.


