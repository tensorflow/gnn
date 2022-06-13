description: A node state update with input from convolutions or other edge set inputs.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfgnn.keras.layers.NodeSetUpdate" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# tfgnn.keras.layers.NodeSetUpdate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/graph_update.py#L338-L428">
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


#### Init args:


* <b>`edge_set_inputs`</b>: A dict `{edge_set_name: edge_set_input, ...}` of Keras
  layers (such as convolutions) that return values shaped like node features
  with information aggregated from the given edge set.
  They are run in parallel on the input graph tensor as
  `edge_set_input(graph, edge_set_name=edge_set_name)`.
* <b>`next_state`</b>: A Keras layer to compute the new node state from a tuple of
  inputs that contains, in this order:

    - the `node_input_feature` (see there),
    - a dict `{edge_set_name: input}` with the results of `edge_set_inputs`,
      in which each result is a tensor or dict of tensors,
    - if context_input_feature is not `None`, those feature(s).
* <b>`node_input_feature`</b>: The feature name(s) of inputs from the node set to
  `next_state`, defaults to <a href="../../../tfgnn.md#HIDDEN_STATE"><code>tfgnn.HIDDEN_STATE</code></a>.
  If set to a single feature name, a single tensor is passed.
  If set to `None` or an empty sequence, an empty dict is passed.
  Otherwise, a dict of tensors keyed by feature names is passed.
* <b>`context_input_feature`</b>: The feature name(s) of inputs from the context to
  `next_state`. Defaults to `None`, which passes an empty dict.
  If set to a single feature name, a single tensor is passed.
  Otherwise, a dict of tensors keyed by feature names is passed.
  To pass the default state tensor of the context, set this to
  <a href="../../../tfgnn.md#HIDDEN_STATE"><code>tfgnn.HIDDEN_STATE</code></a>.


#### Call result:

The tensor or dict of tensors with the new node state, as returned by
next_state.


