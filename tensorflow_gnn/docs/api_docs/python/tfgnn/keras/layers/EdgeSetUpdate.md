description: Computes the new state of an EdgeSet from select input features.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfgnn.keras.layers.EdgeSetUpdate" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# tfgnn.keras.layers.EdgeSetUpdate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/layers/graph_update.py#L247-L335">
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


#### Init args:


* <b>`next_state`</b>: The NextState layer to apply.
* <b>`edge_input_feature`</b>: The feature name(s) of inputs from the edge set to
  `next_state`, defaults to <a href="../../../tfgnn.md#HIDDEN_STATE"><code>tfgnn.HIDDEN_STATE</code></a>.
  If set to a single feature name, a single tensor is passed.
  If set to `None` or an empty sequence, an empty dict is passed.
  Otherwise, a dict of tensors keyed by feature names is passed.
* <b>`node_input_tags`</b>: The incident nodes of each edge whose states are used
  as an input, specified as IncidentNodeTags (tfgnn.SOURCE and tfgnn.TARGET
  by default).
* <b>`node_input_feature`</b>: The feature name of the input from node sets to
  `next_state`, defaults to <a href="../../../tfgnn.md#HIDDEN_STATE"><code>tfgnn.HIDDEN_STATE</code></a>.
  Setting this to `None` passes an empty dict of node inputs.
  This class supports only a single input feature from nodes. For more
  complex settings, you need to write your own, or start a design discussion
  about a node_input_map from tags to the respective features for each.
* <b>`context_input_feature`</b>: The feature name(s) of inputs from the context to
  `next_state`. Defaults to `None`, which passes an empty dict.
  If set to a single feature name, a single tensor is passed.
  Otherwise, a dict of tensors keyed by feature names is passed.
  To pass the default state tensor of the context, set this to
  <a href="../../../tfgnn.md#HIDDEN_STATE"><code>tfgnn.HIDDEN_STATE</code></a>.


#### Call returns:

The result of next_state called on the configured inputs.


