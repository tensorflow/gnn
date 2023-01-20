# gat_v2.GATv2Conv

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/gat_v2/layers.py#L22-L323">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

The multi-head attention from Graph Attention Networks v2 (GATv2).

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gat_v2.GATv2Conv(
    *,
    num_heads: int,
    per_head_channels: int,
    receiver_tag: Optional[tfgnn.IncidentNodeOrContextTag] = None,
    receiver_feature: tfgnn.FieldName = tfgnn.HIDDEN_STATE,
    sender_node_feature: Optional[tfgnn.FieldName] = tfgnn.HIDDEN_STATE,
    sender_edge_feature: Optional[tfgnn.FieldName] = None,
    use_bias: bool = True,
    edge_dropout: float = 0.0,
    attention_activation: Union[str, Callable[..., Any]] = &#x27;leaky_relu&#x27;,
    heads_merge_type: str = &#x27;concat&#x27;,
    activation: Union[str, Callable[..., Any]] = &#x27;relu&#x27;,
    kernel_initializer: Union[None, str, tf.keras.initializers.Initializer] = None,
    kernel_regularizer: Union[None, str, tf.keras.regularizers.Regularizer] = None,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

GATv2 (https://arxiv.org/abs/2105.14491) improves upon the popular GAT
architecture (https://arxiv.org/abs/1710.10903) by allowing the network to
compute a more expressive "dynamic" instead of just "static" attention, each of
whose heads is described by Equations (7), (3) and (4) in
https://arxiv.org/abs/2105.14491.

Example: GATv2-style attention on incoming edges whose result is concatenated
with the old node state and passed through a Dense layer to compute the new node
state.

```python
dense = tf.keras.layers.Dense
graph = tfgnn.keras.layers.GraphUpdate(
    node_sets={"paper": tfgnn.keras.layers.NodeSetUpdate(
        {"cites": tfgnn.keras.layers.GATv2Conv(
             message_dim, receiver_tag=tfgnn.TARGET)},
        tfgnn.keras.layers.NextStateFromConcat(dense(node_state_dim)))}
)(graph)
```

This layer implements the multi-head attention of GATv2 with the following
generalizations:

*   This implementation of GATv2 attends only to edges that are explicitly
    stored in the input GraphTensor. Attention of a node to itself is enabled or
    disabled by storing or not storing an explicit loop in the edge set. The
    example above uses a separate layer to combine the old node state with the
    attention result to form the new node state.
*   Attention values can be computed from a sender node state that gets
    broadcast onto the edge (see arg `sender_node_feature`), from an edge
    feature (see arg `sender_edge_feature`), or from their concatenation (by
    setting both arguments). This choice is used in place of the sender node
    state $h_j$ in the defining equations cited above.
*   This layer can be used with `receiver_tag=tfgnn.CONTEXT` to perform a
    convolution to the context, with graph components as receivers and the
    containment in graph components used in lieu of edges.
*   An `edge_dropout` option is provided.

This layer can also be configured to do attention pooling from edges to context
or to receiver nodes (without regard for source nodes) by setting
`sender_node_feature=None` and setting `sender_edge_feature=...` to the
applicable edge feature name (e.g., `tfgnn.HIDDEN_STATE`).

Like the Keras Dense layer, if the input features have rank greater than 2, this
layer computes a point-wise attention along the last axis of the inputs. For
example, if the input features have shape `[num_nodes, 2, 4, 1]`, then it will
perform an identical computation on each of the `num_nodes * 2 * 4` input
values.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Init args</h2></th></tr>

<tr>
<td>
`num_heads`<a id="num_heads"></a>
</td>
<td>
The number of attention heads.
</td>
</tr><tr>
<td>
`per_head_channels`<a id="per_head_channels"></a>
</td>
<td>
The number of channels for each attention head. This
means:
  if `heads_merge_type == "concat"`, then final output size will be:
    `per_head_channels * num_heads`.
  if `heads_merge_type == "mean"`, then final output size will be:
    `per_head_channels`.
</td>
</tr><tr>
<td>
`receiver_tag`<a id="receiver_tag"></a>
</td>
<td>
one of `tfgnn.SOURCE`, `tfgnn.TARGET` or `tfgnn.CONTEXT`.
The results of attention are aggregated for this graph piece.
If set to `tfgnn.SOURCE` or `tfgnn.TARGET`, the layer can be called for
an edge set and will aggregate results at the specified endpoint of the
edges.
If set to `tfgnn.CONTEXT`, the layer can be called for an edge set or
node set.
If left unset for init, the tag must be passed at call time.
</td>
</tr><tr>
<td>
`receiver_feature`<a id="receiver_feature"></a>
</td>
<td>
Can be set to override `tfgnn.HIDDEN_STATE` for use as
the receiver's input feature to attention. (The attention key is derived
from this input.)
</td>
</tr><tr>
<td>
`sender_node_feature`<a id="sender_node_feature"></a>
</td>
<td>
Can be set to override `tfgnn.HIDDEN_STATE` for use as
the input feature from sender nodes to attention.
IMPORANT: Must be set to `None` for use with `receiver_tag=tfgnn.CONTEXT`
on an edge set, or for pooling from edges without sender node states.
</td>
</tr><tr>
<td>
`sender_edge_feature`<a id="sender_edge_feature"></a>
</td>
<td>
Can be set to a feature name of the edge set to select
it as an input feature. By default, this set to `None`, which disables
this input.
IMPORTANT: Must be set for use with `receiver_tag=tfgnn.CONTEXT`
on an edge set.
</td>
</tr><tr>
<td>
`use_bias`<a id="use_bias"></a>
</td>
<td>
If true, a bias term is added to the transformations of query and
value inputs.
</td>
</tr><tr>
<td>
`edge_dropout`<a id="edge_dropout"></a>
</td>
<td>
Can be set to a dropout rate for edge dropout. (When pooling
nodes to context, it's the node's membership in a graph component that
is dropped out.)
</td>
</tr><tr>
<td>
`attention_activation`<a id="attention_activation"></a>
</td>
<td>
The nonlinearity used on the transformed inputs
before multiplying with the trained weights of the attention layer.
This can be specified as a Keras layer, a tf.keras.activations.*
function, or a string understood by tf.keras.layers.Activation().
Defaults to "leaky_relu", which in turn defaults to a negative slope
of `alpha=0.2`.
</td>
</tr><tr>
<td>
`heads_merge_type`<a id="heads_merge_type"></a>
</td>
<td>
The merge operation for combining output from
all `num_heads` attention heads. By default, output of heads will be
concatenated. However, GAT paper (Velickovic et al, Eq 6) recommends *only for output layer* to do mean across attention heads, which is acheivable
by setting to `"mean"`.
</td>
</tr><tr>
<td>
`activation`<a id="activation"></a>
</td>
<td>
The nonlinearity applied to the final result of attention,
specified in the same ways as attention_activation.
</td>
</tr><tr>
<td>
`kernel_initializer`<a id="kernel_initializer"></a>
</td>
<td>
Can be set to a `kerner_initializer` as understood
by `tf.keras.layers.Dense` etc.
</td>
</tr><tr>
<td>
`kernel_regularizer`<a id="kernel_regularizer"></a>
</td>
<td>
If given, will be used to regularize all layer kernels.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr> <td> `receiver_tag`<a id="receiver_tag"></a> </td> <td> one of
`tfgnn.SOURCE`, `tfgnn.TARGET` or `tfgnn.CONTEXT`. The results are aggregated
for this graph piece. If set to `tfgnn.SOURCE` or `tfgnn.TARGET`, the layer can
be called for an edge set and will aggregate results at the specified endpoint
of the edges. If set to `tfgnn.CONTEXT`, the layer can be called for an edge set
or a node set and will aggregate results for context (per graph component). If
left unset for init, the tag must be passed at call time. </td> </tr><tr> <td>
`receiver_feature`<a id="receiver_feature"></a> </td> <td> The name of the
feature that is read from the receiver graph piece and passed as
convolve(receiver_input=...). </td> </tr><tr> <td>
`sender_node_feature`<a id="sender_node_feature"></a> </td> <td> The name of the
feature that is read from the sender nodes, if any, and passed as
convolve(sender_node_input=...). NOTICE this must be `None` for use with
`receiver_tag=tfgnn.CONTEXT` on an edge set, or for pooling from edges without
sender node states. </td> </tr><tr> <td>
`sender_edge_feature`<a id="sender_edge_feature"></a> </td> <td> The name of the
feature that is read from the sender edges, if any, and passed as
convolve(sender_edge_input=...). NOTICE this must not be `None` for use with
`receiver_tag=tfgnn.CONTEXT` on an edge set. </td> </tr><tr> <td>
`extra_receiver_ops`<a id="extra_receiver_ops"></a> </td> <td> A str-keyed
dictionary of Python callables that are wrapped to bind some arguments and then
passed on to `convolve()`. Sample usage: `extra_receiver_ops={"softmax":
tfgnn.softmax}`. The values passed in this dict must be callable as follows,
with two positional arguments:

```python
f(graph, receiver_tag, node_set_name=..., feature_value=..., ...)
f(graph, receiver_tag, edge_set_name=..., feature_value=..., ...)
```

The wrapped callables seen by `convolve()` can be called like

```python
wrapped_f(feature_value, ...)
```

The first three arguments of `f` are set to the input GraphTensor of
the layer and the tag/name pair required by `tfgnn.broadcast()` and
`tfgnn.pool()` to move values between the receiver and the messages that
are computed inside the convolution. The sole positional argument of
`wrapped_f()` is passed to `f()`  as `feature_value=`, and any keyword
arguments are forwarded.
</td>
</tr><tr>
<td>
`**kwargs`<a id="**kwargs"></a>
</td>
<td>
Forwarded to the base class tf.keras.layers.Layer.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`takes_receiver_input`<a id="takes_receiver_input"></a>
</td>
<td>
If `False`, all calls to convolve() will get `receiver_input=None`.
</td>
</tr><tr>
<td>
`takes_sender_edge_input`<a id="takes_sender_edge_input"></a>
</td>
<td>
If `False`, all calls to convolve() will get `sender_edge_input=None`.
</td>
</tr><tr>
<td>
`takes_sender_node_input`<a id="takes_sender_node_input"></a>
</td>
<td>
If `False`, all calls to convolve() will get `sender_node_input=None`.
</td>
</tr>
</table>

## Methods

<h3 id="convolve"><code>convolve</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/gat_v2/layers.py#L242-L301">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>convolve(
    *,
    sender_node_input: Optional[tf.Tensor],
    sender_edge_input: Optional[tf.Tensor],
    receiver_input: Optional[tf.Tensor],
    broadcast_from_sender_node: Callable[[tf.Tensor], tf.Tensor],
    broadcast_from_receiver: Callable[[tf.Tensor], tf.Tensor],
    pool_to_receiver: Callable[..., tf.Tensor],
    extra_receiver_ops: Optional[Mapping[str, Callable[..., Any]]] = None,
    **kwargs
) -> tf.Tensor
</code></pre>

Overridden internal method of the base class.
