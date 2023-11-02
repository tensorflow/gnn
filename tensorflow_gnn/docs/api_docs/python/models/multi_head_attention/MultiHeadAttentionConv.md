<!-- lint-g3mark -->

# multi_head_attention.MultiHeadAttentionConv

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/multi_head_attention/layers.py#L24-L563">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Transformer-style (dot-product) multi-head attention on GNNs.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>multi_head_attention.MultiHeadAttentionConv(
    *,
    num_heads: int,
    per_head_channels: int,
    receiver_tag: Optional[tfgnn.IncidentNodeOrContextTag] = None,
    receiver_feature: tfgnn.FieldName = tfgnn.HIDDEN_STATE,
    sender_node_feature: Optional[tfgnn.FieldName] = tfgnn.HIDDEN_STATE,
    sender_edge_feature: Optional[tfgnn.FieldName] = None,
    use_bias: bool = True,
    edge_dropout: float = 0.0,
    inputs_dropout: float = 0.0,
    attention_activation: Optional[Union[str, Callable[..., Any]]] = None,
    activation: Union[str, Callable[..., Any]] = &#x27;relu&#x27;,
    kernel_initializer: Any = None,
    kernel_regularizer: Any = None,
    transform_keys: bool = True,
    score_scaling: Literal['none', 'rsqrt_dim', 'trainable_elup1'] = &#x27;rsqrt_dim&#x27;,
    transform_values_after_pooling: bool = False,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

The [Graph Transformer](https://arxiv.org/abs/2012.09699) introduces
[transformer\-style multi-head attention](https://arxiv.org/abs/1706.03762) to
GNN. This class describes a layer of computing such multi-head attention and
produces concatenated multi-head outputs (without positional encoding, clamping
in Softmax, linear transformation for multi-head outputs, the feed-forward
network, the residual connections and normalization layers). Please see
tensorflow_gnn/models/multi_head_attention/README.md for more details. For
the regular sequential transformer attention, please see
`tf.keras.layers.MultiHeadAttention` instead.

This attention is formuated differently depending on the presence of edge
features:

1.  When edge features are NOT considered, this layer is exactly the same as
    Graph Transformer attention, where the receiver node feature is seen as
    'query' and the sender node feature is 'key' and 'value':

    $$Q_v = h_v, K_u = V_u = h_u, \text{where} \enspace u \in N(v)$$

2.  When edge features are considered, this layer still uses the receiver node
    feature as 'query', but uses the concatenation of the sender node feature
    and edge feature as 'key' and 'value':

    $$Q_v = h_v, K_u = V_u = \[h_u||e_{uv}\], \\text{where} \\enspace u
    \\in N(v)$$

Then, similar to what is done in "Attention is all you need" and what is
described in Equations (4) and (5) of "Graph Transformer", the attention output
$O^k_v$ from head $k$ for receiver node $v$ is computed as

$$O^k_v = \sum_{u \in N(v)} \alpha^k_{uv} V_u W_V^k$$

with attention weights

$$(\\alpha^k_{uv} \\mid u \\in N(v)) = Softmax((Q_v W_Q^k)(K_u W_K^k)^T
\\mid u \\in N(v)) / \\sqrt{d}$$

where the softmax is taken over all neighbors $u$ along edges $(u,v)$ into $v$
and $d$ is the dimension of keys and queries as projected by $W_K$ and $W_Q$.
The final output for node $v$ is the concatenation over all heads, that is

$$O_v = ||_k O^k_v$$.

Note that in the context of graph, only nodes with edges connected are attended
to each other, which means we do NOT compute $N^2$ pairs of scores as the
original Transformer-style Attention.

Users are able to remove the scaling of attention scores
(`score_scaling="none"`) or add an activation on the transformed query
(controlled by `attention_activation`). However, we recommend to remove the
scaling when using an `attention_activation` since activating both of them may
lead to degraded accuracy. One can also customize the transformation kernels
with different initializers, regularizers as well as the use of bias terms,
using the other arguments.

Example: Transformer-style attention on neighbors along incoming edges whose
result is concatenated with the old node state and passed through a Dense layer
to compute the new node state.

    dense = tf.keras.layers.Dense
    graph = tfgnn.keras.layers.GraphUpdate(
        node_sets={"paper": tfgnn.keras.layers.NodeSetUpdate(
            {"cites": tfgnn.keras.layers.MultiHeadAttentionConv(
                 message_dim, receiver_tag=tfgnn.TARGET)},
            tfgnn.keras.layers.NextStateFromConcat(dense(node_state_dim)))}
    )(graph)

For now, there is a variant that modifies the inputs transformation part and
could potentially be beneficial:

    1. (transform_keys is False) Instead of projecting both queries and
      keys when computing attention weights, we only project the queries
      because the two linear projections can be collapsed to a single
      projection:

        $$ (Q_v W_Q^k)(K_u W_K^k)^T
          = Q_v (W_Q^k {W_K^k}^T) K_u^T
          = Q_v W_{QK}^k K_u^T $$

      where $d$ is the key width. (Following "Attention is all you need",
      this scaling is meant to achieve unit variance of the results, assuming
      that $Q_v W_{QK}^k$ has unit variance due to the initialization of
      $Q_v W_{QK}^k$.)

      NOTE: The single projection matrix behaves differently in
      gradient-descent training than the product of two matrices.

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
means that the final output size will be per_head_channels * num_heads.
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
Can be set to override `tfgnn.HIDDEN_STATE`
for use as the receiver's input feature to attention. (The attention key
is derived from this input.)
</td>
</tr><tr>
<td>
`sender_node_feature`<a id="sender_node_feature"></a>
</td>
<td>
Can be set to override `tfgnn.HIDDEN_STATE`
for use as the input feature from sender nodes to attention.
IMPORTANT: Must be set to `None` for use with `receiver_tag=tfgnn.CONTEXT`
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
If true, bias terms are added to the transformations of query,
key and value inputs.
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
`inputs_dropout`<a id="inputs_dropout"></a>
</td>
<td>
Dropout rate for random dropout on the inputs to this
convolution layer, i.e. the receiver, sender node, and sender edge inputs.
</td>
</tr><tr>
<td>
`attention_activation`<a id="attention_activation"></a>
</td>
<td>
The nonlinearity used on the transformed inputs
(query, and keys if `transform_keys` is `True`) before computing the
attention scores. This can be specified as a Keras layer, a
tf.keras.activations.* function, or a string understood by
`tf.keras.layers.Activation`. Defaults to None.
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
Can be set to a `kernel_initializer` as understood
by `tf.keras.layers.Dense` etc.
An `Initializer` object gets cloned before use to ensure a fresh seed,
if not set explicitly. For more, see `tfgnn.keras.clone_initializer()`.
</td>
</tr><tr>
<td>
`kernel_regularizer`<a id="kernel_regularizer"></a>
</td>
<td>
Can be set to a `kernel_regularized` as understood
by `tf.keras.layers.Dense` etc.
</td>
</tr><tr>
<td>
`transform_keys`<a id="transform_keys"></a>
</td>
<td>
If true, transform both queries and keys inputs. Otherwise,
only queries are transformed since the two transformations on queries and
keys are equivalent to one. (The presence of transformations on values is
independent of this arg.)
</td>
</tr><tr>
<td>
`score_scaling`<a id="score_scaling"></a>
</td>
<td>
One of either `"rsqrt_dim"` (default), `"trainable_elup1"`,
or `"none"`. If set to `"rsqrt_dim"`, the attention scores are
divided by the square root of the dimension of keys (i.e.,
`per_head_channels` if `transform_keys=True`, otherwise whatever the
dimension of combined sender inputs is). If set to `"trainable_elup1"`,
the scores are scaled with `elu(x) + 1`, where `elu` is the Exponential
Linear Unit (see `tf.keras.activations.elu`), and `x` is a per-head
trainable weight of the model that is initialized to `0.0`. Recall that 
`elu(x) + 1 == exp(x) if x<0 else x+1`, so the
initial scaling factor is `1.0`, decreases exponentially below 1.0, and
grows linearly above 1.0.
</td>
</tr><tr>
<td>
`transform_values_after_pooling`<a id="transform_values_after_pooling"></a>
</td>
<td>
By default, each attention head applies
the value transformation, then pools with attention coefficients.
Setting this option pools inputs with attention coefficients, then applies
the transformation. This is mathematically equivalent but can be faster
or slower to compute, depending on the platform and the dataset.
IMPORTANT: Toggling this option breaks checkpoint compatibility.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`receiver_tag`<a id="receiver_tag"></a>
</td>
<td>
one of `tfgnn.SOURCE`, `tfgnn.TARGET` or `tfgnn.CONTEXT`.
The results are aggregated for this graph piece.
If set to `tfgnn.SOURCE` or `tfgnn.TARGET`, the layer can be called for
an edge set and will aggregate results at the specified endpoint of the
edges.
If set to `tfgnn.CONTEXT`, the layer can be called for an edge set or a
node set and will aggregate results for context (per graph component).
If left unset for init, the tag must be passed at call time.
</td>
</tr><tr>
<td>
`receiver_feature`<a id="receiver_feature"></a>
</td>
<td>
The name of the feature that is read from the receiver
graph piece and passed as convolve(receiver_input=...).
</td>
</tr><tr>
<td>
`sender_node_feature`<a id="sender_node_feature"></a>
</td>
<td>
The name of the feature that is read from the sender
nodes, if any, and passed as convolve(sender_node_input=...).
NOTICE this must be `None` for use with `receiver_tag=tfgnn.CONTEXT`
on an edge set, or for pooling from edges without sender node states.
</td>
</tr><tr>
<td>
`sender_edge_feature`<a id="sender_edge_feature"></a>
</td>
<td>
The name of the feature that is read from the sender
edges, if any, and passed as convolve(sender_edge_input=...).
NOTICE this must not be `None` for use with `receiver_tag=tfgnn.CONTEXT`
on an edge set.
</td>
</tr><tr>
<td>
`extra_receiver_ops`<a id="extra_receiver_ops"></a>
</td>
<td>
A str-keyed dictionary of Python callables that are
wrapped to bind some arguments and then passed on to `convolve()`.
Sample usage: `extra_receiver_ops={"softmax": tfgnn.softmax}`.
The values passed in this dict must be callable as follows, with two
positional arguments:

``` python
f(graph, receiver_tag, node_set_name=..., feature_value=..., ...)
f(graph, receiver_tag, edge_set_name=..., feature_value=..., ...)
```

The wrapped callables seen by `convolve()` can be called like

``` python
wrapped_f(feature_value, ...)
```

The first three arguments of `f` are set to the input GraphTensor of the layer
and the tag/name pair required by `tfgnn.broadcast()` and `tfgnn.pool()` to move
values between the receiver and the messages that are computed inside the
convolution. The sole positional argument of `wrapped_f()` is passed to `f()` as
`feature_value=`, and any keyword arguments are forwarded.

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

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/multi_head_attention/layers.py#L352-L535">View
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

Returns the convolution result.

The Tensor inputs to this function still have their original shapes and need to
be broadcast such that the leading dimension is indexed by the items in the
graph for which messages are computed (usually edges; except when convolving
from nodes to context). In the end, values have to be pooled from there into a
Tensor with a leading dimension indexed by receivers, see `pool_to_receiver`.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`sender_node_input`
</td>
<td>
The input Tensor from the sender NodeSet, or `None`.
If self.takes_sender_node_input is `False`, this arg will be `None`.
(If it is `True`, that depends on how this layer gets called.)
See also broadcast_from_sender_node.
</td>
</tr><tr>
<td>
`sender_edge_input`
</td>
<td>
The input Tensor from the sender EdgeSet, or `None`.
If self.takes_sender_edge_input is `False`, this arg will be `None`.
(If it is `True`, it depends on how this layer gets called.)
If present, this Tensor is already indexed by the items for which
messages are computed.
</td>
</tr><tr>
<td>
`receiver_input`
</td>
<td>
The input Tensor from the receiver NodeSet or Context,
or None. If self.takes_receiver_input is `False`, this arg will be
`None`. (If it is `True`, it depends on how this layer gets called.)
See broadcast_from_receiver.
</td>
</tr><tr>
<td>
`broadcast_from_sender_node`
</td>
<td>
A function that broadcasts a Tensor indexed
like sender_node_input to a Tensor indexed by the items for which
messages are computed.
</td>
</tr><tr>
<td>
`broadcast_from_receiver`
</td>
<td>
Call this as `broadcast_from_receiver(value)`
to broadcast a Tensor indexed like receiver_input to a Tensor indexed
by the items for which messages are computed.
</td>
</tr><tr>
<td>
`pool_to_receiver`
</td>
<td>
Call this as `pool_to_receiver(value, reduce_type=...)`
to pool an item-indexed Tensor to a receiver-indexed tensor, using
a reduce_type understood by tfgnn.pool(), such as "sum".
</td>
</tr><tr>
<td>
`extra_receiver_ops`
</td>
<td>
The extra_receiver_ops passed to init, see there,
wrapped so that they can be called directly on a feature value.
If init did not receive extra_receiver_ops, convolve() will not receive
this argument, so subclass implementors not using it can omit it.
</td>
</tr><tr>
<td>
`training`
</td>
<td>
The `training` boolean that was passed to Layer.call(). If true,
the result is computed for training rather than inference. For example,
calls to tf.nn.dropout() are usually conditioned on this flag.
By contrast, calling another Keras layer (like tf.keras.layers.Dropout)
does not require forwarding this arg, Keras does that automatically.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Tensor whose leading dimension is indexed by receivers, with the
result of the convolution for each receiver.
</td>
</tr>

</table>
