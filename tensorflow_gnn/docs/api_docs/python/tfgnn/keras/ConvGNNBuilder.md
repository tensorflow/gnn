# tfgnn.keras.ConvGNNBuilder

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/builders.py#L29-L142">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Factory of layers that do convolutions on a graph.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.ConvGNNBuilder(
    convolutions_factory: Callable[..., graph_update_lib.EdgesToNodePoolingLayer],
    nodes_next_state_factory: Callable[[const.NodeSetName], next_state_lib.NextStateForNodeSet],
    *,
    receiver_tag: Optional[const.IncidentNodeOrContextTag] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->

ConvGNNBuilder object constructs `GraphUpdate` layers, that apply arbitrary
convolutions and updates on nodes of a graph. The convolutions (created by the
`convolutions_factory`) propagate information to the incident edges of the
graph. The results of the convolution together with the current nodes states
are used to update the nodes, using a layer created by
`nodes_next_state_factory`.

Layers created by ConvGNNBuilder can be (re-)used in any order.

#### Example:

```python
# Model hyper-parameters:
h_dims = {'a': 64, 'b': 32, 'c': 32}
m_dims = {'a->b': 64, 'b->c': 32, 'c->a': 32}

# ConvGNNBuilder initialization:
gnn = tfgnn.keras.ConvGNNBuilder(
  lambda edge_set_name, receiver_tag: tfgnn.keras.layers.SimpleConv(
      tf.keras.layers.Dense(m_dims[edge_set_name]),
      receiver_tag=receiver_tag),
  lambda node_set_name: tfgnn.keras.layers.NextStateFromConcat(
      tf.keras.layers.Dense(h_dims[node_set_name])),
  receiver_tag=tfgnn.TARGET)

# Two rounds of message passing to target node sets:
model = tf.keras.models.Sequential([
    gnn.Convolve({"a", "b", "c"}),  # equivalent to gnn.Convolve()
    gnn.Convolve({"c"}),
])
```

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Init args</h2></th></tr>

<tr>
<td>
`convolutions_factory`<a id="convolutions_factory"></a>
</td>
<td>
called as
`convolutions_factory(edge_set_name, receiver_tag=receiver_tag)`
to return the convolution layer for the edge set towards the specified
receiver. The `receiver_tag` kwarg is omitted from the call if it is
omitted from the init args (but that usage is deprecated).
</td>
</tr><tr>
<td>
`nodes_next_state_factory`<a id="nodes_next_state_factory"></a>
</td>
<td>
called as
`nodes_next_state_factory(node_set_name)` to return the next-state layer
for the respectve NodeSetUpdate.
</td>
</tr><tr>
<td>
`receiver_tag`<a id="receiver_tag"></a>
</td>
<td>
Set this to <a href="../../tfgnn.md#TARGET"><code>tfgnn.TARGET</code></a> or <a href="../../tfgnn.md#SOURCE"><code>tfgnn.SOURCE</code></a> to choose which
incident node of each edge receives the convolution result.
DEPRECATED: This used to be optional and effectively default to TARGET.
New code is expected to set it in any case.
</td>
</tr>
</table>

## Methods

<h3 id="Convolve"><code>Convolve</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/builders.py#L91-L142">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>Convolve(
    node_sets: Optional[Collection[const.NodeSetName]] = None
) -> tf.keras.layers.Layer
</code></pre>

Constructs GraphUpdate layer for the set of receiver node sets.

This method contructs NodeSetUpdate layers from convolutions and next state
factories (specified during the class construction) for the given receiver
node sets. The resulting node set update layers are combined and returned
as one GraphUpdate layer.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`node_sets`
</td>
<td>
By default, the result updates all node sets that receive from
at least one edge set. Passing a set of node set names here (or a
Collection convertible to a set) overrides this (possibly including
node sets that receive from zero edge sets).
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A GraphUpdate layer, with building deferred to the first call.
</td>
</tr>

</table>





