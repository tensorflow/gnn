# tfgnn.keras.ConvGNNBuilder

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/builders.py#L29-L219">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Factory of layers that do convolutions on a graph.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.keras.ConvGNNBuilder(
    convolutions_factory: Callable[..., graph_update_lib.EdgesToNodePoolingLayer],
    nodes_next_state_factory: Callable[[const.NodeSetName], next_state_lib.NextStateForNodeSet],
    *,
    receiver_tag: Optional[const.IncidentNodeTag] = None,
    node_set_update_factory: Optional[Callable[..., graph_update_lib.NodeSetUpdateLayer]] = None,
    graph_update_factory: Optional[Callable[..., tf.keras.layers.Layer]] = None
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
    tf.keras.Input(type_spec=graph_tensor_spec),
    gnn.Convolve({"a", "b", "c"}),  # equivalent to gnn.Convolve()
    gnn.Convolve({"c"}),
])
```

Advanced users can pass additional callbacks to further customize the creation
of node set updates and the complete graph updates. The default values of those
callbacks are equivalent to

```python
def node_set_update_factory(node_set_name, edge_set_inputs, next_state):
  del node_set_name  # Unused by default.
  return tfgnn.keras.layers.NodeSetUpdate(edge_set_inputs, next_state)

def graph_update_factory(deferred_init_callback, name):
  return tfgnn.keras.layers.GraphUpdate(
      deferred_init_callback=deferred_init_callback, name=name)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Init args</h2></th></tr>

<tr>
<td>
<code>convolutions_factory</code><a id="convolutions_factory"></a>
</td>
<td>
called as
<code>convolutions_factory(edge_set_name, receiver_tag=receiver_tag)</code>
to return the convolution layer for the edge set towards the specified
receiver. The <code>receiver_tag</code> kwarg is omitted from the call if it is
omitted from the init args (but that usage is deprecated).
</td>
</tr><tr>
<td>
<code>nodes_next_state_factory</code><a id="nodes_next_state_factory"></a>
</td>
<td>
called as
<code>nodes_next_state_factory(node_set_name)</code> to return the next-state layer
for the respectve NodeSetUpdate.
</td>
</tr><tr>
<td>
<code>receiver_tag</code><a id="receiver_tag"></a>
</td>
<td>
Set this to <a href="../../tfgnn.md#TARGET"><code>tfgnn.TARGET</code></a> or <a href="../../tfgnn.md#SOURCE"><code>tfgnn.SOURCE</code></a> to choose which
incident node of each edge receives the convolution result.
DEPRECATED: This used to be optional and effectively default to TARGET.
New code is expected to set it in any case.
</td>
</tr><tr>
<td>
<code>node_set_update_factory</code><a id="node_set_update_factory"></a>
</td>
<td>
If set, called as
<code>node_set_update_factory(node_set_name, edge_set_inputs, next_state)</code>
to return the node set update for the given <code>node_set_name</code>. The
remaining arguments are as expected by <a href="../../tfgnn/keras/layers/NodeSetUpdate.md"><code>tfgnn.keras.layers.NodeSetUpdate</code></a>.
</td>
</tr><tr>
<td>
<code>graph_update_factory</code><a id="graph_update_factory"></a>
</td>
<td>
If set, called as
<code>graph_update_factory(deferred_init_callback, name)</code> to return the graph
update. The arguments are as expected by <a href="../../tfgnn/keras/layers/GraphUpdate.md"><code>tfgnn.keras.layers.GraphUpdate</code></a>.
</td>
</tr>
</table>

## Methods

<h3 id="Convolve"><code>Convolve</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/keras/builders.py#L127-L219">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>Convolve(
    node_sets: Optional[Collection[const.NodeSetName]] = None,
    name: Optional[str] = None
) -> tf.keras.layers.Layer
</code></pre>

Constructs GraphUpdate layer for the set of receiver node sets.

This method constructs NodeSetUpdate layers from convolutions and next state
factories (specified during the class construction) for the given receiver node
sets. The resulting node set update layers are combined and returned as one
GraphUpdate layer. Auxiliary node sets (e.g., as needed for
`tfgnn.keras.layers.NamedReadout`) are ignored.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<code>node_sets</code>
</td>
<td>
By default, the result updates all node sets that receive from
at least one edge set. Optionally, this argument can specify a subset
of those node sets. It is not allowed to include node sets that do not
receive messages from any edge set. It is also not allowed to include
auxiliary node sets.
</td>
</tr><tr>
<td>
<code>name</code>
</td>
<td>
Optionally, a name for the returned GraphUpdate layer.
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





