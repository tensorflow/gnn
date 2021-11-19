description: Factory of layers that do convolutions on a graph.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gnn.keras.ConvGNNBuilder" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="Convolve"/>
<meta itemprop="property" content="__init__"/>
</div>

# gnn.keras.ConvGNNBuilder

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/keras/builders.py#L15-L100">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Factory of layers that do convolutions on a graph.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gnn.keras.ConvGNNBuilder(
    convolutions_factory: Callable[[const.EdgeSetName], graph_update_lib.EdgesToNodePoolingLayer],
    nodes_next_state_factory: Callable[[const.NodeSetName], next_state_lib.NextStateForNodeSet]
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

# Model hyper-parameters:
h_dims = {'a': 64, 'b': 32, 'c': 32}
m_dims = {'a->b': 64, 'b->c': 32, 'c->a': 32}

# ConvGNNBuilder initialization:
gnn = tfgnn.ConvGNNBuilder(
  lambda edge_set_name: tfgnn.SimpleConvolution(
     tf.keras.layers.Dense(m_dims[edge_set_name])),
  lambda node_set_name: tfgnn.NextStateFromConcat(
     tf.keras.layers.Dense(h_dims[node_set_name]))
)

# Two rounds of message passing to target node sets:
model = tf.keras.models.Sequential([
    gnn.Convolve({"a", "b", "c"}),  # equivalent to gnn.Convolve()
    gnn.Convolve({"c"}),
])



#### Init args:


* <b>`convolutions_factory`</b>: callable that takes as an input edge set name and
  returns graph convolution as EdgesToNodePooling layer.
* <b>`nodes_next_state_factory`</b>: callable that takes as an input node set name and
  returns node set next state as NextStateForNodeSet layer.


## Methods

<h3 id="Convolve"><code>Convolve</code></h3>

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/keras/builders.py#L61-L100">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>Convolve(
    node_sets: Optional[Set[const.NodeSetName]] = None
) -> tf.keras.layers.Layer
</code></pre>

Constructs GraphUpdate layer for the set of target nodes.

This method contructs NodeSetUpdate layers from convolutions and next state
factories (specified during the class construction) for the target node
sets. The resulting node set update layers are combined and returned as a
GraphUpdate layer.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`node_sets`
</td>
<td>
optional set of node set names to be updated. Not setting this
parameter is equivalent to updating all node sets.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
GraphUpdate layer wrapped with OncallBuilder for delayed building.
</td>
</tr>

</table>





