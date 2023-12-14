# tfgnn.convert_to_line_graph

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/graph/graph_tensor_ops.py#L1017-L1129">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Obtain a graph's line graph.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfgnn.convert_to_line_graph(
    graph_tensor: gt.GraphTensor,
    *,
    connect_from: const.IncidentNodeTag = const.TARGET,
    connect_to: const.IncidentNodeTag = const.SOURCE,
    connect_with_original_nodes: bool = False,
    non_backtracking: bool = False,
    use_node_features_as_line_graph_edge_features: bool = False
) -> gt.GraphTensor
</code></pre>

<!-- Placeholder for "Used in" -->

In the line graph, every edge in the original graph becomes a node, see
https://en.wikipedia.org/wiki/Line_graph. Line graph nodes are connected
whenever the corresponding edges share a specified endpoint. The *node* sets of
the resulting graph are the *edge* sets of the original graph, with the same
name. The resulting edge sets are named `{edge_set_name1}=>{edge_set_name2}`,
for every pair of edge sets that connects through a common node set (as selected
by the args). In particular, a pair of edges `u_0->u_1`, `v_0->v_1` will be
connected if `u_i == v_j`, where the index `i in {0, 1}` is specified by
`connect_from` and `j in {0, 1}` is specified by `connect_to`.

If `non_backtracking=True`, edges will only be connected if they also fulfill
`u_{1-i} != v_{1-j}`.

This function only supports graphs where all edge set adjacencies contain only
one SOURCE and one TARGET end point, i.e. non-hypergraphs. Note that
representing undirected edges {u,v} as a pair of two directed edges u->v and
v->u will result in a pair of separate line graph nodes.

Auxiliary node sets are not converted. This will raise an error if (a) the graph
contains a _readout node set and `preserve_node_sets` is False or (b) it
contains a _shadow node set.

Example: Consider a directed triangle represented as a homogeneous graph. The
node set 'points' contains nodes a, b and c while the edge set 'lines' contains
edges a->b, b->c, and c->a. The resulting line graph will contain a node set(!)
'lines' and an edge set 'lines=>lines'. The nodes in node set 'lines' correspond
to the original edges; let's call them ab, bc, and ca. The edges in edge set
'lines=>lines' represent the connections of lines at points: ab->bc, bc->ca, and
ca->ab.

If `connect_with_original_nodes=True`, the resulting graph will retain the
original nodes and their connection to edges as follows: Node set
'original/points' keeps the original nodes a, b and c, and there are two edge
sets: 'original/to/lines' with edges a->ab, b->bc, c->ca, and
'original/from/lines' with edges ab->b, bc->c, ca->a.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<code>graph_tensor</code><a id="graph_tensor"></a>
</td>
<td>
Graph to convert to a line graph.
</td>
</tr><tr>
<td>
<code>connect_from</code><a id="connect_from"></a>
</td>
<td>
Specifies which endpoint of the original edges
will determine the source for the line graph edges.
</td>
</tr><tr>
<td>
<code>connect_to</code><a id="connect_to"></a>
</td>
<td>
Specifies which endpoint of the original edges
will determine the target for the line graph edges.
</td>
</tr><tr>
<td>
<code>connect_with_original_nodes</code><a id="connect_with_original_nodes"></a>
</td>
<td>
If true, keep the original node sets (not the
original edge sets) and connect them to line graph nodes according to
source and target in the original graph. The node set names will be called
<code>original/{node_set}</code> and the new edges <code>original/to/{edge_set}</code> for the
SOURCE nodes and <code>original/from/{edge_set}</code> for the TARGET nodes.
</td>
</tr><tr>
<td>
<code>non_backtracking</code><a id="non_backtracking"></a>
</td>
<td>
Whether to return the non-backtracking line graph. Setting
this to True will only connect edges where the "outer" nodes are
different, i.e. <code>u_{1-i} != v_{1-j}</code>. For default connection settings,
for every edge u->v this *removes* line graph edges uv->vu. If
connect_to=TARGET, this *removes* line graph edges uv->uv.
</td>
</tr><tr>
<td>
<code>use_node_features_as_line_graph_edge_features</code><a id="use_node_features_as_line_graph_edge_features"></a>
</td>
<td>
Whether to use the original
graph's node features as edge features in the line graph.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A GraphTensor defining the graph's line graph.
</td>
</tr>

</table>
