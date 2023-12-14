# Module: tfgnn.proto

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/proto/__init__.py">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

The protocol message (protobuf) types defined by TensorFlow GNN.

This package is automatically included in the top-level tfgnn package:

```
import tensorflow_gnn as tfgnn
graph_schema = tfgnn.proto.GraphSchema()
```

Users are also free to import it separately as

```
import tensorflow_gnn.proto as tfgnn_proto
graph_schema = tfgnn_proto.GraphSchema()
```

...which, together with using its more targeted BUILD dependency, can help to
shrink the bazel-bin/**/*.runfiles/ directory.

## Classes

[`class BigQuery`](../tfgnn/proto/BigQuery.md): Describes a BigQuery table or
SQL statement as datasource of a graph piece.

[`class Context`](../tfgnn/proto/Context.md): The schema for the features that
apply across the entire input graph.

[`class EdgeSet`](../tfgnn/proto/EdgeSet.md): The schema shared by a set of
edges that connect the same pair of node sets.

[`class Feature`](../tfgnn/proto/Feature.md): The schema entry for a single
feature.

[`class GraphSchema`](../tfgnn/proto/GraphSchema.md): The top-level container
for the schema of a graph dataset.

[`class Metadata`](../tfgnn/proto/Metadata.md): Extra information optionally
provided on a context, node set or edge set.

[`class NodeSet`](../tfgnn/proto/NodeSet.md): The schema shared by a set of
nodes in the graph.

[`class OriginInfo`](../tfgnn/proto/OriginInfo.md): Metadata about the origin of
the graph data.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Other Members</h2></th></tr>

<tr> <td> GraphType<a id="GraphType"></a> </td> <td> ['UNDEFINED', 'FULL',
'SUBGRAPH', 'RANDOM_WALKS']

An enumeration of graph types according to the method of creation.

For detailed documentation, see the comments in the <code>graph_schema.proto</code> file.
</td>
</tr>
</table>
