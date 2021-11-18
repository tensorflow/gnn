description: Public interface for TensorFlow GNN package.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gnn" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="CONTEXT"/>
<meta itemprop="property" content="DEFAULT_STATE_NAME"/>
<meta itemprop="property" content="EDGES"/>
<meta itemprop="property" content="NODES"/>
<meta itemprop="property" content="SIZE_NAME"/>
<meta itemprop="property" content="SOURCE"/>
<meta itemprop="property" content="SOURCE_NAME"/>
<meta itemprop="property" content="TARGET"/>
<meta itemprop="property" content="TARGET_NAME"/>
</div>

# Module: gnn

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Public interface for TensorFlow GNN package.


All the public symbols, data types and functions are provided from this
top-level package. To use the library, you should use a single import statement,
like this:

    import tensorflow_gnn as tfgnn

The various data types provided by the GNN library have corresponding schemas
similar to `tf.TensorSpec`. For example, a `FieldSpec` describes an instance of
`Field`, and a `GraphTensorSpec` describes an instance of `GraphTensor`.

## Modules

[`keras`](./gnn/keras.md) module: The tfgnn.keras package.

## Classes

[`class Adjacency`](./gnn/Adjacency.md): Stores simple binary edges with a source and target.

[`class AdjacencySpec`](./gnn/AdjacencySpec.md): TypeSpec for Adjacency.

[`class Context`](./gnn/Context.md): A container of features for a graph component.

[`class ContextSpec`](./gnn/ContextSpec.md): A type spec for global features for a graph component.

[`class EdgeSet`](./gnn/EdgeSet.md): A container for the features of a single edge set.

[`class EdgeSetSpec`](./gnn/EdgeSetSpec.md): A type spec for the features of a single edge set.

[`class Feature`](./gnn/Feature.md): A schema for a single feature.

[`class GraphSchema`](./gnn/GraphSchema.md): A schema definition for graphs.

[`class GraphTensor`](./gnn/GraphTensor.md): Stores graphs, possibly heterogeneous (i.e., with multiple node sets).

[`class GraphTensorSpec`](./gnn/GraphTensorSpec.md): A type spec for a `GraphTensor` instance.

[`class HyperAdjacency`](./gnn/HyperAdjacency.md): Stores edges as indices of nodes in node sets.

[`class HyperAdjacencySpec`](./gnn/HyperAdjacencySpec.md): TypeSpec for HyperAdjacency.

[`class NodeSet`](./gnn/NodeSet.md): A container for the features of a single node set.

[`class NodeSetSpec`](./gnn/NodeSetSpec.md): A type spec for the features of a single node set.

[`class ValidationError`](./gnn/ValidationError.md): A schema validation error.

## Functions

[`assert_constraints(...)`](./gnn/assert_constraints.md): Validate the shape constaints of a graph's features at runtime.

[`broadcast_context_to_edges(...)`](./gnn/broadcast_context_to_edges.md): Broadcasts a context value to the `edge_set` edges.

[`broadcast_context_to_nodes(...)`](./gnn/broadcast_context_to_nodes.md): Broadcasts a context value to the `node_set` nodes.

[`broadcast_node_to_edges(...)`](./gnn/broadcast_node_to_edges.md): Broadcasts values from nodes to incident edges.

[`check_required_features(...)`](./gnn/check_required_features.md): Checks the requirements of a given schema against another.

[`create_graph_spec_from_schema_pb(...)`](./gnn/create_graph_spec_from_schema_pb.md): Converts a graph schema proto message to a scalar GraphTensorSpec.

[`gather_first_node(...)`](./gnn/gather_first_node.md): Gathers feature value from the first node of each graph component.

[`get_io_spec(...)`](./gnn/get_io_spec.md): Returns tf.io parsing features for GraphTensorSpec.

[`get_registered_reduce_operation_names(...)`](./gnn/get_registered_reduce_operation_names.md): Returns the registered list of supported reduce operation names.

[`graph_tensor_to_values(...)`](./gnn/graph_tensor_to_values.md): Convert an eager `GraphTensor` to a mapping of mappings of PODTs.

[`is_graph_tensor(...)`](./gnn/is_graph_tensor.md): Returns whether `value` is a GraphTensor (possibly wrapped for Keras).

[`iter_features(...)`](./gnn/iter_features.md): Utility function to iterate over the features of a graph schema.

[`iter_sets(...)`](./gnn/iter_sets.md): Utility function to iterate over all the sets present in a graph schema.

[`parse_example(...)`](./gnn/parse_example.md): Parses a batch of serialized Example protos into a single `GraphTensor`.

[`parse_schema(...)`](./gnn/parse_schema.md): Parse a schema from text-formatted protos.

[`parse_single_example(...)`](./gnn/parse_single_example.md): Parses a single serialized Example proto into a single GraphTensor.

[`pool_edges_to_context(...)`](./gnn/pool_edges_to_context.md): Aggregates (pools) edge values to graph context.

[`pool_edges_to_node(...)`](./gnn/pool_edges_to_node.md): Aggregates (pools) edge values to incident nodes.

[`pool_nodes_to_context(...)`](./gnn/pool_nodes_to_context.md): Aggregates (pools) node values to graph context.

[`random_graph_tensor(...)`](./gnn/random_graph_tensor.md): Generate a graph tensor from a schema, with random features.

[`read_schema(...)`](./gnn/read_schema.md): Read a proto schema from a file with text-formatted contents.

[`register_reduce_operation(...)`](./gnn/register_reduce_operation.md): Register a new reduction operation for pooling.

[`softmax_edges_per_node(...)`](./gnn/softmax_edges_per_node.md): Softmaxes all the edges in the graph over their incident nodes.

[`validate_schema(...)`](./gnn/validate_schema.md): Validates the correctness of a graph schema instance.

[`write_example(...)`](./gnn/write_example.md): Encode an eager `GraphTensor` to a tf.train.Example proto.

[`write_schema(...)`](./gnn/write_schema.md): Write a `GraphSchema` to a text-formatted proto file.

## Type Aliases

[`Field`](./gnn/Field.md): The central part of internal API.

[`FieldSpec`](./gnn/FieldSpec.md): The central part of internal API.

[`Fields`](./gnn/Fields.md): The central part of internal API.

[`FieldsSpec`](./gnn/FieldsSpec.md): The central part of internal API.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Other Members</h2></th></tr>

<tr>
<td>
CONTEXT<a id="CONTEXT"></a>
</td>
<td>
`'context'`
</td>
</tr><tr>
<td>
DEFAULT_STATE_NAME<a id="DEFAULT_STATE_NAME"></a>
</td>
<td>
`'hidden_state'`
</td>
</tr><tr>
<td>
EDGES<a id="EDGES"></a>
</td>
<td>
`'edges'`
</td>
</tr><tr>
<td>
NODES<a id="NODES"></a>
</td>
<td>
`'nodes'`
</td>
</tr><tr>
<td>
SIZE_NAME<a id="SIZE_NAME"></a>
</td>
<td>
`'#size'`
</td>
</tr><tr>
<td>
SOURCE<a id="SOURCE"></a>
</td>
<td>
`0`
</td>
</tr><tr>
<td>
SOURCE_NAME<a id="SOURCE_NAME"></a>
</td>
<td>
`'#source'`
</td>
</tr><tr>
<td>
TARGET<a id="TARGET"></a>
</td>
<td>
`1`
</td>
</tr><tr>
<td>
TARGET_NAME<a id="TARGET_NAME"></a>
</td>
<td>
`'#target'`
</td>
</tr>
</table>

