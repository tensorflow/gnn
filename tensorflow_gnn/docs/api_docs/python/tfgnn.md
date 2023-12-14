# Module: tfgnn

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/__init__.py">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Public interface for TensorFlow GNN package.

All the public symbols, data types and functions are provided from this
top-level package. To use the library, you should use a single import statement,
like this:

```
import tensorflow_gnn as tfgnn
```

## Modules

[`experimental`](./tfgnn/experimental.md) module: Experimental (unstable) parts
of the public interface of TensorFlow GNN.

[`keras`](./tfgnn/keras.md) module: The tfgnn.keras package.

[`proto`](./tfgnn/proto.md) module: The protocol message (protobuf) types
defined by TensorFlow GNN.

[`sampler`](./tfgnn/sampler.md) module: Public interface for GNN Sampler.

## Classes

[`class Adjacency`](./tfgnn/Adjacency.md): Stores how edges connect pairs of
nodes from source and target node sets.

[`class AdjacencySpec`](./tfgnn/AdjacencySpec.md): A type spec for
<a href="./tfgnn/Adjacency.md"><code>tfgnn.Adjacency</code></a>.

[`class Context`](./tfgnn/Context.md): A composite tensor for graph context
features.

[`class ContextSpec`](./tfgnn/ContextSpec.md): A type spec for
<a href="./tfgnn/Context.md"><code>tfgnn.Context</code></a>.

[`class EdgeSet`](./tfgnn/EdgeSet.md): A composite tensor for edge set features,
size and adjacency information.

[`class EdgeSetSpec`](./tfgnn/EdgeSetSpec.md): A type spec for
<a href="./tfgnn/EdgeSet.md"><code>tfgnn.EdgeSet</code></a>.

[`class Feature`](./tfgnn/proto/Feature.md): The schema entry for a single
feature.

[`class FeatureDefaultValues`](./tfgnn/FeatureDefaultValues.md): Default values
for graph context, node sets and edge sets features.

[`class GraphSchema`](./tfgnn/proto/GraphSchema.md): The top-level container for
the schema of a graph dataset.

[`class GraphTensor`](./tfgnn/GraphTensor.md): A composite tensor for
heterogeneous directed graphs with features.

[`class GraphTensorSpec`](./tfgnn/GraphTensorSpec.md): A type spec for
<a href="./tfgnn/GraphTensor.md"><code>tfgnn.GraphTensor</code></a>.

[`class HyperAdjacency`](./tfgnn/HyperAdjacency.md): Stores how (hyper-)edges
connect tuples of nodes from incident node sets.

[`class HyperAdjacencySpec`](./tfgnn/HyperAdjacencySpec.md): A type spec for
<a href="./tfgnn/HyperAdjacency.md"><code>tfgnn.HyperAdjacency</code></a>.

[`class NodeSet`](./tfgnn/NodeSet.md): A composite tensor for node set features
plus size information.

[`class NodeSetSpec`](./tfgnn/NodeSetSpec.md): A type spec for
<a href="./tfgnn/NodeSet.md"><code>tfgnn.NodeSet</code></a>.

[`class SizeConstraints`](./tfgnn/SizeConstraints.md): Constraints on the number
of entities in the graph.

[`class ValidationError`](./tfgnn/ValidationError.md): A schema validation
error.

## Functions

[`add_readout_from_first_node(...)`](./tfgnn/add_readout_from_first_node.md):
Adds a readout structure equivalent to
<a href="./tfgnn/gather_first_node.md"><code>tfgnn.gather_first_node()</code></a>.

[`add_self_loops(...)`](./tfgnn/add_self_loops.md): Adds self-loops for
`edge_set_name` EVEN if they already exist.

[`assert_constraints(...)`](./tfgnn/assert_constraints.md): Validate the shape
constaints of a graph's features at runtime.

[`assert_satisfies_size_constraints(...)`](./tfgnn/assert_satisfies_size_constraints.md):
Raises InvalidArgumentError if graph_tensor exceeds size_constraints.

[`assert_satisfies_total_sizes(...)`](./tfgnn/assert_satisfies_size_constraints.md):
Raises InvalidArgumentError if graph_tensor exceeds size_constraints.

[`broadcast(...)`](./tfgnn/broadcast.md): Broadcasts values from nodes to edges,
or from context to nodes or edges.

[`broadcast_context_to_edges(...)`](./tfgnn/broadcast_context_to_edges.md):
Broadcasts a context value to the `edge_set` edges.

[`broadcast_context_to_nodes(...)`](./tfgnn/broadcast_context_to_nodes.md):
Broadcasts a context value to the `node_set` nodes.

[`broadcast_node_to_edges(...)`](./tfgnn/broadcast_node_to_edges.md): Broadcasts
values from nodes to incident edges.

[`check_compatible_with_schema_pb(...)`](./tfgnn/check_compatible_with_schema_pb.md):
Checks that the given spec or value is compatible with the graph schema.

[`check_homogeneous_graph_tensor(...)`](./tfgnn/check_homogeneous_graph_tensor.md):
Raises ValueError when tfgnn.get_homogeneous_node_and_edge_set_name() does.

[`check_required_features(...)`](./tfgnn/check_required_features.md): Checks the
requirements of a given schema against another.

[`check_scalar_graph_tensor(...)`](./tfgnn/check_scalar_graph_tensor.md): Checks
that graph tensor is scalar (has rank 0).

[`combine_values(...)`](./tfgnn/combine_values.md): Combines a list of tensors
into one (by concatenation or otherwise).

[`convert_to_line_graph(...)`](./tfgnn/convert_to_line_graph.md): Obtain a
graph's line graph.

[`create_graph_spec_from_schema_pb(...)`](./tfgnn/create_graph_spec_from_schema_pb.md):
Converts a graph schema proto message to a scalar GraphTensorSpec.

[`create_schema_pb_from_graph_spec(...)`](./tfgnn/create_schema_pb_from_graph_spec.md):
Converts scalar GraphTensorSpec to a graph schema proto message.

[`dataset_filter_with_summary(...)`](./tfgnn/dataset_filter_with_summary.md):
Dataset filter with a summary for the fraction of dataset elements removed.

[`dataset_from_generator(...)`](./tfgnn/dataset_from_generator.md): Creates
dataset from generator of any nest of scalar graph pieces.

[`disable_graph_tensor_validation(...)`](./tfgnn/disable_graph_tensor_validation.md):
Disables both static and runtime checks of graph tensors.

[`disable_graph_tensor_validation_at_runtime(...)`](./tfgnn/disable_graph_tensor_validation_at_runtime.md):
Disables runtime checks (`tf.debugging.Assert`) of graph tensors.

[`enable_graph_tensor_validation(...)`](./tfgnn/enable_graph_tensor_validation.md):
Enables static checks of graph tensors.

[`enable_graph_tensor_validation_at_runtime(...)`](./tfgnn/enable_graph_tensor_validation_at_runtime.md):
Enables both static and runtime checks of graph tensors.

[`find_tight_size_constraints(...)`](./tfgnn/find_tight_size_constraints.md):
Returns smallest possible size constraints that allow dataset padding.

[`gather_first_node(...)`](./tfgnn/gather_first_node.md): Gathers feature value
from the first node of each graph component.

[`get_aux_type_prefix(...)`](./tfgnn/get_aux_type_prefix.md): Returns type
prefix of aux node or edge set names, or `None` if non-aux.

[`get_homogeneous_node_and_edge_set_name(...)`](./tfgnn/get_homogeneous_node_and_edge_set_name.md):
Returns the sole `node_set_name, edge_set_name` or raises `ValueError`.

[`get_io_spec(...)`](./tfgnn/get_io_spec.md): Returns tf.io parsing features for
`GraphTensorSpec` type spec.

[`get_registered_reduce_operation_names(...)`](./tfgnn/get_registered_reduce_operation_names.md):
Returns the registered list of supported reduce operation names.

[`graph_tensor_to_values(...)`](./tfgnn/graph_tensor_to_values.md): Convert an
eager `GraphTensor` to a mapping of mappings of PODTs.

[`homogeneous(...)`](./tfgnn/homogeneous.md): Constructs a homogeneous
`GraphTensor` with node features and one edge_set.

[`is_dense_tensor(...)`](./tfgnn/is_dense_tensor.md): Returns whether a tensor
(TF or Keras) is a Tensor.

[`is_graph_tensor(...)`](./tfgnn/is_graph_tensor.md): Returns whether `value` is
a GraphTensor (possibly wrapped for Keras).

[`is_ragged_tensor(...)`](./tfgnn/is_ragged_tensor.md): Returns whether a tensor
(TF or Keras) is a RaggedTensor.

[`iter_features(...)`](./tfgnn/iter_features.md): Utility function to iterate
over the features of a graph schema.

[`iter_sets(...)`](./tfgnn/iter_sets.md): Utility function to iterate over all
the sets present in a graph schema.

[`learn_fit_or_skip_size_constraints(...)`](./tfgnn/learn_fit_or_skip_size_constraints.md):
Learns the optimal size constraints for the fixed size batching with retry.

[`mask_edges(...)`](./tfgnn/mask_edges.md): Creates a GraphTensor after applying
edge_mask over the specified edge-set.

[`node_degree(...)`](./tfgnn/node_degree.md): Returns the degree of each node
w.r.t. one side of an edge set.

[`pad_to_total_sizes(...)`](./tfgnn/pad_to_total_sizes.md): Pads graph tensor to
the total sizes by inserting fake graph components.

[`parse_example(...)`](./tfgnn/parse_example.md): Parses a batch of serialized
Example protos into a single `GraphTensor`.

[`parse_schema(...)`](./tfgnn/parse_schema.md): Parse a schema from
text-formatted protos.

[`parse_single_example(...)`](./tfgnn/parse_single_example.md): Parses a single
serialized Example proto into a single `GraphTensor`.

[`pool(...)`](./tfgnn/pool.md): Pools values from edges to nodes, or from nodes
or edges to context.

[`pool_edges_to_context(...)`](./tfgnn/pool_edges_to_context.md): Aggregates
(pools) edge values to graph context.

[`pool_edges_to_node(...)`](./tfgnn/pool_edges_to_node.md): Aggregates (pools)
edge values to incident nodes.

[`pool_neighbors_to_node(...)`](./tfgnn/pool_neighbors_to_node.md): Aggregates
(pools) neighbor node values along one or more edge sets.

[`pool_neighbors_to_node_feature(...)`](./tfgnn/pool_neighbors_to_node_feature.md):
Aggregates (pools) sender node feature to receiver nodes feature.

[`pool_nodes_to_context(...)`](./tfgnn/pool_nodes_to_context.md): Aggregates
(pools) node values to graph context.

[`random_graph_tensor(...)`](./tfgnn/random_graph_tensor.md): Generate a graph
tensor from a spec, with random features.

[`read_schema(...)`](./tfgnn/read_schema.md): Read a proto schema from a file
with text-formatted contents.

[`reorder_nodes(...)`](./tfgnn/reorder_nodes.md): Reorders nodes within node
sets according to indices.

[`reverse_tag(...)`](./tfgnn/reverse_tag.md): Flips tfgnn.SOURCE to tfgnn.TARGET
and vice versa.

[`satisfies_size_constraints(...)`](./tfgnn/satisfies_size_constraints.md):
Returns whether the input `graph_tensor` satisfies `total_sizes`.

[`satisfies_total_sizes(...)`](./tfgnn/satisfies_size_constraints.md): Returns
whether the input `graph_tensor` satisfies `total_sizes`.

[`shuffle_features_globally(...)`](./tfgnn/shuffle_features_globally.md):
Shuffles context, node set and edge set features of a scalar GraphTensor.

[`shuffle_nodes(...)`](./tfgnn/shuffle_nodes.md): Randomly reorders nodes of
given node sets, within each graph component.

[`softmax(...)`](./tfgnn/softmax.md): Computes softmax over a many-to-one
relationship in a GraphTensor.

[`softmax_edges_per_node(...)`](./tfgnn/softmax_edges_per_node.md): Returns
softmax() of edge values per common `node_tag` node.

[`structured_readout(...)`](./tfgnn/structured_readout.md): Reads out a feature
value from select nodes (or edges) in a graph.

[`structured_readout_into_feature(...)`](./tfgnn/structured_readout_into_feature.md):
Reads out a feature value from select nodes (or edges) in a graph.

[`validate_graph_tensor_for_readout(...)`](./tfgnn/validate_graph_tensor_for_readout.md):
Checks `graph` supports `structured_readout()` from `required_keys`.

[`validate_graph_tensor_spec_for_readout(...)`](./tfgnn/validate_graph_tensor_spec_for_readout.md):
Checks `graph_spec` supports `structured_readout()` from `required_keys`.

[`validate_schema(...)`](./tfgnn/validate_schema.md): Validates the correctness
of a graph schema instance.

[`write_example(...)`](./tfgnn/write_example.md): Encode an eager `GraphTensor`
to a tf.train.Example proto.

[`write_schema(...)`](./tfgnn/write_schema.md): Write a `GraphSchema` to a
text-formatted proto file.

## Type Aliases

[`Field`](./tfgnn/Field.md)

[`FieldOrFields`](./tfgnn/FieldOrFields.md)

[`FieldSpec`](./tfgnn/FieldSpec.md)

[`Fields`](./tfgnn/Fields.md)

[`FieldsSpec`](./tfgnn/FieldsSpec.md)

[`IncidentNodeOrContextTag`](./tfgnn/IncidentNodeOrContextTag.md)

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Other Members</h2></th></tr>

<tr>
<td>
CONTEXT<a id="CONTEXT"></a>
</td>
<td>
<code>'context'</code>
</td>
</tr><tr>
<td>
EDGES<a id="EDGES"></a>
</td>
<td>
<code>'edges'</code>
</td>
</tr><tr>
<td>
HIDDEN_STATE<a id="HIDDEN_STATE"></a>
</td>
<td>
<code>'hidden_state'</code>
</td>
</tr><tr>
<td>
NODES<a id="NODES"></a>
</td>
<td>
<code>'nodes'</code>
</td>
</tr><tr>
<td>
SIZE_NAME<a id="SIZE_NAME"></a>
</td>
<td>
<code>'#size'</code>
</td>
</tr><tr>
<td>
SOURCE<a id="SOURCE"></a>
</td>
<td>
<code>0</code>
</td>
</tr><tr>
<td>
SOURCE_NAME<a id="SOURCE_NAME"></a>
</td>
<td>
<code>'#source'</code>
</td>
</tr><tr>
<td>
TARGET<a id="TARGET"></a>
</td>
<td>
<code>1</code>
</td>
</tr><tr>
<td>
TARGET_NAME<a id="TARGET_NAME"></a>
</td>
<td>
<code>'#target'</code>
</td>
</tr><tr>
<td>
**version**<a id="__version__"></a>
</td>
<td>
<code>'1.0.0.dev2'</code>
</td>
</tr>
</table>
