"""Public interface for TensorFlow GNN package.

All the public symbols, data types and functions are provided from this
top-level package. To use the library, you should use a single import statement,
like this:

    import tensorflow_gnn as tfgnn

The various data types provided by the GNN library have corresponding schemas
similar to `tf.TensorSpec`. For example, a `FieldSpec` describes an instance of
`Field`, and a `GraphTensorSpec` describes an instance of `GraphTensor`.
"""

from tensorflow_gnn.graph import adjacency
from tensorflow_gnn.graph import graph_constants
from tensorflow_gnn.graph import graph_tensor
from tensorflow_gnn.graph import graph_tensor_encode
from tensorflow_gnn.graph import graph_tensor_io
from tensorflow_gnn.graph import graph_tensor_ops
from tensorflow_gnn.graph import graph_tensor_pprint
from tensorflow_gnn.graph import graph_tensor_random
from tensorflow_gnn.graph import keras  # For use as a subpackage.
from tensorflow_gnn.graph import normalization_ops
from tensorflow_gnn.graph import schema_utils
from tensorflow_gnn.graph import schema_validation
from tensorflow_gnn.proto import graph_schema

# String constants for feature name components, and special feature names.
CONTEXT = graph_constants.CONTEXT
NODES = graph_constants.NODES
EDGES = graph_constants.EDGES
DEFAULT_STATE_NAME = graph_constants.DEFAULT_STATE_NAME

# Integer tags.
SOURCE = graph_constants.SOURCE
TARGET = graph_constants.TARGET

# Encoded names of implicit features.
SIZE_NAME = '#size'
SOURCE_NAME = '#source'
TARGET_NAME = '#target'

# Field values, specs, and dictionaries containing them.
Field = graph_constants.Field
FieldName = graph_constants.FieldName
FieldSpec = graph_constants.FieldSpec
Fields = graph_constants.Fields
FieldsSpec = graph_constants.FieldsSpec

# Names and types of node sets and edge sets.
SetName = graph_constants.SetName
SetType = graph_constants.SetType
NodeSetName = graph_constants.NodeSetName
EdgeSetName = graph_constants.EdgeSetName

# Context, node and edge set objects.
Context = graph_tensor.Context
ContextSpec = graph_tensor.ContextSpec
NodeSet = graph_tensor.NodeSet
NodeSetSpec = graph_tensor.NodeSetSpec
EdgeSet = graph_tensor.EdgeSet
EdgeSetSpec = graph_tensor.EdgeSetSpec

# Adjacency data types.
Adjacency = adjacency.Adjacency
AdjacencySpec = adjacency.AdjacencySpec
HyperAdjacency = adjacency.HyperAdjacency
HyperAdjacencySpec = adjacency.HyperAdjacencySpec

# Principal container and spec type.
GraphTensor = graph_tensor.GraphTensor
GraphTensorSpec = graph_tensor.GraphTensorSpec

# Proto description of schema.
GraphSchema = graph_schema.GraphSchema
Feature = graph_schema.Feature

# I/O functions (input parsing).
parse_example = graph_tensor_io.parse_example
parse_single_example = graph_tensor_io.parse_single_example
get_io_spec = graph_tensor_io.get_io_spec

# I/O functions (output encoding).
write_example = graph_tensor_encode.write_example

# Pretty-printing.
graph_tensor_to_values = graph_tensor_pprint.graph_tensor_to_values

# Random generation.
random_graph_tensor = graph_tensor_random.random_graph_tensor

# Operations.
broadcast_node_to_edges = graph_tensor_ops.broadcast_node_to_edges
is_graph_tensor = graph_tensor_ops.is_graph_tensor
pool_edges_to_node = graph_tensor_ops.pool_edges_to_node
broadcast_context_to_nodes = graph_tensor_ops.broadcast_context_to_nodes
broadcast_context_to_edges = graph_tensor_ops.broadcast_context_to_edges
pool_nodes_to_context = graph_tensor_ops.pool_nodes_to_context
pool_edges_to_context = graph_tensor_ops.pool_edges_to_context
gather_first_node = graph_tensor_ops.gather_first_node
get_registered_reduce_operation_names = (
    graph_tensor_ops.get_registered_reduce_operation_names)
register_reduce_operation = graph_tensor_ops.register_reduce_operation

# Normalization operations.
softmax_edges_per_node = normalization_ops.softmax_edges_per_node

# Schema conversion and I/O functions.
parse_schema = schema_utils.parse_schema
read_schema = schema_utils.read_schema
write_schema = schema_utils.write_schema
create_graph_spec_from_schema_pb = schema_utils.create_graph_spec_from_schema_pb
iter_sets = schema_utils.iter_sets
iter_features = schema_utils.iter_features

# Schema validation.
ValidationError = schema_validation.ValidationError
validate_schema = schema_validation.validate_schema
check_required_features = schema_validation.check_required_features
assert_constraints = schema_validation.assert_constraints

# Prune imported module symbols so they're not accessible implicitly,
# except those meant to be used as subpackages, like tfgnn.keras.*.
del adjacency
del graph_constants
del graph_tensor
del graph_tensor_encode
del graph_tensor_io
del graph_tensor_ops
del graph_tensor_pprint
del graph_tensor_random
del normalization_ops
del graph_schema
del schema_utils
del schema_validation
