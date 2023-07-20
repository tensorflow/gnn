# Copyright 2021 The TensorFlow GNN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Public interface for TensorFlow GNN package.

All the public symbols, data types and functions are provided from this
top-level package. To use the library, you should use a single import statement,
like this:

    import tensorflow_gnn as tfgnn

The various data types provided by the GNN library have corresponding schemas
similar to `tf.TensorSpec`. For example, a `FieldSpec` describes an instance of
`Field`, and a `GraphTensorSpec` describes an instance of `GraphTensor`.
"""
# pylint: disable=line-too-long

from tensorflow_gnn import experimental  # Exposed as submodule. pylint: disable=unused-import
from tensorflow_gnn import keras  # Exposed as submodule. pylint: disable=unused-import
from tensorflow_gnn import sampler  # Exposed as submodule. pylint: disable=unused-import
from tensorflow_gnn import version
from tensorflow_gnn.graph import adjacency
from tensorflow_gnn.graph import batching_utils
from tensorflow_gnn.graph import broadcast_ops
from tensorflow_gnn.graph import graph_constants
from tensorflow_gnn.graph import graph_tensor
from tensorflow_gnn.graph import graph_tensor_encode
from tensorflow_gnn.graph import graph_tensor_io
from tensorflow_gnn.graph import graph_tensor_ops
from tensorflow_gnn.graph import graph_tensor_pprint
from tensorflow_gnn.graph import graph_tensor_random
from tensorflow_gnn.graph import normalization_ops
from tensorflow_gnn.graph import padding_ops
from tensorflow_gnn.graph import pool_ops
from tensorflow_gnn.graph import preprocessing_common
from tensorflow_gnn.graph import readout
from tensorflow_gnn.graph import schema_utils
from tensorflow_gnn.graph import schema_validation
from tensorflow_gnn.graph import tag_utils
from tensorflow_gnn.graph import tensor_utils
from tensorflow_gnn.proto import graph_schema

# Package version.
__version__ = version.__version__

# String constants for feature name components, and special feature names.
CONTEXT = graph_constants.CONTEXT
NODES = graph_constants.NODES
EDGES = graph_constants.EDGES
HIDDEN_STATE = graph_constants.HIDDEN_STATE
DEFAULT_STATE_NAME = graph_constants.DEFAULT_STATE_NAME  # Deprecated.

# Integer tags.
SOURCE = graph_constants.SOURCE
TARGET = graph_constants.TARGET

# Type annotations for tags.
IncidentNodeTag = graph_constants.IncidentNodeTag
IncidentNodeOrContextTag = graph_constants.IncidentNodeOrContextTag

# Utils for tags.
reverse_tag = tag_utils.reverse_tag

# Encoded names of implicit features.
SIZE_NAME = graph_constants.SIZE_NAME
SOURCE_NAME = graph_constants.SOURCE_NAME
TARGET_NAME = graph_constants.TARGET_NAME

# Field values, specs, and dictionaries containing them.
Field = graph_constants.Field
FieldName = graph_constants.FieldName
FieldOrFields = graph_constants.FieldOrFields
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
homogeneous = graph_tensor.homogeneous

# Proto description of schema.
GraphSchema = graph_schema.GraphSchema
Feature = graph_schema.Feature

# Preprocessing (batching and padding) types.
FeatureDefaultValues = preprocessing_common.FeatureDefaultValues
SizeConstraints = preprocessing_common.SizeConstraints

# General preprocessing helpers.
dataset_filter_with_summary = preprocessing_common.dataset_filter_with_summary
dataset_from_generator = batching_utils.dataset_from_generator

# I/O functions (input parsing).
parse_example = graph_tensor_io.parse_example
parse_single_example = graph_tensor_io.parse_single_example
get_io_spec = graph_tensor_io.get_io_spec

# GraphTensor batching and padding.
pad_to_total_sizes = padding_ops.pad_to_total_sizes
assert_satisfies_size_constraints = padding_ops.assert_satisfies_size_constraints
satisfies_size_constraints = padding_ops.satisfies_size_constraints

assert_satisfies_total_sizes = padding_ops.assert_satisfies_size_constraints
satisfies_total_sizes = padding_ops.satisfies_size_constraints

# Learned batching and padding.
find_tight_size_constraints = batching_utils.find_tight_size_constraints
learn_fit_or_skip_size_constraints = batching_utils.learn_fit_or_skip_size_constraints

# I/O functions (output encoding).
write_example = graph_tensor_encode.write_example

# Pretty-printing.
graph_tensor_to_values = graph_tensor_pprint.graph_tensor_to_values

# Random generation.
random_graph_tensor = graph_tensor_random.random_graph_tensor

# Broadcast and pool.
broadcast_node_to_edges = broadcast_ops.broadcast_node_to_edges
broadcast_context_to_nodes = broadcast_ops.broadcast_context_to_nodes
broadcast_context_to_edges = broadcast_ops.broadcast_context_to_edges
broadcast = broadcast_ops.broadcast_v2
pool_edges_to_node = pool_ops.pool_edges_to_node
pool_nodes_to_context = pool_ops.pool_nodes_to_context
pool_edges_to_context = pool_ops.pool_edges_to_context
pool = pool_ops.pool_v2
get_registered_reduce_operation_names = pool_ops.get_registered_reduce_operation_names

# Misc operations.
mask_edges = graph_tensor_ops.mask_edges
add_self_loops = graph_tensor_ops.add_self_loops
gather_first_node = graph_tensor_ops.gather_first_node
shuffle_features_globally = graph_tensor_ops.shuffle_features_globally
combine_values = graph_tensor_ops.combine_values
reorder_nodes = graph_tensor_ops.reorder_nodes
shuffle_nodes = graph_tensor_ops.shuffle_nodes
node_degree = graph_tensor_ops.node_degree
convert_to_line_graph = graph_tensor_ops.convert_to_line_graph

# Normalization operations.
softmax = normalization_ops.softmax
softmax_edges_per_node = normalization_ops.softmax_edges_per_node

# Readout.
validate_graph_tensor_spec_for_readout = readout.validate_graph_tensor_spec_for_readout
validate_graph_tensor_for_readout = readout.validate_graph_tensor_for_readout
add_readout_from_first_node = readout.add_readout_from_first_node
structured_readout = readout.structured_readout
structured_readout_into_feature = readout.structured_readout_into_feature
# DO NOT USE the obsolete aliases `readout_named*`.
readout_named = readout.structured_readout
readout_named_into_feature = readout.structured_readout_into_feature

# Schema conversion and I/O functions.
parse_schema = schema_utils.parse_schema
read_schema = schema_utils.read_schema
write_schema = schema_utils.write_schema
check_compatible_with_schema_pb = schema_utils.check_compatible_with_schema_pb
create_graph_spec_from_schema_pb = schema_utils.create_graph_spec_from_schema_pb
create_schema_pb_from_graph_spec = schema_utils.create_schema_pb_from_graph_spec
iter_sets = schema_utils.iter_sets
iter_features = schema_utils.iter_features

# Schema validation.
ValidationError = schema_validation.ValidationError
validate_schema = schema_validation.validate_schema
check_required_features = schema_validation.check_required_features
assert_constraints = schema_validation.assert_constraints

# Tensor Validation Utils
check_scalar_graph_tensor = graph_tensor.check_scalar_graph_tensor
check_homogeneous_graph_tensor = graph_tensor.check_homogeneous_graph_tensor
get_homogeneous_node_and_edge_set_name = graph_tensor.get_homogeneous_node_and_edge_set_name
get_aux_type_prefix = graph_tensor.get_aux_type_prefix

# Type check helpers
is_dense_tensor = tensor_utils.is_dense_tensor
is_ragged_tensor = tensor_utils.is_ragged_tensor
is_graph_tensor = graph_tensor_ops.is_graph_tensor

# Prune imported module symbols so they're not accessible implicitly,
# except those meant to be used as subpackages, like tfgnn.keras.*.
# Please use the same order as for the import statements at the top.
del version
del adjacency
del batching_utils
del broadcast_ops
del graph_constants
del graph_tensor
del graph_tensor_encode
del graph_tensor_io
del graph_tensor_ops
del graph_tensor_pprint
del graph_tensor_random
del normalization_ops
del padding_ops
del pool_ops
del preprocessing_common
del readout
del schema_utils
del schema_validation
del tag_utils
del tensor_utils
del graph_schema

