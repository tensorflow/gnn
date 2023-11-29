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

```
import tensorflow_gnn as tfgnn
```
"""
# pylint: disable=line-too-long

from tensorflow_gnn import keras  # Exposed as submodule. pylint: disable=unused-import
from tensorflow_gnn.graph import adjacency
from tensorflow_gnn.graph import graph_constants
from tensorflow_gnn.graph import graph_tensor
from tensorflow_gnn.graph import tag_utils
from tensorflow_gnn.utils import api_utils

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

# Global state flags that controls GraphPieces checks.
enable_graph_tensor_validation = graph_constants.enable_graph_tensor_validation
enable_graph_tensor_validation_at_runtime = (
    graph_constants.enable_graph_tensor_validation_at_runtime
)
disable_graph_tensor_validation = (
    graph_constants.disable_graph_tensor_validation
)
disable_graph_tensor_validation_at_runtime = (
    graph_constants.disable_graph_tensor_validation_at_runtime
)

# Remove all names added by module imports, unless explicitly allowed here.
api_utils.remove_submodules_except(
    __name__,
    [
        "keras",
    ],
)
# LINT.ThenChange(api_def/tfgnn-symbols.txt)
