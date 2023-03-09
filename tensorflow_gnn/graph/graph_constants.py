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
"""Constant strings used throughout the package."""

from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import tensorflow as tf

# TODO(b/245729353) Migrate string constants to enums where appropriate.
# Formatting string for feature names.
CONTEXT_FMT = '{stype}/{fname}'
FEATURE_FMT = '{stype}/{sname}.{fname}'

# Names of set types.
CONTEXT = 'context'
NODES = 'nodes'
EDGES = 'edges'
SetType = str  # A value of CONTEXT, NODES or EDGES.

# Name of the implicitly-defined set size feature, source and target index
# features, for serialization.
SIZE_NAME = '#size'
SOURCE_NAME = '#source'
TARGET_NAME = '#target'
RESERVED_FEATURES = frozenset({SIZE_NAME, SOURCE_NAME, TARGET_NAME})
# The pattern of feature names (present and future) that are not allowed on a
# graph tensor and schema, for use with re.fullmatch(pattern, feature_name).
RESERVED_FEATURE_NAME_PATTERN = r'#.*'

# The conventional feature name for the hidden state (neuron activations) of
# an edge set, a node set or the context. Not special in GraphTensor, but used
# in some modeling libraries on top if explicit names are not needed.
HIDDEN_STATE = 'hidden_state'

# The internal metadata key prefix to use for hyper adjacency.
INDEX_KEY_PREFIX = '#index.'

# All edges in an EdgeSet have the same number of incident nodes. Each incident
# node is identified by a unique tag, a small integer. For ordinary graphs,
# these are SOURCE and TARGET, by convention. Other or additional
# numbers can be used, e.g., for hypergraphs.
IncidentNodeTag = int

# Integer tags for selecting the specific endpoints of an edge in a
# HyperAdjacency.
SOURCE: IncidentNodeTag = 0
TARGET: IncidentNodeTag = 1

# Generic pool and broadcast ops allow the special case tfgnn.CONTEXT (a str)
# in addition to pooling from or broadcasting to tfgnn.SOURCE and tfgnn.TARGET.
IncidentNodeOrContextTag = Union[IncidentNodeTag, str]

FieldName = str  # Name of a context, node set or edge set field
NodeSetName = str  # Name of a NodeSet within a GraphTensor
EdgeSetName = str  # Name of an EdgeSet within a GraphTensor
SetName = str  # A NodeSetName or EdgeSetName

FieldNameOrNames = Union[FieldName, Sequence[FieldName]]

ShapeLike = Union[tf.TensorShape, Tuple[Optional[int], ...],
                  List[Optional[int]]]
Field = Union[tf.Tensor, tf.RaggedTensor]
FieldSpec = Union[tf.TensorSpec, tf.RaggedTensorSpec]

Fields = Mapping[FieldName, Field]
FieldsSpec = Mapping[FieldName, FieldSpec]

FieldOrFields = Union[Field, Fields]
# An arbitrarily deep nest of fields. Pytype cannot express this.
FieldsNest = Union[Field, List[Any], Tuple[Any], Mapping[str, Any]]

# If set, enables validation for objects contructed within the library. This
# flag does not interfere with validation flags controlled by user. It is used
# to better control library self-consistency.
#
# TODO(aferludin): disable in prod as those checks may be expensive.
validate_internal_results = True

# Default representation type for indices and size integers.
# Can be either tf.int32 or tf.int64.
#
# IMPORTANT: On TPUs tf.int64 is not implemented.
default_indices_dtype = tf.int32

# DEPRECATED

# An older name used before tensorflow_gnn 0.2.
DEFAULT_STATE_NAME = HIDDEN_STATE
