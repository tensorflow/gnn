"""Constant strings used throughout the package."""

import re
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import tensorflow as tf

# Formatting string for feature names.
CONTEXT_FMT = '{stype}/{fname}'
FEATURE_FMT = '{stype}/{sname}.{fname}'

# Names of set types.
CONTEXT = 'context'
NODES = 'nodes'
EDGES = 'edges'
SetType = str  # A value of CONTEXT, NODES or EDGES.

# Integer tags for selecting the specific endpoints of an edge in a
# HyperAdjacency.
SOURCE = 0
TARGET = 1

# Name of the implicitly-defined set size feature, source and target index
# features, for serialization.
SIZE_NAME = '#size'
SOURCE_NAME = '#source'
TARGET_NAME = '#target'
RESERVED_FEATURES = frozenset({SIZE_NAME, SOURCE_NAME, TARGET_NAME})

# The conventional feature name for the hidden state (neuron activations) of
# an edge set, a node set or the context. Not special in GraphTensor, but used
# in some modeling libraries on top if explicit names are not needed.
DEFAULT_STATE_NAME = 'hidden_state'

# The pattern of feature names that are not allowed on a graph tensor and
# schema.
RESERVED_REGEX = re.compile(r'#.*')

# The internal metadata key prefix to use for hyper adjacency.
INDEX_KEY_PREFIX = '#index.'

# All edges in an EdgeSet have the same number of incident nodes. Each incident
# node is identified by a unique tag, a small integer. For ordinary graphs,
# these are SOURCE and TARGET, by convention. Other or additional
# numbers can be used, e.g., for hypergraphs.
IncidentNodeTag = int

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
