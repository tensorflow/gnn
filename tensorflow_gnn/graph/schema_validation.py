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
"""Graph schema validation routines."""

import re
from typing import List, Optional, Sequence

from absl import logging  # TODO(blais): Remove, see below.
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import readout
from tensorflow_gnn.graph import schema_utils as su
import tensorflow_gnn.proto.graph_schema_pb2 as schema_pb2


# The supported data types. Note that these are currently limited to the ones
# supported by `tensorflow.Example` but we can eventually extend the list by
# adding casting transformations, and supporting other data formats for
# encoding.
VALID_DTYPES = (tf.string, tf.int64, tf.float32)


class ValidationError(ValueError):
  """A schema validation error.

  This exception is raised if in the course of validating the schema for
  correctness some errors are found.
  """


def validate_schema(
    schema: schema_pb2.GraphSchema,
    readout_node_sets: Optional[Sequence[const.NodeSetName]] = None,
) -> List[Exception]:
  """Validates the correctness of a graph schema instance.

  `GraphSchema` configuration messages are created by users in order to describe
  the topology of a graph. This function checks various aspects of the schema
  for correctness, e.g. prevents usage of reserved feature names, ensures given
  shapes are fully-defined, ensures set name references are found, etc.

  Args:
    schema: An instance of the graph schema.
    readout_node_sets: By default, this function checks the "_readout" node set,
      if present, if it meets the requirements of `tfgnn.structured_readout()`.
      That's sufficient for most cases. Optionally, you can pass a list of
      `readout_node_set` names to (a) require their presence and (b) check them.

  Returns:
    A list of exceptions describing optional warnings.
    Render those to your favorite stream (or ignore).

  Raises:
    ValidationError: If a validation check fails.
  """
  _validate_schema_feature_dtypes(schema)
  _validate_schema_shapes(schema)
  _validate_schema_descriptions(schema)
  _validate_schema_reserved_feature_names(schema)
  _validate_schema_context_references(schema)
  _validate_schema_node_set_references(schema)
  _validate_schema_readout(schema, readout_node_sets)
  return _warn_schema_scalar_shapes(schema)


def check_required_features(requirements: schema_pb2.GraphSchema,
                            actual: schema_pb2.GraphSchema):
  """Checks the requirements of a given schema against another.

  This function is used to enable the specification of required features to a
  function. A function accepting a `GraphTensor` instance can this way document
  what features it is expecting to find on it. The function accepts two schemas:
  a `requirements` schema which describes what the function will attempt to
  fetch and use on the `GraphTensor`, and an `actual` schema instance, which is
  the schema describing the dataset. You can use this in your model code to
  ensure that a dataset contains all the expected node sets, edge sets and
  features that the model uses.

  Note that a dimension with a size of `0` in a feature from the `requirements`
  schema is interpreted specially: it means "accept any value for this
  dimension." The special value `-1` is still used to represent a ragged
  dimension.

  (Finally, note that this function predates the existence of `GraphTensorSpec`,
  which is a runtime descriptor for a `GraphTensor`. We may eventually perovide
  an equivalent construct using the `GraphTensorSpec.)

  Args:
    requirements: An instance of a GraphSchema object, with optional shapes.
    actual: The instance of actual schema to check is a matching superset
      of the required schema.

  Raises:
    ValidationError: If the given schema does not fulfill the requirements.
  """
  # Create maps of the required and provided features.
  def build_schema_map(schema_):
    mapping = {}
    for (set_type, set_name, feature_name,
         feature) in su.iter_features(schema_):
      key = (set_type, set_name, feature_name)
      mapping[key] = feature
    return mapping
  required = build_schema_map(requirements)
  given = build_schema_map(actual)
  for key, required_feature in required.items():
    set_type, set_name, feature_name = key
    try:
      given_feature = given[key]
    except KeyError:
      raise ValidationError(
          "{} feature '{}' from set '{}' is missing from given schema".format(
              set_type.capitalize(), feature_name, set_name))
    else:
      if required_feature.HasField("dtype") and (
          required_feature.dtype != given_feature.dtype):
        raise ValidationError(
            "{} feature '{}' from set '{}' has invalid type: {}".format(
                set_type.capitalize(), feature_name, set_name,
                given_feature.dtype))
      if required_feature.HasField("shape"):
        if len(given_feature.shape.dim) != len(required_feature.shape.dim):
          raise ValidationError(
              "{} feature '{}' from set '{}' has invalid shape: {}".format(
                  set_type.capitalize(), feature_name, set_name,
                  given_feature.shape))
        for required_dim, given_dim in zip(required_feature.shape.dim,
                                           given_feature.shape.dim):
          if required_dim.size == 0:  # Accept any dimension.
            continue
          elif given_dim.size != required_dim.size:
            raise ValidationError(
                "{} feature '{}' from set '{}' has invalid shape: {}".format(
                    set_type.capitalize(), feature_name, set_name,
                    given_feature.shape))


def _validate_schema_feature_dtypes(schema: schema_pb2.GraphSchema):
  """Verify that dtypes are set and from our list of supported types."""
  for set_type, set_name, feature_name, feature in su.iter_features(schema):
    if not feature.HasField("dtype"):
      raise ValidationError(
          "Missing 'dtype' field on {} set '{}' feature '{}'".format(
              set_type, set_name, feature_name))
    if feature.dtype not in {dtype.as_datatype_enum
                             for dtype in VALID_DTYPES}:
      raise ValidationError(
          ("Invalid 'dtype' field {} on {} set '{}' feature '{}': {}; "
           "valid types include: {}").format(
               feature.dtype, set_type, set_name, feature_name, feature.dtype,
               ", ".join(map(str, VALID_DTYPES))))


def _validate_schema_shapes(schema: schema_pb2.GraphSchema):
  """Check for the validity of shape protos."""
  for set_type, set_name, feature_name, feature in su.iter_features(schema):
    if feature.shape.unknown_rank:
      raise ValidationError(
          "Shapes must have a known rank; on {} set '{}' feature '{}'".format(
              set_type, set_name, feature_name))


def _warn_schema_scalar_shapes(schema: schema_pb2.GraphSchema):
  """Return warnings on unnecessary shapes of size 1. This is a common error.

  Note that strictly speaking this should parse fine, the problem is that
  clients will inevitably configure shapes of [1] where scalar shapes would be
  sufficient. This check is there to nudge them in the right direction.

  Args:
    schema: A GraphSchema instance to validate.
  Returns:
    A list of ValidationError warnings to issue conditionally.
  """
  warnings = []
  for set_type, set_name, feature_name, feature in su.iter_features(schema):
    if len(feature.shape.dim) == 1 and feature.shape.dim[0].size == 1:
      warnings.append(ValidationError(
          "Unnecessary shape of [1] in {} set '{}' / '{}'; use scalar feature "
          "instead (i.e., specify an empty shape proto).".format(
              set_type, set_name, feature_name)))
  return warnings


def _validate_schema_descriptions(schema: schema_pb2.GraphSchema):
  """Verify that the descriptions aren't set on the shapes' .name fields."""
  # This seems to be a common error.
  name_fields = []
  for set_type, set_name, feature_name, feature in su.iter_features(schema):
    if feature.HasField("description"):
      continue
    for dim in feature.shape.dim:
      if dim.name:
        name_fields.append((set_type, set_name, feature_name))
  if name_fields:
    field_names = ",".join([str(ntuple) for ntuple in name_fields])
    raise ValidationError(
        "The following features are incorrectly locating the description on "
        "the shape dimensions 'name' field: {}; use the 'description' field of "
        "the feature instead".format(field_names))


def _validate_schema_reserved_feature_names(schema: schema_pb2.GraphSchema):
  """Check that reserved feature names aren't being used as explicit features."""
  node_set_dicts = [("nodes", name, node_set.features)
                    for name, node_set in schema.node_sets.items()]
  edge_set_dicts = [("edges", name, edge_set.features)
                    for name, edge_set in schema.edge_sets.items()]
  for set_type, set_name, feature_dict in node_set_dicts + edge_set_dicts:
    if const.SIZE_NAME in feature_dict:
      raise ValidationError(
          "Feature '{}' from {} set '{}' is reserved".format(
              const.SIZE_NAME, set_type, set_name))
  for set_type, set_name, feature_dict in edge_set_dicts:
    for name in const.SOURCE_NAME, const.TARGET_NAME:
      # Invalidate reserved feature names.
      if name in feature_dict:
        raise ValidationError(
            "Feature '{}' from {} set '{}' is reserved".format(
                name, set_type, set_name))

  # TODO(blais): Make this compulsory after we remove the hardcoded
  # feature names from the sampler.
  for set_type, set_name, feature_name, feature in su.iter_features(schema):
    if re.fullmatch(const.RESERVED_FEATURE_NAME_PATTERN, feature_name):
      logging.error("Invalid %s feature name '%s' on set '%s': reserved names "
                    "are not allowed", set_type, feature_name, set_name)


def _validate_schema_context_references(schema: schema_pb2.GraphSchema):
  """Verify the cross-references to context features from node and edge sets."""
  for set_name, node_set in schema.node_sets.items():
    for feature in node_set.context:
      if feature not in schema.context.features:
        raise ValidationError("Context feature '{}' does not exist "
                              "(from node set '{}')".format(feature, set_name))
  for set_name, edge_set in schema.edge_sets.items():
    for feature in edge_set.context:
      if feature not in schema.context.features:
        raise ValidationError("Context feature '{}' does not exist "
                              "(from edge set '{}')".format(feature, set_name))


def _validate_schema_node_set_references(schema: schema_pb2.GraphSchema):
  """Verify the source and target set references from the edge sets."""
  for set_name, edge_set in schema.edge_sets.items():

    if not set_name:
      raise ValidationError("Edge set name cannot be empty.")

    if not edge_set.source:
      raise ValidationError(f"Edge set {set_name} must specify a `source`.")

    if not edge_set.target:
      raise ValidationError(f"Edge set {set_name} must specify a `target`.")

    for feature_name in edge_set.source, edge_set.target:
      if feature_name not in schema.node_sets:
        raise ValidationError(
            "Edge set '{}' referencing unknown node set '{}'".format(
                set_name, feature_name))


def _validate_schema_readout(
    schema: schema_pb2.GraphSchema,
    readout_node_sets: Optional[Sequence[const.NodeSetName]] = None,
) -> None:
  """Applies validate_graph_tensor_spec_for_readout()."""
  if readout_node_sets is None:
    if "_readout" not in schema.node_sets:
      return
    readout_node_sets = ["_readout"]

  spec = su.create_graph_spec_from_schema_pb(schema)
  for readout_node_set in readout_node_sets:
    try:
      readout.validate_graph_tensor_spec_for_readout(
          spec, readout_node_set=readout_node_set)
    except (ValueError, KeyError) as e:
      raise ValidationError(
          "tfgnn.validate_graph_tensor_spec_for_readout(..., "
          f"readout_node_set='{readout_node_set}') failed: {e}") from e


# TODO(blais): This code could eventually be folded into the various
# constructors of `GraphTensor` pieces.
def assert_constraints(graph: gt.GraphTensor) -> tf.Operation:
  """Validate the shape constaints of a graph's features at runtime.

  This code returns a TensorFlow op with debugging assertions that ensure the
  parsed data has valid shape constraints for a graph. This can be instantiated
  in your TensorFlow graph while debugging if you believe that your data may be
  incorrectly shaped, or simply applied to a manually produced dataset to ensure
  that those constraints have been applied correctly.

  Args:
    graph: An instance of a `GraphTensor`.
  Returns:
    A list of check operations.
  """
  return tf.group(
      _assert_constraints_feature_shape_prefix(graph),
      _assert_constraints_edge_shapes(graph),
      _assert_constraints_edge_indices_range(graph),
  )


def _assert_constraints_feature_shape_prefix(
    graph: gt.GraphTensor) -> tf.Operation:
  """Validates the number of nodes or edges of feature tensors."""
  with tf.name_scope("constraints_feature_shape_prefix"):
    checks = []
    for set_type, set_dict in [("node", graph.node_sets),
                               ("edge", graph.edge_sets)]:
      for set_name, feature_set in set_dict.items():
        sizes = feature_set.sizes

        # Check the rank is at least 1.
        checks.append(tf.debugging.assert_rank_at_least(sizes, 1))
        rank = tf.rank(sizes)

        for feature_name, tensor in feature_set.features.items():
          # Check that each tensor has greater or equal rank to the parent
          # piece.
          checks.append(tf.debugging.assert_greater_equal(
              tf.rank(tensor), rank,
              "Rank too small for {} feature '{}/{}'".format(
                  set_type, set_name, feature_name)))

          # Check the prefix shape of the tensor matches.
          checks.append(tf.debugging.assert_equal(
              tensor.shape[:rank], sizes,
              "Invalid prefix shape for {} feature: {}/{}".format(
                  set_type, set_name, feature_name)))

    return tf.group(*checks)


def _assert_constraints_edge_indices_range(
    graph: gt.GraphTensor) -> tf.Operation:
  """Validates that edge indices are within the bounds of node set sizes."""

  with tf.name_scope("constraints_edge_indices_range"):
    checks = []
    for set_name, edge_set in graph.edge_sets.items():
      adjacency = edge_set.adjacency
      if not issubclass(type(adjacency), adj.HyperAdjacency):
        raise ValueError(f"Adjacency type for constraints assertions must be "
                         f"HyperAdjacency: {adjacency}")

      for tag, (node_set_name, indices) in sorted(adjacency
                                                  .get_indices_dict().items()):
        # Check that the indices are positive.
        flat_indices = (indices.flat_values
                        if isinstance(indices, tf.RaggedTensor)
                        else indices)
        checks.append(tf.debugging.Assert(
            tf.math.reduce_all(
                tf.math.greater_equal(indices,
                                      tf.constant(0, dtype=indices.dtype))),
            ["Index underflow",
             "edges/{} {} indices:".format(set_name, tag), flat_indices],
            name="check_indices_underflow", summarize=-1))

        # Check the indices are smaller than the node tensor sizes.
        sizes = graph.node_sets[node_set_name].sizes
        checks.append(tf.debugging.Assert(
            tf.math.reduce_all(
                tf.math.less(indices, tf.expand_dims(sizes, axis=-1))),
            ["Index overflow",
             "edges/{} {} indices:".format(set_name, tag), flat_indices,
             "nodes/{} {}:".format(node_set_name, "size"), sizes],
            name="check_indices_overflow", summarize=-1))
    return tf.group(*checks)


def _assert_constraints_edge_shapes(graph: gt.GraphTensor) -> tf.Operation:
  """Validates edge shapes and that they contain a scalar index per node."""
  with tf.name_scope("constraints_edge_indices_range"):
    checks = []
    for set_name, edge_set in graph.edge_sets.items():
      adjacency = edge_set.adjacency
      if not issubclass(type(adjacency), adj.HyperAdjacency):
        raise ValueError(f"Adjacency type for constraints assertions must be "
                         f"HyperAdjacency: {adjacency}")

      for tag, (_, indices) in sorted(adjacency.get_indices_dict().items()):
        # Check the shape of the edge indices matches the size, and that the
        # shape is scalar on the indices.
        checks.append(tf.debugging.assert_equal(
            indices.shape, edge_set.sizes,
            "Invalid shape for edge indices: {}/{}".format(set_name, tag)))
    return tf.group(*checks)
