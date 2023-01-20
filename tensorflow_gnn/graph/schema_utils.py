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
"""Misc graph tensor utilities."""

from typing import Any, Iterator, Mapping, Optional, Text, Tuple, Union

import tensorflow as tf
from tensorflow_gnn.graph import adjacency
from tensorflow_gnn.graph import graph_constants as gc
from tensorflow_gnn.graph import graph_tensor as gt
import tensorflow_gnn.proto.graph_schema_pb2 as schema_pb2

from google.protobuf import text_format


def parse_schema(schema_text: Union[bytes, str]) -> schema_pb2.GraphSchema:
  """Parse a schema from text-formatted protos.

  Args:
    schema_text: A string containing a text-formatted protocol buffer rendition
      of a `GraphSchema` message.

  Returns:
    A `GraphSchema` instance.
  """
  return text_format.Parse(schema_text, schema_pb2.GraphSchema())


def read_schema(filename: str) -> schema_pb2.GraphSchema:
  """Read a proto schema from a file with text-formatted contents.

  Args:
    filename: A string, the path to a file containing a text-formatted protocol
      buffer rendition of a `GraphSchema` message.

  Returns:
    A `GraphSchema` instance.
  """
  with tf.io.gfile.GFile(filename) as infile:
    return text_format.Parse(infile.read(), schema_pb2.GraphSchema())


def write_schema(schema: schema_pb2.GraphSchema, filename: str):
  """Write a `GraphSchema` to a text-formatted proto file.

  Args:
    schema: A `GraphSchema` instance to write out.
    filename: A string, the path to a file to render a text-formatted rendition
      of the `GraphSchema` message to.
  """
  with tf.io.gfile.GFile(filename, 'w') as schema_file:
    schema_file.write(text_format.MessageToString(schema))


def create_graph_spec_from_schema_pb(
    schema: schema_pb2.GraphSchema,
    indices_dtype: tf.dtypes.DType = gc.default_indices_dtype
) -> gt.GraphTensorSpec:
  """Converts a graph schema proto message to a scalar GraphTensorSpec.

  A `GraphSchema` message contains shape information in a serializable format.
  The `GraphTensorSpec` is a runtime object fulfilling the type spec
  requirements, that accompanies each `GraphTensor` instance and fulfills much
  of the same goal. This function converts the proto to the corresponding type
  spec.

  It is guranteed that the output graph spec is compatible with the input graph
  schema (as `tfgnn.check_compatible_with_schema_pb()`.)

  Args:
    schema: An instance of the graph schema proto message.
    indices_dtype: A `tf.dtypes.DType` for GraphTensor edge set source and
      target indices, node and edge sets sizes.

  Returns:
    A `GraphTensorSpec` specification for the scalar graph tensor (of rank 0).
  """

  size_value_spec = tf.TensorSpec(shape=(1,), dtype=indices_dtype)
  index_spec = tf.TensorSpec(shape=(None,), dtype=indices_dtype)

  context_spec = gt.ContextSpec.from_field_specs(
      features_spec=_create_fields_spec_from_schema(schema.context.features, 1,
                                                    indices_dtype),
      shape=tf.TensorShape([]),
      indices_dtype=indices_dtype)

  nodes_spec = {}
  for set_name, node in schema.node_sets.items():
    nodes_spec[set_name] = gt.NodeSetSpec.from_field_specs(
        sizes_spec=size_value_spec,
        features_spec=_create_fields_spec_from_schema(node.features, None,
                                                      indices_dtype))

  edges_spec = {}
  for set_name, edge in schema.edge_sets.items():
    edges_spec[set_name] = gt.EdgeSetSpec.from_field_specs(
        sizes_spec=size_value_spec,
        adjacency_spec=adjacency.AdjacencySpec.from_incident_node_sets(
            source_node_set=edge.source,
            target_node_set=edge.target,
            index_spec=index_spec),
        features_spec=_create_fields_spec_from_schema(edge.features, None,
                                                      indices_dtype))

  return gt.GraphTensorSpec.from_piece_specs(
      context_spec=context_spec,
      node_sets_spec=nodes_spec,
      edge_sets_spec=edges_spec)


def create_schema_pb_from_graph_spec(
    graph: Union[gt.GraphTensor, gt.GraphTensorSpec]) -> schema_pb2.GraphSchema:
  """Converts scalar GraphTensorSpec to a graph schema proto message.

  The result graph schema containts entires for all graph pieces. The features
  proto field contains type (`dtype`) and shape (`shape`) information for all
  features. For edge sets their `source` and `target` fields are populated. All
  other fields are left unset. (Callers can set them separately before writing
  out the schema.)

  It is guranteed that the input graph is compatible with the output graph
  schema (as `tfgnn.check_compatible_with_schema_pb()`.)

  Args:
    graph: The scalar graph tensor or its spec with single graph component.

  Returns:
    An instance of the graph schema proto message.

  Raises:
    ValueError: if graph has multiple graph components or rank > 0.
    ValueError: if adjacency types is not an instance of `fgnn.Adjacency`.
  """
  graph_spec = gt.get_graph_tensor_spec(graph)
  gt.check_scalar_singleton_graph_tensor(graph_spec,
                                         'create_schema_pb_from_graph()')

  def _to_feature(spec: gc.FieldSpec) -> schema_pb2.Feature:
    result = schema_pb2.Feature()
    result.dtype = spec.dtype.as_datatype_enum
    feature_shape = spec.shape[1:]  # Drop num_items dimension.
    if feature_shape.rank > 0:  # For empty, use default.
      result.shape.CopyFrom(feature_shape.as_proto())
    return result

  def _add_features_spec(features_spec: gc.FieldsSpec,
                         target: Mapping[str, schema_pb2.Feature]) -> None:
    for name, spec in features_spec.items():
      target[name].MergeFrom(_to_feature(spec))

  result = schema_pb2.GraphSchema()

  _add_features_spec(graph_spec.context_spec.features_spec,
                     result.context.features)

  for name, node_set_spec in graph_spec.node_sets_spec.items():
    node_set_schema = result.node_sets[name]
    _add_features_spec(node_set_spec.features_spec, node_set_schema.features)

  for name, edge_set_spec in graph_spec.edge_sets_spec.items():
    edge_set_schema = result.edge_sets[name]
    _add_features_spec(edge_set_spec.features_spec, edge_set_schema.features)
    adjacency_spec = edge_set_spec.adjacency_spec
    if not isinstance(adjacency_spec, adjacency.AdjacencySpec):
      raise ValueError(f'Adjacency type `{adjacency_spec.value_type.__name__}`'
                       f' of the edge set \'{name}\' is not supported.'
                       ' Expected an instance of `tfgnn.Adjacency`.')

    edge_set_schema.source = edge_set_spec.adjacency_spec.source_name
    edge_set_schema.target = edge_set_spec.adjacency_spec.target_name

  return result


def check_compatible_with_schema_pb(graph: Union[gt.GraphTensor,
                                                 gt.GraphTensorSpec],
                                    schema: schema_pb2.GraphSchema) -> None:
  """Checks that the given spec or value is compatible with the graph schema.

  The `graph` is compatible with the `schema` if

  * it is scalar (rank=0) graph tensor;
  * has single graph component;
  * has matching sets of nodes and edges;
  * has matching sets of features on all node sets, edge sets, and the context,
    and their types and shapes are compatible;
  * all adjacencies are of type `tfgnn.Adjacency`.

  Args:
    graph: The graph tensor or graph tensor spec.
    schema: The graph schema.

  Raises:
    ValueError: if `spec_or_value` is not represented by the graph schema.
  """
  gt.check_scalar_singleton_graph_tensor(graph,
                                         'check_compatible_with_schema_pb()')

  actual_spec = gt.get_graph_tensor_spec(graph).relax(
      num_edges=True, num_nodes=True)
  expected_spec = create_graph_spec_from_schema_pb(schema, graph.indices_dtype)

  # We can require an exact equality here, because graph pieces must be equal,
  # their feature specs must be equal, and that means:
  # * dtypes are exactly equal (for indices_dtype by construction);
  # * shapes are exactly equal without special-casing None, because fields have
  #    - no graph dimensions,
  #    - an items dimension as outermost dimension that is set exactly to None,
  #    - feature dimensions that, per GraphTensor requirements, are None if and
  #      only if it is a ragged dimension of a RaggedTensor.
  if actual_spec != expected_spec:
    raise ValueError('Graph is not compatible with the graph schema.'
                     f' Graph spec: {actual_spec},'
                     f' spec imposed by schema: {expected_spec}.')


def _is_ragged_dim(dim) -> bool:
  if dim.size < -1:
    raise ValueError(f'Dimension size must be >= -1, got {dim}.')
  return dim.size == -1


def _get_ragged_rank(feature: schema_pb2.Feature) -> int:
  return sum(_is_ragged_dim(dim) for dim in feature.shape.dim)


def _is_ragged(feature: schema_pb2.Feature) -> int:
  return any(_is_ragged_dim(dim) for dim in feature.shape.dim)


def _create_fields_spec_from_schema(
    features_schema: Any, dim0: Optional[int],
    indices_dtype: tf.dtypes.DType) -> gc.FieldsSpec:
  """Convers a features schema to fields specification."""

  def _get_shape(feature: schema_pb2.Feature) -> tf.TensorShape:
    dim_fn = lambda dim: (None if dim.size == -1 else dim.size)
    dims = [dim_fn(dim) for dim in feature.shape.dim]
    dims = [dim0] + dims
    return tf.TensorShape(dims)

  result = {}
  for fname, feature in features_schema.items():
    shape = _get_shape(feature)
    dtype = tf.dtypes.as_dtype(feature.dtype)
    if _is_ragged(feature):
      value_spec = tf.RaggedTensorSpec(
          shape=shape,
          dtype=dtype,
          ragged_rank=_get_ragged_rank(feature),
          row_splits_dtype=indices_dtype)
    else:
      value_spec = tf.TensorSpec(shape=shape, dtype=dtype)

    result[fname] = value_spec

  return result


def iter_sets(
    schema: Union[schema_pb2.GraphSchema, gt.GraphTensor]
) -> Iterator[Tuple[str, str, Any]]:
  """Utility function to iterate over all the sets present in a graph schema.

  This function iterates over the context set, each of the node sets, and
  finally each of the edge sets.

  Args:
    schema: An instance of a `GraphSchema` proto message.

  Yields:
    Triplets of (set-type, set-name, features) where

    * set-type: A type of set, which is either of "context", "nodes" or "edges".
    * set-name: A string, the name of the set.
    * features: A dict of feature-name to feature-value.
  """
  if (not isinstance(schema, schema_pb2.GraphSchema) or
      schema.HasField('context')):
    yield (gc.CONTEXT, '', schema.context)
  for set_name, set_ in schema.node_sets.items():
    yield (gc.NODES, set_name, set_)
  for set_name, set_ in schema.edge_sets.items():
    yield (gc.EDGES, set_name, set_)


def iter_features(
    schema: Union[schema_pb2.GraphSchema, gt.GraphTensor]
) -> Iterator[Tuple[Text, Text, Text, Union[schema_pb2.Feature, gt.Field]]]:
  """Utility function to iterate over the features of a graph schema.

  This function iterates over all the feature values of each of the context set,
  each of the node sets, and each of the edge sets.

  Args:
    schema: An instance of a `GraphSchema` proto message.

  Yields:
    Triplets of (set-type, set-name, feature-name, feature-value) where

    * set-type: A type of set, which is either of "context", "nodes" or "edges".
    * set-name: A string, the name of the set.
    * feature-name: A string, the name of the feature in the set.
    * feature-value: A potentially ragged tensor (either a `tf.Tensor` or a
      `tf.RaggedTensor`).
  """
  if schema.HasField('context'):
    for feature_name, feature in schema.context.features.items():
      yield (gc.CONTEXT, '', feature_name, feature)
  for set_name, set_ in schema.node_sets.items():
    for feature_name, feature in set_.features.items():
      yield (gc.NODES, set_name, feature_name, feature)
  for set_name, set_ in schema.edge_sets.items():
    for feature_name, feature in set_.features.items():
      yield (gc.EDGES, set_name, feature_name, feature)
