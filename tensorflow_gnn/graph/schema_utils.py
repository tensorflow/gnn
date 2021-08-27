"""Misc graph tensor utilities."""

from typing import Any, Iterator, Optional, Text, Tuple, Union

import tensorflow as tf
from tensorflow_gnn.graph import adjacency
from tensorflow_gnn.graph import graph_constants as gc
from tensorflow_gnn.graph import graph_tensor as gt
import tensorflow_gnn.proto.graph_schema_pb2 as schema_pb2

from google.protobuf import text_format


def parse_schema(schema_text: str) -> schema_pb2.GraphSchema:
  """Parse a schema from text-formatted protos.

  Args:
    schema_text: A string containing a text-formatted protocol buffer
      rendition of a `GraphSchema` message.

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
    filename: A string, the path to a file to render a text-formatted
      rendition of the `GraphSchema` message to.
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
