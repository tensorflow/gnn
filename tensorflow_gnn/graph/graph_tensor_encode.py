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
"""Encoder for an eager instance of GraphTensor to tf.train.Example protos.

The code in this module may be used to produce streams of tf.train.Example proto
messages that will parse with `tfgnn.parse_example()`.
"""

import functools
from typing import Optional

import tensorflow as tf

from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as gc
from tensorflow_gnn.graph import graph_piece as gp
from tensorflow_gnn.graph import graph_tensor as gt


def write_example(graph: gt.GraphTensor,
                  prefix: Optional[str] = None) -> tf.train.Example:
  """Encode an eager `GraphTensor` to a tf.train.Example proto.

  This routine can be used to create a stream of training data for GNNs from a
  Python job. Create instances of scalar `GraphTensor` with a single graph
  component and call this to write them out. It is recommended to always
  accompany serialized graph tensor tensorflow examples by their graph schema
  file (see `tfgnn.create_schema_pb_from_graph_spec()`). TF-GNN library provides
  `tfgnn.check_compatible_with_schema_pb()` to check that graph tensor instances
  (or their specs) are compatible with the graph schema. The graph tensors
  materialized in this way will be parseable by `tfgnn.parse_example()`
  (using the spec deserialized from the schema) and have the same contents
  (up to the choice of indices_dtype).

  Args:
    graph: An eager instance of `GraphTensor` to write out.
    prefix: An optional prefix string over all the features. You may use
      this if you are encoding other data in the same protocol buffer.
  Returns:
    A reference to `result`, if provided, or a to a freshly created instance
    of `tf.train.Example`.
  """
  result = tf.train.Example()
  if prefix is None:
    prefix = ''
  _encode(graph, prefix, result)
  return result


@functools.singledispatch
def _encode(piece: gp.GraphPieceBase, prefix: str,
            result: tf.train.Example):
  """Recursively flattens GraphPieceSpec into map of its field specs.

  Args:
    piece: Subclass of the GraphPieceBase.
    prefix: string prefix to append to all result field specs name.
    result: A tf.train.Example proto message which gets mutated directly.
  """
  raise NotImplementedError(
      f'Encoding is not defined for {type(piece).__name__}')


@_encode.register(gt.GraphTensor)
def _(graph: gt.GraphTensor, prefix: str, result: tf.train.Example):
  _encode(graph.context, f'{prefix}{gc.CONTEXT}/', result)
  for set_name, node_set in sorted(graph.node_sets.items()):
    _encode(node_set, f'{prefix}{gc.NODES}/{set_name}.', result)
  for set_name, edge_set in sorted(graph.edge_sets.items()):
    _encode(edge_set, f'{prefix}{gc.EDGES}/{set_name}.', result)


@_encode.register(gt.Context)
def _(context: gt.Context, prefix: str, result: tf.train.Example):
  _encode_features(context.features, prefix, result)


@_encode.register(gt.NodeSet)
def _(node_set: gt.NodeSet, prefix: str, result: tf.train.Example):
  _copy_feature_values(node_set.sizes, f'{prefix}{gc.SIZE_NAME}', result)
  _encode_features(node_set.features, prefix, result)


@_encode.register(gt.EdgeSet)
def _(edge_set: gt.EdgeSet, prefix: str, result: tf.train.Example):
  _copy_feature_values(edge_set.sizes, f'{prefix}{gc.SIZE_NAME}', result)
  _encode_features(edge_set.features, prefix, result)
  _encode(edge_set.adjacency, prefix, result)


@_encode.register(adj.Adjacency)
def _(adjacency: adj.Adjacency, prefix: str, result: tf.train.Example):
  for name, values in [(f'{prefix}{gc.SOURCE_NAME}', adjacency.source),
                       (f'{prefix}{gc.TARGET_NAME}', adjacency.target)]:
    _copy_feature_values(values, name, result)


def _encode_features(features: gc.Fields, template: str,
                     result: tf.train.Example):
  """Encode features `features` by mutating `result`."""
  for fname, values in sorted(features.items()):
    _copy_feature_values(values, f'{template}{fname}', result)


def _copy_feature_values(values: gc.Field, fname: str,
                         result: tf.train.Example):
  """Copy the values of an eager tensor to a `Feature` object."""

  # Flatten out the tensor to a rank-1 array.
  flat_values = values
  if isinstance(flat_values, tf.RaggedTensor):
    flat_values = values.flat_values
  flat_values = tf.reshape(flat_values, [-1])
  array = flat_values.numpy()

  # Convert the values to the proper type and set them.
  feature = result.features.feature[fname]
  if flat_values.dtype is tf.int32:
    flat_values = tf.cast(flat_values, tf.int64)
    feature.int64_list.value.extend(array)
  elif flat_values.dtype is tf.int64:
    feature.int64_list.value.extend(array)
  elif flat_values.dtype is tf.float32:
    feature.float_list.value.extend(array)
  elif flat_values.dtype is tf.float64:
    flat_values = tf.cast(flat_values, tf.float32)
    feature.float_list.value.extend(array)
  elif flat_values.dtype is tf.string:
    feature.bytes_list.value.extend(array)
  else:
    raise ValueError(f'Invalid type for tf.Example: {flat_values}')

  # If the tensor has ragged dimensions, serialize those into features to be
  # parsed as partitions.
  if isinstance(values, tf.RaggedTensor):
    for i, (dim, row_lengths) in enumerate(
        zip(values.shape[1:], values.nested_row_lengths()), start=1):
      if dim is not None:
        continue
      feature = result.features.feature[f'{fname}.d{i}']
      feature.int64_list.value.extend(row_lengths.numpy())
