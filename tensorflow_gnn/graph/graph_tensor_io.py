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
"""Parsing of serialized TF Example(s) using GraphTensorSpec.

Example1. Parsing of multiple serialized examples.

```python
ds = tf.data.TFRecordDataset(data_path)
ds = ds.batch(batch_size, True)
ds = ds.map(functools.partial(tfgnn.parse_example, graph_tensor_spec))
```

Example2. Parsing of a single serialized example. The resulting dataset is
identical to Example1 (but this approach could be less efficient because it
does not leverage parsing batches of graphs at once).

```python
ds = tf.data.TFRecordDataset(data_path)
ds = ds.map(functools.partial(tfgnn.parse_single_example,
                              graph_tensor_spec))
ds = ds.batch(batch_size, True)
```
"""
import functools
from typing import Any, Dict, List, Optional, Tuple, Union

import tensorflow as tf

from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as gc
from tensorflow_gnn.graph import graph_piece as gp
from tensorflow_gnn.graph import graph_tensor as gt

# pytype: disable=attribute-error
IOFeature = Union[tf.io.FixedLenFeature, tf.io.RaggedFeature]
RaggedPartition = Union[tf.io.RaggedFeature.RowLengths,
                        tf.io.RaggedFeature.UniformRowLength]
AssertOp = Any
# pytype: enable=attribute-error


def parse_example(spec: gt.GraphTensorSpec,
                  serialized: tf.Tensor,
                  prefix: Optional[str] = None,
                  validate: bool = True) -> gt.GraphTensor:
  """Parses a batch of serialized Example protos into a single `GraphTensor`.

  We expect `serialized` to be a string tensor batched with `batch_size` many
  entries of individual `tf.train.Example` protos. Each example contains
  serialized graph tensors with the `spec` graph tensor type specification.

  See `tf.io.parse_example()`. In contrast to the regular tensor parsing routine
  which operates directly from `tf.io` feature configuration objects, this
  function accepts a type spec for a graph tensor and implements an encoding for
  all container tensors, including ragged tensors, from a batched sequence of
  `tf.train.Example` protocol buffer messages.

  The encoded examples shapes and features are expected to conform to the
  encoding defined by `get_io_spec()`. The `validate` flag exists to implement
  verifications of this encoding.

  Args:
    spec: A graph tensor type specification of a single serialized graph tensor
      value.
    serialized: A rank-1 dense tensor of strings with serialized Example protos,
      where each example is a graph tensor object with type corresponding `spec`
      type spec.
    prefix: An optional prefix string over all the features. You may use
      this if you are encoding other data in the same protocol buffer.
    validate: A boolean indicating whether or not to validate that the input
      values form a valid GraphTensor. Defaults to `True`.

  Returns:
    A graph tensor object with `spec.batch(serialized.shape[0])` type spec.
  """
  if serialized.shape.rank != 1:
    raise ValueError(
        f'`serialized` must have rank=1, got {serialized.shape.rank}')
  batch_size = serialized.shape[0]

  fields_io_spec = get_io_spec(spec, prefix, validate)
  flat_fields = tf.io.parse_example(serialized, fields_io_spec)

  spec = spec._batch(batch_size)  # pylint: disable=protected-access
  with tf.control_dependencies(
      _check_size_fields(spec, flat_fields) if validate else []):
    flat_fields = _restore_types(spec, prefix, flat_fields)
    return tf.identity(_unflatten_graph_fields(spec, flat_fields, prefix or ''))


def parse_single_example(spec: gt.GraphTensorSpec,
                         serialized: tf.Tensor,
                         prefix: Optional[str] = None,
                         validate: bool = True) -> gt.GraphTensor:
  """Parses a single serialized Example proto into a single `GraphTensor`.

  Like `parse_example()`, but for a single graph tensor.
  See `tfgnn.parse_example()` for reference.

  Args:
    spec: A graph tensor type specification.
    serialized: A scalar string tensor with a serialized Example proto
      containing a graph tensor object with the `spec` type spec.
    prefix: An optional prefix string over all the features. You may use
      this if you are encoding other data in the same protocol buffer.
    validate: A boolean indicating whether or not to validate that the input
      fields form a valid `GraphTensor`. Defaults to `True`.

  Returns:
    A graph tensor object with a matching type spec.
  """
  fields_io_spec = get_io_spec(spec, prefix, validate)
  flat_fields = tf.io.parse_single_example(serialized, fields_io_spec)
  with tf.control_dependencies(
      _check_size_fields(spec, flat_fields) if validate else None):
    flat_fields = _restore_types(spec, prefix, flat_fields)
    return tf.identity(_unflatten_graph_fields(spec, flat_fields, prefix or ''))


def get_io_spec(spec: gt.GraphTensorSpec,
                prefix: Optional[str] = None,
                validate: bool = True) -> Dict[str, IOFeature]:
  """Returns tf.io parsing features for `GraphTensorSpec` type spec.

  This function returns a mapping of `tf.train.Feature` names to configuration
  objects that can be used to parse instances of `tf.train.Example` (see
  https://www.tensorflow.org/api_docs/python/tf/io). The resulting mapping can
  be used with `tf.io.parse_example()` for reading the individual fields of a
  `GraphTensor` instance. This essentially forms our encoding of a `GraphTensor`
  to a `tf.train.Example` proto.

  (This is an internal function. You are not likely to be using this if you're
  decoding graph tensors. Instead, you should use the `tfgnn.parse_example()`
  routine directly, which handles this process for you.)

  Args:
    spec: A graph tensor type specification.
    prefix: An optional prefix string over all the features. You may use
      this if you are encoding other data in the same protocol buffer.
    validate: A boolean indicating whether or not to validate that the input
      fields form a valid `GraphTensor`. Defaults to `True`.

  Returns:
    A dict of `tf.train.Feature` name to feature configuration object, to be
    used in `tf.io.parse_example()`.
  """

  def get_io_ragged_partitions(
      fname: str, shape: tf.TensorShape) -> Tuple[RaggedPartition, ...]:
    partitions = []
    for i, dim in enumerate(shape.as_list()[1:], start=1):
      # pytype: disable=attribute-error
      if dim is None:
        partitions.append(tf.io.RaggedFeature.RowLengths(f'{fname}.d{i}'))
      else:
        partitions.append(tf.io.RaggedFeature.UniformRowLength(dim))
      # pytype: enable=attribute-error
    return tuple(partitions)

  def get_io_feature(fname: str, value_spec: gt.FieldSpec) -> IOFeature:
    io_dtype = _get_io_type(value_spec.dtype)
    if isinstance(value_spec, tf.RaggedTensorSpec):
      return tf.io.RaggedFeature(
          value_key=fname,
          dtype=io_dtype,
          partitions=get_io_ragged_partitions(fname, value_spec.shape),
          row_splits_dtype=value_spec.row_splits_dtype,
          validate=validate)

    if isinstance(value_spec, tf.TensorSpec):
      if None not in value_spec.shape.as_list():
        # If shape is [d0..dk], where di is static (compile-time constant), the
        # value is parsed as a dense tensor.

        return tf.io.FixedLenFeature(
            dtype=io_dtype,
            shape=value_spec.shape,
            default_value=tf.zeros(value_spec.shape, io_dtype)
            if _is_size_field(fname) else None)

      if value_spec.shape[1:].is_fully_defined():
        # If shape is [None, d1..dk], where di is static (compile-time
        # constant), the value is parsed as a ragged tensor with ragged rank 0.
        # For single example parsing this result in a dense tensor, for multiple
        # examples parsing - in ragged.
        partitions = get_io_ragged_partitions(fname, value_spec.shape)
        # pytype: disable=attribute-error
        assert all(
            isinstance(p, tf.io.RaggedFeature.UniformRowLength)
            for p in partitions)
        # pytype: enable=attribute-error
        return tf.io.RaggedFeature(
            value_key=fname,
            dtype=io_dtype,
            partitions=partitions,
            row_splits_dtype=spec.indices_dtype,
            validate=validate)

      raise ValueError(
          ('Expected dense tensor with static non-leading dimensions'
           f', got shape={value_spec.shape}, fname={fname}'))

    raise ValueError(
        f'Unsupported type spec {type(value_spec).__name__}, fname={fname}')

  out = {}

  for fname, value_spec in _flatten_graph_field_specs(spec, '').items():
    if prefix:
      fname = f'{prefix}{fname}'
    # pylint: disable=protected-access
    out[fname] = get_io_feature(
        fname, gp._box_spec(spec.rank, value_spec, spec.indices_dtype))

  return out


@functools.singledispatch
def _flatten_graph_field_specs(piece_spec: gp.GraphPieceSpecBase,
                               prefix: str) -> gc.FieldsSpec:
  """Recursively flattens GraphPieceSpec into map of its field specs.

  Args:
    piece_spec: subclass of the GraphPieceSpecBase.
    prefix: string prefix to append to all result field specs name.

  Returns:
    Mapping from string keys to field type specs (as FieldSpec).
  """
  raise NotImplementedError(
      f'Dispatching is not defined for {type(piece_spec).__name__}')


@_flatten_graph_field_specs.register(gt.GraphTensorSpec)
def _(graph_spec: gt.GraphTensorSpec, prefix: str) -> gc.FieldsSpec:
  """Specialization for GraphTensorSpec."""
  result = {}
  result.update(
      _flatten_graph_field_specs(graph_spec.context_spec,
                                 f'{prefix}{gc.CONTEXT}/'))
  for name, spec in graph_spec.node_sets_spec.items():
    result.update(
        _flatten_graph_field_specs(spec, f'{prefix}{gc.NODES}/{name}.'))
  for name, spec in graph_spec.edge_sets_spec.items():
    result.update(
        _flatten_graph_field_specs(spec, f'{prefix}{gc.EDGES}/{name}.'))
  return result


@_flatten_graph_field_specs.register(gt.ContextSpec)
def _(context_spec: gt.ContextSpec, prefix: str) -> gc.FieldsSpec:
  return _prefix_keys(context_spec.features_spec, prefix)


@_flatten_graph_field_specs.register(gt.NodeSetSpec)
def _(node_set_spec: gt.NodeSetSpec, prefix: str) -> gc.FieldsSpec:
  result = {f'{prefix}{gc.SIZE_NAME}': node_set_spec.sizes_spec}
  result.update(_prefix_keys(node_set_spec.features_spec, prefix))
  return result


@_flatten_graph_field_specs.register(gt.EdgeSetSpec)
def _(edge_set_spec: gt.EdgeSetSpec, prefix: str) -> gc.FieldsSpec:
  result = {f'{prefix}{gc.SIZE_NAME}': edge_set_spec.sizes_spec}
  result.update(_prefix_keys(edge_set_spec.features_spec, prefix))
  result.update(
      _flatten_graph_field_specs(edge_set_spec.adjacency_spec, prefix))
  return result


# Note: HyperAdjacency I/O is not yet supported.
@_flatten_graph_field_specs.register(adj.AdjacencySpec)
def _(adjacency_spec: adj.AdjacencySpec, prefix: str) -> gc.FieldsSpec:
  return {
      f'{prefix}{gc.SOURCE_NAME}': adjacency_spec.source,
      f'{prefix}{gc.TARGET_NAME}': adjacency_spec.target
  }


def _prefix_keys(features_spec: gc.FieldsSpec, prefix: str) -> gc.FieldsSpec:
  # NOTE: Registering `_flatten_graph_field_specs` for gc.FieldsSpec is more
  #       elegant, but `singledispatch` does not support parametrized generics,
  #       like Mapping.
  return {f'{prefix}{key}': value for key, value in features_spec.items()}


@functools.singledispatch
def _unflatten_graph_fields(piece_spec: gp.GraphPieceSpecBase,
                            flat_fields: gc.Fields,
                            prefix: str) -> gp.GraphPieceBase:
  """Converts flat fields specifications to GraphPiece subclass.

  Reverses `_flatten_graph_field_specs`.

  Args:
    piece_spec: superclass of the GraphPieceSpecBase.
    flat_fields: mapping from field names to values as Field.
    prefix: string prefix appended to all flat field specs names.

  Returns:
    GraphPiece subclass compatible with the piece_spec.
  """
  raise NotImplementedError(type(piece_spec).__name__)


@_unflatten_graph_fields.register(gt.GraphTensorSpec)
def _(graph_spec: gt.GraphTensorSpec,
      flat_fields: gc.Fields,
      prefix: str) -> gt.GraphTensor:
  """Specialization for GraphTensorSpec."""
  context = _unflatten_graph_fields(graph_spec.context_spec, flat_fields,
                                    f'{prefix}{gc.CONTEXT}/')
  node_sets = {}
  for name, spec in graph_spec.node_sets_spec.items():
    node_sets[name] = _unflatten_graph_fields(spec, flat_fields,
                                              f'{prefix}{gc.NODES}/{name}.')
  edge_sets = {}
  for name, spec in graph_spec.edge_sets_spec.items():
    edge_sets[name] = _unflatten_graph_fields(spec, flat_fields,
                                              f'{prefix}{gc.EDGES}/{name}.')

  return gt.GraphTensor.from_pieces(
      context=context, node_sets=node_sets, edge_sets=edge_sets)


@_unflatten_graph_fields.register(gt.ContextSpec)
def _(context: gt.ContextSpec,
      flat_fields: gc.Fields,
      prefix: str) -> gt.Context:
  return gt.Context.from_fields(
      features=_match_fields(context.features_spec, flat_fields, prefix),
      shape=context.shape,
      indices_dtype=context.indices_dtype)


@_unflatten_graph_fields.register(gt.NodeSetSpec)
def _(node_set_spec: gt.NodeSetSpec,
      flat_fields: gc.Fields,
      prefix: str) -> gt.NodeSet:
  return gt.NodeSet.from_fields(
      sizes=_get_prefixed_field(flat_fields, gc.SIZE_NAME, prefix),
      features=_match_fields(node_set_spec.features_spec, flat_fields, prefix))


@_unflatten_graph_fields.register(gt.EdgeSetSpec)
def _(edge_set_spec: gt.EdgeSetSpec, flat_fields: gc.Fields,
      prefix: str) -> gt.EdgeSet:
  return gt.EdgeSet.from_fields(
      sizes=_get_prefixed_field(flat_fields, gc.SIZE_NAME, prefix),
      adjacency=_unflatten_graph_fields(edge_set_spec.adjacency_spec,
                                        flat_fields, prefix),
      features=_match_fields(edge_set_spec.features_spec, flat_fields, prefix))


@_unflatten_graph_fields.register(adj.AdjacencySpec)
def _(adjacency_spec: adj.AdjacencySpec, flat_fields: gc.Fields,
      prefix: str) -> adj.Adjacency:
  return adj.Adjacency.from_indices(
      source=(adjacency_spec.node_set_name(gc.SOURCE),
              _get_prefixed_field(flat_fields, gc.SOURCE_NAME, prefix)),
      target=(adjacency_spec.node_set_name(gc.TARGET),
              _get_prefixed_field(flat_fields, gc.TARGET_NAME, prefix)))


def _match_fields(features_spec: gc.FieldsSpec, flat_fields: gc.Fields,
                  prefix: str) -> gc.Fields:
  # NOTE: Registering `_unflatten_graph_fields` for gc.FieldsSpec is more
  #       elegant, but `singledispatch` does not support parametrized generics,
  #       like Mapping.
  return {
      name: _get_prefixed_field(flat_fields, name, prefix)
      for name in features_spec
  }


def _get_prefixed_field(fields: gc.Fields, field_name: str,
                        prefix: str) -> gc.Field:
  full_name = f'{prefix}{field_name}'
  assert full_name in fields, full_name
  return fields[full_name]


def _get_io_type(dtype: tf.dtypes.DType) -> tf.dtypes.DType:
  """Maps `dtype` on one of the supported by the tf.io types."""
  if dtype in (tf.int8, tf.int16, tf.int32, tf.int64):
    return tf.int64
  if dtype in (tf.uint8, tf.uint16, tf.uint32):
    return tf.int64
  if dtype in (tf.float16, tf.float32):
    return tf.float32
  if dtype in (tf.string,):
    return tf.string
  raise TypeError((f'Unsupported type {dtype}.'
                   ' Type must be safely castable to one'
                   ' of the following supported IO types:'
                   ' tf.int64, tf.float32, tf.string'))


def _restore_types(spec: gt.GraphTensorSpec,
                   prefix: Optional[str],
                   flat_values: gt.Fields) -> gt.Fields:
  """Casts `flat_values` to types expected in the graph tensor spec, if needed.

  Parsing of tensorflow examples using tf.io is limited to one of the following
  types: tf.int64, tf.float32, tf.string. This function insures that parsed
  values have types expected by the graph tensor `spec`.

  Args:
    spec: graph tensor specification.
    prefix: An optional prefix string over all the features. You may use
      this if you are encoding other data in the same protocol buffer.
    flat_values: flattened graph tensor values matching the `spec` except maybe
      values types (must be safely castable to the `spec` types).

  Returns:
    flattened graph tensor values with types matching the graph tensor spec.
  """
  # pylint: disable=protected-access
  flat_spec = _flatten_graph_field_specs(spec, prefix or '')
  result = dict()
  for fname, value in flat_values.items():
    field_dtype = flat_spec[fname].dtype
    if value.dtype != field_dtype:
      value = tf.cast(value, field_dtype)
    result[fname] = value
  return result


def _check_size_fields(spec: gt.GraphTensorSpec,
                       flat_values: gt.Fields) -> List[AssertOp]:
  """Checks special size fields for all node and edge sets.

  Size fields must

  * be set for all node and edge set defined in the `spec`;
  * have identical shapes (so identical partitions of graph components);
  * always describe number of nodes/edges in graph components, in particular,
    be set to zero when there are no node/edge instances in some set.

  Args:
    spec: graph tensor specification.
    flat_values: flattened graph tensor values to check matching the `spec`.

  Returns:
    List of assertion operations.
  """
  asserts = []
  size_shapes = []

  size_error_message = (
      'The `{size_name}` field must always be present when parsing graph '
      'tensor with non-static number of graph components. E.g. it is required '
      'that `nodes/{{node_set}}.{size_name}` and '
      '`edges/{{edge_set}}.{size_name}` features are present for all node and '
      'edge sets in each Tensorflow example.').format(size_name=gc.SIZE_NAME)

  def check_ragged_shapes(size: tf.RaggedTensor):
    assert isinstance(size, tf.RaggedTensor)

    dim = 0
    while isinstance(size.values, tf.RaggedTensor):
      size = size.values
      size_shapes.append((size.row_splits, (f'R{dim}',)))
      dim += 1

    assert isinstance(size.values, tf.Tensor)
    asserts.append(
        tf.debugging.assert_positive(
            size.row_lengths(), message=size_error_message))
    size_shapes.append((size.values, ('C',)))

  def check_dense_shapes(size: tf.Tensor):
    assert isinstance(size, tf.Tensor)
    outer_shape = size.shape[(spec.rank + 1):]
    assert None not in outer_shape.as_list(), (
        'Undefined inner dimensions for dense `{}` field.'.format(gc.SIZE_NAME))

    if size.shape[spec.rank:(spec.rank + 1)].as_list() == [None]:
      asserts.append(
          tf.debugging.assert_positive(
              tf.size(size), message=size_error_message))
    size_shapes.append((size, [f'D{d}' for d in range(size.shape.rank)]))

  def check_shapes(size: gt.Field):
    if isinstance(size, tf.RaggedTensor):
      check_ragged_shapes(size)
    elif isinstance(size, tf.Tensor):
      check_dense_shapes(size)
    else:
      raise ValueError('Unsupported `{}` field type {}'.format(
          gc.SIZE_NAME, type(size).__name__))

  for fname, value in flat_values.items():
    if _is_size_field(fname):
      check_shapes(value)

  # Note: This does not return an op.
  tf.debugging.assert_shapes(
      size_shapes,
      message=('All `{}` fields must have identical shapes for all node and '
               'edge sets.').format(gc.SIZE_NAME))

  return asserts


def _is_size_field(fname: str) -> bool:
  # TODO(b/189087785): provide more robust size feature matching.
  return fname.endswith(f'.{gc.SIZE_NAME}')
