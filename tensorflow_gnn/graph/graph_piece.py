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
"""Base classes for all CompositeTensors used inside a GraphTensor."""

import abc
import functools
from typing import Any, Callable, Mapping, Optional, Union
from typing import cast
from typing import List, Tuple

import tensorflow as tf

from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import tensor_utils as utils
from tensorflow_gnn.graph import tf_internal

ShapeLike = const.ShapeLike
Field = const.Field
FieldSpec = const.FieldSpec

# Data is a multi-level nest (see tf.nest) of Tensors, RaggedTensors and
# subclasses of the GraphPieceBase.
Data = Any
# DataSpec is a multi-level nest (see tf.nest) of TensorSpecs, RaggedTensorSpecs
# and subclasses of the GraphPieceSpecBase. Structure of the spec should match
# values in Data (as of tf.nest.assert_same_structure).
DataSpec = Any
Metadata = Optional[Mapping[str, Union[bool, int, str, float, tf.dtypes.DType]]]
FieldMapFn = Callable[[Field, FieldSpec], Tuple[Field, FieldSpec]]
PieceMapFn = Callable[['GraphPieceBase'], 'GraphPieceBase']


def convert_to_tensor_or_ragged(value):
  """Coerce objects other than ragged tensors to tensors."""
  return (value
          if isinstance(value, (tf.RaggedTensor, GraphPieceBase))
          else tf.convert_to_tensor(value))


class GraphPieceBase(tf_internal.CompositeTensor, metaclass=abc.ABCMeta):
  """The base class for all `CompositeTensors` used inside a `GraphTensor`.

  A `GraphPieceBase` is a `CompositeTensor` whose value is a multi-level nest of
  tensors, ragged tensors (fields) and other subclasses of the `GraphPieceBase`.

   - Each `GraphPiece` (subclass of `GraphPieceBase`) is a `CompositeTensor` and
     has a matching `GraphPieceSpec` (a subclass of `GraphPieceSpecBase`) as its
     type spec.

   - `GraphPiece` and `GraphPieceSpec` are immutable objects.

   - Each `GraphPiece` defines a `shape` attribute that reflects common batching
     dimensions. These are dense (i.e., non-ragged) dimensions, a dimension
     `None` is statically unknown (but must be the same for all constituent
     tensors of one `GraphPiece` tensor).

   - `GraphPiece` rank is by definition its `shape.rank`.

   - All `GraphPiece` field shapes must have the shape of the `GraphPiece` as a
     prefix. Fields rank is strictly greater than the `GraphPiece` rank.

   - Each `GraphPiece` supports batching, which may introduce ragged dimensions
     after the batching dimensions (e.g., when batching node features of graphs
     with different numbers of nodes).

   - The first field dimension after the `GraphPiece` rank ranges over graph
     items (nodes, edges or components; see `GraphTensor` for an explanation of
     components). For example, the node data fields in a `NodeSet` have an item
     dimension ranging over the nodes.

   - The following dimensions (> rank + 1), if any, apply to the field value
     for the respective item.

   - Dense field dimensions must be fully defined (have static shapes).

   - If for some field its item dimension or following dimensions are not
     statically defined (None), the field is batched as a potentially ragged
     tensor. Otherwise the field is batched as a dense tensor.

   - A `GraphPiece` object stores a nest of immutable tensors (tensors, ragged
     tensors, and/or other `GraphPiece`s) together with a `GraphPieceSpec`. The
     `GraphPieceSpec` contains a nest of specs that mirrors this nest of
     tensors.

   - A `GraphPiece` object stores optional metadata as a flat mapping from a
     string key to hashable values (see the `Metadata` typedef). Metadata allows
     to attach to the `GraphPiece` some extra static information. For example, a
     list of names of node sets that are connected by an edge set; the number of
     nodes/edges in a graph component (as a python integer) if graph components
     are of fixed-size.

   - Each `GraphPiece` shares the following graph-wide attributes with a
     GraphPiece it belongs to (or could belong to): shape, indices_dtype and
     metadata.

   - If `GraphPiece` X is a part of `GraphPiece` Y (X->Y), all metadata entries
     of X and Y with the same keys must have exactly the same values. If X->Y->Z
     metadata values with the same key `K` from Z and X may have different
     values if and only if entry with the key `K` is not present in metadata Y.
     This rule allows to automatically propagate metadata entries from parent
     graph pieces to all their children.

  """
  # TODO(b/188399175): Migrate to the new Extension Type API.

  __slots__ = ['_data', '_spec']

  def __init__(self,
               data: Data,
               spec: 'GraphPieceSpecBase',
               validate: bool = False):
    """Internal constructor, use `from_*` class factory methods instead.

    Creates object from `data` and matching data `spec`.

    NOTE: Any subclass `GraphPiece` is expected to support initialization
    as `GraphPiece(data, spec)` from data (suitably nested) and a spec with
    the matching subclass of GraphPieceSpecBase. Avoid redefining `__init__`.

    Args:
      data: Nest of Field or subclasses of GraphPieceBase.
      spec: A subclass of GraphPieceSpecBase with a `_data_spec` that matches
        `data`.
      validate: if set, checks that data and spec are aligned, compatible and
        supported.
    """
    super().__init__()
    assert data is not None
    assert spec is not None
    assert isinstance(spec, GraphPieceSpecBase), type(spec).__name__
    if validate or const.validate_internal_results:
      tf.nest.assert_same_structure(data, spec._data_spec)
      tf.nest.map_structure(_assert_value_compatible_with_spec, data,
                            spec._data_spec)
    self._data = data
    self._spec = spec

  @abc.abstractstaticmethod
  def _type_spec_cls():
    """Returns type spec class (sublass fo the `GraphPieceBase`)."""
    raise NotImplementedError('`_type_spec_cls` is not implemented')

  @classmethod
  def _from_data(cls,
                 data: Data,
                 shape: tf.TensorShape,
                 indices_dtype: tf.dtypes.DType,
                 metadata: Metadata = None) -> 'GraphPieceBase':
    """Creates a GraphPiece from its data and attributes.

    Args:
      data: a nest of Field and GraphPiece objects. The batch dimensions of all
        Fields (incl. those nested in GraphPiece objects) must be exactly equal.
        (In particular, if a dimension is None for one, it must be None for
        all.)
      shape: A hint for the shape of the result. This shape must have a known
        rank. It must be compatible with (but not necessary equal to) the
        common batch dimensions of the Fields nested in data. (This is meant to
        align this function with the requirements of TypeSpec._from_components()
        and BatchableTypeSpec._from_compatible_tensor_list().)
      indices_dtype: indices type to use for potentially ragged fields batching.
      metadata: optional mapping from a string key to hashable values.

    Returns:
      An instance of GraphPieceBase, holding the data, after GraphPieces in the
      data has been transformed to match `indices_dtype` and `metadata`.
      The shape of the result and its constituent GraphPieces is the common
      shape of all data Fields if there are any, or else the `shape` passed
      in as an argument. In either case, the shape of the result is compatible
      with the passed-in shape (but not necessarily equal).

    Raises:
      ValueError: if the data Fields do not have equal batch shapes.
    """
    # TODO(aferludin,edloper): Clarify the requirements of
    # TypeSpec._from_components(). Why can I safely construct from components
    # with a different dynamic shape, but only if that is statically unknown?

    # pylint: disable=protected-access
    def update_fn(value: Union['GraphPieceBase', Field], shape: tf.TensorShape,
                  indices_dtype: tf.dtypes.DType,
                  metadata: Metadata) -> Union['GraphPieceBase', Field]:
      """Updates data with new attributes."""
      if isinstance(value, GraphPieceBase):
        return value._with_attributes(shape, indices_dtype, metadata)
      if not isinstance(value, (tf.RaggedTensor, tf.Tensor)):
        raise TypeError(
            f'Invalid type for: {value}; should be tensor or ragged tensor')
      return value

    shape_from_data = _get_batch_shape_from_fields(data, shape.rank)
    if shape_from_data is not None:
      if shape.is_compatible_with(shape_from_data):
        shape = shape_from_data
      else:
        raise ValueError('Fields have batch dimensions that are not compatible'
                         ' with the GraphPiece shape,'
                         f' fields batch dimensions {shape_from_data},'
                         f' GraphPiece shape {shape}')

    data = tf.nest.map_structure(
        functools.partial(
            update_fn,
            shape=shape,
            indices_dtype=indices_dtype,
            metadata=metadata), data)
    data_spec = tf.nest.map_structure(tf.type_spec_from_value, data)

    cls_spec = cls._type_spec_cls()
    assert issubclass(cls_spec, GraphPieceSpecBase), cls_spec
    return cls(data, cls_spec(data_spec, shape, indices_dtype, metadata))

  def _apply(self,
             field_map_fn: Optional[FieldMapFn] = None,
             piece_map_fn: Optional[PieceMapFn] = None,
             shape: Optional[tf.TensorShape] = None) -> 'GraphPieceBase':
    """Constructs a new instance, mapping values while preserving attributes.

    This method allows transforming GraphPiece fields with their specs.

    Args:
      field_map_fn: takes as an input field (as Field) and its spec (as
        FieldSpec) tuple and returns new field and new spec tuple.
      piece_map_fn: takes as an input GraphPiece and returns new GraphPiece.
      shape: optional updated GraphPiece shape.

    Returns:
      A transformed instance of the GraphPieceBase object.
    """

    # pylint: disable=protected-access
    def update_fn(data, spec):
      if isinstance(data, GraphPieceBase):
        assert isinstance(spec, GraphPieceSpecBase)
        if piece_map_fn is None:
          return data, spec
        data = piece_map_fn(data)
        return data, tf.type_spec_from_value(data)

      if not isinstance(data, (tf.Tensor, tf.RaggedTensor)):
        raise TypeError(
            f'Invalid type for: {data}; should be tensor or ragged tensor')

      if field_map_fn is None:
        return data, spec
      return field_map_fn(cast(Field, data), cast(FieldSpec, spec))

    data, data_spec = _apply(update_fn, self._data, self.spec._data_spec)
    cls = self.__class__
    cls_spec = cls._type_spec_cls()
    assert issubclass(cls_spec, GraphPieceSpecBase), cls_spec
    return cls(
        data,
        cls_spec(data_spec, self.shape if shape is None else shape,
                 self.indices_dtype, self.spec._metadata))

  def _with_attributes(self, shape: tf.TensorShape,
                       indices_dtype: tf.dtypes.DType,
                       metadata: Metadata) -> 'GraphPieceBase':
    """Returns an instance of this object with replaced attributes."""
    # pylint: disable=protected-access
    if self.spec._are_compatible_attributes(shape, indices_dtype, metadata):
      return self
    new_spec = self.spec._with_attributes(shape, indices_dtype, metadata)

    def update_fn(value):
      if isinstance(value, GraphPieceBase):
        return value._with_attributes(new_spec.shape, new_spec.indices_dtype,
                                      new_spec._metadata)
      return value

    return self.__class__(  # pytype: disable=not-instantiable
        tf.nest.map_structure(update_fn, self._data), new_spec)

  @property
  def shape(self) -> tf.TensorShape:
    """A possibly-partial shape specification for this Tensor.

    The returned `TensorShape` is guaranteed to have a known rank, but the
    individual dimension sizes may be unknown.

    Returns:
      A `tf.TensorShape` containing the statically known shape of the
      GraphTensor.
    """
    return self._spec.shape

  def set_shape(self, new_shape: ShapeLike) -> 'GraphPieceSpecBase':
    """Enforce the common prefix shape on all the contained features."""
    # pylint: disable=protected-access
    if not isinstance(new_shape, tf.TensorShape):
      new_shape = tf.TensorShape(new_shape)
    return self._with_attributes(new_shape, self.spec._indices_dtype,
                                 self.spec._metadata)

  @property
  def rank(self) -> int:
    """The rank of this Tensor. Guaranteed not to be `None`."""
    assert self._spec.rank is not None
    return self._spec.rank

  @property
  def indices_dtype(self) -> tf.dtypes.DType:
    """The integer type to represent ragged splits."""
    return self._spec.indices_dtype

  @property
  def spec(self) -> 'GraphPieceSpecBase':
    """The public type specification of this tensor."""
    return self._type_spec

  @property
  def _type_spec(self) -> 'GraphPieceSpecBase':
    """Like .spec, part of the CompositeTensor API."""
    return self._spec

  def _merge_batch_to_components(self, *args, **kwargs) -> 'GraphPieceBase':
    """Merges components from all batch dimensions into a single scalar piece.

    For example, flattening of
      GraphPiece(
        shape=[3],
        # Three graphs with
        #   - 1st graph having two components (values 'a.1' and 'a.2');
        #   - 2nd graph having 1 component (value 'b');
        #   - 3rd graph having 1 component (value 'c');
        data: [['a.1', 'a.2'], ['b'], ['c']]
      )

    would result in

      GraphPiece(
        shape=[],
        # Single graphs with 4 componests (2 + 1 + 1).
        data: ['a.1', 'a.2', 'b', 'c']
      )

    NOTE: Data items that are GraphPieces themselves can override this method to
    merge values in specific ways. See the GraphTensor documentation for the
    purpose of this (merging separately indexed node sequences into one).

    Args:
      *args: The list of arguments to be passed to nested GraphPieces.
      **kwargs: Any keyword arguments to be passed to nested GraphPieces.

    Returns:
      scalar (rank-0) GraphPiece.
    """
    # pylint: disable=protected-access
    return self._apply(
        functools.partial(remove_batch_dimensions, self.rank),
        lambda piece: piece._merge_batch_to_components(*args, **kwargs),
        tf.TensorShape([]))


class GraphPieceSpecBase(tf_internal.BatchableTypeSpec, metaclass=abc.ABCMeta):
  """The base class for TypeSpecs of GraphPieces."""

  __slots__ = ['_data_spec', '_shape', '_indices_dtype', '_metadata']

  def __init__(self,
               data_spec: DataSpec,
               shape: tf.TensorShape,
               indices_dtype: tf.dtypes.DType,
               metadata: Metadata = None,
               validate: bool = False):
    """Constructs GraphTensor type spec."""
    super().__init__()
    assert isinstance(shape, tf.TensorShape), f'got: {type(shape).__name__}'
    if not shape[1:].is_fully_defined():
      raise ValueError(
          ('All shape dimensions except the outermost must be fully defined,'
           f' got shape={shape}.'))

    if metadata is not None and const.validate_internal_results:
      assert isinstance(metadata, Mapping)
      for k, v in metadata.items():
        assert isinstance(k, str), 'See b/187015015'
        assert isinstance(v, (bool, int, str, float, tf.dtypes.DType))

    if validate or const.validate_internal_results:
      # TODO(b/187011656): currently ragged-rank0 dimensions are not supported.
      tf.nest.map_structure(_assert_not_rank0_ragged, data_spec)
      tf.nest.map_structure(
          functools.partial(_assert_batch_shape_compatible_with_spec, shape),
          data_spec)

    self._shape = shape
    self._indices_dtype = indices_dtype
    self._metadata = metadata
    self._data_spec = data_spec

  @classmethod
  def _from_data_spec(cls,
                      data_spec: DataSpec,
                      shape: tf.TensorShape,
                      indices_dtype: tf.dtypes.DType,
                      metadata: Metadata = None) -> 'GraphPieceSpecBase':
    """Counterpart of `GraphPiece.from_data` with data type spec."""

    # pylint: disable=protected-access
    def update_fn(value_spec):
      if isinstance(value_spec, GraphPieceSpecBase):
        return value_spec._with_attributes(shape, indices_dtype, metadata)
      return _set_batch_shape_in_spec(shape, value_spec)

    return cls(
        tf.nest.map_structure(update_fn, data_spec), shape, indices_dtype,
        metadata)

  def _with_attributes(self, shape: tf.TensorShape,
                       indices_dtype: tf.dtypes.DType,
                       metadata: Metadata) -> 'GraphPieceSpecBase':
    """Returns object instance with replaced attributes."""
    # pylint: disable=protected-access
    if self._are_compatible_attributes(shape, indices_dtype, metadata):
      return self

    metadata = self._merge_metadata(metadata)

    def update_fn(value_spec):
      if isinstance(value_spec, GraphPieceSpecBase):
        return value_spec._with_attributes(shape, indices_dtype, metadata)
      return _set_batch_shape_in_spec(shape, value_spec)

    return self.__class__(
        tf.nest.map_structure(update_fn, self._data_spec), shape, indices_dtype,
        metadata)

  def _are_compatible_attributes(self, shape: tf.TensorShape,
                                 indices_dtype: tf.dtypes.DType,
                                 metadata: Metadata) -> bool:
    """Checks that the new set of attributes is equivalent to the current."""
    return self._is_equal_shape(shape) and self._is_compatible_metadata(
        metadata) and self.indices_dtype == indices_dtype

  def _is_equal_shape(self, shape: tf.TensorShape) -> bool:
    """Shapes are equal if their dimension lists are equal."""
    return self.shape.as_list() == shape.as_list()

  def _is_compatible_metadata(self, metadata: Metadata) -> bool:
    if metadata is None or self._metadata is None:
      return metadata is None
    return all(new_v == self._metadata[k]
               for k, new_v in metadata.items()
               if k in self._metadata)

  def _merge_metadata(self, metadata: Metadata) -> Metadata:
    if metadata is None or self._metadata is None:
      return self._metadata
    return {
        k: (metadata[k] if k in metadata else v)
        for k, v in self._metadata.items()
    }

  @property
  def shape(self) -> tf.TensorShape:
    """A possibly-partial shape specification of the GraphPiece.

    The returned `TensorShape` is guaranteed to have a known rank, but the
    individual dimension sizes may be unknown.

    Returns:
      A `tf.TensorShape` containing the statically known shape of the
      `GraphTensor`.
    """
    return self._shape

  @property
  def rank(self) -> int:
    """The rank of the GraphPiece. Guaranteed not to be `None`."""
    return self.shape.rank

  @property
  def indices_dtype(self) -> tf.dtypes.DType:
    """The integer type to represent ragged splits."""
    return self._indices_dtype

  @classmethod
  def from_value(cls, value: GraphPieceBase):
    """Extension Types API: Factory method."""
    return value._type_spec  # pylint: disable=protected-access

  def _serialize(self) -> Tuple[Any, ...]:
    """Extension Types API: Serialization as a nest of simpler types."""
    return (self._data_spec, self._shape, self._indices_dtype, self._metadata)

  @classmethod
  def _deserialize(cls, serialization):
    """Extension Types API: Deserialization from a nest of simpler types."""
    data_spec, shape, indices_dtype, metadata = serialization
    # Reinstate types lost by Keras Model serialization.
    # TODO(b/241917040): Remove if/when fixed in Keras.
    if not isinstance(shape, tf.TensorShape):
      shape = tf.TensorShape(shape)
    if not isinstance(indices_dtype, tf.dtypes.DType):
      indices_dtype = tf.dtypes.as_dtype(indices_dtype)
    return cls(data_spec, shape, indices_dtype, metadata)

  @property
  def _component_specs(self) -> Any:
    """Extension Types API: Specs matching _to_components values."""
    return self._data_spec

  def _to_components(self, value: GraphPieceBase) -> Any:
    """Extension Types API: Nest of GraphPiece components. No TF OPS."""
    return value._data  # pylint: disable=protected-access

  def _from_components(self, components: Any) -> GraphPieceBase:
    """Extension Types API: Inverse to `_to_components`. No TF OPS."""
    # pylint: disable=protected-access
    cls = self.value_type
    assert issubclass(cls, GraphPieceBase), cls
    return cls._from_data(components, self._shape, self._indices_dtype,
                          self._metadata)

  def _batch(self, batch_size: Optional[int]) -> 'GraphPieceSpecBase':
    """Extension Types API: Batching."""
    if not self._shape.is_fully_defined():
      raise NotImplementedError((
          f'Batching a graph piece without fully defined shape={self.shape} is'
          ' not supported. This error is typically a result of two consecutive'
          ' dataset batching, as ds.batch(batch_size1).batch(batch_size2),'
          ' where the first batch operation produces graph pieces with an'
          ' undefined (None) outermost dimension as it allows incomplete'
          ' batches. This indicates a potential error, as it is not generally'
          ' possible to stack two pieces with different shapes. If that is the'
          ' case, consider setting drop_remainder=True, for'
          ' ds.batch(batch_size1, drop_remainder=True).batch(batch_size2).'
      ))

    def batch_fn(spec):
      # Convert all specs for potentially ragged tensors to ragged rank-0 type
      # specifications to allow variable size (ragged) batching.
      # pylint: disable=protected-access
      spec = _box_spec(self.rank, spec, self.indices_dtype)
      spec = spec._batch(batch_size)
      return _unbox_spec(self.rank + 1, spec)

    batched_data_spec = tf.nest.map_structure(batch_fn, self._data_spec)
    shape = tf.TensorShape([batch_size]).concatenate(self._shape)
    return self.__class__(batched_data_spec, shape, self._indices_dtype,
                          self._metadata)

  def _unbatch(self) -> 'GraphPieceSpecBase':
    """Extension Types API: Unbatching."""
    if self.rank == 0:
      raise ValueError('Could not unbatch scalar (rank=0) GraphPiece.')

    def unbatch_fn(spec):
      # Convert all ragged rank-0 specs to simple tensor type specs to ensure
      # that it is not leaking to the `to_components`.
      # pylint: disable=protected-access
      spec = _box_spec(self.rank, spec, self.indices_dtype)
      spec = spec._unbatch()
      return _unbox_spec(self.rank - 1, spec)

    unbatched_data_spec = tf.nest.map_structure(unbatch_fn, self._data_spec)

    batch_size = self._shape.as_list()[0]
    batch_size = int(batch_size) if batch_size is not None else None
    shape = self._shape[1:]
    return self.__class__(unbatched_data_spec, shape, self.indices_dtype,
                          self._metadata)

  def _from_compatible_tensor_list(self,
                                   tensor_list: List[Any]) -> GraphPieceBase:
    """Extension Types API: Decodes from a list of (possibly stacked) Tensors.

    Args:
      tensor_list: A list of `Tensors` that was returned by `_to_tensor_list`;
        or a list of `Tensors` that was formed by stacking, unstacking, and
        concatenating the values returned by `_to_tensor_list`.

    Returns:
      A value compatible with this TypeSpec.
    """
    # pylint: disable=protected-access
    tensor_list = list(tensor_list)

    flat_values = list()
    for spec in tf.nest.flatten(self._data_spec):
      spec = _box_spec(self.rank, spec, self.indices_dtype)
      num_tensors_for_feature = len(spec._flat_tensor_specs)
      feature_tensors = tensor_list[:num_tensors_for_feature]
      del tensor_list[:num_tensors_for_feature]
      value = spec._from_compatible_tensor_list(feature_tensors)
      flat_values.append(value)
    assert not tensor_list

    fields = tf.nest.pack_sequence_as(self._data_spec, flat_values)
    cls = self.value_type
    assert issubclass(cls, GraphPieceBase)
    return cls._from_data(fields, self._shape, self._indices_dtype,
                          self._metadata)

  @property
  def _flat_tensor_specs(self) -> List[tf.TensorSpec]:
    """Extension Types API: Specs matching `_to_tensor_list`."""
    # pylint: disable=protected-access
    out = []
    for spec in tf.nest.flatten(self._data_spec):
      spec = _box_spec(self.rank, spec, self.indices_dtype)
      out.extend(spec._flat_tensor_specs)
    return out

  def _to_tensor_list(self, value: GraphPieceBase) -> List[tf.Tensor]:
    """Extension Types API: Encodes `value` as stackable Tensors.

    Args:
      value: A value compatible with this TypeSpec. (Caller is responsible for
        ensuring compatibility.)

    Returns:
      A list of `Tensors` that encodes `value`. These `Tensors` can be stacked,
      unstacked, or concatenated before passing them to `.from_tensor_list()`,
      resulting in a value that has been stacked, unstacked or concatenated
      in the same way.
    """
    return self._to_tensor_list_impl(value, '_to_tensor_list')

  def _to_batched_tensor_list(self, value: GraphPieceBase) -> List[tf.Tensor]:
    """Extension Types API: Encodes non-scalar `value` as stackable Tensors."""
    return self._to_tensor_list_impl(value, '_to_batched_tensor_list')

  def _to_tensor_list_impl(self, value: GraphPieceBase,
                           spec_method_name: str) -> List[tf.Tensor]:
    # pylint: disable=protected-access
    data_spec = self._data_spec
    data = value._data
    out = []

    def map_fn(spec, value):
      spec = _box_spec(self.rank, spec, self.indices_dtype)
      to_list_fn = getattr(spec, spec_method_name)
      out.extend(to_list_fn(value))

    tf.nest.map_structure(map_fn, data_spec, data)
    return out

  def _to_legacy_output_types(self):
    """Extension Types API: Legacy compatibility method."""
    return self

  def _to_legacy_output_shapes(self):
    """Extension Types API: Legacy compatibility method."""
    return self._shape

  def _to_legacy_output_classes(self):
    """Extension Types API: Legacy compatibility method."""
    return self

  def _create_empty_value(self) -> GraphPieceBase:
    """Creates minimal empty GraphPiece allowed by this spec.

    Rules:
      1. all unknown dimensions are assumed to be 0.
      2. field values for fixed size dimensions are set to empty with tf.zeros.
      3. resulting tensor should have no values (empty values of flat values).

    NOTE: this is temporary workaround to allow to contruct GraphTensors with
    empty batch dimensions to use with TF distribution strategy (b/183969859).
    The method could be removed in the future without notice, PLEASE DO NOT USE.

    Returns:
      GraphPiece compatible with this spec.
    """

    def create_empty_dense_field(shape: tf.TensorShape,
                                 dtype: tf.dtypes.DType) -> tf.Tensor:
      dims = [(0 if d is None else d) for d in shape.as_list()]
      if 0 not in dims:
        raise ValueError(
            f'Could not create empty tensor for non-empty shape {shape}')
      return tf.zeros(tf.TensorShape(dims), dtype)

    def create_empty_ragged_field(spec: tf.RaggedTensorSpec) -> Field:
      if spec.value_type == tf.Tensor:
        # For ragged rank-0 tensors values are dense tensors.
        return create_empty_dense_field(spec.shape, spec.dtype)

      assert spec.value_type == tf.RaggedTensor
      assert spec.ragged_rank > 0

      # Set components dimension to 0 (the outer-most flat values dimension).
      flat_values_shape = spec.shape[spec.ragged_rank:]
      assert flat_values_shape[1:].is_fully_defined(), flat_values_shape
      if flat_values_shape[0] not in (None, 0):
        raise ValueError(f'Could not create empty flat values for {spec}')

      # Use empty tensors for ragged dimensions row splits. Keep uniform
      # dimensions unchaged.
      empty_row_splits = tf.constant([0], dtype=spec.row_splits_dtype)
      result = create_empty_dense_field(flat_values_shape, spec.dtype)
      for dim in reversed(spec.shape[1:(spec.ragged_rank + 1)].as_list()):
        if dim is None:
          result = tf.RaggedTensor.from_row_splits(
              result,
              empty_row_splits,
              validate=const.validate_internal_results)
        else:
          result = tf.RaggedTensor.from_uniform_row_length(
              result,
              tf.convert_to_tensor(dim, dtype=spec.row_splits_dtype),
              validate=const.validate_internal_results)
      return result

    def create_empty_field(spec):
      if isinstance(spec, GraphPieceSpecBase):
        return spec._create_empty_value()  # pylint: disable=protected-access

      if isinstance(spec, tf.RaggedTensorSpec):
        return create_empty_ragged_field(cast(tf.RaggedTensorSpec, spec))

      if isinstance(spec, tf.TensorSpec):
        return create_empty_dense_field(spec.shape, spec.dtype)

      raise ValueError(f'Unsupported field type {type(spec).__name__}')

    dummy_fields = tf.nest.map_structure(create_empty_field, self._data_spec)

    cls = self.value_type
    assert issubclass(cls, GraphPieceBase), cls
    result = self.value_type(dummy_fields, self)
    if const.validate_internal_results:
      assert self.is_compatible_with(result)
    return result


_Value = Union[Field, GraphPieceBase]
_ValueSpec = Union[FieldSpec, GraphPieceSpecBase]


def _is_ragged_rank0(spec: _ValueSpec) -> bool:
  """Ragged rank-0 spec is used to batch dense to ragged rank-1 tensors."""
  return isinstance(spec, tf.RaggedTensorSpec) and spec.value_type == tf.Tensor


def _is_var_size_dense(batch_rank: int, spec: _ValueSpec) -> bool:
  """Returns True if the `spec` corresponds to the variable-size value."""
  # Dense tensor is variable size if and only if its component dimension is not
  # defined.
  return isinstance(spec, tf.TensorSpec) and (spec.shape[batch_rank] is None)


def _box_spec(batch_rank: int, spec: _ValueSpec,
              indices_dtype: tf.dtypes.DType) -> _ValueSpec:
  """Returns ragged rank-0 specification for potentially ragged tensors.

  Dense fields with variable-size components dimension (with static size None)
  must be batched into the ragged tensors with the ragged rank 1, similar to the
  tf.data.experimental.dense_to_ragged_batch. This is achieved by converting
  variable size dense field specifications from the tf.TensorSpec to the ragged
  rank-0 tf.RaggedTensorSpec.

  Args:
    batch_rank: number of batch dimensions.
    spec: value specification (FieldSpec or GraphPieceSpec).
    indices_dtype: ragged splits type (tf.int64 or tf.int32).

  Returns:
    Type specification to use for field batching.
  """
  if batch_rank > 0:
    return spec

  if _is_ragged_rank0(spec) or _is_var_size_dense(batch_rank, spec):
    inner_shape = spec.shape[1:]
    if not inner_shape.is_fully_defined():
      raise ValueError('Inner field dimensions must have static sizes,'
                       f' got shape={spec.shape}')

    return tf.RaggedTensorSpec(
        shape=tf.TensorShape([None]).concatenate(spec.shape[1:]),
        ragged_rank=0,
        dtype=spec.dtype,
        row_splits_dtype=indices_dtype)
  return spec


def _unbox_spec(batch_rank: int, spec: _ValueSpec) -> _ValueSpec:
  """Converts ragged rank-0 specification to a tf.TensorSpec.

  Dense fields with variable-size components dimension (with static size None)
  are batched using ragged rank-0 tf.RaggedTensorSpec (see _box_spec).
  Unfortunatelly ragged rank-0 specs are not well supported internally, so we
  are converting them to the tf.TensorSpec after batching (e.g. b/187011656).

  Args:
    batch_rank: number of batch dimensions.
    spec: value specification (FieldSpec or GraphPieceSpec).

  Returns:
    Type specification converted from the type spec used during the batching.
  """
  if batch_rank > 0:
    assert not _is_ragged_rank0(spec), spec
    return spec

  if _is_ragged_rank0(spec):
    return tf.TensorSpec(shape=spec.shape, dtype=spec.dtype)
  return spec


def _assert_not_rank0_ragged(spec: _ValueSpec) -> None:
  assert not _is_ragged_rank0(spec), ('b/187011656, use `_box_spec` and '
                                      '`_unbox_spec` for batching/unbatching')


def _assert_value_compatible_with_spec(value: _Value, spec: _ValueSpec) -> None:
  if not spec.is_compatible_with(value):
    value_spec = tf.type_spec_from_value(value)
    raise ValueError(f'Spec {spec} is not compatible with value {value_spec}')


def _assert_batch_shape_compatible_with_spec(batch_shape: tf.TensorShape,
                                             spec: _ValueSpec) -> None:
  """Checks that spec is compatible with the batch shape."""
  if isinstance(spec, GraphPieceSpecBase):
    if not spec.shape.is_compatible_with(batch_shape):
      raise ValueError(
          'Graph piece spec shape is not compatible with the batch shape,'
          f' spec.shape={spec.shape}, batch_shape={batch_shape}')
    return

  if spec.shape.rank <= batch_shape.rank:
    raise ValueError('Field spec rank must be greater than the batch rank:'
                     f' spec.shape={spec.shape}, batch_shape={batch_shape}')
  if not spec.shape[:batch_shape.rank].is_compatible_with(batch_shape):
    raise ValueError('Field spec shape is not compatible with the batch shape,'
                     f' spec.shape={spec.shape}, batch_shape={batch_shape}')


def _apply(map_fn: FieldMapFn, data_struct: Data,
           spec_struct: DataSpec) -> Tuple[Data, DataSpec]:
  """Applies function to each pair of data and its spec from structures."""
  assert callable(map_fn)
  flat_data = []
  flat_spec = []
  for data, spec in zip(
      tf.nest.flatten(data_struct), tf.nest.flatten(spec_struct)):
    new_data, new_spec = map_fn(data, spec)
    flat_data.append(new_data)
    flat_spec.append(new_spec)

  new_data_struct = tf.nest.pack_sequence_as(data_struct, flat_data)
  new_spec_struct = tf.nest.pack_sequence_as(spec_struct, flat_spec)
  return new_data_struct, new_spec_struct


def relax_dim(dim_index: int, value: Field,
              spec: FieldSpec) -> Tuple[Field, FieldSpec]:
  """Sets dimension with dim_index to None in the `spec`."""
  old_shape = spec.shape
  new_shape = old_shape[:dim_index].concatenate(tf.TensorShape(
      [None])).concatenate(old_shape[(dim_index + 1):])

  if isinstance(spec, tf.RaggedTensorSpec):
    new_spec = tf.RaggedTensorSpec(
        shape=new_shape,
        dtype=spec.dtype,
        ragged_rank=spec.ragged_rank,
        row_splits_dtype=spec.row_splits_dtype)
  elif isinstance(spec, tf.TensorSpec):
    new_spec = tf.TensorSpec(shape=new_shape, dtype=spec.dtype)
  else:
    raise ValueError(f'Unsupported spec type {type(spec).__name__}')

  return value, new_spec


def field_remove_batch_dimensions(rank: int, field: Field) -> Field:
  """Flattens the `rank` outer most batch dimensions from the `field`."""
  for dim in range(rank, 0, -1):
    if isinstance(field, (tf.RaggedTensor,)):
      field = field.values
    elif isinstance(field, (tf.Tensor,)):
      shape = utils.dims_list(field)
      return tf.reshape(field, shape=[-1] + shape[(dim + 1):])
    else:
      raise ValueError(f'Unsupported type {type(field).__name__}')
  return field


def spec_remove_batch_dimensions(rank: int, field_spec: FieldSpec) -> FieldSpec:
  """Flattens the `rank` outer most batch dimensions from the `field_spec`."""
  assert field_spec.shape.rank > rank, field_spec
  inner_shape = field_spec.shape[(rank + 1):]
  outer_shape = field_spec.shape[:(rank + 1)]
  squashed_shape = tf.TensorShape([utils.static_size(outer_shape)])
  shape = tf.TensorShape(squashed_shape).concatenate(inner_shape)
  if isinstance(field_spec, (tf.TensorSpec,)):
    return tf.TensorSpec(shape=shape, dtype=field_spec.dtype)

  if isinstance(field_spec, (tf.RaggedTensorSpec,)):
    ragged_rank = sum(d is None for d in inner_shape.as_list())
    if ragged_rank == 0:
      return tf.TensorSpec(shape=shape, dtype=field_spec.dtype)
    return tf.RaggedTensorSpec(
        shape=shape,
        dtype=field_spec.dtype,
        ragged_rank=ragged_rank,
        row_splits_dtype=field_spec.row_splits_dtype)

  raise ValueError(f'Unsupported field spec type {type(field_spec).__name__}')


def remove_batch_dimensions(rank: int, field: Field,
                            field_spec: FieldSpec) -> Tuple[Field, FieldSpec]:

  new_field = field_remove_batch_dimensions(rank, field)
  new_field_spec = spec_remove_batch_dimensions(rank, field_spec)
  return new_field, new_field_spec


def _set_batch_shape_in_spec(batch_shape: tf.TensorShape,
                             field_spec: FieldSpec) -> FieldSpec:
  """Ensures batch dimensions in the `field_spec` to match the `batch_shape`.


  Args:
    batch_shape: graph piece batch shape.
    field_spec: field specification.

  Returns:
    Field spec with the first shape dimensions set to the batch_shape.
  """
  _assert_batch_shape_compatible_with_spec(batch_shape, field_spec)
  old_shape = field_spec.shape
  new_shape = batch_shape.concatenate(old_shape[batch_shape.rank:])
  if isinstance(field_spec, tf.RaggedTensorSpec):
    return tf.RaggedTensorSpec(
        shape=new_shape,
        dtype=field_spec.dtype,
        ragged_rank=field_spec.ragged_rank,
        row_splits_dtype=field_spec.row_splits_dtype)

  if isinstance(field_spec, tf.TensorSpec):
    return tf.TensorSpec(shape=new_shape, dtype=field_spec.dtype)

  raise ValueError(f'Unsupported field spec type {type(field_spec).__name__}')


def _get_fields_list(data: Data) -> List[Field]:
  """Extracts all nested fields from the data as a flat list."""
  result = []

  def map_fn(value):
    if isinstance(value, GraphPieceBase):
      # pylint: disable=protected-access
      tf.nest.map_structure(map_fn, value._data)
    else:
      result.append(value)

  tf.nest.map_structure(map_fn, data)
  return result


def _get_batch_shape_from_fields(data: Data,
                                 batch_rank: int) -> Optional[tf.TensorShape]:
  """Extracts common batch dimensions from data fields.

  Args:
    data: nest of GraphPiece fields.
    batch_rank: number of batch dimensions.

  Returns:
    Returns common batch dimensions (as TensorShape) for all fields or None if
    data has no fields.

  Raises:
    ValueError: if batch dimensions are unequal between fields. In particular,
      if some dimension is None for one field, it must be None for all.
  """

  def get_batch_shape(field: Field) -> tf.TensorShape:
    if field.shape.rank is None or field.shape.rank <= batch_rank:
      raise ValueError('Field rank must be greater than the batch rank:'
                       f' field shape={field.shape}, batch_rank={batch_rank}')
    return field.shape[:batch_rank]

  fields = _get_fields_list(data)
  if not fields:
    return None

  result = get_batch_shape(fields[0])
  for field in fields[1:]:
    shape = get_batch_shape(field)
    if shape.as_list() != result.as_list():
      raise ValueError('Fields batch dimensions do not match:'
                       f' batch_rank={batch_rank},'
                       f' 1st field shape: {fields[0].shape},'
                       f' 2nd field shape: {field.shape}')
  return result


@tf.autograph.experimental.do_not_convert
def check_scalar_graph_piece(piece: Union[GraphPieceBase,
                                          GraphPieceSpecBase],
                             name='This operation') -> None:
  if isinstance(piece, GraphPieceSpecBase):
    piece_name: str = piece.value_type.__name__
  else:
    piece_name: str = type(piece).__name__
  if piece.rank != 0:
    raise ValueError(
        (f'{name} requires a scalar {piece_name}, that is,'
         f' with `{piece_name}.rank=0`, but got `rank={piece.rank}`.'
         f' Use GraphTensor.merge_batch_to_components() to merge all contained'
         ' graphs into one contiguously indexed graph of the scalar'
         ' GraphTensor.'))

