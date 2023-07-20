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
"""The GraphTensor composite tensor and its pieces."""
from __future__ import annotations

import abc
import collections.abc
import re
from typing import Any, Callable, cast, Dict, Mapping, Optional, Sequence, Union

import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_piece as gp
from tensorflow_gnn.graph import tensor_utils as utils
from tensorflow_gnn.graph import tf_internal

FieldName = const.FieldName
NodeSetName = const.NodeSetName
EdgeSetName = const.EdgeSetName
ShapeLike = const.ShapeLike
Field = const.Field
Fields = const.Fields
FieldOrFields = const.FieldOrFields
FieldSpec = const.FieldSpec
FieldsSpec = const.FieldsSpec
EDGES = const.EDGES
NODES = const.NODES
HIDDEN_STATE = const.HIDDEN_STATE

# TODO(b/189057503): use adjacency interface class instead.
Adjacency = Any
AdjacencySpec = Any


class _GraphPieceWithFeatures(gp.GraphPieceBase, metaclass=abc.ABCMeta):
  """Base class for graph pieces that hold user-defined features."""
  _DATAKEY_FEATURES = 'features'  # A Mapping[FieldName, Field].
  _DATAKEY_SIZES = 'sizes'  # A Field with `sizes`.

  def __getitem__(self, feature_name: FieldName) -> Field:
    """Indexing operator `[]` to access feature values by their name."""
    return self._get_features_ref[feature_name]

  @property
  def features(self) -> Mapping[FieldName, Field]:
    """A read-only mapping of feature name to feature specs."""
    return _as_immutable_mapping(self._get_features_ref)

  @property
  def sizes(self) -> Field:
    """The number of items in each graph component.

    Returns:
      A potentially ragged int tensor of shape `[*graph_shape, num_components]`
      where the `graph_shape` is the graph piece shape and its containing
      GraphTensor, `num_components` is the number of graph components contained
      in each graph (could be ragged).
    """
    return self._data[_GraphPieceWithFeatures._DATAKEY_SIZES]  # pylint: disable=protected-access

  @property
  def total_size(self) -> tf.Tensor:
    """The total number of items.

    Returns:
      A scalar integer tensor equal to `tf.math.reduce_sum(sizes)`. If result is
      statically known (`spec.total_size is not None`), the output is a constant
      tensor (suitable for environments in which constant shapes are required,
      like TPU).
    """
    dtype = self.spec.sizes_spec.dtype
    return _fast_alternative(
        self.spec.total_size is not None,
        lambda: tf.constant(self.spec.total_size, dtype, shape=[]),
        lambda: tf.math.reduce_sum(self.sizes), 'total_size != spec.total_size')

  @property
  def num_components(self) -> tf.Tensor:
    """The number of graph components for each graph.

    Returns:
      A dense integer tensor with the same shape as the graph piece.
    """
    result = tf.reduce_sum(tf.ones_like(self.sizes), axis=self.rank)
    if utils.is_ragged_tensor(result):
      # TODO(b/232914703): reconsider if ragged batch dimensions are supported.
      result_dense = result.to_tensor()
      check_ops = []
      if const.validate_internal_results:
        check_ops.append(
            tf.debugging.assert_equal(
                tf.size(result),
                tf.size(result_dense),
                message='`sizes` shape is not compatible with the piece shape'))

      with tf.control_dependencies(check_ops):
        result = tf.identity(result_dense)
    return result

  @property
  def total_num_components(self) -> tf.Tensor:
    """The total number of graph components.

    Returns:
      A scalar integer tensor equal to `tf.math.reduce_sum(num_components)`. If
      result is statically known (`spec.total_num_components is not None`), the
      output is a constant Tensor (suitable for environments in which constant
      shapes are required, like TPU).
    """
    dtype = self.spec.sizes_spec.dtype
    return _fast_alternative(
        self.spec.total_num_components is not None,
        lambda: tf.constant(self.spec.total_num_components, dtype, []),
        lambda: tf.size(self.sizes, out_type=dtype),
        'total_num_components != spec.total_num_components')

  @property
  def _get_features_ref(self) -> Fields:
    return self._data[_GraphPieceWithFeatures._DATAKEY_FEATURES]

  @classmethod
  def _from_features_and_sizes(cls, features: Fields, sizes: Field,
                               **extra_data) -> '_GraphPieceWithFeatures':
    """Constructs graph piece from features and component sizes."""
    assert isinstance(features, Mapping)
    sizes = gp.convert_to_tensor_or_ragged(sizes)
    prepared_features = {
        key: gp.convert_to_tensor_or_ragged(value)
        for key, value in features.items()
    }
    data = {
        _GraphPieceWithFeatures._DATAKEY_FEATURES: prepared_features,
        _GraphPieceWithFeatures._DATAKEY_SIZES: sizes
    }
    data.update({
        key: gp.convert_to_tensor_or_ragged(value)
        for key, value in extra_data.items()
    })
    # Note that this graph piece does not use any metadata fields.
    return cls._from_data(
        data=data, shape=sizes.shape[:-1], indices_dtype=sizes.dtype)

  def get_features_dict(self) -> Dict[FieldName, Field]:
    """Returns features copy as a dictionary."""
    return dict(self._get_features_ref)


class _GraphPieceWithFeaturesSpec(gp.GraphPieceSpecBase):
  """A type spec for `_GraphPieceWithFeatures`."""

  def __getitem__(self, feature_name: FieldName) -> FieldSpec:
    return self._get_features_spec_ref[feature_name]

  @property
  def features_spec(self) -> Mapping[FieldName, FieldSpec]:
    """A read-only mapping of feature name to feature spec."""
    return _as_immutable_mapping(self._get_features_spec_ref)

  @property
  def sizes_spec(self) -> FieldSpec:
    """The type spec for the sizes that provides num. elements per component."""
    return self._data_spec[_GraphPieceWithFeatures._DATAKEY_SIZES]  # pylint: disable=protected-access

  @property
  def total_num_components(self) -> Optional[int]:
    """The total number of graph components if known."""
    return self.sizes_spec.shape.num_elements()

  @property
  def total_size(self) -> Optional[int]:
    """The total number of graph items if known."""
    indicative_feature_spec = _get_indicative_feature(
        self._get_features_spec_ref)
    if indicative_feature_spec is None:
      return None
    else:
      return indicative_feature_spec.shape[:(self.rank + 1)].num_elements()

  @property
  def _get_features_spec_ref(self) -> FieldsSpec:
    return self._data_spec[_GraphPieceWithFeatures._DATAKEY_FEATURES]  # pylint: disable=protected-access

  @classmethod
  def _from_feature_and_size_specs(
      cls, features_spec: FieldsSpec, sizes_spec: FieldSpec,
      **extra_data) -> '_GraphPieceWithFeaturesSpec':
    """Constructs GraphPieceSpec from specs of features and component sizes."""
    # pylint: disable=protected-access
    assert isinstance(features_spec, Mapping)
    data_spec = {
        _NodeOrEdgeSet._DATAKEY_FEATURES: features_spec.copy(),
        _NodeOrEdgeSet._DATAKEY_SIZES: sizes_spec
    }
    data_spec.update(extra_data)
    # Note that this graph piece does not use any metadata fields.
    return cls._from_data_spec(
        data_spec, shape=sizes_spec.shape[:-1], indices_dtype=sizes_spec.dtype)


def resolve_value(graph_piece: _GraphPieceWithFeatures,
                  *,
                  feature_value: Optional[Field] = None,
                  feature_name: Optional[FieldName] = None) -> Field:
  """Resolves feature value by its name or provided value."""
  if (feature_value is None) == (feature_name is None):
    raise ValueError('One of feature name of feature value must be specified.')

  if feature_value is not None:
    # TODO(b/189087785): Check that the value shape is valid for `graph_piece`.
    return feature_value
  if feature_name is not None:
    return graph_piece[feature_name]
  assert False, 'This should never happen, please file a bug with TF-GNN.'


class Context(_GraphPieceWithFeatures):
  """A composite tensor for graph context features.

  The items of the context are the graph components (just like the items of a
  node set are the nodes and the items of an edge set are the edges). The
  `Context` is a composite tensor. It stores features that belong to a graph
  component as a whole, not any particular node or edge. Each context feature
  has a shape `[*graph_shape, num_components, ...]`, where `num_components` is
  the number of graph components in a graph (could be ragged).
  """

  @classmethod
  def from_fields(cls,
                  *,
                  features: Optional[Fields] = None,
                  sizes: Optional[Field] = None,
                  shape: Optional[ShapeLike] = None,
                  indices_dtype: Optional[tf.dtypes.DType] = None) -> 'Context':
    """Constructs a new instance from context fields.

    Example:

    ```python
    tfgnn.Context.from_fields(features={'country_code': ['CH']})
    ```

    Args:
      features: A mapping from feature name to feature Tensor or RaggedTensor.
        All feature tensors must have shape `[*graph_shape, num_components,
        *feature_shape]`, where `num_components` is the number of graph
        components (could be ragged); `feature_shape` are feature-specific inner
        dimensions.
      sizes: A Tensor of 1's with shape `[*graph_shape, num_components]`, where
        `num_components` is the number of graph components (could be ragged).
        For symmetry with `sizes` in NodeSet and EdgeSet, this counts the items
        per graph component, but since the items of Context are the components
        themselves, each value is 1. Must be compatible with `shape`, if that is
        specified.
      shape: The shape of this tensor and a GraphTensor containing it, also
        known as the `graph_shape`. If not specified, the shape is inferred from
        `sizes` or set to `[]` if the `sizes` is not specified.
      indices_dtype: An `indices_dtype` of a GraphTensor containing this object,
        used as `row_splits_dtype` when batching potentially ragged fields. If
        `sizes` are specified they are casted to that type.

    Returns:
      A `Context` composite tensor.

    """

    if indices_dtype is not None:
      if indices_dtype not in (tf.int64, tf.int32):
        raise ValueError(f'Expected indices_dtype={indices_dtype}'
                         ' to be tf.int64 or tf.int32')
    if shape is not None:
      shape = shape if isinstance(shape,
                                  tf.TensorShape) else tf.TensorShape(shape)

    if sizes is not None:
      sizes = gp.convert_to_tensor_or_ragged(sizes)
      if indices_dtype is not None and indices_dtype != sizes.dtype:
        sizes = tf.cast(sizes, dtype=indices_dtype)

    if shape is not None and sizes is not None:
      if sizes.shape.rank != shape.rank + 1:
        raise ValueError('The `sizes` `rank != shape.rank + 1`: '
                         f' shape={shape}'
                         f' sizes.shape={sizes.shape}')

      if not shape.is_compatible_with(sizes.shape[:shape.rank]):
        raise ValueError('The `sizes` is not compatible with the `shape`: '
                         f' shape={shape}'
                         f' sizes.shape={sizes.shape}')

    if features is None:
      features = {}
    else:
      features = {
          k: gp.convert_to_tensor_or_ragged(v) for k, v in features.items()
      }
    if sizes is None:
      shape = _ifnone(shape, tf.TensorShape([]))
      indices_dtype = _ifnone(indices_dtype, const.default_indices_dtype)

      indicative_feature = _get_indicative_feature(features)
      if indicative_feature is None:
        # There are no features to use for sizes inference. Assume that the
        # Context has no components and set sizes accordingly.
        size_dims = [_ifnone(dim, 0) for dim in shape.concatenate([0])]
        sizes = tf.ones(shape=size_dims, dtype=indices_dtype)
      else:
        sizes = utils.ones_like_leading_dims(
            indicative_feature, shape.rank + 1, dtype=indices_dtype)

    return cls._from_features_and_sizes(features=features, sizes=sizes)

  def replace_features(self, features: Fields) -> 'Context':
    """Returns a new instance with a new set of features."""
    assert isinstance(features, Mapping)
    return self.__class__.from_fields(
        features=features,
        sizes=self.sizes,
        shape=self.shape,
        indices_dtype=self.indices_dtype)

  @staticmethod
  def _type_spec_cls():
    return ContextSpec

  def __repr__(self):
    return (f'Context('
            f'features={utils.short_features_repr(self.features)}, '
            f'sizes={self.sizes}, '
            f'shape={self.shape}, '
            f'indices_dtype={self.indices_dtype!r})')


@tf_internal.type_spec_register('tensorflow_gnn.ContextSpec.v2')
class ContextSpec(_GraphPieceWithFeaturesSpec):
  """A type spec for `tfgnn.Context`."""

  @classmethod
  def from_field_specs(
      cls,
      *,
      features_spec: Optional[FieldsSpec] = None,
      sizes_spec: Optional[FieldSpec] = None,
      shape: ShapeLike = tf.TensorShape([]),
      indices_dtype: tf.dtypes.DType = const.default_indices_dtype
  ) -> 'ContextSpec':
    """The counterpart of `Context.from_fields()` for field type specs."""
    shape = shape if isinstance(shape,
                                tf.TensorShape) else tf.TensorShape(shape)

    if features_spec is None:
      features_spec = {}

    if sizes_spec is None:
      indicative_feature_spec = _get_indicative_feature(features_spec)
      is_ragged = False
      if indicative_feature_spec is None:
        sizes_shape = shape.concatenate([0])
      else:
        components_dim = indicative_feature_spec.shape[shape.rank]
        sizes_shape = shape.concatenate(tf.TensorShape([components_dim]))
        if isinstance(indicative_feature_spec, tf.RaggedTensorSpec):
          is_ragged = (shape.rank > 0) and (components_dim is None)

      if is_ragged:
        sizes_spec = tf.RaggedTensorSpec(
            shape=sizes_shape,
            ragged_rank=shape.rank + 1,
            dtype=indices_dtype,
            row_splits_dtype=indices_dtype)
      else:
        sizes_spec = tf.TensorSpec(shape=sizes_shape, dtype=indices_dtype)

    return cls._from_feature_and_size_specs(features_spec, sizes_spec)

  @property
  def value_type(self):
    return Context

  def relax(self, *, num_components: bool = False) -> 'ContextSpec':
    """Allows variable number of graph components.

    Calling with all default parameters keeps the spec unchanged.

    Args:
      num_components: if True, allows variable number of graph components by
        setting the outermost sizes dimension to `None`.

    Returns:
      Relaxed compatible context spec.

    Raises:
      ValueError: if contex is not scalar (rank > 0).
    """
    gp.check_scalar_graph_piece(self, 'ContextSpec.relax()')
    if not num_components:
      return self

    return self.from_field_specs(
        features_spec=_relax_outer_dim_if(num_components, self.features_spec),
        sizes_spec=_relax_outer_dim_if(num_components, self.sizes_spec))


class _NodeOrEdgeSet(_GraphPieceWithFeatures):
  """Base class for node set or edge set."""

  def replace_features(self, features: Mapping[FieldName,
                                               Field]) -> '_NodeOrEdgeSet':
    """Returns a new instance with a new set of features."""
    assert isinstance(features, Mapping)
    new_data = self._data.copy()
    new_data.update({_NodeOrEdgeSet._DATAKEY_FEATURES: features})
    return self.__class__.from_fields(**new_data)


class _NodeOrEdgeSetSpec(_GraphPieceWithFeaturesSpec):
  """A type spec for _NodeOrEdgeSet."""
  pass


class NodeSet(_NodeOrEdgeSet):
  """A composite tensor for node set features plus size information.

  The items of the node set are subset of graph nodes.

  All nodes in a node set have the same features, identified by a string key.
  Each feature is stored as one tensor and has shape `[*graph_shape, num_nodes,
  *feature_shape]`. The `num_nodes` is the number of nodes in a graph (could be
  ragged). The `feature_shape` is the shape of the feature value for each node.
  NodeSet supports both fixed-size and variable-size features. The fixed-size
  features must have fully defined feature_shape. They are stored as `tf.Tensor`
  if `num_nodes` is fixed-size or `graph_shape.rank = 0`. Variable-size node
  features are always stored as `tf.RaggedTensor`.

  Note that node set features are indexed without regard to graph components.
  The information which node belong to which graph component is contained in
  the `.sizes` tensor which defines the number of nodes in each graph component.
  """

  @classmethod
  def from_fields(cls,
                  *,
                  features: Optional[Fields] = None,
                  sizes: Field) -> 'NodeSet':
    """Constructs a new instance from node set fields.

    Example:

    ```python
    tfgnn.NodeSet.from_fields(
        sizes=tf.constant([3]),
        features={
            "tokenized_title": tf.ragged.constant(
                [["Anisotropic", "approximation"],
                 ["Better", "bipartite", "bijection", "bounds"],
                 ["Convolutional", "convergence", "criteria"]]),
            "embedding": tf.zeros([3, 128]),
            "year": tf.constant([2018, 2019, 2020]),
        })
    ```

    Args:
      features: A mapping from feature name to feature Tensors or RaggedTensors.
        All feature tensors must have shape `[*graph_shape, num_nodes,
        *feature_shape]`, where `num_nodes` is the number of nodes in the node
        set (could be ragged) and feature_shape is a shape of the feature value
        for each node.
      sizes: A number of nodes in each graph component. Has shape
        `[*graph_shape, num_components]`, where `num_components` is the number
        of graph components (could be ragged).

    Returns:
      A `NodeSet` composite tensor.
    """
    if features is None:
      features = {}
    return cls._from_features_and_sizes(features=features, sizes=sizes)

  @staticmethod
  def _type_spec_cls():
    return NodeSetSpec

  def __repr__(self):
    return (f'NodeSet('
            f'features={utils.short_features_repr(self.features)}, '
            f'sizes={self.sizes})')


@tf_internal.type_spec_register('tensorflow_gnn.NodeSetSpec')
class NodeSetSpec(_NodeOrEdgeSetSpec):
  """A type spec for `tfgnn.NodeSet`."""

  @classmethod
  def from_field_specs(cls,
                       *,
                       features_spec: Optional[FieldsSpec] = None,
                       sizes_spec: FieldSpec) -> 'NodeSetSpec':
    """The counterpart of `NodeSet.from_fields()` for values type specs."""
    if features_spec is None:
      features_spec = {}

    return cls._from_feature_and_size_specs(
        features_spec=features_spec, sizes_spec=sizes_spec)

  @property
  def value_type(self):
    return NodeSet

  def relax(self,
            *,
            num_components: bool = False,
            num_nodes: bool = False) -> 'NodeSetSpec':
    """Allows variable number of nodes or/and graph components.

    Calling with all default parameters keeps the spec unchanged.

    Args:
      num_components: if True, allows variable number of graph components by
        setting the outermost sizes dimension to `None`.
      num_nodes: if True, allows variable number of nodes by setting the
        outermost features dimensions to `None`.

    Returns:
      Relaxed compatible edge set spec.

    Raises:
      ValueError: if edge set is not scalar (rank > 0).
    """
    gp.check_scalar_graph_piece(self, 'NodeSetSpec.relax()')

    return self.from_field_specs(
        features_spec=_relax_outer_dim_if(num_nodes, self.features_spec),
        sizes_spec=_relax_outer_dim_if(num_components, self.sizes_spec))


class EdgeSet(_NodeOrEdgeSet):
  """A composite tensor for edge set features, size and adjacency information.

  Each edge set contains edges as its items that connect nodes from particular
  node sets. The information which edges connect which nodes is encapsulated in
  the `EdgeSet.adjacency` composite tensor (see adjacency.py).

  All edges in a edge set have the same features, identified by a string key.
  Each feature is stored as one tensor and has shape `[*graph_shape, num_edges,
  *feature_shape]`. The `num_edges` is a number of edges in a graph (could be
  ragged). The `feature_shape` is a shape of the feature value for each edge.
  EdgeSet supports both fixed-size and variable-size features. The fixed-size
  features must have fully defined feature_shape. They are stored as `tf.Tensor`
  if `num_edges` is fixed-size or `graph_shape.rank = 0`. Variable-size edge
  features are always stored as `tf.RaggedTensor`.

  Note that edge set features are indexed without regard to graph components.
  The information which edge belong to which graph component is contained in
  the `.sizes` tensor which defines the number of edges in each graph component.
  """

  _DATAKEY_ADJACENCY = 'adjacency'  # An Adjacency GraphPiece.

  @classmethod
  def from_fields(cls,
                  *,
                  features: Optional[Fields] = None,
                  sizes: Field,
                  adjacency: Adjacency) -> 'EdgeSet':
    """Constructs a new instance from edge set fields.

    Example 1:

    ```python
    tfgnn.EdgeSet.from_fields(
        sizes=tf.constant([3]),
        adjacency=tfgnn.Adjacency.from_indices(
            source=("paper", [1, 2, 2]),
            target=("paper", [0, 0, 1])))
    ```

    Example 2:

    ```python
    tfgnn.EdgeSet.from_fields(
        sizes=tf.constant([4]),
        adjacency=tfgnn.Adjacency.from_indices(
            source=("paper", [1, 1, 1, 2]),
            target=("author", [0, 1, 1, 3])))
    ```

    Args:
      features: A mapping from feature name to feature Tensor or RaggedTensor.
        All feature tensors must have shape `[*graph_shape, num_edges,
        *feature_shape]`, where num_edge is the number of edges in the edge set
        (could be ragged) and feature_shape is a shape of the feature value for
        each edge.
      sizes: The number of edges in each graph component. Has shape
        `[*graph_shape, num_components]`, where `num_components` is the number
        of graph components (could be ragged).
      adjacency: One of the supported adjacency types (see adjacency.py).

    Returns:
      An `EdgeSet` composite tensor.
    """
    if features is None:
      features = {}
    return cls._from_features_and_sizes(
        features=features, sizes=sizes, adjacency=adjacency)

  @property
  def adjacency(self) -> Adjacency:
    """The information which edges connect which nodes (see tfgnn.Adjacency)."""
    return self._data[EdgeSet._DATAKEY_ADJACENCY]

  @staticmethod
  def _type_spec_cls():
    return EdgeSetSpec

  def __repr__(self):
    return (f'EdgeSet('
            f'features={utils.short_features_repr(self.features)}, '
            f'sizes={self.sizes}, '
            f'adjacency={self.adjacency})')


@tf_internal.type_spec_register('tensorflow_gnn.EdgeSetSpec')
class EdgeSetSpec(_NodeOrEdgeSetSpec):
  """A type spec for `tfgnn.EdgeSet`."""

  @classmethod
  def from_field_specs(cls,
                       *,
                       features_spec: Optional[FieldsSpec] = None,
                       sizes_spec: FieldSpec,
                       adjacency_spec: AdjacencySpec) -> 'EdgeSetSpec':
    """The counterpart of `EdgeSet.from_fields()` for values type specs."""
    # pylint: disable=protected-access
    if features_spec is None:
      features_spec = {}

    return cls._from_feature_and_size_specs(
        features_spec=features_spec,
        sizes_spec=sizes_spec,
        **{EdgeSet._DATAKEY_ADJACENCY: adjacency_spec})

  @property
  def value_type(self):
    return EdgeSet

  @property
  def adjacency_spec(self) -> AdjacencySpec:
    """A type spec for the adjacency composite tensor."""
    return self._data_spec[EdgeSet._DATAKEY_ADJACENCY]  # pylint: disable=protected-access

  @property
  def total_size(self) -> Optional[int]:
    """The total number of edges if known."""
    return _ifnone(super().total_size, self.adjacency_spec.total_size)

  def relax(self,
            *,
            num_components: bool = False,
            num_edges: bool = False) -> 'EdgeSetSpec':
    """Allows variable number of edge or/and graph components.

    Calling with all default parameters keeps the spec unchanged.

    Args:
      num_components: if True, allows variable number of graph components by
        setting the outermost sizes dimension to `None`.
      num_edges: if True, allows variable number of edges by setting the
        outermost features dimensions to `None`.

    Returns:
      Relaxed compatible edge set spec.

    Raises:
      ValueError: if edge set is not scalar (rank > 0).
    """
    gp.check_scalar_graph_piece(self, 'EdgeSetSpec.relax()')

    return self.from_field_specs(
        features_spec=_relax_outer_dim_if(num_edges, self.features_spec),
        adjacency_spec=self.adjacency_spec.relax(num_edges=num_edges),
        sizes_spec=_relax_outer_dim_if(num_components, self.sizes_spec))


class GraphTensor(gp.GraphPieceBase):
  """A composite tensor for heterogeneous directed graphs with features.

  A GraphTensor is an immutable container (as any composite tensor) to represent
  one or more heterogeneous directed graphs, as defined in the GraphTensor guide
  (or even hypergraphs). A GraphTensor consists of NodeSets, EdgeSets and a
  Context (collectively known as graph pieces), which are also composite
  tensors. The graph pieces consist of fields, which are `tf.Tensor`s and/or
  `tf.RaggedTensor`s that store the graph structure (esp. the edges between
  nodes) and user-defined features.

  In the same way as `tf.Tensor` has numbers as its elements, the elements of
  the GraphTensor are graphs. Its `shape` of `[]` describes a scalar (single)
  graph, a shape of `[d0]` describes a `d0`-vector of graphs, a shape of
  `[d0, d1]` a `d0` x `d1` matrix of graphs, and so on.

  RULE: In the shape of a GraphTensor, no dimension except the outermost is
  allowed to be `None` (that is, of unknown size). Future versions of
  GraphTensor may lift that requirement.

  Each of those graphs in a GraphTensor consists of 0 or more disjoint
  (sub-)graphs called graph components. The number of components could vary
  from graph to graph or be fixed to a value known statically. On a batched
  GraphTensor, one can call the method merge_batch_to_components() to merge all
  graphs of the batch into one, contiguously indexed graph containing the same
  components as the original graph tensor. See the GraphTensor guide for the
  typical usage that has motivated this design (going from input graphs with one
  component each to a batch of input graphs and on to one merged graph with
  multiple components for use in GNN model).

  Example 1:

  ```python
  # A homogeneous scalar graph tensor with 1 graph component, 10 nodes and 3
  # edges. Edges connect nodes 0 and 3, 5 and 7, 9 and 1. There are no features.
  tfgnn.GraphTensor.from_pieces(
      node_sets = {
          'node': tfgnn.NodeSet.from_fields(sizes=[10], features={})},
      edge_sets = {
          'edge': tfgnn.EdgeSet.from_fields(
              sizes=[3],
              features={},
              adjacency=tfgnn.Adjacency.from_indices(
                  source=('node', [0, 5, 9]),
                  target=('node', [3, 7, 1])))})
  ```

  All graph pieces provide a mapping interface to access their features by name
  as `graph_piece[feature_name]`. Each graph piece feature has the shape
  `[*graph_shape, num_items, *feature_shape]`, where `graph_shape` is the shape
  of the GraphTensor, `num_items` is the number of items in a piece (number of
  graph components, number of nodes in a node set or edges in an edge set). The
  `feature_shape` is the shape of the feature value for each item.

  Naturally, the first `GraphTensor.rank` dimensions of all graph tensor fields
  must index the same graphs, the item dimension must correspond to the same
  item (graph component, node or edge) within the same graph piece (context,
  node set or edge set).

  RULE: 'None' always denotes ragged or outermost field dimension. Uniform
  dimensions must have a fixed size that is given in the dimension.

  In particular this rule implies that if a feature has `tf.Tensor` type its
  `feature_shape` must by fully defined.

  Example 2:

  ```python
  # A scalar graph tensor with edges between authors, papers and their venues
  # (journals or conferences). Each venue belongs to one graph component. The
  # 1st venue (1980519) has 2 authors and 3 papers.
  # The 2nd venue (9756463) has 2 authors and 1 paper.
  # The paper 0 is written by authors 0 and 2; paper 1 - by authors 0 and 1;
  # paper 2 - by author 2; paper 3 - by author 3.
  venues = tfgnn.GraphTensor.from_pieces(
      context=tfgnn.Context.from_fields(
          features={'venue': [1980519, 9756463]}),
      node_sets={
          'author': tfgnn.NodeSet.from_fields(sizes=[2, 2], features={}),
          'paper': tfgnn.NodeSet.from_fields(
              sizes=[3, 1], features={'year': [2018, 2017, 2017, 2022]})},
      edge_sets={
          'is_written': tfgnn.EdgeSet.from_fields(
              sizes=[4, 2],
              features={},
              adjacency=tfgnn.Adjacency.from_indices(
                  source=('paper', [0, 1, 1, 0, 2, 3]),
                  target=('author', [0, 0, 1, 2, 3, 3])))})
  ```

  The assignment of an item to its graph components is stored as the `sizes`
  attribute of the graph piece. Its shape is `[*graph_shape, num_components]`
  (the same for all pieces). The stored values are number of items in each graph
  component (so the `.sizes` has all 1s as its values).

  Example 3:

  ```python
  # The year of publication of the first article in each venue from the
  # previous example.
  papers = venues.node_sets['paper']
  years_by_venue = tf.RaggedTensor.from_row_lengths(
      papers['year'], papers.sizes
  )
  first_paper_year = tf.reduce_min(years_by_venue, -1)  # [2017, 2022]
  ```

  The GraphTensor, as a composite tensor, can be used directly in a
  tf.data.Dataset, as an input or output of a Keras Layer or a tf.function,
  and so on. As any other tensor, the GraphTensor has an associated type
  specification object, a GraphTensorSpec, that holds the `tf.TensorSpec` and
  `tf.RaggedTensorSpec` objects for all its fields, plus a bit of metadata such
  as graph connectivity (see tfgnn.GraphTensorSpec).

  The GraphTensor allows batching of graphs. Batching changes a GraphTensor
  instance's shape to `[batch_size, *graph_shape]` and the GraphTensor's
  rank is increased by 1. Unbatching removes dimension-0, as if truncating with
  `shape[1:]`, and the GraphTensor's rank is decreased by 1. This works
  naturally with the batch and unbatch methods of tf.data.Datatset.

  RULE: Batching followed by unbatching results in a dataset with equal
  GraphTensors as before, except for the last incomplete batch (if batching
  used `drop_remainder=True`).

  For now, GraphTensor requires that GraphTensor.shape does not contain `None`,
  except maybe as the outermost dimension. That means repeated calls to
  `.batch()` must set `drop_remainder=True` in all but the last one. Future
  versions of GraphTensor may lift that requirement.

  All pieces and its fields are batched together with their GraphTensor so that
  shapes of a graph tensor, its pieces and features are all in sync.

  RULE: Batching fields with an outermost dimension of `None` turns it into a
  ragged dimension of a RaggedTensor. (Note this is only allowed for the items
  dimension, not a graph dimension.) In all other cases, the type of the field
  (Tensor or RaggedTensor) is preserved.
  """
  _DATAKEY_CONTEXT = 'context'  # A Context.
  _DATAKEY_NODE_SETS = 'node_sets'  # A Mapping[NodeSetName, NodeSet].
  _DATAKEY_EDGE_SETS = 'edge_sets'  # A Mapping[EdgeSetName, EdgeSet].

  @classmethod
  def from_pieces(
      cls,
      context: Optional[Context] = None,
      node_sets: Optional[Mapping[NodeSetName, NodeSet]] = None,
      edge_sets: Optional[Mapping[EdgeSetName, EdgeSet]] = None,
  ) -> 'GraphTensor':
    """Constructs a new `GraphTensor` from context, node sets and edge sets."""
    context = _ifnone(context, Context.from_fields(features={}))
    node_sets = _ifnone(node_sets, dict()).copy()
    edge_sets = _ifnone(edge_sets, dict()).copy()
    indicative_piece = _get_indicative_graph_piece(context, node_sets,
                                                   edge_sets)
    if isinstance(context.spec, ContextSpec) and isinstance(
        indicative_piece.spec, _NodeOrEdgeSetSpec):
      # Reevaluate context sizes from the node or edge set sizes. The latter are
      # directly set by user, whereas the context sizes are indirectly inferred
      # from the context feature shapes or set to zero-shaped tensor if the
      # context has no features.
      context_sizes = tf.ones_like(indicative_piece.sizes,
                                   indicative_piece.indices_dtype)
      context = context.from_fields(
          features=context.features, sizes=context_sizes)

    assert isinstance(indicative_piece, _GraphPieceWithFeatures)
    return cls._from_data(
        data={
            GraphTensor._DATAKEY_CONTEXT: context,
            GraphTensor._DATAKEY_NODE_SETS: node_sets,
            GraphTensor._DATAKEY_EDGE_SETS: edge_sets
        },
        shape=indicative_piece.shape,
        indices_dtype=indicative_piece.indices_dtype)

  def merge_batch_to_components(self) -> 'GraphTensor':
    """Merges all contained graphs into one contiguously indexed graph.

    On a batched GraphTensor, one can call this method merge all graphs of the
    batch into one, contiguously indexed graph. The resulting GraphTensor has
    shape [] (i.e., is scalar) and its features have the shape
    `[total_num_items, *feature_shape]` where `total_num_items` is the sum of
    the previous `num_items` per batch element. Most TF-GNN models expect scalar
    GraphTensors. There is no function to reverse this method.


    Example: Flattening of

    ```python
    tfgnn.GraphTensor.from_pieces(
        node_sets={
            'node': tfgnn.NodeSet.from_fields(
                # Three graphs:
                #   - 1st graph has two components with 2 and 1 nodes;
                #   - 2nd graph has 1 component with 1 node;
                #   - 3rd graph has 1 component with 1 node;
                sizes=tf.ragged.constant([[2, 1], [1], [1]]),
                features={
                   'id': tf.ragged.constant([['a11', 'a12', 'a21'],
                                             ['b11'],
                                             ['c11']])})},
        edge_sets={
            'edge': tfgnn.EdgeSet.from_fields(
                sizes=tf.ragged.constant([[3, 1], [1], [1]]),
                features={},
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('node', tf.ragged.constant([[0, 1, 1, 2],
                                                        [0],
                                                        [0]])),
                    target=('node', tf.ragged.constant([[0, 0, 1, 2],
                                                        [0],
                                                        [0]]))))})
    ```

    results in the equivalent graph of

    ```python
    tfgnn.GraphTensor.from_pieces(
        node_sets={
            'node': tfgnn.NodeSet.from_fields(
                # One graph with 4 components with 2, 1, 1, 1 nodes.
                sizes=[2, 1, 1, 1],
                features={'id': ['a11', 'a12', 'a21', 'b11', 'c11']})},
        edge_sets={
            'edge': tfgnn.EdgeSet.from_fields(
                sizes=[3, 2, 1, 1],
                features={},
                # Note how node indices have changes to reference nodes
                # withing the same graph ignoring its components.
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('node', [0, 1, 1, 2, 3 + 0, 3 + 1 + 0]),
                    target=('node', [0, 0, 1, 2, 3 + 0, 3 + 1 + 0])))})
    ```

    Returns:
      A scalar (rank 0) graph tensor.
    """

    if self.rank == 0:
      return self

    def num_elements(node_or_edge_set) -> tf.Tensor:
      sizes = node_or_edge_set.sizes
      sizes = tf.math.reduce_sum(sizes, axis=-1)
      assert isinstance(sizes, tf.Tensor)
      return tf.reshape(sizes, [-1])

    def edge_set_merge_batch_to_components(edge_set: EdgeSet) -> EdgeSet:
      return edge_set._merge_batch_to_components(  # pylint: disable=protected-access
          num_edges_per_example=num_elements(edge_set),
          num_nodes_per_example=num_nodes)

    num_nodes = {
        set_name: num_elements(n) for set_name, n in self.node_sets.items()
    }
    return self.__class__.from_pieces(
        context=self.context._merge_batch_to_components(),  # pylint: disable=protected-access
        node_sets=tf.nest.map_structure(
            lambda n: n._merge_batch_to_components(), self.node_sets.copy()),  # pylint: disable=protected-access
        edge_sets=tf.nest.map_structure(edge_set_merge_batch_to_components,
                                        self.edge_sets.copy()))

  @property
  def context(self) -> Context:
    """The graph context."""
    return self._data[GraphTensor._DATAKEY_CONTEXT]

  @property
  def node_sets(self) -> Mapping[NodeSetName, NodeSet]:
    """A read-only mapping from node set name to the node set."""
    return _as_immutable_mapping(self._data[GraphTensor._DATAKEY_NODE_SETS])

  @property
  def edge_sets(self) -> Mapping[EdgeSetName, EdgeSet]:
    """A read-only mapping from node set name to the node set."""
    return _as_immutable_mapping(self._data[GraphTensor._DATAKEY_EDGE_SETS])

  @property
  def num_components(self) -> tf.Tensor:
    """The number of graph components for each graph.

    Returns:
      A dense integer tensor with the same shape as the GraphTensor.
    """
    indicative_piece = _get_indicative_graph_piece(self.context, self.node_sets,
                                                   self.edge_sets)
    return indicative_piece.num_components

  @property
  def total_num_components(self) -> tf.Tensor:
    """The total number of graph components.

    Returns:
      A scalar integer tensor equal to `tf.math.reduce_sum(num_components)`. If
      result is statically known (`spec.total_num_components is not None`), the
      output is a constant Tensor (suitable for environments in which constant
      shapes are required, like TPU).
    """
    indicative_piece = _get_indicative_graph_piece(self.context, self.node_sets,
                                                   self.edge_sets)
    return cast(_GraphPieceWithFeatures, indicative_piece).total_num_components

  def replace_features(
      self,
      context: Optional[Fields] = None,
      node_sets: Optional[Mapping[NodeSetName, Fields]] = None,
      edge_sets: Optional[Mapping[EdgeSetName, Fields]] = None,
  ) -> 'GraphTensor':
    """Returns a new instance with a new set of features for the same topology.

    Example 1. Replaces all features for node set 'node.a' but not 'node.b'.

    ```python
    graph = tfgnn.GraphTensor.from_pieces(
        context=tfgnn.Context.from_fields(
            features={'label': tf.ragged.constant([['A'], ['B']])}),
        node_sets={
            'node.a': tfgnn.NodeSet.from_fields(
                features={'id': ['a1', 'a3']},
                sizes=[2]),
            'node.b': tfgnn.NodeSet.from_fields(
                features={'id': ['b4', 'b1']},
                sizes=[2])})
    result = graph.replace_features(
        node_sets={
            'node.a': {
                'h': tf.ragged.constant([[1., 0.], [3., 0.]])
             }
        })
    ```

    Result:

    ```python
    tfgnn.GraphTensor.from_pieces(
        context=tfgnn.Context.from_fields(
            features={'label': tf.ragged.constant([['A'], ['B']])}),
        node_sets={
            'node.a': tfgnn.NodeSet.from_fields(
                features={
                    'h': tf.ragged.constant([[1., 0.], [3., 0.]])
                },
                sizes=[2]),
            'node.b': tfgnn.NodeSet.from_fields(
                features={
                    'id': ['b4', 'b1']
                },
                sizes=[2])
        })
    ```

    Args:
      context: A substitute for the context features, or `None` (which keeps the
        prior features). Their tensor shapes must match the number of existing
        components, which remains unchanged.
      node_sets: Substitutes for the features of the specified node sets. Their
        tensor shapes must match the existing number of nodes, which remains
        unchanged. Features on node sets that are not included remain unchanged.
      edge_sets: Substitutes for the features of the specified edge sets. Their
        tensor shapes must match the existing number of edges. The number of
        edges and their incident nodes are unchanged. Features on edge sets that
        are not included remain unchanged.

    Returns:
      A `GraphTensor` instance with feature maps replaced according to the
      arguments.
    Raises:
      ValueError: if some node sets or edge sets are not present in the graph
        tensor.
    """
    if context is None:
      new_context = self.context
    else:
      new_context = self.context.replace_features(context)

    if node_sets is None:
      new_node_sets = self.node_sets.copy()
    else:
      not_present = set(node_sets.keys()) - set(self.node_sets.keys())
      if not_present:
        raise ValueError(('Some node sets in the `node_sets` are not present'
                          f' in the graph tensor: {sorted(not_present)}'))
      new_node_sets = {
          set_name: (node_set.replace_features(node_sets[set_name])
                     if set_name in node_sets else node_set)
          for set_name, node_set in self.node_sets.items()
      }

    if edge_sets is None:
      new_edge_sets = self.edge_sets.copy()
    else:
      not_present = set(edge_sets.keys()) - set(self.edge_sets.keys())
      if not_present:
        raise ValueError(('Some edge sets in the `edge_sets` are not present'
                          f' in the graph tensor: {sorted(not_present)}'))
      new_edge_sets = {
          set_name: (edge_set.replace_features(edge_sets[set_name])
                     if set_name in edge_sets else edge_set)
          for set_name, edge_set in self.edge_sets.items()
      }

    return self.__class__.from_pieces(new_context, new_node_sets, new_edge_sets)

  def remove_features(
      self,
      context: Optional[Sequence[FieldName]] = None,
      node_sets: Optional[Mapping[NodeSetName, Sequence[FieldName]]] = None,
      edge_sets: Optional[Mapping[NodeSetName, Sequence[FieldName]]] = None,
  ) -> 'GraphTensor':
    """Returns a new GraphTensor with some features removed.

    The graph topology and the other features remain unchanged.

    Example 1. Removes the id feature from node set 'node.a'.

    ```python
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            'node.a': tfgnn.NodeSet.from_fields(
                features={'id': ['a1', 'a3']},
                sizes=[2]),
            'node.b': tfgnn.NodeSet.from_fields(
                features={'id': ['b4', 'b1']},
                sizes=[2])})
    result = graph.remove_features(node_sets={'node.a': ['id']})
    ```

    Result:

    ```python
    tfgnn.GraphTensor.from_pieces(
        node_sets={
            'node.a': tfgnn.NodeSet.from_fields(
                features={},
                sizes=[2]),
            'node.b': tfgnn.NodeSet.from_fields(
                features={'id': ['b4', 'b1']},
                sizes=[2])})
    ```

    Args:
      context: A list of feature names to remove from the context, or `None`.
      node_sets: A mapping from node set names to lists of feature names to be
        removed from the respective node sets.
      edge_sets: A mapping from edge set names to lists of feature names to be
        removed from the respective edge sets.

    Returns:
      A `GraphTensor` with the same graph topology as the input and a subset
      of its features. Each feature of the input either was named as a feature
      to be removed or is still present in the output.

    Raises:
      ValueError: if some feature names in the arguments were not present in the
        input graph tensor.
    """
    context_features = None
    if context:
      context_features = self.context.get_features_dict()
      for feature_name in set(context):
        try:
          context_features.pop(feature_name)
        except KeyError as e:
          raise ValueError(  # TODO(b/226560215): ...or KeyError?
              f'GraphTensor has no feature context[\'{feature_name}\']') from e
    node_sets_features = {}
    if node_sets:
      for node_set_name, feature_names in node_sets.items():
        node_sets_features[node_set_name] = self.node_sets[
            node_set_name].get_features_dict()
        for feature_name in set(feature_names):
          try:
            node_sets_features[node_set_name].pop(feature_name)
          except KeyError as e:
            raise ValueError(
                'GraphTensor has no feature '
                f'node_sets[\'{node_set_name}\'][\'{feature_name}\']') from e

    edge_sets_features = {}
    if edge_sets:
      for edge_set_name, feature_names in edge_sets.items():
        edge_sets_features[edge_set_name] = self.edge_sets[
            edge_set_name].get_features_dict()
        for feature_name in set(feature_names):
          try:
            edge_sets_features[edge_set_name].pop(feature_name)
          except KeyError as e:
            raise ValueError(
                'GraphTensor has no feature '
                f'edge_sets[\'{edge_set_name}\'][\'{feature_name}\']') from e

    return self.replace_features(
        context=context_features,
        node_sets=node_sets_features,
        edge_sets=edge_sets_features)

  @staticmethod
  def _type_spec_cls():
    return GraphTensorSpec

  def __repr__(self):
    # We define __repr__ instead of __str__ here because it's the default
    # for jupyter notebooks and interactive analysis. Keeping the full
    # tensor representations makes the __repr__ unreadably long very quickly
    # so we truncate it but still keep it unique.
    return (f'GraphTensor(\n'
            f'  context={self.context},\n'
            f'  node_set_names={list(self.node_sets.keys())},\n'
            f'  edge_set_names={list(self.edge_sets.keys())})')


@tf_internal.type_spec_register('tensorflow_gnn.GraphTensorSpec')
class GraphTensorSpec(gp.GraphPieceSpecBase):
  """A type spec for `tfgnn.GraphTensor`."""

  @classmethod
  def from_piece_specs(
      cls,
      context_spec: Optional[ContextSpec] = None,
      node_sets_spec: Optional[Mapping[NodeSetName, NodeSetSpec]] = None,
      edge_sets_spec: Optional[Mapping[EdgeSetName, EdgeSetSpec]] = None,
  ) -> 'GraphTensorSpec':
    """The counterpart of `GraphTensor.from_pieces` for pieces type specs."""
    context_spec = _ifnone(context_spec, ContextSpec.from_field_specs())
    node_sets_spec = _ifnone(node_sets_spec, dict())
    edge_sets_spec = _ifnone(edge_sets_spec, dict())
    indicative_piece = _get_indicative_graph_piece(context_spec, node_sets_spec,
                                                   edge_sets_spec)
    assert isinstance(indicative_piece, _GraphPieceWithFeaturesSpec)
    if isinstance(context_spec,
                  ContextSpec) and context_spec != indicative_piece:
      entity_sizes_spec = indicative_piece.sizes_spec
      if entity_sizes_spec is not None:
        context_spec = context_spec.from_field_specs(
            features_spec=context_spec.features_spec,
            sizes_spec=entity_sizes_spec)

    # pylint: disable=protected-access
    return cls._from_data_spec(
        {
            GraphTensor._DATAKEY_CONTEXT: context_spec,
            GraphTensor._DATAKEY_NODE_SETS: node_sets_spec,
            GraphTensor._DATAKEY_EDGE_SETS: edge_sets_spec
        },
        shape=indicative_piece.shape,
        indices_dtype=indicative_piece.indices_dtype)

  @property
  def value_type(self):
    return GraphTensor

  @property
  def context_spec(self) -> ContextSpec:
    """The graph context type spec."""
    return self._data_spec[GraphTensor._DATAKEY_CONTEXT]  # pylint: disable=protected-access

  @property
  def node_sets_spec(self) -> Mapping[NodeSetName, NodeSetSpec]:
    """A read-only mapping form node set name to the node set type spec."""
    return _as_immutable_mapping(
        self._data_spec[GraphTensor._DATAKEY_NODE_SETS])  # pylint: disable=protected-access

  @property
  def edge_sets_spec(self) -> Mapping[EdgeSetName, EdgeSetSpec]:
    """A read-only mapping form edge set name to the edge set type spec."""
    return _as_immutable_mapping(
        self._data_spec[GraphTensor._DATAKEY_EDGE_SETS])  # pylint: disable=protected-access

  @property
  def total_num_components(self) -> Optional[int]:
    """The total number of graph components if known."""
    indicative_piece = _get_indicative_graph_piece(self.context_spec,
                                                   self.node_sets_spec,
                                                   self.edge_sets_spec)
    assert indicative_piece is not None
    return cast(_GraphPieceWithFeaturesSpec,
                indicative_piece).total_num_components

  def relax(
      self,
      *,
      num_components: bool = False,
      num_nodes: bool = False,
      num_edges: bool = False,
  ) -> 'GraphTensorSpec':
    """Allows variable number of graph nodes, edges or/and graph components.

    Calling with all default parameters keeps the spec unchanged.

    Args:
      num_components: if True, allows a variable number of graph components.
      num_nodes: if True, allows a variable number of nodes in each node set.
      num_edges: if True, allows a variable number of edges in each edge set.

    Returns:
      Relaxed compatible graph tensor spec.

    Raises:
      ValueError: if graph tensor is not scalar (rank > 0).
    """
    check_scalar_graph_tensor(self, 'GraphTensorSpec.relax()')

    result = self.from_piece_specs(
        self.context_spec.relax(num_components=num_components), {
            name: spec.relax(
                num_components=num_components, num_nodes=num_nodes)
            for name, spec in self.node_sets_spec.items()
        }, {
            name: spec.relax(
                num_components=num_components, num_edges=num_edges)
            for name, spec in self.edge_sets_spec.items()
        })
    if const.validate_internal_results:
      assert self.is_compatible_with(
          result), f'{result} is not compatible with {self}.'
    return result


def _ifnone(value, default):
  return value if value is not None else default


def _get_indicative_feature(
    field_specs: Union[Field, FieldsSpec]) -> Optional[Union[Field, FieldSpec]]:
  """Deterministically selects one of the field specs."""

  def key_fn(item):
    """First not ragged by field name."""
    name, value = item
    if isinstance(value, tf.RaggedTensorSpec) or utils.is_ragged_tensor(value):
      ragged_rank = value.ragged_rank
    else:
      ragged_rank = 0
    return (ragged_rank, name)

  _, result = min(field_specs.items(), key=key_fn, default=('', None))
  return result


def _get_indicative_graph_piece(
    context: Union[Context,
                   ContextSpec], node_sets: Mapping[NodeSetName,
                                                    Union[NodeSet,
                                                          NodeSetSpec]],
    edge_sets: Mapping[EdgeSetName, Union[EdgeSet, EdgeSetSpec]]
) -> Union[_GraphPieceWithFeatures, _GraphPieceWithFeaturesSpec]:
  """Deterministically selects one of the graph pieces."""

  def first_by_name(
      elements: Mapping[EdgeSetName, Union[_GraphPieceWithFeatures,
                                           _GraphPieceWithFeaturesSpec]]
  ) -> Union[_GraphPieceWithFeatures, _GraphPieceWithFeaturesSpec]:
    _, result = min(elements.items(), key=lambda item: item[0])
    return result

  if edge_sets:
    return first_by_name(edge_sets)
  if node_sets:
    return first_by_name(node_sets)
  return context


class _ImmutableMapping(Mapping):
  """`tf.nest` friendly immutable mapping view implementation."""

  def __init__(self, data):
    """Constructs immutable mapping view from dictionary or generator.

    Wraps internal mutable dictionaries as immutable mapping containers without
    copying them. Compared to the `types.MappingProxyType` this implementation
    is compatible with `tf.nest.map_structure` as it could be instantiated from
    a generator. The latter:
     1. allows to use object as tf.function parameter or return value;
     2. makes it compatible with KerasTensor.

    Args:
      data: Mapping or generator object returning key-value pairs. When data is
        mapping the object preserve its reference without copying its content.
        When data is a generator its values are copied into internal dictionary.
    """
    super().__init__()
    if isinstance(data, Mapping):
      self._data = data
    else:
      self._data = dict(data)

  def __getitem__(self, key):
    return self._data[key]

  def __iter__(self):
    return iter(self._data)

  def __len__(self):
    return len(self._data)

  def __repr__(self):
    return repr(self._data)

  def copy(self):
    """Returns copy as a dictionary object."""
    return dict(self._data)


def _as_immutable_mapping(input_map):
  return _ImmutableMapping(input_map)


def _fast_alternative(use_fast_path: bool,
                      fast_eval_path_fn: Callable[[], tf.Tensor],
                      default_eval_path_fn: Callable[[], tf.Tensor],
                      debug_message: str) -> tf.Tensor:
  """Uses fast alternative computation path if `use_fast_path` is true."""
  if not use_fast_path:
    return default_eval_path_fn()

  if not const.validate_internal_results:
    return fast_eval_path_fn()

  fast_result = fast_eval_path_fn()
  result = default_eval_path_fn()
  with tf.control_dependencies(
      [tf.debugging.assert_equal(fast_result, result, message=debug_message)]):
    return tf.identity(fast_result)


def _relax_outer_dim_if(cond: bool, features_nest: Any) -> Any:
  if not cond:
    return features_nest
  return tf.nest.map_structure(utils.with_undefined_outer_dimension,
                               features_nest)


def check_scalar_graph_tensor(graph: Union[GraphTensor, GraphTensorSpec],
                              name='This operation') -> None:
  gp.check_scalar_graph_piece(graph, name=name)


def get_graph_tensor_spec(
    graph: Union[GraphTensor, GraphTensorSpec]) -> GraphTensorSpec:
  if isinstance(graph, GraphTensorSpec):
    return graph

  if hasattr(graph, 'spec') and isinstance(graph.spec, GraphTensorSpec):
    return graph.spec

  raise ValueError(f'Unsupported value type {type(graph).__name__}.')


def check_scalar_singleton_graph_tensor(graph: Union[GraphTensor,
                                                     GraphTensorSpec],
                                        name='This operation') -> None:
  """Checks that graph tensor is scalar with single graph component."""
  spec = get_graph_tensor_spec(graph)
  check_scalar_graph_tensor(graph, name=name)
  if spec.total_num_components != 1:
    raise ValueError(
        (f'{name} requires scalar GraphTensor with a single graph component,'
         f' got a scalar GraphTensor with {spec.total_num_components}'
         ' components.'))


def _fields_and_size_from_fieldorfields(
    features: FieldOrFields,
    default_feature_name: FieldName,
) -> tuple[Fields, Optional[Union[int, tf.Tensor]]]:
  """Returns a mapping from a default feature name if needed."""
  if isinstance(features, collections.abc.Mapping):
    num_entities = tf.stack(
        [utils.outer_dimension_size(_get_indicative_feature(features))])
  elif features is None:
    features, num_entities = {}, None
  else:
    num_entities = tf.stack([utils.outer_dimension_size(features)])
    features = {default_feature_name: features}
  return features, num_entities


def homogeneous(
    source: tf.Tensor,
    target: tf.Tensor,
    *,
    node_features: Optional[FieldOrFields] = None,
    edge_features: Optional[FieldOrFields] = None,
    context_features: Optional[FieldOrFields] = None,
    node_set_name: Optional[FieldName] = const.NODES,
    edge_set_name: Optional[FieldName] = const.EDGES,
    node_set_sizes: Optional[Field] = None,
    edge_set_sizes: Optional[Field] = None,
) -> GraphTensor:
  """Constructs a homogeneous `GraphTensor` with node features and one edge_set.

  Args:
    source: A dense Tensor with the source indices for edges
    target: A dense Tensor with the target indices for edges
    node_features: A Tensor or mapping from feature name to Tensor of features
      corresponding to graph nodes.
    edge_features: Optional Tensor or mapping from feature name to Tensor of
      features corresponding to graph edges.
    context_features: Optional Tensor or mapping from name to Tensor for the
      context (entire graph)
    node_set_name: Optional name for the node set
    edge_set_name: Optional name for the edge set
    node_set_sizes: Optional Tensor with the number of nodes per component. If
      this is provided, edge_set_sizes should also be passed and it should be
      the same length.
    edge_set_sizes: Optional Tensor with the number of edges per component. If
      this is provided, node_set_sizes should also be passed and it should be
      the same length.

  Returns:
    A scalar `GraphTensor` with a single node set and edge set.
  """

  # Featureless edges with no sizes are okay because we can compute sizes
  # from the adjacency source and target.
  # This is impossible for node sets with no features
  if node_features is None and node_set_sizes is None:
    raise ValueError('node_set_sizes must be provided if node_features is not')

  if source.shape.rank != 1 or target.shape.rank != 1:
    raise ValueError('source and target must be rank-1 dense tensors')

  node_features, num_nodes = _fields_and_size_from_fieldorfields(
      node_features, HIDDEN_STATE)
  edge_features, _ = _fields_and_size_from_fieldorfields(
      edge_features, HIDDEN_STATE)
  context_features, _ = _fields_and_size_from_fieldorfields(
      context_features, HIDDEN_STATE)

  num_edges = tf.shape(source)
  node_sizes = (num_nodes if node_set_sizes is None else node_set_sizes)
  edge_sizes = (num_edges if edge_set_sizes is None else edge_set_sizes)

  return GraphTensor.from_pieces(
      node_sets={
          node_set_name:
              NodeSet.from_fields(
                  sizes=node_sizes,
                  features=node_features,
              )
      },
      edge_sets={
          edge_set_name:
              EdgeSet.from_fields(
                  sizes=edge_sizes,
                  adjacency=adj.Adjacency.from_indices(
                      source=(node_set_name, source),
                      target=(node_set_name, target),
                  ),
                  features=edge_features,
              )
      },
      context=Context.from_fields(
          features=context_features,
          sizes=tf.ones_like(node_sizes),
      ),
  )


def get_aux_type_prefix(set_name: const.SetName) -> Optional[str]:
  """Returns type prefix of aux node or edge set names, or `None` if non-aux.

  Auxiliary node sets and edge sets in a `tfgnn.GraphTensor` have names that
  begin with an underscore `_` (preferred) or any of the characters `#`, `!`,
  `%`, `.`, `^`, and `~` (reserved for future use). They store structural
  information needed by helper functions like `tfgnn.structured_readout()`,
  beyond the node sets and edge sets that represent graph data from the
  application domain.

  By convention, the names of auxiliary graph pieces begin with a type prefix
  that contains the leading special character and all letters, digits and
  underscores following it.

  Users can define their own aux types and handle them in their own code.
  The TF-GNN library uses the following types:

    * `"_readout"` and (rarely) `"_shadow"`, for `tfgnn.structured_readout()`.

  See the named function(s) for how the respective types of auxiliary edge sets
  and node sets are formed.

  Args:
    set_name: The name of a node set or edge set in a `tfgnn.GraphTensor`,
      `tfgnn.GraphTensorSpec` or `tfgnn.GraphSchema`.

  Returns:
    For an auxiliary node set or edge set, a non-empty prefix that identifies
    its type; for other node sets or edge sets, `None`.
  """
  match = re.match(r'[_#!%.^~][a-zA-Z0-9_]*', set_name)
  if match:
    return match.group()


def get_homogeneous_node_and_edge_set_name(
    graph: Union[GraphTensor, GraphTensorSpec],
    name: str = 'This operation') -> tuple[str, str]:
  """Returns the sole `node_set_name, edge_set_name` or raises `ValueError`.

  By default, this function ignores auxiliary node sets and edge sets for which
  `tfgnn.get_aux_type_prefix(set_name) is not None` (e.g., those needed for
  `tfgnn.structured_readout()`), as appropriate for model-building code.

  Args:
    graph: the `GraphTensor` or `GraphTensorSpec` to check.
    name: Optionally, the name of the operation (library function, class, ...)
      to mention in the user-visible error message in case an exception is
      raised.

  Returns:
    A tuple `node_set_name, edge_set_name` with the unique node set and edge
    set, resp., in `graph` for which `tfgnn.get_aux_type_prefix(set_name)` is
    `None`.
  """
  spec = get_graph_tensor_spec(graph)
  node_set_names = [node_set_name for node_set_name in spec.node_sets_spec
                    if not get_aux_type_prefix(node_set_name)]
  edge_set_names = [edge_set_name for edge_set_name in spec.edge_sets_spec
                    if not get_aux_type_prefix(edge_set_name)]
  if len(node_set_names) != 1 or len(edge_set_names) != 1:
    raise ValueError(
        f'{name} requires a graph with 1 node set and 1 edge set '
        f'but got {len(node_set_names)} node sets and '
        f'{len(edge_set_names)} edge sets.')
  return node_set_names[0], edge_set_names[0]


def check_homogeneous_graph_tensor(
    graph: Union[GraphTensor, GraphTensorSpec],
    name: str = 'This operation') -> None:
  """Raises ValueError when tfgnn.get_homogeneous_node_and_edge_set_name() does.
  """
  _ = get_homogeneous_node_and_edge_set_name(graph, name=name)
