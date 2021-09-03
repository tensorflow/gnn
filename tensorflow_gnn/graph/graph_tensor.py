"""Implementation of GraphTensor data type.
"""

import abc
from typing import Any, Dict, Mapping, Optional, Union

import tensorflow as tf
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_piece as gp

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import type_spec
# pylint: enable=g-direct-tensorflow-import

FieldName = const.FieldName
NodeSetName = const.NodeSetName
EdgeSetName = const.EdgeSetName
ShapeLike = const.ShapeLike
Field = const.Field
Fields = const.Fields
FieldSpec = const.FieldSpec
FieldsSpec = const.FieldsSpec

# TODO(b/189057503): use adjacency interface class instead.
Adjacency = Any
AdjacencySpec = Any


class _GraphPieceWithFeatures(gp.GraphPieceBase, metaclass=abc.ABCMeta):
  """Base class for graph pieces that hold user-defined features."""

  def __getitem__(self, feature_name: FieldName) -> Field:
    """Indexing operator `[]` to access feature values by their name."""
    return self._get_features_ref[feature_name]

  @property
  def features(self) -> Mapping[FieldName, Field]:
    """Read-only view for features."""
    return _as_immutable_mapping(self._get_features_ref)

  def get_features_dict(self) -> Dict[FieldName, Field]:
    """Returns features copy as a dictionary."""
    return dict(self._get_features_ref)

  @abc.abstractproperty
  def _get_features_ref(self) -> Fields:
    """Returns the mutable features dict. Subclass controls location in data."""
    raise NotImplementedError


class _GraphPieceWithFeaturesSpec(gp.GraphPieceSpecBase):
  """TypeSpec for _GraphPieceWithFeatures."""

  def __getitem__(self, feature_name: FieldName) -> FieldSpec:
    return self._get_features_spec_ref[feature_name]

  @property
  def features_spec(self) -> Mapping[FieldName, FieldSpec]:
    """A mapping of feature name to feature specs."""
    return _as_immutable_mapping(self._get_features_spec_ref)

  @abc.abstractproperty
  def total_num_components(self) -> Optional[int]:
    """The total number of graph components across dimensions if known."""
    raise NotImplementedError

  @abc.abstractproperty
  def _get_features_spec_ref(self) -> FieldsSpec:
    raise NotImplementedError


class Context(_GraphPieceWithFeatures):
  """A container of features for a graph component.

  This class is a container for the shapes of the context features associated
  with each component of a graph in a `GraphTensor` instance. Note that the
  number of components of those features is always explicitly set to `1` (in
  lieu of the number of nodes, we've got one such feature per graph).

  (Note that this graph piece does not use any metadata fields.)

  """

  @classmethod
  def from_fields(
      cls, *,
      features: Optional[Fields] = None,
      shape: ShapeLike = tf.TensorShape([]),
      indices_dtype: tf.dtypes.DType = const.default_indices_dtype
  ) -> 'Context':
    """Constructs a new instance from context fields.

    Args:
      features: mapping from feature names to feature Tensors or RaggedTensors.
        All feature tensors must have shape = graph_shape + [num_components] +
        feature_shape, where num_components is a number of graph components
        (could be ragged); feature_shape are field-specific inner dimensions.
      shape: the shape of this tensor and a GraphTensor containing it, also
        known as the graph_shape.
      indices_dtype: The `indices_dtype` of a GraphTensor containing this
        object, used as `row_splits_dtype` when batching potentially ragged
        fields.

    Returns:
      A `Context` tensor.

    """
    if features is None:
      features = {}
    assert isinstance(features, Mapping)
    prepared_features = {key: gp.convert_to_tensor_or_ragged(value)
                         for key, value in features.items()}
    return cls._from_data(
        prepared_features,
        shape=shape
        if isinstance(shape, tf.TensorShape) else tf.TensorShape(shape),
        indices_dtype=indices_dtype)

  def replace_features(self, features: Fields) -> 'Context':
    """Returns a new instance with a new set of features."""
    assert isinstance(features, Mapping)
    return self.__class__.from_fields(
        features=features, shape=self.shape, indices_dtype=self.indices_dtype)

  @property
  def _get_features_ref(self) -> Fields:
    return self._data

  @staticmethod
  def _type_spec_cls():
    return ContextSpec


@type_spec.register('tensorflow_gnn.ContextSpec')
class ContextSpec(_GraphPieceWithFeaturesSpec):
  """A type spec for global features for a graph component.

  This class is a type descriptor for the shapes of the context features
  associated with each component of a graph in a `GraphTensor` instance. Note
  that the prefix shape of those features is always explicitly set to either `1`
  for a single graph, or to the number of components for a batched graph.

  (Note that this graph piece does not use any metadata fields.)
  """

  @classmethod
  def from_field_specs(
      cls, *,
      features_spec: Optional[FieldsSpec] = None,
      shape: ShapeLike = tf.TensorShape([]),
      indices_dtype: tf.dtypes.DType = const.default_indices_dtype
  ) -> 'ContextSpec':
    """Counterpart of `Context.from_fields()` for values type specs."""
    if features_spec is None:
      features_spec = {}
    assert isinstance(features_spec, Mapping)
    return cls._from_data_spec(
        features_spec,
        shape=shape
        if isinstance(shape, tf.TensorShape) else tf.TensorShape(shape),
        indices_dtype=indices_dtype)

  @property
  def value_type(self):
    return Context

  @property
  def total_num_components(self) -> Optional[int]:
    """The total number of graph components across dimensions if known."""
    indicative_feature_spec = _get_indicative_feature_spec(self._data_spec)
    if indicative_feature_spec is None:
      return None
    else:
      return indicative_feature_spec.shape[:(self.rank + 1)].num_elements()

  @property
  def _get_features_spec_ref(self) -> FieldsSpec:
    return self._data_spec


class _NodeOrEdgeSet(_GraphPieceWithFeatures):
  """Base class for node set or edge set."""
  _DATAKEY_FEATURES = 'features'  # A Mapping[FieldName, Field].
  _DATAKEY_SIZES = 'sizes'  # A Field with `sizes`.

  @classmethod
  def _from_fields(cls, features: Fields, sizes: Field,
                   **extra_data) -> '_NodeOrEdgeSet':
    assert isinstance(features, Mapping)
    sizes = gp.convert_to_tensor_or_ragged(sizes)
    prepared_features = {key: gp.convert_to_tensor_or_ragged(value)
                         for key, value in features.items()}
    data = {
        _NodeOrEdgeSet._DATAKEY_FEATURES: prepared_features,
        _NodeOrEdgeSet._DATAKEY_SIZES: sizes
    }
    data.update({key: gp.convert_to_tensor_or_ragged(value)
                 for key, value in extra_data.items()})
    return cls._from_data(
        data=data, shape=sizes.shape[:-1], indices_dtype=sizes.dtype)

  def replace_features(self, features: Mapping[FieldName,
                                               Field]) -> '_NodeOrEdgeSet':
    """Returns a new instance with a new set of features."""
    assert isinstance(features, Mapping)
    new_data = self._data.copy()
    new_data.update({_NodeOrEdgeSet._DATAKEY_FEATURES: features})
    return self.__class__.from_fields(**new_data)

  @property
  def sizes(self) -> Field:
    """Tensor with a number of elements in each graph component."""
    return self._data[_NodeOrEdgeSet._DATAKEY_SIZES]

  @property
  def total_size(self) -> tf.Tensor:
    """Returns the total number of elements across dimensions.

    Returns:
      Scalar integer tensor equal to `tf.math.reduce_sum(sizes)`.
    """
    result = tf.math.reduce_sum(self.sizes)
    assert isinstance(result, tf.Tensor) and result.shape.rank == 0
    return result

  @property
  def _get_features_ref(self) -> Fields:
    return self._data[_NodeOrEdgeSet._DATAKEY_FEATURES]


class _NodeOrEdgeSetSpec(_GraphPieceWithFeaturesSpec):
  """TypeSpec for _NodeOrEdgeSet."""

  @classmethod
  def _from_field_specs(cls, features_spec: FieldsSpec, sizes_spec: FieldSpec,
                        **extra_data) -> '_NodeOrEdgeSetSpec':
    # pylint: disable=protected-access
    assert isinstance(features_spec, Mapping)
    data_spec = {
        _NodeOrEdgeSet._DATAKEY_FEATURES: features_spec,
        _NodeOrEdgeSet._DATAKEY_SIZES: sizes_spec
    }
    data_spec.update(extra_data)
    return cls._from_data_spec(
        data_spec, shape=sizes_spec.shape[:-1], indices_dtype=sizes_spec.dtype)

  @property
  def sizes_spec(self) -> FieldSpec:
    """A type spec for the sizes that provides num. elements per component."""
    return self._data_spec[_NodeOrEdgeSet._DATAKEY_SIZES]  # pylint: disable=protected-access

  @property
  def total_num_components(self) -> Optional[int]:
    """The total number of graph components across dimensions if known."""
    return self.sizes_spec.shape.num_elements()

  @property
  def total_size(self) -> Optional[int]:
    """Returns the total number of graph entities across dimensions if known."""
    indicative_feature_spec = _get_indicative_feature_spec(
        self._get_features_spec_ref)
    if indicative_feature_spec is None:
      return None
    else:
      return indicative_feature_spec.shape[:(self.rank + 1)].num_elements()

  @property
  def _get_features_spec_ref(self) -> FieldsSpec:
    return self._data_spec[_NodeOrEdgeSet._DATAKEY_FEATURES]  # pylint: disable=protected-access


class NodeSet(_NodeOrEdgeSet):
  """A container for the features of a single node set.

  This class is a container for the shapes of the features associated with a
  graph's node set from a `GraphTensor` instance. This graph piece stores
  features that belong to an edge set, and a `sizes` tensor with the number of
  edges in each graph component.

  (This graph piece does not use any metadata fields.)
  """

  @classmethod
  def from_fields(cls, *,
                  features: Optional[Fields] = None,
                  sizes: Field) -> 'NodeSet':
    """Constructs a new instance from node set fields.

    Args:
      features: mapping from feature names to feature Tensors or RaggedTensors.
        All feature tensors must have shape = graph_shape + [num_nodes] +
        feature_shape, where num_nodes is the number of graph nodes in this set
        (could be ragged) and feature_shape are feature-specific inner
        dimensions.
      sizes: the number of nodes in each graph component. Has shape =
        graph_shape + [num_components], where num_components is the number of
        graph components (could be ragged).

    Returns:
      A `NodeSet` tensor.
    """
    if features is None:
      features = {}
    return cls._from_fields(features=features, sizes=sizes)

  @staticmethod
  def _type_spec_cls():
    return NodeSetSpec


@type_spec.register('tensorflow_gnn.NodeSetSpec')
class NodeSetSpec(_NodeOrEdgeSetSpec):
  """A type spec for the features of a single node set.

  This class is a type descriptor for the shapes of the features associated with
  a graph's node set from a `GraphTensor` instance. This graph piece stores
  features that belong to an edge set, and a `sizes` tensor with the number of
  edges in each graph component.

  (This graph piece does not use any metadata fields.)
  """

  @classmethod
  def from_field_specs(cls, *,
                       features_spec: Optional[FieldsSpec] = None,
                       sizes_spec: FieldSpec) -> 'NodeSetSpec':
    """Counterpart of `NodeSet.from_fields()` for values type specs."""
    if features_spec is None:
      features_spec = {}
    return cls._from_field_specs(
        features_spec=features_spec, sizes_spec=sizes_spec)

  @property
  def value_type(self):
    return NodeSet


class EdgeSet(_NodeOrEdgeSet):
  """A container for the features of a single edge set.

  This class is a container for the shapes of the features associated with a
  graph's edge set from a `GraphTensor` instance. This graph piece stores
  features that belong to an edge set, a `sizes` tensor with the number of edges
  in each graph component and an `adjacency` `GraphPiece` tensor describing how
  this edge set connects node sets (see adjacency.py).

  (This graph piece does not use any metadata fields.)
  """

  _DATAKEY_ADJACENCY = 'adjacency'  # An Adjacency GraphPiece.

  @classmethod
  def from_fields(cls, *,
                  features: Optional[Fields] = None,
                  sizes: Field,
                  adjacency: Adjacency) -> 'EdgeSet':
    """Constructs a new instance from edge set fields.

    Args:
      features: mapping from feature names to feature Tensors or RaggedTensors.
        All feature tensors must have shape = graph_shape + [num_edges] +
        feature_shape, where num_edges is the number of edges in the edge set
        (could be ragged) and feature_shape are feature-specific inner
        dimensions.
      sizes: the number of edges in each graph component. Has shape =
        graph_shape + [num_components], where num_components is the number of
        graph components (could be ragged).
      adjacency: one of supported adjacency types (see adjacency.py).

    Returns:
      A `EdgeSet` tensor.
    """
    if features is None:
      features = {}
    return cls._from_fields(features=features, sizes=sizes, adjacency=adjacency)

  @property
  def adjacency(self) -> Adjacency:
    return self._data[EdgeSet._DATAKEY_ADJACENCY]

  @staticmethod
  def _type_spec_cls():
    return EdgeSetSpec


@type_spec.register('tensorflow_gnn.EdgeSetSpec')
class EdgeSetSpec(_NodeOrEdgeSetSpec):
  """A type spec for the features of a single edge set.

  This class is a type descriptor for the shapes of the features associated with
  a graph's edge set from a `GraphTensor` instance. This graph piece stores
  features that belong to an edge set, a `sizes` tensor with the number of edges
  in each graph component and an `adjacency` `GraphPiece` tensor describing how
  this edge set connects node sets (see adjacency.py).

  (This graph piece does not use any metadata fields.)
  """

  @classmethod
  def from_field_specs(cls, *,
                       features_spec: Optional[FieldsSpec] = None,
                       sizes_spec: FieldSpec,
                       adjacency_spec: AdjacencySpec) -> 'EdgeSetSpec':
    """Counterpart of `EdgeSet.from_fields()` for values type specs."""
    # pylint: disable=protected-access
    if features_spec is None:
      features_spec = {}
    return cls._from_field_specs(
        features_spec=features_spec,
        sizes_spec=sizes_spec,
        **{EdgeSet._DATAKEY_ADJACENCY: adjacency_spec})

  @property
  def value_type(self):
    return EdgeSet

  @property
  def adjacency_spec(self) -> AdjacencySpec:
    """A type spec for the adjacency indices container."""
    return self._data_spec[EdgeSet._DATAKEY_ADJACENCY]  # pylint: disable=protected-access

  @property
  def total_size(self) -> Optional[int]:
    """Returns the total number of edges across dimensions if known."""
    return _ifnone(super().total_size, self.adjacency_spec.total_size)


class GraphTensor(gp.GraphPieceBase):
  """Stores graphs, possibly heterogeneous (i.e., with multiple node sets).

  A `GraphTensor` consists of

  * A `GraphTensorSpec` object that provides its type information. It defines
    the node and edge sets, how node sets are connected by edge sets, and the
    type and shape constraints of graph field values. The `GraphTensorSpec`s of
    two graphs are equal if they agree in the features and shapes, independent
    of the variable number of nodes and edges in an actual graph tensor.

  * Graph data, or "fields", which can be instances of `Tensor`s or
    `RaggedTensor`s. Fields are stored on the `NodeSet`, `EdgeSet` and `Context`
    tensors that make up the `GraphTensor`. Each of those tensors have fields to
    represent user-defined data features. In addition, there are fields storing
    the graph topology:  NodeSets and EdgeSets have a special `size` field that
    provides a tensor of the number of nodes (or edges) of each graph component.
    Moreover, adjacency information is stored in the `adjacency` property of the
    EdgeSet.

  A `GraphTensor` object is a tensor with graphs as its elements. Its `.shape`
  attribute describes the shape of the graph tensor, where a shape of `[]`
  describes a scalar (single) graph, a shape of `[d0]` describes a `d0`-vector
  of graphs, a shape of `[d0, d1]` a `d0` x `d1` matrix of graphs, and so on.

  Context, node set, and edge set features are accessed via the `context`,
  `node_sets` and `edge_sets` properties, respectively. Note that
  the node sets and edge sets are mappings of a set name (a string) to either a
  `NodeSet` or `EdgeSet` object. These containers provide a mapping interface
  (via `getitem`, i.e., `[]`) to access individual feature values by their name,
  and a `features` property that provides an immutable mapping of feature names
  to their values. These features are those you defined in your schema.

  A "scalar" graph tensor describes a single graph with `C` disjoint components.
  When utilized in building models, this usually represents `C` example graphs
  bundled into a single graph with multiple disjoint graph components. This
  allows you to build models that work on batches of graphs all at once, with
  vectorized operations, as if they were a single graph with multiple
  components. The shapes of the tensors have elided the prefix batch dimension,
  but conceptually it is still present, and recoverable. The number of
  components (`C`) could vary from graph to graph, or if necessary for custom
  hardware, be fixed to a value known statically. In the simplest case of `C =
  1`, this number is constrained only by the available RAM and example sizes.
  The number of components in a graph corresponds to the concept of "batch-size"
  in a regular neural network context.

  Note that since context features store data for each graph, the first
  dimension of all *context* features always index the graph component and has
  size `C`.

  Conceptually (but not in practice - see below), for scalar graphs, each
  node/edge set feature could be described as a ragged tensor with a shape
  `[c, n_c, f1..fk]` where `c` indexes the individual graph components, `n_c`
  indexes the nodes or edges within each component `c`, and `f1..fk` are inner
  dimensions of features, with `k` being the rank of the feature tensor.
  Dimensions `f1..fk` are typically fully defined, but the `GraphTensor`
  container also supports ragged features (of a variable size), in which case
  instances of `tf.RaggedTensor`s are provided in those mappings. The actual
  number of nodes in each graph is typically different for each `c` graph
  (variable number of nodes), so this dimension is normally ragged.

  Please note some limitations inherent in the usage of `tf.RaggedTensor`s to
  represent features; it is not ideal, in that

    * `tf.RaggedTensor`s are not supported on XLA compilers, and when used on
      accelerators (e.g., TPUs), can cause error messages that are difficult to
      understand;

    * Slices of features for individual graph components are rarely needed in
      practice;

    * The ragged partitions (see docs on `tf.RaggedTensor`) are the same for all
      features within the same node/edge set, hence they would be redundant to
      represent as individual ragged tensor instances.

  For these reasons, the `GraphTensor` extracts component partitions into a
  special node set field called 'size'. For scalar `GraphTensor` instances this
  is a rank-1 integer tensor containing the number of nodes/edges in each graph
  component.

  It is important to know that feature values are stored with their component
  dimension flattened away, leading to shapes like `[n, f1..fk]`, where `n`
  (instead of `c` and `n_c`) indexes a node within a graph over all of its
  components. For the most common case of features with fully-defined shape of
  dimensions `f1..fk`, this allows us to represent those features as simple
  dense tensors. Finally, when all the dimensions including `n` are also
  fully-defined, the `GraphTensor` is XLA compatible (and this provides
  substantial performance opportunities). The same principle also applies to the
  edge set features.

  In general, for non-scalar graph tensors, the feature values can be a dense
  tensor (an instance of `tf.Tensor`) or a ragged tensors (an instance of a
  `tf.RaggedTensor`). This union is usually referred to as a "potentially ragged
  tensor" (mainly due to the recursive nature of the definition of ragged
  tensors). For our purposes, the leading dimensions of the shapes of a set of
  feature tensors must match the shape of their containing graph tensor.

  `GraphTensor` allows batching of graphs. Batching changes a `GraphTensor`
  instance's shape to `[batch_size] + shape` and the graph tensor's rank is
  increased by 1. Unbatching removes dimension-0, as if truncating with
  `shape[1:]`, and the `GraphTensor`'s rank is decreased by 1. This works
  naturally with the batch and unbatch methods of tf.data.Datatset.

  Batching and unbatching are equivalent to the batching and unbatching of
  individual fields. Dense fields with static shapes (that is, fully-defined
  shapes known at compile time) are always batched to `(rank + 1)` dense
  tensors. If a field has ragged dimensions, batching results in `(rank + 1)`
  ragged tensors. In general, graph tensor operations always try to preserve
  fully-defined field shapes and dense representations wherever possible, as
  this makes it possible to leverage as XLA optimizations where possible.

  A `GraphTensor` of any rank can be converted to a scalar graph using the
  'merge_batch_to_components()' method. This method is a graph transformation
  operation that merges the graph components of each graph into a single
  disjoint graph. Typically, this happens after the input pipeline is done with
  shuffling and batching the graphs from individual training examples and before
  the actual model treats them as components of a single graph with contiguous
  indexing.

  Example 1: A homogeneous scalar graph with one component having 10 nodes and
  20 edges and no values.

      gnn.GraphTensor.from_pieces(
        node_sets = {
          'node': gnn.NodeSet.from_fields(sizes=[10], features={})
        },
        edge_sets = {
          'edge': gnn.EdgeSet.from_fields(
             sizes=[10],
             features={},
             adjacency=gnn.Adjacency.from_indices(
               source=('node', [0, 5, 9]),
               target=('node', [19, 10, 0])))})

  Example 2: A rank-1 graph tensor with three graphs. Each graph is a tree with
  a single scalar label.

      rt = tf.ragged.constant

      gnn.GraphTensor.from_pieces(
        context=gnn.Context.from_fields(features={
          'label': rt([['GOOD'], ['BAD'], ['UGLY']])
        }),
        node_sets={
          'root': gnn.NodeSet.from_fields(
                    sizes=rt([[1], [1], [1]]),
                    features={}),
          'leaf': gnn.NodeSet.from_fields(
                    sizes=rt([[2], [3], [1]]),
                    features={'id': rt([['a', 'b'], ['c', 'a', 'd'], ['e']])})},
        edge_sets={
          'leaf->root': gnn.EdgeSet.from_fields(
             sizes=rt([[2], [3], [1]]),
             features={'weight': rt([[.5, .6], [.3, .4, .5], [.9]])},
             adjacency=gnn.Adjacency.from_indices(
               source=('leaf', rt([[0, 1], [0, 1, 2], [0]])),
               target=('root', rt([[0, 0], [0, 0, 0], [0]]))))})

  Example 3: An application of `merge_batch_to_components()` to the previous
  example. Please note how the source and target edge indices have changed to
  reference nodes within a graph.

      gnn.GraphTensor.from_pieces(
        context=gnn.Context.from_fields(features={
          'label': ['GOOD', 'BAD', 'UGLY']
        }),
        node_sets={
          'root': gnn.NodeSet.from_fields(sizes=[1, 1, 1], features={}),
          'leaf': gnn.NodeSet.from_fields(
                    sizes=[2, 3, 1],
                    features={'id': ['a', 'b', 'c', 'a', 'd', 'e']}
                  ),
        },
        edge_sets={
          'leaf->root': gnn.EdgeSet.from_fields(
                          sizes=[2, 3, 1],
                          features={'weight': [.5, .6, .3, .4, .5, .9]},
                          adjacency=gnn.Adjacency.from_indices(
                            source=('leaf', [0, 1, 0, 1, 2, 0]),
                            target=('root', [0, 0, 0, 0, 0, 0]),
                          ))})

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
    indicative_entity = _get_indicative_graph_entity(context, node_sets,
                                                     edge_sets)
    assert isinstance(indicative_entity, _GraphPieceWithFeatures)
    return cls._from_data(
        data={
            GraphTensor._DATAKEY_CONTEXT: context,
            GraphTensor._DATAKEY_NODE_SETS: node_sets,
            GraphTensor._DATAKEY_EDGE_SETS: edge_sets
        },
        shape=indicative_entity.shape,
        indices_dtype=indicative_entity.indices_dtype)

  def merge_batch_to_components(self) -> 'GraphTensor':
    """Merges the contained graphs into a single scalar `GraphTensor`.

    For example, flattening of

        GraphTensor.from_pieces(
          node_sets={
            'node': NodeSet.from_fields(
              # Three graphs with
              #   - 1st graph having two components with 3 and 2 nodes;
              #   - 2nd graph having 1 component with 2 nodes;
              #   - 3rd graph having 1 component with 3 nodes;
              sizes=tf.ragged.constant([[3, 2], [2], [3]]),
              features={...},
            )
          }
          edge_sets={
            'edge': EdgeSet.from_fields(
              sizes=tf.ragged.constant([[6, 7], [8], [3]]),
              features={...},
              adjacency=...,
            )
          }
        )

    would result in the equivalent graph of

        GraphTensor.from_pieces(
          node_sets={
            'node': NodeSet.from_fields(
              # One graph with 4 components with 3, 2, 2, 3 nodes each.
              sizes=[3, 2, 2, 3],
              features={...},
            )
          }
          edge_sets={
            'edge': EdgeSet.from_fields(
              sizes=[6, 7, 8, 3],
              features={...},
              adjacency=...,
            )
          }
        )

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
    """The graph context feature container."""
    return self._data[GraphTensor._DATAKEY_CONTEXT]

  @property
  def node_sets(self) -> Mapping[NodeSetName, NodeSet]:
    """A read-only view of node sets."""
    return _as_immutable_mapping(self._data[GraphTensor._DATAKEY_NODE_SETS])

  @property
  def edge_sets(self) -> Mapping[EdgeSetName, EdgeSet]:
    """A read-only view for edge sets."""
    return _as_immutable_mapping(self._data[GraphTensor._DATAKEY_EDGE_SETS])

  def replace_features(
      self,
      context: Optional[Fields] = None,
      node_sets: Optional[Mapping[NodeSetName, Fields]] = None,
      edge_sets: Optional[Mapping[EdgeSetName, Fields]] = None,
  ) -> 'GraphTensor':
    """Returns a new instance with a new set of features for the same topology.

    Example 1. Replaces all features for node set 'node.a' but not 'node.b'.

        graph = gnn.GraphTensor.from_pieces(
          context=gnn.Context.from_fields(features={
            'label': tf.ragged.constant([['A'], ['B']])
          }),
          node_sets={
              'node.a': gnn.NodeSet.from_fields(features={
                'id': ['a1', 'a3']
              }, sizes=[2]),
              'node.b': gnn.NodeSet.from_fields(features={
                'id': ['b4', 'b1']
              }, sizes=[2])
          }
        )
        result = graph.replace_features(
          node_sets={'node.a': {'h': tf.ragged.constant([[1., 0.], [3., 0.]])}}
        )

    Result:

        gnn.GraphTensor.from_pieces(
          context=gnn.Context.from_fields(features={
            'label': tf.ragged.constant([['A'], ['B']])
          }),
          node_sets={
              'node.a': gnn.NodeSet.from_fields(features={
                'h': tf.ragged.constant([[1., 0.], [3., 0.]])
              }, sizes=[2]),
              'node.b': gnn.NodeSet.from_fields(features={
                'id': ['b4', 'b1']
              }, sizes=[2])
          }
        )

    Args:
      context: A substitute for the context features, or None (which keeps the
        prior features).
      node_sets: A substitute for specified node set features. Node sets which
        are not included remain unchanged.
      edge_sets: A substitute for specified edge set features. Edge sets which
        are not included remain unchanged.

    Returns:
      A `GraphTensor` instance with features overridden according to the
      arguments.
    """
    if context is None:
      new_context = self.context
    else:
      new_context = self.context.replace_features(context)

    if node_sets is None:
      new_node_sets = self.node_sets.copy()
    else:
      new_node_sets = {
          set_name: (node_set.replace_features(node_sets[set_name])
                     if set_name in node_sets else node_set)
          for set_name, node_set in self.node_sets.items()
      }

    if edge_sets is None:
      new_edge_sets = self.edge_sets.copy()
    else:
      new_edge_sets = {
          set_name: (edge_set.replace_features(edge_sets[set_name])
                     if set_name in edge_sets else edge_set)
          for set_name, edge_set in self.edge_sets.items()
      }

    return self.__class__.from_pieces(new_context, new_node_sets, new_edge_sets)

  @staticmethod
  def _type_spec_cls():
    return GraphTensorSpec


@type_spec.register('tensorflow_gnn.GraphTensorSpec')
class GraphTensorSpec(gp.GraphPieceSpecBase):
  """A type spec for a `GraphTensor` instance."""

  @classmethod
  def from_piece_specs(
      cls,
      context_spec: Optional[ContextSpec] = None,
      node_sets_spec: Optional[Mapping[NodeSetName, NodeSetSpec]] = None,
      edge_sets_spec: Optional[Mapping[EdgeSetName, EdgeSetSpec]] = None,
  ) -> 'GraphTensorSpec':
    """Counterpart of `GraphTensor.from_pieces` for values type specs."""
    context_spec = _ifnone(context_spec, ContextSpec.from_field_specs())
    node_sets_spec = _ifnone(node_sets_spec, dict())
    edge_sets_spec = _ifnone(edge_sets_spec, dict())
    indicative_entity = _get_indicative_graph_entity(context_spec,
                                                     node_sets_spec,
                                                     edge_sets_spec)
    assert isinstance(indicative_entity, _GraphPieceWithFeaturesSpec)
    # pylint: disable=protected-access
    return cls._from_data_spec(
        {
            GraphTensor._DATAKEY_CONTEXT: context_spec,
            GraphTensor._DATAKEY_NODE_SETS: node_sets_spec,
            GraphTensor._DATAKEY_EDGE_SETS: edge_sets_spec
        },
        shape=indicative_entity.shape,
        indices_dtype=indicative_entity.indices_dtype)

  @property
  def value_type(self):
    return GraphTensor

  @property
  def context_spec(self) -> ContextSpec:
    """Type spec for the container of context features."""
    return self._data_spec[GraphTensor._DATAKEY_CONTEXT]  # pylint: disable=protected-access

  @property
  def node_sets_spec(self) -> Mapping[NodeSetName, NodeSetSpec]:
    """Type spec for the containers of node features for each node set."""
    return _as_immutable_mapping(
        self._data_spec[GraphTensor._DATAKEY_NODE_SETS])  # pylint: disable=protected-access

  @property
  def edge_sets_spec(self) -> Mapping[EdgeSetName, EdgeSetSpec]:
    """Type spec for the containers of node features for each edge set."""
    return _as_immutable_mapping(
        self._data_spec[GraphTensor._DATAKEY_EDGE_SETS])  # pylint: disable=protected-access

  @property
  def total_num_components(self) -> Optional[int]:
    """The total number of graph components across dimensions if known."""
    indicative_entity = _get_indicative_graph_entity(self.context_spec,
                                                     self.node_sets_spec,
                                                     self.edge_sets_spec)
    assert indicative_entity is not None
    return indicative_entity.total_num_components


def _ifnone(value, default):
  return value if value is not None else default


def _get_indicative_feature_spec(
    field_specs: FieldsSpec) -> Optional[FieldSpec]:
  """Deterministically selects one of the field specs."""

  def key_fn(item):
    """First not ragged by field name."""
    name, spec = item
    ragged_rank = spec.ragged_rank if isinstance(spec, tf.RaggedTensor) else 0
    return (ragged_rank, name)

  _, result = min(field_specs.items(), key=key_fn, default=('', None))
  return result


def _get_indicative_graph_entity(
    context: Union[Context, ContextSpec],
    node_sets: Mapping[NodeSetName, Union[NodeSet, NodeSetSpec]],
    edge_sets: Mapping[EdgeSetName, Union[EdgeSet, EdgeSetSpec]]
) -> Union[_GraphPieceWithFeatures, _GraphPieceWithFeaturesSpec]:
  """Deterministically selects one of the graph entities."""

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

  def copy(self):
    """Returns copy as a dictionary object."""
    return dict(self._data)


def _as_immutable_mapping(input_map):
  return _ImmutableMapping(input_map)
