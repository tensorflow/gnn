"""GraphTensor adjacency types.
"""

from typing import Dict, Mapping, Optional, Tuple, Union

import tensorflow as tf
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_piece as gp
from tensorflow_gnn.graph import tensor_utils as utils

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import type_spec
# pylint: enable=g-direct-tensorflow-import

Field = const.Field
FieldSpec = const.FieldSpec
IncidentNodeTag = const.IncidentNodeTag
NodeSetName = const.NodeSetName

Index = Tuple[NodeSetName, Field]
Indices = Mapping[IncidentNodeTag, Index]


class HyperAdjacency(gp.GraphPieceBase):
  """Stores edges as indices of nodes in node sets.

  Node adjacency is represented as a mapping of unique node tags to pairs of
  (node set names, index tensors) into them. The tags are `SOURCE` and
  `TARGET` for ordinary graphs but there can be more of them for hypergraphs
  (e.g., edges linking more than two nodes, also known as "hyper-edges"). All
  index tensors must agree in their type (`tf.Tensor` or `tf.RaggedTensor`),
  integer dtype, and shape. Corresponding values are the indices of nodes that
  belong to the same hyper-edge.
  """

  @classmethod
  @tf.__internal__.dispatch.add_dispatch_support
  def from_indices(cls,
                   indices: Indices,
                   *,
                   validate: bool = True) -> 'HyperAdjacency':
    """Constructs a new instance from the `indices` tensors.

    Example 1. Single graph (rank is 0). Connects pairs of nodes (a.0, b.2),
    (a.1, b.1), (a.2, b.0) from node sets a and b:

        gnn.HyperAdjacency.from_indices({
            gnn.SOURCE: ('a', [0, 1, 2]),
            gnn.TARGET: ('b', [2, 1, 0])
        })

    Example 2. Single hypergraph (rank is 0). Connects triplets of nodes
    (a.0, b.2, c.1), (a.1, b.1, c.0) from the node sets a, b and c:

        gnn.HyperAdjacency.from_indices({
            0: ('a', [0, 1]),
            1: ('b', [2, 1]),
            2: ('c', [1, 0]),
        })

    Example 3. Batch of two graphs (rank is 1). Connects pairs of nodes
    graph 0: (a.0, b.2), (a.1, b.1); graph 1: (a.2, b.0):

        gnn.HyperAdjacency.from_indices({
            gnn.SOURCE: ('a', tf.ragged.constant([[0, 1], [2]])),
            gnn.TARGET: ('b', tf.ragged.constant([[2, 1], [0]])),
        })

    Args:
      indices: Mapping from node tags to tuples of node set names and integer
        Tensors or RaggedTensors with the indices of nodes in the respective
        node set. All tensors must have shape = graph_shape + [num_edges], where
        num_edges is a number of edges in each graph. If graph_shape.rank > 0
        and num_edges has variable size, the tensors are ragged.
      validate: if set, checks that node indices have the same shapes.

    Returns:
      A `HyperAdjacency` tensor with a shape and an indices_dtype being inferred
      from the `indices` values.
    """
    indices = {key: (name, gp.convert_to_tensor_or_ragged(index))
               for key, (name, index) in indices.items()}

    if validate or const.validate_internal_results:
      indices = _validate_indices(indices)

    data = {
        _node_tag_to_index_key(tag): index
        for tag, (_, index) in indices.items()
    }
    metadata = {
        _node_tag_to_index_key(tag): name for tag, (name, _) in indices.items()
    }
    indicative_index_tensor = _get_indicative_index(data)
    return cls._from_data(
        data,
        shape=indicative_index_tensor.shape[:-1],
        indices_dtype=indicative_index_tensor.dtype,
        metadata=metadata)

  def __getitem__(self, node_set_tag: IncidentNodeTag) -> Field:
    """Returns index tensor for a given node set tag."""
    return self._data[_node_tag_to_index_key(node_set_tag)]

  def node_set_name(self, node_set_tag: IncidentNodeTag) -> NodeSetName:
    """Returns node set name for a given node set tag."""
    return self.spec.node_set_name(node_set_tag)

  def get_indices_dict(
      self) -> Dict[IncidentNodeTag, Tuple[NodeSetName, Field]]:
    """Returns copy of indices tensor."""
    return {
        _index_key_to_node_tag(key):
        (self.node_set_name(_index_key_to_node_tag(key)), index)
        for key, index in self._data.items()
    }

  def _merge_batch_to_components(
      self, num_edges_per_example: Field,
      num_nodes_per_example: Mapping[NodeSetName, Field]) -> 'HyperAdjacency':
    if self.rank == 0:
      return self

    flat_adj = super()._merge_batch_to_components(
        num_edges_per_example=num_edges_per_example,
        num_nodes_per_example=num_nodes_per_example)
    assert isinstance(flat_adj, HyperAdjacency)

    def flatten_indices(node_tag_key, index: Field) -> Field:
      node_set_name = self.spec._metadata[node_tag_key]  # pylint: disable=protected-access
      return utils.flatten_indices(index, num_edges_per_example,
                                   num_nodes_per_example[node_set_name])

    new_data = {
        node_tag_key: flatten_indices(node_tag_key, index)
        for node_tag_key, index in flat_adj._data.items()  # pylint: disable=protected-access
    }
    return self.__class__(new_data, flat_adj.spec)

  @staticmethod
  def _type_spec_cls():
    return HyperAdjacencySpec


@type_spec.register('tensorflow_gnn.HyperAdjacencySpec')
class HyperAdjacencySpec(gp.GraphPieceSpecBase):
  """TypeSpec for HyperAdjacency."""

  @classmethod
  def from_incident_node_sets(
      cls,
      incident_node_sets: Mapping[IncidentNodeTag, NodeSetName],
      index_spec: FieldSpec = tf.TensorSpec((None,),
                                            const.default_indices_dtype)
  ) -> 'HyperAdjacencySpec':
    """Constructs a new instance from the `incident_node_sets`.

    Args:
      incident_node_sets: mapping from incident node tags to node set names.
      index_spec: type spec for all index tensors. Its shape must be graph_shape
        + [num_edges], where num_edges is the number of edges in each graph. If
        graph_shape.rank > 0 and num_edges has variable size, the spec should be
        an instance of tf.RaggedTensorSpec.

    Returns:
      A `HyperAdjacencySpec` TypeSpec.
    """
    if not (index_spec.shape.rank > 0 and
            index_spec.dtype in (tf.int32, tf.int64)):
      raise ValueError(
          'Index spec must have rank > 0 and dtype in (tf.int32, tf.int64),'
          f' got {index_spec}')

    data_spec = {
        _node_tag_to_index_key(tag): index_spec for tag in incident_node_sets
    }
    metadata = {
        _node_tag_to_index_key(tag): name
        for tag, name in incident_node_sets.items()
    }
    return cls._from_data_spec(
        data_spec,
        shape=index_spec.shape[:-1],
        indices_dtype=index_spec.dtype,
        metadata=metadata)

  @property
  def value_type(self):
    return HyperAdjacency

  def __getitem__(self, node_set_tag: IncidentNodeTag) -> FieldSpec:
    """Returns index tensor type spec for a given node set tag."""
    return self._data_spec[_node_tag_to_index_key(node_set_tag)]

  def get_index_specs_dict(
      self) -> Dict[IncidentNodeTag, Tuple[NodeSetName, FieldSpec]]:
    """Returns copy of index type specs."""
    return {
        _index_key_to_node_tag(key):
        (self.node_set_name(_index_key_to_node_tag(key)), index)
        for key, index in self._data_spec.items()
    }

  def node_set_name(self, node_set_tag: IncidentNodeTag) -> NodeSetName:
    """Returns node set name for a given node set tag."""
    return self._metadata[_node_tag_to_index_key(node_set_tag)]

  @property
  def total_size(self) -> Optional[int]:
    """Returns the total number of edges across dimensions if known."""
    ind_spec = _get_indicative_index(self._data_spec)
    assert ind_spec is not None
    return ind_spec.shape[:(self.rank + 1)].num_elements()


class Adjacency(HyperAdjacency):
  """Stores simple binary edges with a source and target.

  Node adjacency is represented as mapping of source and target edge endpoints
  to pairs of (node set names, index tensors) into them. All index tensors must
  agree in their type (`tf.Tensor` or `tf.RaggedTensor`), integer dtype, and
  shape. Corresponding values are the indices of nodes that belong to the same
  edge.
  """

  @classmethod
  @tf.__internal__.dispatch.add_dispatch_support
  def from_indices(cls,
                   source: Index,
                   target: Index,
                   *,
                   validate: bool = True) -> 'Adjacency':
    """Constructs a new instance from the `indices` tensors.

    Example 1. Single graph (rank is 0). Connects pairs of nodes (a.0, b.2),
    (a.1, b.1), (a.2, b.0) from node sets a and b:

        gnn.Adjacency.from_indices(('a', [0, 1, 2]),
                                   ('b', [2, 1, 0]))

    Example 2. Batch of two graphs (rank is 1). Connects pairs of nodes
    graph 0: (a.0, b.2), (a.1, b.1); graph 1: (a.2, b.0):

        gnn.Adjacency.from_indices(('a', tf.ragged.constant([[0, 1], [2]])),
                                   ('b', tf.ragged.constant([[2, 1], [0]])))

    Args:

      source: Tuple of (node set name, integer Tensors or RaggedTensors with the
        indices of nodes in the respective node set). Must have shape =
        graph_shape + [num_edges], where num_edges is a number of edges in each
        graph. If graph_shape.rank > 0 and num_edges has variable size, the
        tensors are ragged.
      target: Like `source` field, but for target edge endpoint.
      validate: if set, checks that node indices have the same shapes.

    Returns:
      An `Adjacency` tensor with a shape and an indices_dtype being inferred
      from the `indices` values.
    """
    return super().from_indices({const.SOURCE: source,
                                 const.TARGET: target})

  @property
  def source(self) -> Field:
    return self[const.SOURCE]

  @property
  def target(self) -> Field:
    return self[const.TARGET]

  @property
  def source_name(self) -> NodeSetName:
    """Returns the node set name for source nodes."""
    return self.node_set_name(const.SOURCE)

  @property
  def target_name(self) -> NodeSetName:
    """Returns the node set name for target nodes."""
    return self.node_set_name(const.TARGET)

  @staticmethod
  def _type_spec_cls():
    return AdjacencySpec


@type_spec.register('tensorflow_gnn.AdjacencySpec')
class AdjacencySpec(HyperAdjacencySpec):
  """TypeSpec for Adjacency."""

  @classmethod
  def from_incident_node_sets(
      cls,
      source_node_set: NodeSetName,
      target_node_set: NodeSetName,
      index_spec: FieldSpec = tf.TensorSpec((None,),
                                            const.default_indices_dtype)
  ) -> 'AdjacencySpec':
    """Constructs a new instance from the `incident_node_sets`.

    Args:
      source_node_set: A string, the name of the source node set.
      target_node_set: A string, the name of the target node set.
      index_spec: type spec for all index tensors. Its shape must be graph_shape
        + [num_edges], where num_edges is the number of edges in each graph. If
        graph_shape.rank > 0 and num_edges has variable size, the spec should be
        an instance of tf.RaggedTensorSpec.

    Returns:
      A `AdjacencySpec` TypeSpec.
    """
    return super().from_incident_node_sets(
        {const.SOURCE: source_node_set,
         const.TARGET: target_node_set}, index_spec)

  @property
  def value_type(self):
    return Adjacency

  @property
  def source(self) -> FieldSpec:
    return self[const.SOURCE]

  @property
  def target(self) -> FieldSpec:
    return self[const.TARGET]

  @property
  def source_name(self) -> NodeSetName:
    """Returns the node set name for source nodes."""
    return self.node_set_name(const.SOURCE)

  @property
  def target_name(self) -> NodeSetName:
    """Returns the node set name for target nodes."""
    return self.node_set_name(const.TARGET)


def _validate_indices(indices: Indices) -> Indices:
  """Checks that indices have compatible shapes."""
  if not indices:
    raise ValueError('`indices` must contain at least one entry.')

  assert_ops = []

  def check_index(tag, name, index):
    if index.dtype not in (tf.int32, tf.int64):
      raise ValueError((f'Adjacency indices ({tag_0}, {name_0}) must have '
                        f'tf.int32 or tf.int64 dtype, got {index.dtype}'))
    if isinstance(index, tf.RaggedTensor):
      if index.flat_values.shape.rank != 1:
        raise ValueError(
            (f'Adjacency indices ({tag_0}, {name_0}) as ragged tensor must'
             f' have flat values rank 1, got {index.flat_values.shape.rank}'))

  def check_compatibility(tag_0, name_0, index_0, tag_i, name_i, index_i):
    err_message = ('Adjacency indices are not compatible:'
                   f' ({tag_0}, {name_0}) and ({tag_i}, {name_i})')
    try:
      if index_0.dtype != index_i.dtype:
        raise ValueError(err_message)

      if isinstance(index_0, tf.Tensor) and isinstance(index_i, tf.Tensor):
        assert_ops.append(
            tf.assert_equal(
                tf.shape(index_0), tf.shape(index_i), message=err_message))
        return

      if isinstance(index_0, tf.RaggedTensor) and isinstance(
          index_i, tf.RaggedTensor):
        if index_0.ragged_rank != index_i.ragged_rank:
          raise ValueError(err_message)
        for partition_0, partition_i in zip(index_0.nested_row_splits,
                                            index_i.nested_row_splits):
          assert_ops.append(
              tf.assert_equal(partition_0, partition_i, message=err_message))

        assert_ops.append(
            tf.assert_equal(
                tf.shape(index_0.flat_values),
                tf.shape(index_i.flat_values),
                message=err_message))
        return
    except:
      raise ValueError(err_message)

    raise ValueError(err_message)

  indices = sorted(list(indices.items()), key=lambda i: i[0])
  tag_0, (name_0, index_0) = indices[0]
  check_index(tag_0, name_0, index_0)
  for tag_i, (name_i, index_i) in indices[1:]:
    check_index(tag_i, name_i, index_i)
    check_compatibility(tag_0, name_0, index_0, tag_i, name_i, index_i)

  # Apply identity operations to all index tensors to ensure that assertions are
  # executed in the graph mode.
  with tf.control_dependencies(assert_ops):
    result = {}
    for node_tag, (node_set, index) in indices:
      result[node_tag] = (node_set, tf.identity(index))

    return result


def _node_tag_to_index_key(node_tag: IncidentNodeTag) -> str:
  """Converts node incident tag to internal string representation.

  GraphPiece requires that all its metadata entries are keyed by string keys.
  This function converts node incident tags to their string representations
  which are used to store incident node set names in metadata and indices
  tensors in the GraphPiece data.

  See `GraphPiece` class for more information.

  Args:
    node_tag: node incident tag.

  Returns:
    Internal string key representation.
  """
  if not isinstance(node_tag, IncidentNodeTag):
    raise ValueError(
        f'Node set tag must be integer, got {type(node_tag).__name__}')
  return f'{const.INDEX_KEY_PREFIX}{node_tag}'


def _index_key_to_node_tag(index_key: str) -> IncidentNodeTag:
  """Recovers node incident tag from internal string representation.

  See `_node_tag_to_index_key`.

  Args:
    index_key: internal node incident tag string representation.

  Returns:
    Node incident tag.
  """
  assert index_key.startswith(const.INDEX_KEY_PREFIX)
  return int(index_key[len(const.INDEX_KEY_PREFIX):])


def _get_indicative_index(
    indices: Mapping[str, Union[Field, FieldSpec]]) -> Union[Field, FieldSpec]:
  """Deterministically selects one of the index tensors from the `indices`."""
  assert indices
  _, result = min(indices.items(), key=lambda item: item[0])

  assert isinstance(
      result, (tf.Tensor, tf.RaggedTensor, tf.TensorSpec, tf.RaggedTensorSpec))
  assert result.shape.rank >= 1
  return result
