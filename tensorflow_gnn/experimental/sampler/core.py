# Copyright 2023 The TensorFlow GNN Authors. All Rights Reserved.
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
"""Baseline implementations for core sampling layers."""
import abc
import base64
import collections
import functools
from typing import Any, cast, Collection, List, Mapping, Optional, Tuple, Union

import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn.experimental.sampler import ext_ops
from tensorflow_gnn.experimental.sampler import interfaces
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import composite_tensor
# pylint: enable=g-direct-tensorflow-import


NODE_ID_NAME = '#id'

Features = interfaces.Features
FeaturesSpec = interfaces.FeaturesSpec
GraphPieces = collections.namedtuple('GraphPieces', [
    'context',    # : Features
    'node_sets',  # : Mapping[tfgnn.NodeSetName,Union[Features, List[Features]]]
    'edge_sets'   # : Mapping[str, Union[Features, List[Features]]
])


class CompositeLayer(tf.keras.layers.Layer, metaclass=abc.ABCMeta):
  r"""Base class for layers implemented on top of other sampling layers.

  This class must be used as a base class if a sampling layer has other sampling
  layers as its class members. Compared to the `tf.keras.layers.Layer` layer,
  the instances of this class must implement the `symbolic_call()` method, which
  is alike implementing the `call` method of `tf.keras.layers.Layer` but for
  symbolic Keras inputs using Functional API.

  Example: Wrapper for any `KeyToBytesAccessor` that adds fixed prefix to keys.

  ```python
  class WithPrefix(CompositeLayer, KeyToBytesAccessor):

    def __init__(self, prefix: str, accessor: KeyToBytesAccessor, **kwargs):
      super().__init__(**kwargs)
      self._prefix = prefix
      self._accessor = cast(tf.keras.layers.Layer, accessor)

    def get_config(self):
      return dict(
        prefix=self._prefix, accessor=self._accessor, **super().get_config()
      )

    def symbolic_call(self, keys):
      keys = tf.strings.as_string(keys)
      keys = tf.strings.join(prefix, keys)
      return self._accessor(keys)

  animals = InMemStringKeyToBytesAccessor(
      keys_to_values={
          'mammal0': 'porcupine',
          'mammal1': 'beaver',
          'marsupials0': 'kangaroo',
          'marsupials1': 'koala',
      }
  )
  mammals = WithPrefix('mammal', animals)
  output = mammals(tf.ragged.constant([[1, 0]]))
  # [["beaver", "porcupine"]]
  ```

  TL;DR. Keras has two computational models: "imperative" and "functional".
  The latter expresses computations as a graph with Keras layers as its nodes
  and their inputs and outputs as graph edges. This graph could be introspected
  which allows sampling API to map Keras layers on sampling stages, e.g. to
  export Keras sampling model as a bulk execution plan. Unfortunatelly this
  graph does not have the concept of modularity: if some computations are
  grouped as Python functions or classes it is not reflected in Keras graph and
  completely lost if the model is serialized. The "imperative" model adds
  modularity to Keras as it allows to create new custom layers using other
  layers as its building blocks. Unfortunately (for Sampler) custom layers do
  not allow introspection: in the general case it is not possible to restore the
  Keras computation graph for a custom keras Layer. The `CompositeLayer` retains
  modularity but allows for introspection. The computational graph of any of its
  instances is available in the `wrapped_model` property as `tf.keras.Model`.
  This structure is preserved if sampling model saved and then loaded.
  """

  def __init__(self, **kwargs):
    # NOTE: even though when the layer is restored from config the
    # `_wrapped_model_input_spec` is known in init time, we must delay the model
    # construction until all subclasses are fully initialized as model building
    # depends on their `symbolic_call()` method implementation.
    wrapped_model_input_spec = kwargs.pop('wrapped_model_input_spec', None)
    super().__init__(**kwargs)
    self._wrapped_model_input_spec = wrapped_model_input_spec
    self._wrapped_model_is_built = False

  @abc.abstractmethod
  def symbolic_call(self, *args, **kwargs):
    """Builds the composed model using symbolic inputs.

    The instances of the `CompositeLayer` layer must implement their logic here.
    The input arguments are Keras symbolic tensors mirroring Layer's `__call__`
    arguments. The implementation must express layer logic using Keras
    Functional API and return result tensors.

    This function is called from the `CompositeLayer.call()`, only at the first
    time it is called. Once the sub-model built, it is re-used in the following
    CompositeLayer.call() and it can be accessed using the `wrapped_model`
    property of the object.


    Args:
      *args: positional arguments as `tf.keras.Input` objects, with the first
        argument always present.
      **kwargs: named arguments as `tf.keras.Input` objects.

    Returns:
      Result value(s), as a single Keras tensor or nest of Keras tensors.
    """
    raise NotImplementedError

  @property
  def wrapped_model(self) -> tf.keras.Model:
    if not self._wrapped_model_is_built:
      self._build_wrapped_model()
    return self._model

  def get_config(self):
    if not self._wrapped_model_is_built:
      if self._wrapped_model_input_spec is None:
        raise ValueError(
            'Cannot get a config for saving a composite layer '
            'before it has been built (during the first call).'
        )

      self._build_wrapped_model()
    return {
        'wrapped_model_input_spec': self._wrapped_model_input_spec,
        **super().get_config(),
    }

  def call(self, *args, **kwargs):
    training = kwargs.pop('training', None)

    input_spec = tf.nest.map_structure(_get_type_spec, _pack_args(args, kwargs))
    if not self._wrapped_model_is_built:
      self._wrapped_model_input_spec = input_spec
      self._build_wrapped_model()
    else:
      assert self._wrapped_model_input_spec is not None
      _check_same_input_struct(
          self.name, self._wrapped_model_input_spec, input_spec
      )

    return self._model(tf.nest.flatten([args, kwargs]), training=training)

  def _build_wrapped_model(self):
    assert not self._wrapped_model_is_built
    assert self._wrapped_model_input_spec is not None

    with tf.init_scope():
      args, kwargs = tf.nest.map_structure(
          lambda type_spec: tf.keras.Input(type_spec=type_spec),
          self._wrapped_model_input_spec,
      )
      outputs = self.symbolic_call(*args, **kwargs)
      self._model = tf.keras.Model(
          inputs=tf.nest.flatten([args, kwargs]), outputs=outputs
      )

    self._wrapped_model_is_built = True


class _InMemKeyToBytesAccessor(
    tf.keras.layers.Layer, interfaces.KeyToBytesAccessor
):
  """Base class to lookup serialized values by their keys from memory.

  In case of missing values layer allows to either return fixed default value or
  raise `tf.errors.InvalidArgumentError` in runtime.
  """

  def __init__(
      self,
      *,
      index_cls=None,
      keys_to_values: Optional[Mapping[Any, bytes]] = None,
      default_value: Optional[bytes] = b'',
      **kwargs,
  ):
    """Constructor.

    Args:
      index_cls: Keras layer class to use for input keys indexing. Either
        `tf.keras.layers.IntegerLookup` or `tf.keras.layers.StringLookup`.
      keys_to_values: A mapping from string key to bytes values.
      default_value: The value to use in place of missing values. If `None`, any
        missing value results in `tf.errors.InvalidArgumentError`.
      **kwargs: Other arguments for the base class.
    """
    keys_to_index = kwargs.pop('keys_to_index', None)
    index_to_values = kwargs.pop('index_to_values', None)

    super().__init__(**kwargs)

    self._default_value = default_value

    if index_to_values is not None:
      # Object is restored from Keras saved model (`from_config``).
      assert keys_to_index is not None
      self._keys_to_index = keys_to_index
      self._index_to_values = index_to_values
      return

    assert index_cls is not None
    keys_to_values = keys_to_values or {}

    num_oov = 0 if default_value is None else 1
    keys = list(keys_to_values.keys())
    values = [default_value] * num_oov + list(keys_to_values.values())
    self._keys_to_index = index_cls(vocabulary=keys, num_oov_indices=num_oov)
    self._index_to_values = _BytesArray(values)

  @property
  def resource_name(self) -> str:
    return self.name

  def get_config(self):
    return {
        'default_value': base64.b64encode(self._default_value).decode('utf-8'),
        'index_to_values': self._index_to_values,
        'keys_to_index': self._keys_to_index,
        **super().get_config(),
    }

  @classmethod
  def from_config(cls, config):
    config['default_value'] = base64.b64decode(
        config['default_value'].encode('utf-8'))
    return cls(**config)

  def call(self, keys: tf.RaggedTensor) -> tf.RaggedTensor:
    _check_ragged_rank1(keys, type(self).__name__)
    indices = self._keys_to_index(keys)
    return self._index_to_values(indices)


@tf.keras.utils.register_keras_serializable(package='GNN')
class InMemStringKeyToBytesAccessor(_InMemKeyToBytesAccessor):
  """Lookup serialized values by their string keys from memory.

  In case of missing values layer allows to either return fixed default value or
  raise `tf.errors.InvalidArgumentError` in runtime.

  Example:

  ```python
    layer = InMemStringKeyToBytesAccessor(keys_to_values={'a': b'A', 'b': b'B'})
    layer(tf.ragged.constant([['a', 'b'], ['c']]))
    # [['A', 'B'], ['']]
  ```
  """

  def __init__(
      self,
      *,
      keys_to_values: Optional[Mapping[str, bytes]] = None,
      default_value: Optional[bytes] = b'',
      **kwargs,
  ):
    """Constructor.

    Args:
      keys_to_values: A mapping from string key to bytes values.
      default_value: The value to use in place of missing values. If `None`, any
        missing value results in `tf.errors.InvalidArgumentError`.
      **kwargs: Other arguments for the base class.
    """
    super().__init__(
        index_cls=tf.keras.layers.StringLookup,
        keys_to_values=keys_to_values,
        default_value=default_value,
        **kwargs,
    )


@tf.keras.utils.register_keras_serializable(package='GNN')
class InMemIntegerKeyToBytesAccessor(_InMemKeyToBytesAccessor):
  """Lookup serialized values by their `int32` or `int64` keys from memory.


  In case of missing values layer allows to either return fixed default value or
  raise `tf.errors.InvalidArgumentError` in runtime.

  Example:

  ```python
    layer = InMemIntegerKeyToBytesAccessor(keys_to_values={10: b'A', 30: b'C'})
    layer(tf.ragged.constant([[10, 20], [30]]))
    # [['A', ''], ['C']]
  ```
  """

  def __init__(
      self,
      *,
      keys_to_values: Optional[Mapping[int, bytes]] = None,
      default_value: Optional[bytes] = b'',
      **kwargs,
  ):
    """Constructor.

    Args:
      keys_to_values: A mapping from integer key to bytes values.
      default_value: The value to use in place of missing values. If `None`, any
        missing value results in `tf.errors.InvalidArgumentError`.
      **kwargs: Other arguments for the base class.
    """
    super().__init__(
        index_cls=tf.keras.layers.IntegerLookup,
        keys_to_values=keys_to_values,
        default_value=default_value,
        **kwargs,
    )


@tf.keras.utils.register_keras_serializable(package='GNN')
class KeyToTfExampleAccessor(CompositeLayer, interfaces.KeyToFeaturesAccessor):
  r"""Accessor for features stored as `Example` proto.

  Example:

  ```python
    serialized_features = core.InMemStringKeyToBytesAccessor(
        keys_to_values={
            'a': pbtext.Merge(
                \"""
                features {
                  feature {
                      key: "class" value {int64_list {value: [1]} }
                  }
                  feature {
                      key: "words" value {bytes_list {value: ["to", "be"]}}
                  }
                }
                \""",
                tf.train.Example(),
            ).SerializeToString(),
        }
    )
    features_layer = tfgnn.sampler.KeyToTfExampleAccessor(
        serialized_features,
        features_spec={
            'class': tf.TensorSpec([], tf.int64),
            'words': tf.TensorSpec([None], tf.string),
        },
    )

    features = features_layer(tf.ragged.constant([["a", "x"], ["y"]]))
    # {
    #     'words': tf.ragged.constant([[['to', 'be'], []], [[]]]),
    #     'class': tf.ragged.constant([[1, 0], [0]]),
    # }
  ```
  """

  def __init__(
      self,
      key_to_serialized: interfaces.KeyToBytesAccessor,
      *,
      features_spec: FeaturesSpec,
      default_values: Optional[Mapping[str, Any]] = None,
      **kwargs,
  ):
    """Constructor.

    Args:
      key_to_serialized: An accessor to serialized `Example` proto.
      features_spec: A mapping from a feature name to its type spec.
      default_values: An optional mapping between a feature name and a value to
        be used if the feature is not present in the example. By default, unless
        otherwise specified, 0 values are used for numeric types and empty
        strings for `tf.string` type. For features for which missing values are
        not supported and must result in the runtime errir, the default values
        must be explicitly set to `None`.
      **kwargs: Other arguments for the base class.
    """
    super().__init__(**kwargs)
    self._key_to_serialized = cast(tf.keras.layers.Layer, key_to_serialized)
    self._features_spec = features_spec.copy()
    self._default_values = default_values.copy() if default_values else {}

    self._parser = TfExamplesParser(
        features_spec, default_values=self._default_values
    )

  @property
  def resource_name(self) -> str:
    return self._key_to_serialized.resource_name

  def get_config(self):
    return dict(
        key_to_serialized=self._key_to_serialized,
        features_spec=self._features_spec,
        default_values=self._default_values,
        **super().get_config(),
    )

  def symbolic_call(self, keys):
    values = self._key_to_serialized(keys)
    return self._parser(values)

  def call(self, keys: tf.RaggedTensor) -> Features:
    return super().call(keys)


class InMemIndexToFeaturesAccessor(
    tf.keras.layers.Layer, interfaces.KeyToFeaturesAccessor
):
  """Extracts features by their indices from the features tensor.

  A single `call()` has O(tf.size(indices)) time complexity.

  Example:

  ```python
    paper_features = core.InMemIndexToFeaturesAccessor(
        {
            'year': [2018, 2019, 2017],
            'feat': [[1., 2.], [3., 4.], [5., 6.]]
        },
        name='paper',
    )


    features = paper_features(tf.ragged.constant([[2, 1], [0]]))
    # {
    #     'year': tf.ragged.constant([[2017, 2019], [2018]]),
    #     'feat': tf.ragged.constant([[[5., 6.], [3., 4.]], [[1., 2.]]]),
    # }
  ```


  Call returns:
      `Features` for values having `indices`. All returned features have shape
      `[batch_size, (num_keys), ...]`.
  """

  def __init__(
      self,
      features: tfgnn.Fields,
      **kwargs,
  ):
    """Constructor.

    Args:
      features: A dictionary of node features for all nodes. Each feature must
        have shape `[num_values, **feature_dims]`.
      **kwargs: Other arguments for the base class.
    """
    super().__init__(**kwargs)
    self._features = {k: tf.convert_to_tensor(v) for k, v in features.items()}

  @property
  def resource_name(self) -> tfgnn.NodeSetName:
    return self.name

  @classmethod
  def from_graph_tensor(
      cls,
      graph_tensor: tfgnn.GraphTensor,
      node_set_name: tfgnn.NodeSetName,
      *,
      name: Optional[str] = None,
  ) -> 'InMemIndexToFeaturesAccessor':
    """Creates an accessor to `graph_tensor` node set features by their indices.

    Args:
      graph_tensor: A scalar (rank 0) `GraphTensor`.
      node_set_name: The node set name to access.
      name: The name of returned Keras layer. If not specified, the
        `node_set_name` is used.

    Returns:
       Keras layer, instance fo the `InMemIndexToFeaturesAccessor`.
    """
    if graph_tensor.rank != 0:
      raise ValueError(
          f'Expected scalar graph tensor, got rank={graph_tensor.rank}'
      )
    name = name or node_set_name
    node_set = graph_tensor.node_sets[node_set_name]
    return cls(node_set.get_features_dict(), name=name)

  def call(self, indices: tf.RaggedTensor) -> Features:
    return tf.nest.map_structure(
        functools.partial(tf.gather, indices=indices), self._features
    )


@tf.keras.utils.register_keras_serializable(package='GNN')
class UniformEdgesSampler(CompositeLayer, interfaces.OutgoingEdgesSampler):
  """Samples edges uniformly at random from adjacency lists without replacement.

  Example: For each input papers samples up to 2 cited papers.

  ```python
    cited_papers = tfgnn.KeyToTfExampleAccessor(
      serialized_cited_papers,
      features_spec={
          '#target': tf.TensorSpec([None], tf.string),
          'weight': tf.TensorSpec([None], tf.float32),
      },
    )
    edge_sampler = tfgnn.UniformEdgesSampler(cited_papers, sample_size=2)
    cites = edge_sampler(tf.ragged.constant([['paper1', 'paper2'], ['paper1']]))
    #
    # #   edges:       1->3      1->4      2->1        1->4      1->5
    # {
    #   '#source': [['paper1', 'paper1', 'paper2'], ['paper1', 'paper1']]
    #   '#target': [['paper3', 'paper4', 'paper1'], ['paper4', 'paper5']]
    #   'weight':  [[  0.3,      0.4,      0.1   ], [  0.4,      0.5   ]]
    # }
  ```

  Call returns:
      `Features` containing the subset of all edges whose source nodes are in
      `source_node_ids`. All returned features have shape `[batch_size,
      (num_edges), ...]`. Result must include two special features
      "#source" and "#target" of rank 2 containing, correspondigly, source node
      ids and targert node ids of the sampled edges.
  """

  def __init__(
      self,
      outgoing_edges_accessor: interfaces.KeyToFeaturesAccessor,
      *,
      sample_size: int,
      edge_target_feature_name: str = tfgnn.TARGET_NAME,
      seed: Optional[int] = None,
      **kwargs,
  ):
    """Constructor.

    Args:
      outgoing_edges_accessor: The Keras layer that for each source node returns
        all its outgoing edges as a `Features` dictionary. Features must include
        `edge_target_feature_name` feature with target node ids of the edges
        allong with other edge features. All returned features must have a shape
        `[batch_size, (num_source_nodes), (num_outgoing_edges), *feature_dims]`.
      sample_size: The maximum number of edges to sample for each source node.
      edge_target_feature_name: The name of the feature returned by the
        `outgoing_edges_accessor` containing target node ids of the edges.
      seed: A Python integer. Used to create a random seed for sampling.
      **kwargs: Other arguments for the base class.
    """
    super().__init__(**kwargs)
    self._outgoing_edges_accessor = cast(
        tf.keras.layers.Layer, outgoing_edges_accessor
    )
    self._sample_size = sample_size
    self._edge_target_feature_name = edge_target_feature_name
    self._seed = seed
    self._sampler = _UniformEdgesSelector(
        sample_size=sample_size,
        edge_target_feature_name=edge_target_feature_name,
        seed=seed,
    )

  @property
  def sample_size(self) -> int:
    return self._sample_size

  @property
  def resource_name(self) -> tfgnn.EdgeSetName:
    return self._outgoing_edges_accessor.resource_name

  def get_config(self):
    return dict(
        outgoing_edges_accessor=self._outgoing_edges_accessor,
        sample_size=self._sample_size,
        edge_target_feature_name=self._edge_target_feature_name,
        seed=self._seed,
        **super().get_config(),
    )

  def symbolic_call(self, source_node_ids):
    outgoing_edges = self._outgoing_edges_accessor(source_node_ids)
    if self._edge_target_feature_name not in outgoing_edges:
      raise ValueError(
          f'Expected {self._edge_target_feature_name} feature '
          'with target node ids of an outgoing edges.'
      )

    return self._sampler([source_node_ids, outgoing_edges])

  def call(self, source_node_ids: tf.RaggedTensor) -> Features:
    return super().call(source_node_ids)


class InMemUniformEdgesSampler(
    tf.keras.layers.Layer, interfaces.OutgoingEdgesSampler
):
  """Samples edges uniformly at random from in-memory edge features.

  A single `call()` has O(tf.size(indices)) time complexity.

  NOTE: the class allocates two auxiliary integer tensors with shapes
  `[num_edges]`, `[num_source_nodes, 2]`.

  TODO(aferludin): consider optimizations if edges are sorted by their sources.

  Example: Samples up to 2 edges uniformly at random for each input source node.

  ```python
    cited_papers = tfgnn.InMemUniformEdgesSampler(
      num_source_nodes=2,
      source=[1, 0, 0, 0, 1],
      target=[2, 1, 2, 3, 3],
      features_spec={
          'weight': [0.2, 0.1, 0.2, 0.3, 0.3]
      },
      sample_size = 2
    )
    cites = edge_sampler(tf.ragged.constant([[0], [0, 1]]))
    # #   edges:     0->1,  0->3,   0->3,  0->2, 1->2, 1->3
    # {
    #   '#source': [[  0,    0  ], [  0,    0,    1,    1  ]]
    #   '#target': [[  1,    3  ], [  3,    2,    2,    3  ]]
    #   'weight':  [[ 0.1,  0.3 ], [ 0.3,  0.2,  0.2,  0.3 ]]
    # }
  ```

  Call returns:
      Edge features containing the subset of all edges whose source nodes are in
      `source_node_ids`. All returned features have shape `[batch_size,
      (num_edges), ...]`. Result includes two special features "#source" and
      "#target" of rank 2 containing, correspondigly, source node ids and
      targert node ids of the sampled edges.
  """

  def __init__(
      self,
      num_source_nodes: tf.Tensor,
      source: tf.Tensor,
      target: tf.Tensor,
      edge_features: Optional[tfgnn.Fields] = None,
      *,
      sample_size: int,
      seed: Optional[int] = None,
      **kwargs,
  ):
    """Constructor.

    Args:
      num_source_nodes: The number of source nodes. Scalar integer tensor.
      source: The indices of source nodes of edges. Tensor of `int32` or `int64`
        dtype and shape `[num_edges]`.
      target: The indices of target nodes of edges. Tensor the same dtype and
        shape as `source`.
      edge_features: An optional dictionary of edge features. Each feature must
        have shape `[num_edges, **feature_dims]`.
      sample_size: The maximum number of edges to sample for each source node.
      seed: A Python integer. Used to create a random seed for sampling.
      **kwargs: Other arguments for the base class.
    """
    super().__init__(**kwargs)
    num_source_nodes = tf.convert_to_tensor(num_source_nodes)
    source = tf.convert_to_tensor(source)
    target = tf.convert_to_tensor(target)
    edge_features = {
        k: tf.convert_to_tensor(v) for k, v in (edge_features or {}).items()
    }
    self._fields = {
        tfgnn.SOURCE_NAME: source,
        tfgnn.TARGET_NAME: target,
        **edge_features,
    }

    self._sample_size = sample_size
    self._seed = seed

    self._sort_index = tf.argsort(source)
    sorted_source = tf.gather(source, self._sort_index)

    splits_by_source = tf.ragged.segment_ids_to_row_splits(
        sorted_source, num_source_nodes
    )
    starts, limits = splits_by_source[:-1], splits_by_source[1:]
    outgoing_edges_count = limits - starts
    # Partition of `self._fields` if sorted by self._sort_index by source nodes.
    self._partition_by_source = tf.stack([starts, outgoing_edges_count], axis=1)

  @property
  def sample_size(self) -> int:
    return self._sample_size

  @property
  def resource_name(self) -> tfgnn.EdgeSetName:
    return self.name

  @classmethod
  def from_graph_tensor(
      cls,
      graph_tensor: tfgnn.GraphTensor,
      edge_set_name: tfgnn.EdgeSetName,
      *,
      sample_size: int,
      source_tag: tfgnn.IncidentNodeTag = tfgnn.SOURCE,
      seed: Optional[int] = None,
      name: Optional[str] = None,
  ) -> 'InMemUniformEdgesSampler':
    """Creates a uniform outgoing edges sampler for a `graph_tensor`'s edge set.

    Args:
      graph_tensor: A scalar (rank 0) `GraphTensor`.
      edge_set_name: The edge set to sample edges from.
      sample_size: The maximum number of edges to sample from each `source_tag`
        node.
      source_tag: The incident node set to sample outgoing edge from.
      seed: A Python integer. Used to create a random seed for sampling.
      name: The name of returned Keras layer. If not specified, the
        `edge_set_name` is used.

    Returns:
       Keras layer, instance fo the `InMemUniformEdgesSampler`.
    """
    if graph_tensor.rank != 0:
      raise ValueError(
          f'Expected scalar graph tensor, got rank={graph_tensor.rank}'
      )
    target_tag = tfgnn.reverse_tag(source_tag)
    name = name or edge_set_name
    edge_set = graph_tensor.edge_sets[edge_set_name]
    adj = edge_set.adjacency
    if not isinstance(adj, tfgnn.Adjacency):
      raise ValueError(
          'Expected adjacency of `tfgnn.Adjacency` type, got'
          f' {type(adj).__name__}'
      )
    source_node_set = graph_tensor.node_sets[adj.node_set_name(source_tag)]
    return cls(
        num_source_nodes=source_node_set.total_size,
        source=adj[source_tag],
        target=adj[target_tag],
        edge_features=edge_set.features,
        name=name,
        sample_size=sample_size,
        seed=seed,
    )

  def call(self, source_node_ids: tf.RaggedTensor) -> Features:
    # First sample edge indices as if they are sorted by their source nodes.
    # Then remap obtained indices on the real edge positions in `self._fields`
    # using `self._sort_index`.

    outgoing_edges_offsets, outgoing_edges_count = tf.unstack(
        tf.gather(self._partition_by_source, source_node_ids.values), 2, axis=1
    )
    samples_count = tf.minimum(
        tf.constant(self.sample_size, dtype=outgoing_edges_count.dtype),
        outgoing_edges_count,
    )
    targets_splits = _row_lengths_to_row_splits(outgoing_edges_count)
    # Indices of sampled target nodes within each group of target nodes.
    edges_idx = ext_ops.ragged_choice(
        samples_count, targets_splits, global_indices=False, seed=self._seed
    )
    # Indices of sampled edges in `self._fields` if sorted by source.
    edges_idx += tf.expand_dims(outgoing_edges_offsets, -1)
    edges_idx = edges_idx.values
    # Real indices of sampled edges in the original `self._fields` (not sorted).
    edges_idx = tf.gather(self._sort_index, edges_idx)

    result_row_lengths = tf.math.unsorted_segment_sum(
        samples_count,
        source_node_ids.value_rowids(),
        source_node_ids.nrows(),
    )
    result_row_splits = _row_lengths_to_row_splits(result_row_lengths)

    def extract(field: tfgnn.Field) -> tfgnn.Field:
      return tf.RaggedTensor.from_row_splits(
          tf.gather(field, edges_idx), result_row_splits, validate=False
      )

    return tf.nest.map_structure(extract, self._fields)


@tf.keras.utils.register_keras_serializable(package='GNN')
class TfExamplesParser(tf.keras.layers.Layer):
  """Parses serialized Example protos according to features type spec."""

  def __init__(
      self,
      features_spec: FeaturesSpec,
      *,
      default_values: Optional[Mapping[str, Any]] = None,
      **kwargs,
  ):
    """Constructor.

    Args:
      features_spec: A mapping from feature name to its inner dimensions spec,
        excluding batch and ragged dimension of the input values (so scalar
        `tf.float32` feature has a type spec `tf.TensorSpec([], tf.string))`.
      default_values: An optional mapping from feature name to default value as
        a Python constants.
      **kwargs: Other arguments for the tf.keras.layers.Layer base class.
    """
    super().__init__(**kwargs)
    self._features_spec = features_spec.copy()
    self._default_values = default_values.copy() if default_values else {}

  def get_config(self):
    return dict(
        features_spec=self._features_spec,
        default_values=self._default_values,
        **super().get_config(),
    )

  def call(
      self, serialized: Union[tf.RaggedTensor, tf.Tensor]
  ) -> Mapping[str, Union[tf.RaggedTensor, tf.Tensor]]:
    """Parses serialized Example protos from passed strings.

    Args:
      serialized: A potentially ragged tensor of strings (`tf.string`) as
        serialized Example protos.

    Returns:
      Dictionary of parsed Features with the same partitions of outer dimensions
      as `serialized`.
    """
    flat_features_spec = {
        k: _get_io_spec(k, v, self._default_values.get(k, _type_default(v)))
        for k, v in self._features_spec.items()
    }

    flat_values = (
        serialized.flat_values
        if isinstance(serialized, tf.RaggedTensor)
        else tf.reshape(serialized, [-1])
    )
    assert flat_values.shape.rank == 1
    assert flat_values.dtype == tf.string

    flat_features = tf.io.parse_example(flat_values, flat_features_spec)
    if isinstance(serialized, tf.RaggedTensor):
      return tf.nest.map_structure(serialized.with_flat_values, flat_features)

    if isinstance(serialized, tf.Tensor):
      shape = tf.shape(serialized)

      def reshape(value):
        if isinstance(value, tf.Tensor):
          return tf.reshape(
              value, tf.concat([shape, tf.shape(value)[1:]], axis=0)
          )
        if isinstance(value, tf.RaggedTensor):
          for dim in reversed(tf.unstack(shape)[1:]):
            value = tf.RaggedTensor.from_uniform_row_length(
                value, tf.cast(dim, value.row_splits.dtype)
            )
          return value
        raise ValueError(f'Unsupported type {type(value).__name__}')

      return tf.nest.map_structure(reshape, flat_features)

    raise ValueError(f'Unsupported type {type(serialized).__name__}')


def build_graph_tensor(
    *,
    context: Optional[Features] = None,
    node_sets: Optional[
        Mapping[tfgnn.NodeSetName, Union[Features, List[Features]]]
    ] = None,
    edge_sets: Optional[
        Mapping[str, Union[Features, List[Features]]]
    ] = None,
    validate: bool = True,
    remove_parallel_edges: bool = True,
) -> tfgnn.GraphTensor:
  """Builds GraphTensor from its pieces using node ids.

  Convenience wrapper for `GraphTensorBuilder` layer.

  NOTE: edge sets could reference node sets that are missing in the `node_sets`.
  In this case latent nodes sets are added to the result graph tensor.

  Args:
    context: A graph context features as a mapping from feature name to dense or
      ragged value. All features must have `[batch_size, (num_components), ...]`
      shapes.
    node_sets: A mapping from node set name to node features or list of node
      features. For the latter features are assumed to belong to disjoint sets
      of nodes and must be compatible. Features mapping must include the
      `NODE_ID_NAME` feature with unique node identifier (string or integer).
      All features must have shapes `[batch_size, (num_nodes), ...]`.
    edge_sets: A mapping from a edge set definition key to edge features or list
      of edge features. The edge set definition key is created by joining the
      names of the edge's source node set, edge set name, and edge's target node
      set using the comma `,` saparator, e.g.
      "<source_node_set>,<edge_set>,<target_node_set>". If a mapping value is a
      list of features, they are assumed to belong to disjoint sets of edges and
      must be compatible. Features dict must include the `tfgnn.SOURCE_NAME` and
      `tfgnn.TARGET_NAME` with source and target node ids correspondingly. All
      features must have shapes `[batch_size, (num_edges), ...]`.
    validate: If True, runs potentially expensive runtime consitency checks,
      like that node sets have unique ids.
    remove_parallel_edges: if True, deduplicates parallel (duplicate) edges
      incident to the same source and target nodes. Which particular parallel
      edge is selected is not deterministic and it is assumed that they are
      equivalent.

  Returns:
    GraphTensor with `rank=1`.

  Example 1: Homogeneous graph from edge set. Node set is latent (no features).
    tfgnn.build_graph_tensor(
        edge_sets={
            'A,A->A,A': {
                tfgnn.SOURCE_NAME: rt([['a', 'a', 'b'], ['a']]),
                tfgnn.TARGET_NAME: rt([['a', 'b', 'a'], ['b']]),
                'weight': rt([[1.0, 0.7, 0.9], [0.7]]),
            },
        })
    print(graph.node_sets['A'].sizes)                # [[2], [2]]
    print(graph.node_sets['A']['#id'])               # [['a', 'b'], ['a', 'b']
    print(graph.edge_sets['A->A'].sizes)             # [[3], [1]]
    print(graph.edge_sets['A->A'].adjacency.source)  # [[0, 0, 1], [0]]
    print(graph.edge_sets['A->A'].adjacency.target)  # [[0, 1, 0], [1]]
    print(graph.edge_sets['A->A']['weight'])         # [[1.0, 0.7, 0.9], [0.7]]

  Example 2: Heterogeneous graph with two node sets and one edge set.
    tfgnn.build_graph_tensor(
        node_sets={
            'A': {
                '#id': rt([[1, 2, 3], [1, 2], [3]]),
            },
            'B': {
                '#id': rt([[4, 3, 2, 1], [7, 6, 5], [1]]),
            },
        },
        edge_sets={
            'A,A->B,B': {
                tfgnn.SOURCE_NAME: rt([[1, 2, 2, 1], [2, 2, 2], [3]]),
                tfgnn.TARGET_NAME: rt([[1, 2, 3, 4], [5, 6, 7], [1]]),
            }
        })

  Example 3: Joins node features from two disjoint sets of nodes.
    tfgnn.build_graph_tensor(
        node_sets={
            'A': [{
                '#id': rt([['a'], ['c']]),
                'f.s': rt([[0], [2]]),
            }, {
                '#id': rt([['b'], []]),
                'f.s': rt([[1], []]),
            }]
        })
    # The same as:
    tfgnn.build_graph_tensor(
        node_sets={
            'A': {
                '#id': rt([['a', 'b'], ['c']]),
                'f.s': rt([[0, 1], [2]]),
            }
        })
  """
  layer = GraphTensorBuilder(
      remove_parallel_edges=remove_parallel_edges, validate=validate
  )
  layer_input = GraphPieces(context=context or {}, node_sets=node_sets or {},
                            edge_sets=edge_sets or {})
  return layer(layer_input)


@tf.keras.utils.register_keras_serializable(package='GNN')
class GraphTensorBuilder(tf.keras.layers.Layer):
  """Builds GraphTensor from its pieces using node ids.

  See examples of inputs in `tfgnn.build_graph_tensor`.

  Call arguments:
    dict with 3 (required) string keys:
      context: A graph context features as a mapping from feature name to dense
        or ragged value. All features must have shapes
        `[batch_size, (num_components), ...]`.
      node_sets: A mapping from node set name to node features or list of node
        features. For the latter features are assumed to belong to disjoint sets
        of nodes and must be compatible. Features mapping must include the
        `NODE_ID_NAME` feature with unique node identifier (string or integer).
        All features must have shapes `[batch_size, (num_nodes), ...]`.
      edge_sets: A mapping from a edge set definition key to edge features or
        list of edge features. The edge set definition key is created by joining
        the names of the edge's source node set, edge set name, and edge's
        target node set using the comma `,` separator, e.g.
        "<source_node_set>,<edge_set>,<target_node_set>". If a mapping value is
        a list of features, they are assumed to belong to disjoint sets of edges
        and must be compatible. Features dict must include the
        `tfgnn.SOURCE_NAME` and `tfgnn.TARGET_NAME` with source and target node
        ids correspondingly. All features must have shapes
        `[batch_size, (num_edges), ...]`.

  Call returns:
      GraphTensor of rank 1.
  """

  def __init__(
      self,
      *,
      remove_parallel_edges: bool = True,
      validate: bool = True,
      **kwargs
  ):
    super().__init__(**kwargs)
    self._remove_parallel_edges = remove_parallel_edges
    self._validate = validate

  def get_config(self):
    return dict(
        remove_parallel_edges=self._remove_parallel_edges,
        validate=self._validate,
        **super().get_config(),
    )

  def call(self, inputs: GraphPieces) -> tfgnn.GraphTensor:
    # When saving `tf.keras.Model`, the loaded model passes a tuple.
    # Convert to namedtuple.
    inputs = GraphPieces(*inputs)
    context: Features = inputs.context
    node_sets: Mapping[tfgnn.NodeSetName,
                       Union[Features, List[Features]]] = inputs.node_sets
    edge_sets: Mapping[str,
                       Union[Features, List[Features]]] = inputs.edge_sets

    def join(pieces: Union[Features, List[Features]]) -> Features:
      if isinstance(pieces, List):
        return concat_features(pieces)
      return pieces

    context_ = tfgnn.Context.from_fields(features=context, shape=[None])
    node_sets_ = {}
    edge_sets_ = {}

    latent_node_sets = collections.defaultdict(list)
    for key, features in edge_sets.items():
      source_node_set, _, target_node_set = (
          _parse_edge_set_definition(key)
      )

      features = features if isinstance(features, list) else [features]
      if source_node_set not in node_sets:
        latent_node_sets[source_node_set].extend(
            [f[tfgnn.SOURCE_NAME] for f in features]
        )
      if target_node_set not in node_sets:
        latent_node_sets[target_node_set].extend(
            [f[tfgnn.TARGET_NAME] for f in features]
        )

    node_sets_with_id = {
        **{k: v for k, v in node_sets.items()},
        **{
            k: [{NODE_ID_NAME: ext_ops.ragged_unique(tf.concat(v, -1))}]
            for k, v in latent_node_sets.items()
        },
    }

    nodes_ids = dict()
    for node_set_name, features in node_sets_with_id.items():
      features_ = {}
      sizes_ = None
      for fname, fvalue in join(features).items():
        if fname == NODE_ID_NAME:
          check_ops = []
          if self._validate and node_set_name not in latent_node_sets:
            check_ops.append(
                tf.debugging.assert_equal(
                    ext_ops.ragged_unique(fvalue).row_splits,
                    fvalue.row_splits,
                    f'Node set {node_set_name} ids are not unique.',
                )
            )
          with tf.control_dependencies(check_ops):
            sizes_ = tf.expand_dims(fvalue.row_lengths(), -1)
          nodes_ids[node_set_name] = fvalue
        features_[fname] = fvalue
      if sizes_ is None:
        raise ValueError(f'Missing `{NODE_ID_NAME}` in {node_set_name}.')

      node_sets_[node_set_name] = tfgnn.NodeSet.from_fields(
          features=features_, sizes=sizes_
      )

    for key, features in edge_sets.items():
      source_node_set, edge_set_name, target_node_set = (
          _parse_edge_set_definition(key)
      )
      key_to_index_fn = {
          tfgnn.SOURCE_NAME: functools.partial(
              ext_ops.ragged_lookup,
              vocabulary=nodes_ids[source_node_set],
              validate=self._validate,
          ),
          tfgnn.TARGET_NAME: functools.partial(
              ext_ops.ragged_lookup,
              vocabulary=nodes_ids[target_node_set],
              validate=self._validate,
          ),
      }
      features_ = {}
      indices_ = {}
      sizes_ = None
      for fname, fvalue in join(features).items():
        if fname in (tfgnn.SOURCE_NAME, tfgnn.TARGET_NAME):
          sizes_ = tf.expand_dims(fvalue.row_lengths(), -1)
          indices_[fname] = key_to_index_fn.pop(fname)(fvalue)
        else:
          features_[fname] = fvalue

      if key_to_index_fn:
        raise ValueError(
            f'Missing `{list(key_to_index_fn)} in {edge_set_name}.'
        )
      source, target = indices_[tfgnn.SOURCE_NAME], indices_[tfgnn.TARGET_NAME]
      edge_sets_[edge_set_name] = tfgnn.EdgeSet.from_fields(
          features=features_,
          sizes=sizes_,
          adjacency=tfgnn.Adjacency.from_indices(
              source=(source_node_set, source),
              target=(target_node_set, target),
          ),
      )
    result = tfgnn.GraphTensor.from_pieces(
        context=context_, node_sets=node_sets_, edge_sets=edge_sets_
    )
    if self._remove_parallel_edges:
      result = _remove_parallel_edges(result)
    return result


def concat_features(pieces: List[Features]) -> Features:
  """Concatenates features from multiple items along the 1st (item) dimension.

  Args:
    pieces: The list of features of several graph items. All features must have
      the same set of keys and compatible shapes, as `[batch_size,
      (items_dim_i), *feature_dims]`, where `item_dim_i` is potentially ragged
      dimension for the `i`-th element of `pieces`, the `batch_size` and
      `*feature_dims` must be the same for same features.

  Returns:
    Features concatenated along the 1st (item) dimension.
  """
  if not isinstance(pieces, List):
    raise ValueError(f'Expected list of features, got {type(pieces).__name__}')

  return tf.nest.map_structure(lambda *argv: tf.concat(argv, axis=1), *pieces)


def _get_io_spec(
    name: str,
    spec: Union[tf.TensorSpec, tf.RaggedTensorSpec],
    default_value=None,
):
  """Returns TF IO features parsing spec from value type spec."""
  if isinstance(spec, tf.TensorSpec) and spec.shape.is_fully_defined():
    return tf.io.FixedLenFeature(
        shape=spec.shape, dtype=spec.dtype, default_value=default_value
    )

  partitions = []
  for dim in spec.shape[1:]:
    if dim is not None:
      partitions.append(tf.io.RaggedFeature.UniformRowLength(dim))  # pytype: disable=attribute-error
    else:
      partitions.append(tf.io.RaggedFeature.RowLengths(f'd.{dim}'))  # pytype: disable=attribute-error

  return tf.io.RaggedFeature(
      value_key=name,
      dtype=spec.dtype,
      partitions=partitions,
      row_splits_dtype=tf.int64,
  )


def _type_default(spec: tf.TypeSpec) -> Optional[tf.Tensor]:
  if spec.shape.is_fully_defined():
    assert isinstance(spec, tf.TensorSpec)
    return tf.zeros(spec.shape, spec.dtype)
  return None


def _check_ragged_rank1(value, name: str) -> None:
  if (
      hasattr(value, 'ragged_rank')
      and value.shape.rank == 2
      and value.ragged_rank == 1
  ):
    return

  raise ValueError(
      f'{name} requires a `tf.RaggedTensor` of ragged rank 1'
      ' and shape compatible with `[None, None]`'
      f', got type_spec={tf.type_spec_from_value(value)}'
  )


def _get_type_spec(value):
  if not isinstance(value, (tf.Tensor, composite_tensor.CompositeTensor)):
    raise ValueError(
        'Expected tensor or composite tensor arguments'
        f', got {type(value).__name__}'
    )
  return tf.type_spec_from_value(value)


@tf.keras.utils.register_keras_serializable(package='GNN')
class _BytesArray(tf.keras.layers.Layer, interfaces.KeyToBytesAccessor):
  """Accessor for string (bytes) values by their indices.

  NOTE: we could not use `tf.keras.layers.StringLookup` with `invert=True` as
  vocabulary must be proper UTF-8 strings, whereas this implementation allows
  any bytes values (e.g. serialized protos).
  """

  def __init__(self, values: Optional[Collection[bytes]] = None, **kwargs):
    shape = kwargs.pop('values_shape', None)
    super().__init__(**kwargs)

    if values is None:
      assert shape is not None
      initializer = tf.constant_initializer('')
    else:
      assert shape is None
      initializer = tf.constant_initializer(values)
      shape = tf.TensorShape([len(values)])

    # Values are stored internally as `tf.Variable` so they could be saved to
    # and restored from checkpoints. We could not save values as `get_config`
    # entry as Keras serialization only supports UTF-8 strings.
    self._values = self.add_weight(
        name='values',
        shape=shape,
        dtype=tf.string,
        initializer=initializer,
        use_resource=False,
        trainable=False,
    )

  @property
  def resource_name(self) -> str:
    return self.name

  def get_config(self):
    return {
        'values_shape': self._values.shape,
        **super().get_config(),
    }

  def call(self, indices: tf.RaggedTensor) -> tf.RaggedTensor:
    if indices.dtype not in (tf.int32, tf.int64):
      raise ValueError(
          f'Expected indices of tf.int32 or tf.int64 type, got {indices.dtype}'
      )
    return tf.gather(self._values, indices)


@tf.keras.utils.register_keras_serializable(package='GNN')
class _UniformEdgesSelector(tf.keras.layers.Layer):
  """Selects edges uniformly at random from outgoing edges tensor.

  Call arguments:
    inputs: a tuple `(source_node_ids, outgoing_edges)` where:
      `source_node_ids`, a `tf.RaggedTensor` shaped `[batch_size,
      (num_source_nodes)]`, with the source ids; `outgoing_edges`: a `Features`
      object, with one feature named after `edge_target_feature_name`, holding
      the target node ids, and shaped `[batch_size, (num_source_nodes),
      (num_outgoing_edges)]`. It can hold also other features (e.g: "weight"),
      shaped  `[batch_size, (num_source_nodes), (num_outgoing_edges),
      *feature_dim]`. The `batch_size` and `num_source_nodes` must match in
      `source_node_ids` and `outgoing_edges`.

  Call returns:
    Sampled edges as `Features` with two special features: "#source" and
    "#target" shaped `[batch_size, (num_source_nodes), (num_sampled_edges)]`,
    plus extra edge features (e.g. "weight") shaped `[batch_size,
    (num_source_nodes), (num_sampled_edges), *feature_dims]`. The ragged
    dimension `(num_sampled_edges)` index sampled edges for each source node and
    has row lengths less or equal to `sample_size`. Notice that the original
    edge name `edge_target_feature_name` in  `outgoing_edges` is renamed to
    "#target" in the result.
  """

  def __init__(
      self,
      *,
      sample_size: int,
      edge_target_feature_name: str = tfgnn.TARGET_NAME,
      seed: Optional[int] = None,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self._sample_size = sample_size
    self._edge_target_feature_name = edge_target_feature_name
    self._seed = seed

  def get_config(self):
    return dict(
        sample_size=self._sample_size,
        edge_target_feature_name=self._edge_target_feature_name,
        seed=self._seed,
        **super().get_config(),
    )

  def call(self, inputs):
    source_node_ids, outgoing_edges = inputs
    return _sample_edges_without_replacement(
        source_node_ids,
        outgoing_edges,
        self._edge_target_feature_name,
        self._sample_size,
        self._seed,
    )


def _remove_parallel_edges(
    graph_tensor: tfgnn.GraphTensor,
) -> tfgnn.GraphTensor:
  """Removes duplicate edges connecting the same source and target nodes."""
  # TODO(aferludin): allow deduplication using edge id feature.

  assert graph_tensor.rank == 1, graph_tensor.rank

  num_components = graph_tensor.num_components
  with tf.control_dependencies(
      [
          tf.debugging.assert_equal(
              graph_tensor.num_components,
              tf.constant(1, num_components.dtype),
              message='Expected graphs with a single graph component per graph',
          )
      ]
  ):
    graph_tensor = tf.identity(graph_tensor)

  unique_edge_indices = _get_unique_parallel_edges_indices(graph_tensor)

  edge_sets = {}
  for edge_set_name, edge_set in graph_tensor.edge_sets.items():
    adj = edge_set.adjacency
    assert isinstance(adj, tfgnn.Adjacency), edge_set_name
    indices = unique_edge_indices[edge_set_name]
    source, target, features = tf.nest.map_structure(
        functools.partial(tf.gather, indices=indices, batch_dims=1),
        (adj.source, adj.target, edge_set.get_features_dict()),
    )

    new_sizes = tf.expand_dims(
        tf.cast(indices.row_lengths(), edge_set.sizes.dtype), axis=-1
    )
    edge_sets[edge_set_name] = tfgnn.EdgeSet.from_fields(
        features=features,
        sizes=new_sizes,
        adjacency=tfgnn.Adjacency.from_indices(
            (adj.source_name, source),
            (adj.target_name, target),
        ),
    )

  return tfgnn.GraphTensor.from_pieces(
      graph_tensor.context, graph_tensor.node_sets, edge_sets
  )


def _get_unique_parallel_edges_indices(
    graph_tensor: tfgnn.GraphTensor,
) -> Mapping[tfgnn.EdgeSetName, tf.RaggedTensor]:
  """Returns indices of unique parallel edges within each graph.

  Unique indices are returned as ragged rank 1 tensors for each edge set. Each
  ragged row `r` of those tensors contains 0-base indices of unique edges for
  graph `r` of `graph_tensor`.

  Args:
    graph_tensor: graph tensor of rank 1 with single graph component per graph.

  Returns:
    Indices of unique edges within each graph for each edge set.
  """

  assert graph_tensor.rank == 1, graph_tensor.rank
  result = {}

  # Flatten graph, so edges have continuous indices.
  flat = graph_tensor.replace_features({}, {}, {}).merge_batch_to_components()
  for edge_set_name, flat_edge_set in flat.edge_sets.items():
    sizes, adj = flat_edge_set.sizes, flat_edge_set.adjacency
    assert isinstance(sizes, tf.Tensor), edge_set_name
    sizes = tf.cast(sizes, tf.int64)
    assert isinstance(adj, tfgnn.Adjacency), edge_set_name
    # Pack edges (source, target) into int64 ids, such that `ids[i] = ids[j]` if
    # and only if `(source[i], target[i]) = (source[j], target[j])`.
    edge_ids = _edges_to_ids(adj.source, adj.target)
    unique_edge_ids, unique_edge_idx = tf.unique(edge_ids, tf.int64)
    # unique_edge_ids contains flattened unique edges for graph0, graph1, ...
    # unique_edge_idx[i] is an index in unique_edge_ids for original edge `i`
    num_unique_edges = tf.size(unique_edge_ids, out_type=tf.int64)

    # allows to project feature of the original edge to the unique edge.
    map_to_unique_edge = functools.partial(
        tf.math.unsorted_segment_min,
        segment_ids=unique_edge_idx,
        num_segments=num_unique_edges,
    )
    # the total number of graphs of the original input `graph_tensor`.
    num_graphs = tf.size(sizes, out_type=tf.int64)
    # 0-base indices of edges within original graphs
    original_edges_idx = tf.ragged.range(sizes).values
    # assignes to each edge its graph index within the original graph tensor.
    original_graph_idx = tf.repeat(tf.range(num_graphs), sizes)
    # TODO(b/285269757): replace with `graph_tensor.row_splits_dtype`
    row_splits_dtype = graph_tensor.edge_sets[
        edge_set_name
    ].adjacency.source.dtype
    result[edge_set_name] = tf.RaggedTensor.from_value_rowids(
        map_to_unique_edge(original_edges_idx),
        map_to_unique_edge(original_graph_idx),
        nrows=num_graphs,
        validate=False,
    ).with_row_splits_dtype(row_splits_dtype)

  return result


def _edges_to_ids(source: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
  """Returns unique `tf.int64` ids for (source, target) pairs.

  result[i] = result[j] <-> (source[i], target[i]) = (source[j], target[j]).

  Args:
    source: indices of edge source nodes.
    target: indices of edge target nodes.

  Returns:
    `tf.int64` ids tensor.
  """

  source, target = tf.cast(source, tf.int64), tf.cast(target, tf.int64)
  base = tf.math.reduce_max(target)
  base = tf.maximum(base, tf.constant(0, tf.int64)) + tf.constant(1, tf.int64)

  check_ops = tf.debugging.assert_less(
      source,
      tf.dtypes.int64.max // (base + 1),
      message='The number nodes > 2^31 is not supported. Contact TF-GNN team',
  )
  with tf.control_dependencies([check_ops]):
    return source * base + target


def _pack_args(args, kwargs):
  return ((*args,), kwargs)


def _sample_edges_without_replacement(
    source_node_ids: tf.RaggedTensor,
    outgoing_edges: Features,
    edge_target_feature_name: str,
    sample_size: int,
    seed: Optional[int] = None,
) -> Features:
  """Samples up to `sample_size` edges for each source without replacement."""
  target_node_ids = outgoing_edges[edge_target_feature_name]
  assert target_node_ids.ragged_rank == 2

  # Assign indices of source nodes (keys) to extracted target nodes.
  row_splits = target_node_ids.values.row_splits
  num_samples = tf.constant(sample_size, dtype=row_splits.dtype)
  sampling_indices = ext_ops.ragged_choice(
      num_samples, row_splits, global_indices=True, seed=seed
  )

  def sample(feature: tf.RaggedTensor) -> tf.RaggedTensor:
    edge_feature = feature.values
    sampled_edge_feature = tf.gather(edge_feature.values, sampling_indices)
    return feature.with_values(sampled_edge_feature)

  sampled_edges = tf.nest.map_structure(sample, outgoing_edges)
  target_node_ids = sampled_edges.pop(edge_target_feature_name)
  # Repeat source node ids for each sample target node.
  source_node_ids = target_node_ids.with_flat_values(
      tf.repeat(
          source_node_ids.flat_values, target_node_ids.values.row_lengths()
      )
  )
  sampled_edges.update({
      tfgnn.SOURCE_NAME: source_node_ids,
      tfgnn.TARGET_NAME: target_node_ids,
  })
  return tf.nest.map_structure(_flatten_ragged_dim2, sampled_edges)


def _flatten_ragged_dim2(values: tf.RaggedTensor) -> tf.RaggedTensor:
  """Flattens second ragged dimension, making a RaggedTensor of ragged-rank 2 to 1.

  Args:
    values: ragged tensor with ragged rank at least 2.

  Returns:
    Ragged tensor with ragged dimension 2 being merged to ragged dimension 1.

  Example:
    _flatten_ragged_dim2(rt([[[1, 2], [3]], [[4], []]]) # rt([[1, 2, 3], [4]])
  """
  if values.ragged_rank <= 1:
    raise ValueError('Values must have ragged rank > 1.')

  return tf.RaggedTensor.from_row_lengths(
      values.values.values,
      tf.math.unsorted_segment_sum(
          values.values.row_lengths(), values.value_rowids(), values.nrows()
      ),
  )


def _check_same_input_struct(name: str, expected, actual) -> None:
  """Checks that layer is called with the same argument types."""
  try:
    tf.nest.assert_same_structure(
        expected, actual, check_types=True, expand_composites=True
    )
  except ValueError as err:
    raise ValueError(
        f'Layer {name} is called with different arguments'
    ) from err

  # Keras Model does tensors auto-casting (e.g. int64 to int32) which leads to
  # hard to find errors. Prevent this behavior and require exact match.
  def check_types(expected, actual):
    if expected.dtype == actual.dtype:
      return expected, actual
    raise ValueError(
        f'Layer {name} is called with different argument types:'
        f' expected {expected.dtype}, actual {actual.dtype}'
    )
  tf.nest.map_structure(check_types, expected, actual)


def _parse_edge_set_definition(
    edge_set_def: str,
) -> Tuple[tfgnn.NodeSetName, tfgnn.EdgeSetName, tfgnn.NodeSetName]:
  """."""
  pieces = edge_set_def.split(',')
  if len(pieces) != 3:
    raise ValueError(
        'Expected edge set definition as'
        f' "<source_node_set>,<edge_set>,<target_node_set>", got {edge_set_def}'
    )
  source_node_set, edge_set, target_node_set = pieces
  return source_node_set, edge_set, target_node_set


def _row_lengths_to_row_splits(row_lengths: tf.Tensor) -> tf.Tensor:
  return tf.concat(
      [
          tf.constant([0], row_lengths.dtype),
          tf.math.cumsum(row_lengths, exclusive=False),
      ],
      axis=-1,
  )
