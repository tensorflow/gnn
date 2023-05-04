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
"""Defines core interfaces for graph sampling and features extraction.

The core primitive is `KeyToBytesAccessor` which for each input key returns
single `tf.string`. This interface allows generic key to (serialized) data
lookups. It abstracts specifics of a data storage from the sampling logic.
For example, it could be used to extract node or edges features, look up
adjacency lists of source nodes, extract partially sampled subgraphs, etc.
All sampling logic could be implemented using this abstraction.

The library defines also other more specialized abstractions, such as
`KeyToFeaturesAccessor` and `OutgoingEdgesSampler`. Although those primitives
could be implemented based on the `KeyToBytesAccessor`, they could be
specialized for particular backends. For example, random sampling of outgoing
edges, in general, could be implemented with `KeyToFeaturesAccessor`: serialized
adjacency lists are collected for their source nodes, then parsed and containing
edges are sampled. For a Spanner backend it is more efficient to delegate
sampling logic to the server side. So the default implementation of
`OutgoingEdgesSampler` could could be replaced with its more efficient Spanner
specialization without affecting customer logic.



NOTE: Currently all interfaces are based on RaggedTensors of rank 2.
The outer dimension 0 is the batch dimension (separate graphs), the ragged
dimension1 1 represents graph items (like nodes and edges). This is restrictive
and in particular it does not allow to run samplng on TPU. We may add support
for dense inputs in the future.
"""

import abc
from typing import Any, Mapping

import tensorflow as tf

Features = Mapping[str, tf.RaggedTensor]
FeaturesSpec = Mapping[str, tf.RaggedTensorSpec]


class SamplingPrimitive(abc.ABC):
  """Base class for all sampling primitives."""
  pass


class AccessorBase(SamplingPrimitive):
  """Base class for any value accessor."""

  def __call__(self, *args, **kwargs) -> Any:
    raise NotImplementedError

  @abc.abstractmethod
  def call(self, *args, **kwargs) -> Any:
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def resource_name(self) -> str:
    """The resource name this accessor is associated with."""
    raise NotImplementedError


class KeyToBytesAccessor(AccessorBase):
  """Generic key to serialized value accessor.

  Keys can be integers or strings. Returned values are always strings, e.g.,
  serialized tf.Example or TensorProto message.

  For each key, exactly one value is returned by this accessor:

    * If a value is stored for the key, the value is returned.
    * If no value is stored for the key, the default value set at initialization
      time of the concrete subclass is returned instead.
    * If no value is stored for the key and no default has been set, the
      execution of the accessor fails, so its return is never reached.

  That way, code calling this interface does not have to concern itself with
  handling lookup failures (which gets in the way of fast data-parallel
  operations): it never sees the case of a missing value.

  Failures on a systems level (like I/O errors) have to be handled outside
  the TensorFlow model anyways.

  The code that configures a concrete subclass has to choose one of the
  following strategies:

    * Supply a harmless fallback value for missing keys. This is often
      appropriate when dealing with input data that is expected to be
      incomplete.
    * Supply a default value that stands out as such but does not get
      in the way of bulk processing. (Think of a serialized tf.Example
      proto that contains an `is_default` feature not seen in real data.)
    * Not supply a default value and fail on a missing key. This is often
      appropriate when dealing with reference datasets known to be complete,
      for which a bad lookup indicates a logic error in the code.
  """

  @abc.abstractmethod
  def call(self, keys: tf.RaggedTensor) -> tf.RaggedTensor:
    """Look up values for the given keys.

    Args:
      keys: lookup keys. Ragged tensor with shape `[batch_size, (num_keys)]`,
        and tf.int32, tf.int64 or tf.string type.

    Returns:
      ragged string tensor with the same shape as `keys` containing single
      lookup result for each input key.
    """
    raise NotImplementedError


class KeyToFeaturesAccessor(AccessorBase):
  """Generic key to features dict accessor.
  """

  @abc.abstractmethod
  def call(self, keys: tf.RaggedTensor) -> Features:
    """Look up values for the given keys.

    Args:
      keys: lookup keys. Ragged tensor with shape `[batch_size, (num_keys)]`,
        and tf.int32, tf.int64 or tf.string type.

    Returns:
      dictionary of features with shape [batch_size, (num_keys), *inner_dims],
      and fixed set of dictionary keys.
    """
    raise NotImplementedError


class OutgoingEdgesSampler(SamplingPrimitive):
  """Samples outgoing edges for given source nodes.

  Used to create rooted subgraphs, e.g. for node classification.
  """

  def __call__(self, source_node_ids: tf.RaggedTensor) -> Features:
    return self.call(source_node_ids)

  @abc.abstractmethod
  def call(self, source_node_ids: tf.RaggedTensor) -> Features:
    """Samples outgoing edges for the given source node ids.

    Args:
      source_node_ids: node ids for sampling outgoing edges. Ragged tensor with
        shape `[batch_size, (num_source_nodes)]` and tf.int32, tf.int64 or
        tf.string type.


    Returns:
      `Features` containing the subset of all edges whose source nodes are
      in `source_node_ids`. All returned features must have shape
      `[batch_size, (num_edges), ...]`. Result must include two special features
      "#source" and "#target" of rank 2 containing, correspondigly, source node
      ids and targert node ids of the sampled edges.
    """
    raise NotImplementedError


class ConnectingEdgesSampler(SamplingPrimitive):
  """Samples incident edges between given subsets of source and target nodes.

  Extracts all edges that connect from the given subsets of source and
  target nodes. Notice it's ok to have `source_node_ids == target_node_ids`.
  Used to connect nodes with existing edges, e.g. for link prediction.
  """

  @abc.abstractmethod
  def call(self,
           source_node_ids: tf.RaggedTensor,
           target_node_ids: tf.RaggedTensor) -> Features:
    """Samples incident edges *from* source *on* target node ids.

    Each sampled edges has its source in the `source_node_ids` and its target in
    the `target_node_ids`.

    Args:
      source_node_ids: node ids for sampling outgoing edges. Ragged tensor with
        shape `[batch_size, (num_source_nodes)]` and tf.int32, tf.int64 or
        tf.string type.
      target_node_ids: node ids for sampling incoming edges. Ragged tensor with
        shape `[batch_size, (num_target_nodes)]` and tf.int32, tf.int64 or
        tf.string type.

    Returns:
      `Features` containing the subset of all edges whose source nodes are
      in `source_node_ids` and whose target nodes are in `target_node_ids`.
      All returned features must have shape `[batch_size, (num_edges), ...]`.
      Result must include two special features "#source" and "#target" of rank 2
      containing, correspondigly, source node ids and targert node ids of the
      sampled edges.
    """
    raise NotImplementedError
