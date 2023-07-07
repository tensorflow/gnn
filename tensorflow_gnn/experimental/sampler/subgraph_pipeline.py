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
"""Samples multi-hop subgraphs given single-hop `OutgoingEdgesSampler`s.

To use functions in this file, the caller should have access to:
+ Instances of `GraphSchema` and `SamplingSpec`, respectively, that defines the
  input graph and configures the sampling graph to be sampled. You may construct
  the `SamplingSpec` manually or via `sampling_spec_builder`.
+ `OutgoingEdgesSampler` factory (`EdgeSamplerFactory`): function that accepts
  an edge set name to return an instance of `interfaces.OutgoingEdgesSampler`.
+ (optional) node features lookup factory (`NodeFeaturesLookupFactory`):
  function that accepts node set name to return an instance of
  `interfaces.KeyToFeaturesAccessor`.


From highest-level to lowest level:
+ `SamplingPipeline`: Defines callable that can be invoked on "seed" node IDs to
  return `tfgnn.GraphTensor` containing sampled subgraph (edges and node
  features), centered around "seed" node IDs.
+ `create_sampling_model_from_spec`: Packages `SamplingPipeline` as a
  `tf.keras.Model`. After model is created, it can be invoked with seed node IDs
  to return `GraphTensor` of subgraphs rooted around seed node IDs.
+ `sample_edge_sets`: Invokes edge sampling to return only edges and node IDs,
  but no node features. Must be invoked on seed node IDs.
"""

import collections

from typing import Callable, Mapping, Optional, Dict
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.experimental.sampler import core
from tensorflow_gnn.experimental.sampler import interfaces
from tensorflow_gnn.sampler import sampling_spec_pb2


# Given a SamplingOp, returns a sampler instance.
# NOTE: As sampler samples 1 edge set at a time, `op_name` and `input_ops` must
#   be ignored. Regardless, it is the responsibility of `SamplingPipeline`,
#   `sample_edge_sets` and `create_sampling_model_from_spec` to use `op_name`
#   and `input_ops` to forward the correct tensors to the returned
#   `interfaces.OutgoingEdgesSampler`.
EdgeSamplerFactory = Callable[[sampling_spec_pb2.SamplingOp],
                              interfaces.OutgoingEdgesSampler]

# Given node-set-name, must return feature lookup or None.
NodeFeaturesLookupFactory = Callable[[tfgnn.NodeSetName],
                                     Optional[interfaces.KeyToFeaturesAccessor]]


def create_sampling_model_from_spec(
    graph_schema: tfgnn.GraphSchema,
    sampling_spec: sampling_spec_pb2.SamplingSpec,
    edge_sampler_factory: EdgeSamplerFactory,
    node_features_accessor_factory: Optional[NodeFeaturesLookupFactory] = None,
    seed_node_dtype=tf.string) -> tf.keras.Model:
  """Creates Keras model that accepts seed node IDs to output GraphTensor.

  Args:
    graph_schema: Attribute `edge_sets` identifies end-point node set names.
    sampling_spec: The number of nodes sampled from edge set. The spec defines
      the structure of the sampled subgraphs, that look like rooted trees,
      possibly densified adding all pairwise edges between sampled nodes.
    edge_sampler_factory: function accepting args (sample size, edge set name)
      to return an instance of `interfaces.OutgoingEdgesSampler`.
    node_features_accessor_factory: If given, must be a function that maps
      node set name into features accessor for the node set.
    seed_node_dtype: type of seed nodes. It must be understood by
      `sampling_kwargs["edge_sampler_factory"]`.

  Returns:
    tf.keras.Model which accepts input of `tf.RaggedTensor` containing seed node
    IDs. The output will be `GraphTensor` containing subgraph samples around
    seed nodes, as configured in `pipeline_kwargs["sampling_spec"]`.
    Input must have shape [batch_dims, (source_node_ids)] be part of node set
    `sampling_spec.seed_op.node_set_name`. Each seed node will be the root of a
    subgraph.
  """
  pipeline = SamplingPipeline(
      graph_schema=graph_schema, sampling_spec=sampling_spec,
      edge_sampler_factory=edge_sampler_factory,
      node_features_accessor_factory=node_features_accessor_factory,
  )
  seed_nodes = tf.keras.Input(
      type_spec=tf.RaggedTensorSpec(
          shape=[None, None], ragged_rank=1, dtype=seed_node_dtype),
      name='Input')
  subgraph = pipeline(seed_nodes)
  return tf.keras.Model(inputs=seed_nodes, outputs=subgraph)


class SamplingPipeline:
  """Callable that accepts seed node IDs to output GraphTensor."""

  def __init__(
      self,
      graph_schema: tfgnn.GraphSchema,
      sampling_spec: sampling_spec_pb2.SamplingSpec,
      edge_sampler_factory: EdgeSamplerFactory,
      node_features_accessor_factory: Optional[
          NodeFeaturesLookupFactory] = None,
  ):
    assert_sorted_sampling_spec(sampling_spec)
    self._graph_schema = graph_schema
    self._sampling_spec = sampling_spec
    self._edge_sampler_factory = edge_sampler_factory
    self._node_features_accessor_factory = node_features_accessor_factory
    self._node_accessor: Dict[
        tfgnn.NodeSetName, Optional[interfaces.KeyToFeaturesAccessor]
    ] = {}

  def get_node_features_accessor(
      self, node_set_name: tfgnn.NodeSetName
  ) -> Optional[interfaces.KeyToFeaturesAccessor]:
    if not self._node_features_accessor_factory:
      return None
    if node_set_name not in self._node_accessor:
      self._node_accessor[node_set_name] = self._node_features_accessor_factory(
          node_set_name
      )
    return self._node_accessor[node_set_name]

  @property
  def seed_node_set_name(self) -> tfgnn.NodeSetName:
    return self._sampling_spec.seed_op.node_set_name

  def __call__(self, seed_nodes: tf.RaggedTensor) -> tfgnn.GraphTensor:
    """Returns subgraph (as `GraphTensor`) sampled around `seed_nodes`.

    Args:
      seed_nodes: These node IDs (regardless of dtype) are expected to be IDs
        for `sampling_spec.seed_op.node_set_name` and they will be used to
        invoke `edge_sampler_factor("first_hop")(seed_nodes)`.
    """
    edge_sets = dict(sample_edge_sets(
        seed_nodes, self._graph_schema, self._sampling_spec,
        self._edge_sampler_factory))

    if hasattr(seed_nodes, 'to_tensor'):  # RaggedTensor!
      ragged_seed = seed_nodes
    else:  # Dense tf.Tensor.
      seed_nodes_shape = tf.shape(seed_nodes)
      ragged_seed = tf.reshape(seed_nodes, [seed_nodes_shape[0], -1])
      ragged_seed = tf.RaggedTensor.from_tensor(ragged_seed)
    # Determine fake edge set name (ensuring unused name)
    seed_edge_set_names = [edge_set_key.split(',')[1]
                           for edge_set_key in edge_sets
                           if edge_set_key.startswith('_seed')]
    seed_edge_set_names.append('_seed')
    longest_name = max(
        map(lambda name: (len(name), name), seed_edge_set_names))[1]
    fake_edge_set_name = longest_name + '0'  # longer than longest ==> unique.
    fake_edge_key = ','.join((self.seed_node_set_name, fake_edge_set_name,
                              self.seed_node_set_name))
    edge_sets[fake_edge_key] = {
        tfgnn.SOURCE_NAME: ragged_seed,
        tfgnn.TARGET_NAME: ragged_seed,
    }
    graph_tensor = core.build_graph_tensor(edge_sets=edge_sets)

    # Remove fake edge set.
    graph_tensor = tf.keras.layers.Lambda(
        lambda g: tfgnn.GraphTensor.from_pieces(  # pylint: disable=g-long-lambda
            context=g.context, node_sets=g.node_sets,
            edge_sets={n: e for n, e in g.edge_sets.items()
                       if n != fake_edge_set_name}))(graph_tensor)

    features = {}
    for node_set_name, node_set in graph_tensor.node_sets.items():
      accessor = self.get_node_features_accessor(node_set_name)
      if accessor:
        node_set_features = accessor(node_set['#id'])
        if node_set_features is not None:
          # TODO(b/289402863): Remove `dict` in future.
          features[node_set_name] = dict(node_set_features)

    if features:
      graph_tensor = graph_tensor.replace_features(node_sets=features)

    return graph_tensor

  @property
  def node_features_accessor_factory(
      self) -> Optional[NodeFeaturesLookupFactory]:
    return self._node_features_accessor_factory


def sample_edge_sets(
    seed_node_ids: tf.RaggedTensor,
    graph_schema: tfgnn.GraphSchema,
    sampling_spec: sampling_spec_pb2.SamplingSpec,
    edge_sampler_factory: EdgeSamplerFactory,
    ) -> Mapping[str, interfaces.Features]:
  """Returns edge lists and unique node IDs visited in all node sets.

  Args:
    seed_node_ids: Must have shape [batch_dims, (source_node_ids)]. Each seed
      node will be the root of a subgraph. They must be part of node set
      `sampling_spec.seed_op.node_set_name`.
    graph_schema: Attribute `edge_sets` identifies end-point node set names.
    sampling_spec: The number of nodes sampled from edge set. The spec defines
      the structure of the sampled subgraphs, that look like rooted trees,
      possibly densified adding all pairwise edges between sampled nodes.
    edge_sampler_factory: function accepting args (sample size, edge set name)
      to return an instance of `interfaces.OutgoingEdgesSampler`.

  Returns: Tuple of (edge lists, unique nodes), where:
    + edge lists is dict: edge set name -> (source node IDs, target node IDs).
    + unique nodes is dict: node set name -> node IDs.
  All IDs are vector `tf.Tensor`s with dtype that is dictated by `seed_node_ids`
  and `edge_sampler_factory`.
  """
  assert_sorted_sampling_spec(sampling_spec)

  edge_set_sources = collections.defaultdict(list)
  edge_set_targets = collections.defaultdict(list)

  tensors_by_op_name = {
      sampling_spec.seed_op.op_name: seed_node_ids
  }

  for sampling_op in sampling_spec.sampling_ops:
    input_tensors = tf.concat(
        [tensors_by_op_name[op_name] for op_name in sampling_op.input_op_names],
        axis=-1)
    edge_sampler = edge_sampler_factory(sampling_op)
    sampled_edges = edge_sampler(input_tensors)

    edge_set_sources[sampling_op.edge_set_name].append(
        sampled_edges[tfgnn.SOURCE_NAME])
    edge_set_targets[sampling_op.edge_set_name].append(
        sampled_edges[tfgnn.TARGET_NAME])
    tensors_by_op_name[sampling_op.op_name] = sampled_edges[tfgnn.TARGET_NAME]

  edge_sets = {}
  for edge_set_name, source_list in edge_set_sources.items():
    target_list = edge_set_targets[edge_set_name]
    edge_set_key = ','.join((graph_schema.edge_sets[edge_set_name].source,
                             edge_set_name,
                             graph_schema.edge_sets[edge_set_name].target))
    edge_sets[edge_set_key] = {
        tfgnn.SOURCE_NAME: tf.concat(source_list, axis=-1),
        tfgnn.TARGET_NAME: tf.concat(target_list, axis=-1),
    }

  return edge_sets


def _unique_y(x: tf.Tensor) -> tf.Tensor:
  return tf.unique(x).y


def assert_sorted_sampling_spec(sampling_spec: sampling_spec_pb2.SamplingSpec):
  """Raises ValueError if `sampling_spec` is not topologically-sorted."""
  seen_ops = {sampling_spec.seed_op.op_name}
  for sampling_op in sampling_spec.sampling_ops:
    for input_op in sampling_op.input_op_names:
      if input_op not in seen_ops:
        raise ValueError(f'Input op {input_op} is used before defined. '
                         'sampling_spec is not topologically-sorted')
    seen_ops.add(sampling_op.op_name)
