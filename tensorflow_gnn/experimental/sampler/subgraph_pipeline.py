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

from __future__ import annotations
import collections
import functools

from typing import Callable, Mapping, Optional, Dict, Tuple
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.experimental.sampler import core
from tensorflow_gnn.experimental.sampler import ext_ops
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


def create_link_sampling_model_from_spec(
    graph_schema: tfgnn.GraphSchema,
    sampling_spec: sampling_spec_pb2.SamplingSpec,
    edge_sampler_factory: EdgeSamplerFactory,
    node_features_accessor_factory: Optional[NodeFeaturesLookupFactory] = None,
    seed_node_dtype=tf.string,
) -> tf.keras.Model:
  """Creates Keras model that accepts link sampling features and seeds to output GraphTensor."""
  _validate_link_sampling_graph_schema(graph_schema)
  inputs = {
      tfgnn.SOURCE_NAME: tf.keras.Input(
          type_spec=tf.RaggedTensorSpec(
              shape=[None, None], ragged_rank=1, dtype=seed_node_dtype
          ),
          name='SeedSource',
      ),
      tfgnn.TARGET_NAME: tf.keras.Input(
          type_spec=tf.RaggedTensorSpec(
              shape=[None, None], ragged_rank=1, dtype=seed_node_dtype
          ),
          name='SeedTarget',
      ),
  }
  inputs.update(_readout_inputs_from_schema(graph_schema))
  pipeline = LinkSamplingPipeline(
      graph_schema=graph_schema, sampling_spec=sampling_spec,
      edge_sampler_factory=edge_sampler_factory,
      node_features_accessor_factory=node_features_accessor_factory,
  )
  subgraph = pipeline(inputs)
  return tf.keras.Model(inputs=inputs, outputs=subgraph)


def _validate_link_sampling_graph_schema(graph_schema: tfgnn.GraphSchema):
  """Validates that the given graph_schema has only one auxiliary node set and its related edge sets.

  Args:
    graph_schema: tfgnn.GraphSchema object to validate for link prediction.
  """
  aux_node_sets = [
      node_set
      for node_set in graph_schema.node_sets.keys()
      if tfgnn.get_aux_type_prefix(node_set) is not None
  ]
  if len(aux_node_sets) != 1 or '_readout' not in aux_node_sets:
    raise ValueError(
        'There should be exactly one auxiliary node set for link sampling and'
        f' it should be called `_readout`. Instead got {aux_node_sets}'
    )

  aux_edge_sets = [
      edge_set
      for edge_set in graph_schema.edge_sets.keys()
      if tfgnn.get_aux_type_prefix(edge_set) is not None
  ]
  if (
      len(aux_edge_sets) != 2
      or '_readout/source' not in aux_edge_sets
      or '_readout/target' not in aux_edge_sets
  ):
    raise ValueError(
        'There should be exactly two auxiliary edge sets for link sampling and'
        ' they should be called `_readout/source` and `_readout/target`.'
        f' Instead got {aux_edge_sets}'
    )
  if (
      graph_schema.edge_sets[aux_edge_sets[0]].source
      != graph_schema.edge_sets[aux_edge_sets[1]].source
  ):
    raise ValueError(
        '`_readout/source` and `_readout/target` edge sets should point from a'
        ' common source node set to `_readout` as a target. Instead'
        ' `_readout/source` points from'
        f' {graph_schema.edge_sets[aux_edge_sets[0]].source} while'
        ' `_readout/target` points from'
        f' {graph_schema.edge_sets[aux_edge_sets[1]].source}'
    )


def _readout_inputs_from_schema(graph_schema: tfgnn.GraphSchema):
  """Populates inputs for sampling model from a graph schema.

  Args:
    graph_schema: tfgnn.GraphSchema for generating the input specs.

  Returns:
    dict from feature names to input specs
  """
  aux_node_set_name = [
      node_set
      for node_set in graph_schema.node_sets.keys()
      if tfgnn.get_aux_type_prefix(node_set) is not None
  ][0]
  node_set_spec = graph_schema.node_sets[aux_node_set_name]
  feature_names = [
      key
      for key in node_set_spec.features.keys()
  ]
  feature_inputs = {}
  for feature_name in feature_names:
    feature_inputs[feature_name] = tf.keras.Input(
        type_spec=tf.RaggedTensorSpec(
            shape=[None, None],
            ragged_rank=1,
            dtype=node_set_spec.features[feature_name].dtype,
        ),
        name=f'Seed/{feature_name}',
    )
  return feature_inputs


def create_sampling_model_from_spec(
    graph_schema: tfgnn.GraphSchema,
    sampling_spec: sampling_spec_pb2.SamplingSpec,
    edge_sampler_factory: EdgeSamplerFactory,
    node_features_accessor_factory: Optional[NodeFeaturesLookupFactory] = None,
    seed_node_dtype=tf.string,
) -> tf.keras.Model:
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
      seed_node_set_name: Optional[tfgnn.NodeSetName] = None,
  ):
    assert_sorted_sampling_spec(sampling_spec)
    self._graph_schema = graph_schema
    self._sampling_spec = sampling_spec
    self._edge_sampler_factory = edge_sampler_factory
    self._node_features_accessor_factory = node_features_accessor_factory
    self._node_accessor: Dict[
        tfgnn.NodeSetName, Optional[interfaces.KeyToFeaturesAccessor]
    ] = {}

    if seed_node_set_name is None:
      seed_node_set_name = self._sampling_spec.seed_op.node_set_name

    if not seed_node_set_name:
      raise ValueError('Seed node set name is not specified.')

    self._seed_node_set_name = seed_node_set_name

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
    return self._seed_node_set_name

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
    graph_tensor = tfgnn.GraphTensor.from_pieces(
        context=graph_tensor.context,
        node_sets=graph_tensor.node_sets,
        edge_sets={
            n: e
            for n, e in graph_tensor.edge_sets.items()
            if n != fake_edge_set_name
        },
    )

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


class LinkSamplingPipeline:
  """Callable that makes GraphTensor out of seed node IDs and labels."""

  def __init__(
      self,
      graph_schema: tfgnn.GraphSchema,
      sampling_spec: sampling_spec_pb2.SamplingSpec,
      edge_sampler_factory: EdgeSamplerFactory,
      node_features_accessor_factory: Optional[
          NodeFeaturesLookupFactory
      ] = None,
  ):
    self._readout_node_set = '_readout'
    assert self._readout_node_set in graph_schema.node_sets, graph_schema
    assert (
        graph_schema.edge_sets[f'{self._readout_node_set}/source'].target
        == graph_schema.edge_sets[f'{self._readout_node_set}/target'].target
        == self._readout_node_set
    ), graph_schema
    assert (
        graph_schema.edge_sets[f'{self._readout_node_set}/source'].source
        == graph_schema.edge_sets[f'{self._readout_node_set}/target'].source
    ), graph_schema

    self._seed_node_set = graph_schema.edge_sets[
        f'{self._readout_node_set}/source'
    ].source

    self._sampling_pipeline = SamplingPipeline(
        graph_schema=graph_schema,
        sampling_spec=sampling_spec,
        edge_sampler_factory=edge_sampler_factory,
        node_features_accessor_factory=node_features_accessor_factory,
        seed_node_set_name=self._seed_node_set,
    )

  def __call__(self, inputs: dict[str, tf.RaggedTensor]) -> tfgnn.GraphTensor:
    seed_nodes = tf.concat(
        [inputs[tfgnn.SOURCE_NAME], inputs[tfgnn.TARGET_NAME]], axis=-1
    )
    subgraph = self._sampling_pipeline(seed_nodes)
    return AddLinkReadoutStruct(
        readout_node_set=self._readout_node_set,
        seed_node_set=self._seed_node_set,
    )((subgraph, inputs))


class AddLinkReadoutStruct(tf.keras.layers.Layer):
  """Helper class for adding readout structure for the link prediction task.

  NOTE: that the link source and target nodes must belong to the same node set.
  
  The input graph tensor is assumed to be sampled by the `SamplingPipeline`
  starting from the link source and target nodes.

  To facilitate link prediction, the class adds `_readout` node set with one
  readout node for each link. It accepts link features to copy over to their
  corresponding readout nodes. The readout nodes are connected to link source
  and target nodes using a pair of auxiliary readout edge sets.

  NOTE: following the TF-GNN implementation, we direct readout edges from
  seed nodes (source) to readout nodes (target).
  """

  def __init__(
      self, readout_node_set: tfgnn.NodeSet, seed_node_set: tfgnn.NodeSetName
  ):
    super().__init__()
    assert readout_node_set
    assert seed_node_set
    self._readout_node_set = readout_node_set
    self._seed_node_set = seed_node_set

  def get_config(self):
    return {
        'readout_node_set': self._readout_node_set,
        'seed_node_set': self._seed_node_set,
        **super().get_config(),
    }

  def call(
      self,
      inputs: Tuple[tfgnn.GraphTensor, Mapping[str, tf.RaggedTensor]],
  ) -> tfgnn.GraphTensor:
    graph_tensor, readout_features = inputs

    seed_node_set = graph_tensor.node_sets[self._seed_node_set]

    ids = seed_node_set[core.NODE_ID_NAME]
    link_source_idx = tf.cast(
        ext_ops.ragged_lookup(
            readout_features[tfgnn.SOURCE_NAME], ids, global_indices=False
        ),
        graph_tensor.indices_dtype,
    )
    link_target_idx = tf.cast(
        ext_ops.ragged_lookup(
            readout_features[tfgnn.TARGET_NAME], ids, global_indices=False
        ),
        graph_tensor.indices_dtype,
    )

    with tf.control_dependencies(
        [
            tf.debugging.assert_equal(
                link_source_idx.row_lengths(),
                link_target_idx.row_lengths(),
                message=(
                    'The number of link source and target nodes must be the'
                    ' same'
                ),
            )
        ]
    ):
      num_seeds = tf.cast(
          link_source_idx.row_lengths(), graph_tensor.indices_dtype
      )
      readout_index = tf.ragged.range(num_seeds, dtype=num_seeds.dtype)
      sizes = tf.expand_dims(num_seeds, axis=-1)

    readout_node_sets = {
        self._readout_node_set: tfgnn.NodeSet.from_fields(
            features=readout_features, sizes=sizes
        )
    }
    readout_edge_sets = {
        f'{self._readout_node_set}/source': tfgnn.EdgeSet.from_fields(
            sizes=sizes,
            adjacency=tfgnn.Adjacency.from_indices(
                (self._seed_node_set, link_source_idx),
                (self._readout_node_set, readout_index),
            ),
        ),
        f'{self._readout_node_set}/target': tfgnn.EdgeSet.from_fields(
            sizes=sizes,
            adjacency=tfgnn.Adjacency.from_indices(
                (self._seed_node_set, link_target_idx),
                (self._readout_node_set, readout_index),
            ),
        ),
    }
    return tfgnn.GraphTensor.from_pieces(
        node_sets={**graph_tensor.node_sets, **readout_node_sets},
        edge_sets={**graph_tensor.edge_sets, **readout_edge_sets},
        context=graph_tensor.context,
    )


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

  features_by_edge_set = collections.defaultdict(
      lambda: collections.defaultdict(list)
  )
  seed_op = (
      sampling_spec.seed_op
      if sampling_spec.HasField('seed_op')
      else sampling_spec.symmetric_link_seed_op
  )
  seeds_by_op_name = {
      seed_op.op_name: seed_node_ids
  }

  # Concatenates inputs `[batch, item, ...]`` along the item dimension.
  concat_fn = functools.partial(tf.concat, axis=1)
  for sampling_op in sampling_spec.sampling_ops:
    input_tensors = concat_fn(
        [seeds_by_op_name[op_name] for op_name in sampling_op.input_op_names]
    )
    edge_sampler = edge_sampler_factory(sampling_op)
    sampled_edges = edge_sampler(input_tensors)
    features = features_by_edge_set[sampling_op.edge_set_name]
    for feature_name, feature_value in sampled_edges.items():
      features[feature_name].append(feature_value)

    seeds_by_op_name[sampling_op.op_name] = sampled_edges[tfgnn.TARGET_NAME]

  edge_sets = {}
  for edge_set_name, features in features_by_edge_set.items():
    edge_set_key = ','.join((
        graph_schema.edge_sets[edge_set_name].source,
        edge_set_name,
        graph_schema.edge_sets[edge_set_name].target,
    ))
    edge_sets[edge_set_key] = {k: concat_fn(v) for k, v in features.items()}

  return edge_sets


def _unique_y(x: tf.Tensor) -> tf.Tensor:
  return tf.unique(x).y


def assert_sorted_sampling_spec(sampling_spec: sampling_spec_pb2.SamplingSpec):
  """Raises ValueError if `sampling_spec` is not topologically-sorted."""
  seed_op = (
      sampling_spec.seed_op
      if sampling_spec.HasField('seed_op')
      else sampling_spec.symmetric_link_seed_op
  )
  seen_ops = {seed_op.op_name}
  for sampling_op in sampling_spec.sampling_ops:
    for input_op in sampling_op.input_op_names:
      if input_op not in seen_ops:
        raise ValueError(
            f'Input op {input_op} is used before defined. '
            'sampling_spec is not topologically-sorted'
        )
    seen_ops.add(sampling_op.op_name)
