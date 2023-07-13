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
"""Functions for sampling subgraphs around edges for link-prediction tasks."""
import functools
from typing import Optional, Union

import tensorflow as tf
from tensorflow_gnn.experimental.sampler import subgraph_pipeline
from tensorflow_gnn.graph import adjacency
from tensorflow_gnn.graph import graph_constants
from tensorflow_gnn.graph import graph_piece
from tensorflow_gnn.graph import graph_tensor
import tensorflow_gnn.proto.graph_schema_pb2 as schema_pb2

from tensorflow_gnn.sampler import sampling_spec_pb2

GraphTensor = graph_tensor.GraphTensor
EdgeSamplerFactory = subgraph_pipeline.EdgeSamplerFactory
NodeFeaturesLookupFactory = subgraph_pipeline.NodeFeaturesLookupFactory


def create_link_prediction_sampling_model(
    graph_schema: schema_pb2.GraphSchema,
    *,
    source_sampling_spec: sampling_spec_pb2.SamplingSpec,
    target_sampling_spec: sampling_spec_pb2.SamplingSpec,
    source_edge_sampler_factory: EdgeSamplerFactory,
    target_edge_sampler_factory: EdgeSamplerFactory,
    node_features_accessor_factory: Optional[NodeFeaturesLookupFactory] = None,
    source_ids_dtype=tf.int32,
    target_ids_dtype=tf.int32,
    ) -> tf.keras.Model:
  r"""Creates keras sampling model for link prediction.

  The model can be invoked on tuple `(source_ids, target_ids)`. Both must be
  vectors of same length, with dtypes, respectively, of `source_ids_dtype`,
  and `target_ids_dtype`.

  Args:
    graph_schema: `GraphSchema` corresponding to `{src, tgt}_sampling_spec`s.
    source_sampling_spec: `SamplingSpec` used to sample subgraph around source.
    target_sampling_spec: `SamplingSpec` used to sample subgraph around target.
    source_edge_sampler_factory: callable should return an edge-sampling
      model for every edge-set traversed by `source_sampling_spec`.
    target_edge_sampler_factory: callable should return an edge-sampling
      model for every edge-set traversed by `target_sampling_spec`. In most
      cases, this is probably equal to `source_edge_sampler_factory`.
    node_features_accessor_factory: If given, must be a factory function, when
      called with a node set name (present in `graph_schema`), should return a
      callable model (or `None`) that can lookup features of sampled nodes.
      It will be called with node set names:
      `source_sampling_spec.seed_op.node_set_name` and
      `target_sampling_spec.seed_op.node_set_name`, and all node sets incident
      on edge-sets sampled per these `sampling_spec`s.
    source_ids_dtype: The expected dtype of source_ids that the returned model
      should be invoked with.
    target_ids_dtype: The expected dtype of target_ids that the returned model
      should be invoked with.

  Returns:
    Keras model must be invoked on a pair `(source_ids, target_ids)` that must
    be of types, `source_ids_dtype` and `target_ids_dtype`, respectively.
    Importantly, they must be IDs of nodes in the node sets:
    `source_sampling_spec.seed_op.node_set_name` and
    `target_sampling_spec.seed_op.node_set_name`. Further, they both must be
    vectors of same length `B`. The model then returns `GraphTensor` containing
    vector of subgraphs with length `B`. Graph at component `i` contains a
    link-prediction subgraph around node pair `(source_ids[i], target_ids[i])`.
    Specifically, one subgraph will be sampled around `source_ids[i]` (using
    spec `source_sampling_spec` and sampler factory
    `source_edge_sampler_factory`), and another subgraph around
    `target_ids[i]` (using `target_*`).
  """
  source_pipeline = subgraph_pipeline.SamplingPipeline(
      graph_schema, source_sampling_spec, source_edge_sampler_factory)
  target_pipeline = subgraph_pipeline.SamplingPipeline(
      graph_schema, target_sampling_spec, target_edge_sampler_factory)

  source_ids = tf.keras.Input([None], dtype=source_ids_dtype)
  target_ids = tf.keras.Input([None], dtype=target_ids_dtype)

  sampled_subgraph = sample_link_prediction_featureless_subgraph(
      source_ids, target_ids,
      source_pipeline=source_pipeline, target_pipeline=target_pipeline)
  if node_features_accessor_factory is not None:
    new_features = {}
    for node_set_name, node_set in sampled_subgraph.node_sets.items():
      if graph_tensor.get_aux_type_prefix(node_set_name):
        continue
      accessor = node_features_accessor_factory(node_set_name)
      if not accessor:
        continue
      node_set_features = accessor(node_set['#id'])
      if not node_set_features:
        continue
      node_set_features = dict(node_set_features)
      if '#id' in node_set_features:
        raise ValueError(
            '`node_features_accessor_factory` returned feature with forbidden '
            'name "#id".')
      node_set_features['#id'] = node_set['#id']
      new_features[node_set_name] = node_set_features
    sampled_subgraph = sampled_subgraph.replace_features(node_sets=new_features)

  return tf.keras.models.Model(
      inputs=(source_ids, target_ids), outputs=sampled_subgraph)


def sample_link_prediction_featureless_subgraph(
    source_ids: tf.Tensor, target_ids: tf.Tensor, *,
    source_pipeline: subgraph_pipeline.SamplingPipeline,
    target_pipeline: subgraph_pipeline.SamplingPipeline) -> GraphTensor:
  """Samples subgraphs around `source_ids` and `target_ids` w/o node features.

  Featureless: it is expected that `source_pipeline` and `target_pipeline` do
  not obtain node features (as in, their `node_features_accessor_factory` must
  be `None`), due to underlying implementation. NOTE: this requirement can be
  lifted by extending `uniqify_featureless_nodes` to process all node features
  in addition to feature '#id'.

  Args:
    source_ids: Vector of size `B` with `dtype` of the seed node-set that is
      configured by `source_pipeline`.
    target_ids: Vector of size `B` with `dtype` of the seed node-set that is
      configured by `target_pipeline`.
    source_pipeline: Sampling pipeline that will be invoked (once) on
      `source_ids` that should return `GraphTensor` containing vector of `B`
      graphs. Graph `i` must be subgraph rooted at node `source_ids[i]`.
    target_pipeline: Sampling pipeline that will be invoked (once) on
      `target_ids` that should return `GraphTensor` containing vector of `B`
      graphs. Graph `i` must be subgraph rooted at node `target_ids[i]`.

  Returns:
    GraphTensor that contains vector of graphs. There should be `B` graphs,
    where graph at index `i` merges subgraph samples
    `source_pipeline(source_ids[i])` and `target_pipeline(target_ids[i])`.
    In addition to all node- and edge-sets returned by `*_pipeline`, the
    returned graph will have node-set: `'_readout'`; and edge-sets:
    `'_readout/source'` and `'_readout/target'`. Each will have a single entry
    per graph.
  """
  if source_pipeline.node_features_accessor_factory:
    raise ValueError(
        '`source_pipeline` is expected to yield subgraphs without node '
        'features. However, `source_pipeline.node_features_accessor_factory` '
        'is set.')
  if target_pipeline.node_features_accessor_factory:
    raise ValueError(
        '`target_pipeline` is expected to yield subgraphs without node '
        'features. However, `target_pipeline.node_features_accessor_factory` '
        'is set.')
  source_graphs = source_pipeline(tf.expand_dims(source_ids, -1))
  target_graphs = target_pipeline(tf.expand_dims(target_ids, -1))

  spec = source_graphs.merge_batch_to_components().spec.relax(
      num_nodes=True, num_edges=True, num_components=True)
  merged_graphs = tf.keras.layers.Lambda(
      functools.partial(
          tf.map_fn,
          merge_graphs_into_one_component,
          fn_output_signature=spec))([source_graphs, target_graphs])

  merged_graphs = tf.keras.layers.Lambda(
      functools.partial(
          tf.map_fn,
          uniqify_featureless_nodes,
          fn_output_signature=spec))(merged_graphs)

  add_readout_fn = functools.partial(
      add_readout,
      source_node_set_name=source_pipeline.seed_node_set_name,
      target_node_set_name=target_pipeline.seed_node_set_name)
  add_readout_lambda = tf.keras.layers.Lambda(lambda x: add_readout_fn(*x))

  readout_spec = _add_readout_to_spec(spec, source_pipeline.seed_node_set_name,
                                      target_pipeline.seed_node_set_name)
  merged_graphs = tf.keras.layers.Lambda(
      functools.partial(
          tf.map_fn,
          add_readout_lambda,
          fn_output_signature=readout_spec))(
              [merged_graphs, source_ids, target_ids])

  return merged_graphs


def uniqify_featureless_nodes(graph: GraphTensor) -> GraphTensor:
  """Dedups nodes in `graph` using '#id' feature and dropping all others.


  Args:
    graph: Scalar graph with one component, with every node set containing
      feature with name `#id`. Nodes might be duplicated (some `#id`s are
      reused).

  Returns:
    graphs with unique `#id` features. The number of nodes is less than (or
    equal to) nodes of input per node set (equal iff input `#id` are unique).
    Output edges are exactly equal to input, however, edge indicies could change
    though the '#id' features of its (node) endpoints are maintained. As such,
    edges might be duplicated.
  """
  graph_tensor.check_scalar_graph_tensor(graph, 'uniqify_nodes')

  uniques = (
      {node_set_name: tf.unique(node_set['#id'], out_idx=graph.indices_dtype)
       for node_set_name, node_set in graph.node_sets.items()})
  edge_sets = {}
  for edge_set_name, edge_set in graph.edge_sets.items():
    source_name = edge_set.adjacency.source_name
    target_name = edge_set.adjacency.target_name
    source = tf.gather(uniques[source_name].idx, edge_set.adjacency.source)
    target = tf.gather(uniques[target_name].idx, edge_set.adjacency.target)
    source = tf.cast(source, edge_set.adjacency.source.dtype)
    target = tf.cast(target, edge_set.adjacency.target.dtype)
    edge_sets[edge_set_name] = graph_tensor.EdgeSet.from_fields(
        features=edge_set.features,
        sizes=edge_set.sizes,
        adjacency=adjacency.Adjacency.from_indices(
            source=(source_name, source), target=(target_name, target)))

  node_sets = {}
  for ns_name, uniq in uniques.items():
    ns_features = tuple(graph.node_sets[ns_name].features.keys())
    if ns_features != ('#id',):
      raise ValueError('Node sets must be featureless (except #id feature). '
                       f'Found: {", ".join(ns_features)}.')
    node_sets[ns_name] = graph_tensor.NodeSet.from_fields(
        features={'#id': uniq.y},
        sizes=tf.cast(_shape(uniq.y), graph.node_sets[ns_name].sizes.dtype))

  return GraphTensor.from_pieces(
      node_sets=node_sets, edge_sets=edge_sets, context=graph.context)


def add_readout(
    graph: GraphTensor,
    source_id: tf.Tensor, target_id: tf.Tensor,
    source_node_set_name: graph_constants.NodeSetName,
    target_node_set_name: graph_constants.NodeSetName) -> GraphTensor:
  """Adds _readout node set and _readout/{source, target} edge sets.

  Note: This only works for scalar graphs with one component.

  Args:
    graph: Scalar `GraphTensor` with one component. It is expected that "#id"
      of node sets (`source_node_set_name`, `target_node_set_name`) do not
      contain duplicates and they contain scalars (`source_id`, `target_id`).
    source_id: Scalar must exist in vector
      `graph.node_sets[source_node_set_name]['#id']`.
    target_id: Scalar must exist in vector
      `graph.node_sets[target_node_set_name]['#id']`.
    source_node_set_name: Name of node set that `source_id` belongs to.
    target_node_set_name: Name of node set that `target_id` belongs to.

  Returns:
    Input `graph` amended with:
    + nodeset `'_readout'` with one "virtual" node, without features.
    + edge-sets `'_readout/source'` and `'_readout/target'`, connecting from,
      respectively, `'source_id'` and `'target_id'`. They both connect to the
      "virtual" `'_readout'` node.
  """
  graph_tensor.check_scalar_graph_tensor(graph, 'add_readout')
  source_pos = tf.argmax(
      graph.node_sets[source_node_set_name]['#id'] == source_id,
      output_type=graph.context.sizes.dtype)
  target_pos = tf.argmax(
      graph.node_sets[target_node_set_name]['#id'] == target_id,
      output_type=graph.context.sizes.dtype)

  edge_sets = dict(graph.edge_sets)  # copy
  node_sets = dict(graph.node_sets)  # copy

  # _readout node- and edge-sets are of size 1 (one virtual node with one edge).
  sizes = tf.ones([1], dtype=graph.context.sizes.dtype)
  indices = tf.zeros([1], dtype=graph.context.sizes.dtype)
  edge_sets['_readout/source'] = graph_tensor.EdgeSet.from_fields(
      sizes=sizes,
      adjacency=adjacency.Adjacency.from_indices(
          source=(source_node_set_name, tf.expand_dims(source_pos, 0)),
          target=('_readout', indices)
      ))
  edge_sets['_readout/target'] = graph_tensor.EdgeSet.from_fields(
      sizes=sizes,
      adjacency=adjacency.Adjacency.from_indices(
          source=(target_node_set_name, tf.expand_dims(target_pos, 0)),
          target=('_readout', indices)
      ))

  node_sets['_readout'] = graph_tensor.NodeSet.from_fields(
      features={}, sizes=sizes)

  return GraphTensor.from_pieces(
      node_sets=node_sets, edge_sets=edge_sets, context=graph.context)


def _make_adjacency(
    index_dict: dict[
        graph_constants.IncidentNodeTag,
        tuple[graph_constants.NodeSetName, tf.Tensor]]
    ) -> adjacency.HyperAdjacency:
  if tuple(sorted(index_dict.keys())) == (0, 1):
    return adjacency.Adjacency.from_indices(
        source=index_dict[0], target=index_dict[1])
  else:
    return adjacency.HyperAdjacency.from_indices(index_dict)


def _stack(graphs: list[GraphTensor]) -> GraphTensor:
  assert graphs
  specs = [tf.type_spec_from_value(g) for g in graphs]
  common_spec = specs[0].most_specific_common_supertype(specs[1:])
  if common_spec is None:
    raise ValueError('input graphs are not compatible')
  pieces = [common_spec._to_tensor_list(g) for g in graphs]  # pylint: disable=protected-access
  result_pieces = [tf.stack(components) for components in zip(*pieces)]
  result_spec = common_spec._batch(len(graphs))  # pylint: disable=protected-access
  return result_spec._from_compatible_tensor_list(result_pieces)  # pylint: disable=protected-access


def stack_componets(graph_tensors: list[GraphTensor]) -> GraphTensor:
  """Combine graphs each with `k` components into graph with `k` components."""
  if not graph_tensors:
    raise ValueError('Expecting at non-empty list `graph_tensors`.')
  for i, graph in enumerate(graph_tensors):
    graph_tensor.check_scalar_graph_tensor(graph, f'stack_componets at {i}')
  return _stack(graph_tensors).merge_batch_to_components()


def merge_components(graph: GraphTensor) -> GraphTensor:
  """Combines graph with several components into one component."""
  ssum = functools.partial(tf.reduce_sum, axis=0, keepdims=True)
  return GraphTensor.from_pieces(
      context=graph_tensor.Context.from_fields(
          sizes=ssum(graph.context.sizes), features=graph.context.features),
      node_sets=(
          {name: graph_tensor.NodeSet.from_fields(
              sizes=ssum(node_set.sizes), features=node_set.features)
           for name, node_set in graph.node_sets.items()}),
      edge_sets=(
          {name: graph_tensor.EdgeSet.from_fields(
              sizes=ssum(edge_set.sizes), features=edge_set.features,
              adjacency=edge_set.adjacency)
           for name, edge_set in graph.edge_sets.items()}))


def merge_graphs_into_one_component(graphs: list[GraphTensor]) -> GraphTensor:
  """Combines a list of scalar graph tensors into one graph tensor.

  Graph will have only one component.

  Args:
    graphs: list of scalar (rank=0) graph tensors.

  Returns:
    `graph_tensor` that combines all input `graph_tensor`s into one component.
  """
  return merge_components(stack_componets(graphs))


def _add_readout_to_spec(
    spec: graph_piece.GraphPieceSpecBase,
    source_node_set_name: graph_constants.NodeSetName,
    target_node_set_name: graph_constants.NodeSetName
    ) -> graph_tensor.GraphTensorSpec:
  """Amends `spec` with "_readout" node- and edge-sets.

  This function becomes obsolete when b/217514667 is fixed.

  Args:
    spec: Spec of `GraphTensor` upon which _readout node- and edge-sets will be
      added.
    source_node_set_name: Source node set for link prediction.
    target_node_set_name: Target node set for link prediction.

  Returns:
    Copy of input `spec` with additional node-set `'_readout'` and edge-sets
    `'_readout/source'` and `'_readout/target'`. The first edge set will connect
    (`source_node_set_name` -> `'_readout'`) and the second will connect
    (`target_node_set_name` -> `'_readout'`).
  """
  node_sets_spec = dict(spec.node_sets_spec)
  node_sets_spec['_readout'] = graph_tensor.NodeSetSpec.from_field_specs(
      sizes_spec=tf.TensorSpec(shape=(1,), dtype=spec.indices_dtype))
  edge_sets_spec = dict(spec.edge_sets_spec)
  edge_sets_spec['_readout/source'] = graph_tensor.EdgeSetSpec.from_field_specs(
      sizes_spec=tf.TensorSpec(shape=(1,), dtype=spec.indices_dtype),
      adjacency_spec=adjacency.AdjacencySpec.from_incident_node_sets(
          source_node_set_name,
          '_readout',
          index_spec=tf.TensorSpec(shape=(1,), dtype=spec.indices_dtype)))
  edge_sets_spec['_readout/target'] = graph_tensor.EdgeSetSpec.from_field_specs(
      sizes_spec=tf.TensorSpec(shape=(1,), dtype=spec.indices_dtype),
      adjacency_spec=adjacency.AdjacencySpec.from_incident_node_sets(
          target_node_set_name,
          '_readout',
          index_spec=tf.TensorSpec(shape=(1,), dtype=spec.indices_dtype)))
  return graph_tensor.GraphTensorSpec.from_piece_specs(
      context_spec=spec.context_spec, edge_sets_spec=edge_sets_spec,
      node_sets_spec=node_sets_spec)


def _shape(tensor: tf.Tensor) -> Union[list[int], tf.Tensor]:
  """Helper function returns shape of eager or symbolic tensors."""
  if tensor.shape.is_fully_defined():
    return tensor.shape
  else:
    return tf.shape(tensor)
