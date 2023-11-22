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
"""Tests for subgraph_pipeline.py."""
import collections
import functools
from typing import Dict, List, Tuple, Callable, Optional

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

import tensorflow_gnn as tfgnn
from tensorflow_gnn.experimental.sampler import core
from tensorflow_gnn.experimental.sampler import eval_dag
from tensorflow_gnn.experimental.sampler import interfaces
from tensorflow_gnn.experimental.sampler import subgraph_pipeline

from tensorflow_gnn.sampler import sampling_spec_pb2


class _FnEdgeSampler(interfaces.OutgoingEdgesSampler):

  def __init__(self, fn: Callable[[tf.RaggedTensor], interfaces.Features]):
    self._fn = fn

  def call(self, source_node_ids: tf.RaggedTensor) -> interfaces.Features:
    return self._fn(source_node_ids)

  @property
  def edge_set_name(self) -> tfgnn.EdgeSetName:
    raise NotImplementedError()


class StringIdsSampler:
  """Samples without replacement given string -> strings for each edge set."""

  def __init__(
      self, edges: Dict[tfgnn.EdgeSetName, Dict[str, List[str]]],
      add_reverse=True):
    if add_reverse:
      for edge_set_name, node_to_nodes in list(edges.items()):
        rev_edges = collections.defaultdict(list)
        for node, neighbors in node_to_nodes.items():
          for neighbor in neighbors:
            rev_edges[neighbor].append(node)
        edges['rev_' + edge_set_name] = dict(rev_edges)

    self.counts: Dict[tfgnn.EdgeSetName, tf.lookup.StaticHashTable] = {}
    self.edges: Dict[tfgnn.EdgeSetName, tf.lookup.StaticHashTable] = {}

    for edge_set_name, node_to_nodes in edges.items():
      keys, values = zip(*node_to_nodes.items())
      lengths = [len(v) for v in values]
      self.counts[edge_set_name] = tf.lookup.StaticHashTable(
          tf.lookup.KeyValueTensorInitializer(
              tf.convert_to_tensor(keys), tf.convert_to_tensor(lengths)), 0)
      edge_keys = []
      edge_values = []
      for k, neighbors in zip(keys, values):
        for i, neighbor in enumerate(neighbors):
          edge_keys.append(f'{k}.{i}')
          edge_values.append(neighbor)
      self.edges[edge_set_name] = tf.lookup.StaticHashTable(
          tf.lookup.KeyValueTensorInitializer(edge_keys, edge_values), '')

  def sample_nodes(self, sample_size: int, edge_set_name: tfgnn.EdgeSetName,
                   node_id: tf.Tensor) -> tf.Tensor:
    node_degree = self.counts[edge_set_name].lookup(node_id)
    sample_indices = tf.random.shuffle(
        tf.range(tf.maximum(node_degree, sample_size)))[:sample_size]
    query_keys = node_id + '.' + tf.strings.as_string(sample_indices)
    return self.edges[edge_set_name].lookup(query_keys)

  def sample_edges(self, sample_size: int, edge_set_name: tfgnn.EdgeSetName,
                   node_id: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    neighbors = self.sample_nodes(sample_size, edge_set_name, node_id)
    node_id_repeated = tf.expand_dims(node_id, -1) + tf.zeros_like(neighbors)
    return node_id_repeated, neighbors

  def sample_edges_multiple_sources(
      self, sample_size: int, edge_set_name: tfgnn.EdgeSetName,
      node_ids: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    vector_node_ids = tf.reshape(node_ids, [-1])
    nodes_dtype = vector_node_ids.dtype

    src_nodes, tgt_nodes = tf.map_fn(
        functools.partial(self.sample_edges, sample_size, edge_set_name),
        vector_node_ids, fn_output_signature=(
            tf.TensorSpec(shape=[sample_size], dtype=nodes_dtype),
            tf.TensorSpec(shape=[sample_size], dtype=nodes_dtype),
        ))
    src_nodes = tf.reshape(src_nodes, [-1])
    tgt_nodes = tf.reshape(tgt_nodes, [-1])
    valid_positions = (tgt_nodes != '')  # pylint: disable=g-explicit-bool-comparison
    src_nodes = tf.boolean_mask(src_nodes, valid_positions)
    tgt_nodes = tf.boolean_mask(tgt_nodes, valid_positions)
    return src_nodes, tgt_nodes

  def sample_v2(self, sample_size: int, edge_set_name: tfgnn.EdgeSetName,
                source_node_ids: tf.RaggedTensor) -> interfaces.Features:
    endpoint_spec = tf.RaggedTensorSpec(
        shape=[None], dtype=source_node_ids.dtype, ragged_rank=0)
    edges_src, edges_tgt = tf.map_fn(
        functools.partial(self.sample_edges_multiple_sources,
                          sample_size, edge_set_name),
        source_node_ids,
        fn_output_signature=(endpoint_spec, endpoint_spec))
    return {tfgnn.SOURCE_NAME: edges_src, tfgnn.TARGET_NAME: edges_tgt}

  def __call__(
      self, op: sampling_spec_pb2.SamplingOp
      ) -> interfaces.OutgoingEdgesSampler:
    return _FnEdgeSampler(
        functools.partial(self.sample_v2, op.sample_size, op.edge_set_name))


def _get_test_edges_sampler_schema_spec():
  eats_edges = {
      'dog': ['beef', 'water'],
      'cat': ['spider', 'rat', 'water'],
      'rat_eater': ['rat'],
      'spider_eater': ['spider'],
      'monkey': ['banana', 'water'],
      'cow': ['banana', 'water'],
      'pig': ['water', 'banana', 'beef'],
      'unicorn': [],  # unicorns need no food.
  }
  sampler = StringIdsSampler({'eats': eats_edges})

  graph_schema = tfgnn.GraphSchema()
  graph_schema.node_sets['animal'].description = 'animal'
  graph_schema.node_sets['food'].description = 'food'
  graph_schema.edge_sets['eats'].source = 'animal'
  graph_schema.edge_sets['eats'].target = 'food'
  graph_schema.edge_sets['rev_eats'].source = 'food'
  graph_schema.edge_sets['rev_eats'].target = 'animal'

  sampling_spec = sampling_spec_pb2.SamplingSpec()
  sampling_spec.seed_op.op_name = 'seed'
  sampling_spec.seed_op.node_set_name = 'animal'
  sampling_spec.sampling_ops.add(
      op_name='A', edge_set_name='eats',
      sample_size=100).input_op_names.append('seed')
  sampling_spec.sampling_ops.add(
      op_name='B', edge_set_name='rev_eats',
      sample_size=2).input_op_names.append('A')

  return eats_edges, sampler, graph_schema, sampling_spec


def _test_create_feature_accessor(
    node_set_name: tfgnn.NodeSetName) -> Optional[
        interfaces.KeyToFeaturesAccessor]:
  if node_set_name == 'food':
    return None
  return _TestNodeFeatureAccessor()


class _TestNodeFeatureAccessor(
    tf.keras.Model, interfaces.KeyToFeaturesAccessor):

  @property
  def resource_name(self) -> str:
    return '_TestNodeFeatureAccessor'

  def call(self, keys: tf.RaggedTensor) -> interfaces.Features:
    return {
        '#id': keys,  # 'dog', 'cat', ...
        '#ID': tf.ragged.map_flat_values(  # 'DOG', 'CAT', ...
            tf.strings.upper, keys),
    }


class SubgraphPipelineTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    tfgnn.enable_graph_tensor_validation_at_runtime()

  def test_sample_edge_sets(self):
    (eats_edges, sampler, graph_schema,
     sampling_spec) = _get_test_edges_sampler_schema_spec()

    seed_nodes = tf.ragged.constant([['dog'], ['cat'], ['dog', 'cat']])
    sampled_edges = subgraph_pipeline.sample_edge_sets(
        seed_nodes, graph_schema=graph_schema,
        sampling_spec=sampling_spec, edge_sampler_factory=sampler)

    eats_edge = 'animal,eats,food'
    eaten_edge = 'food,rev_eats,animal'

    # Although we asked for 100 samples: dog eats 2 things and cat eats 3!
    self.assertEqual(sampled_edges[eats_edge][tfgnn.SOURCE_NAME][0].shape[0], 2)
    self.assertEqual(sampled_edges[eats_edge][tfgnn.TARGET_NAME][0].shape[0], 2)
    self.assertEqual(sampled_edges[eats_edge][tfgnn.SOURCE_NAME][1].shape[0], 3)
    self.assertEqual(sampled_edges[eats_edge][tfgnn.TARGET_NAME][1].shape[0], 3)
    self.assertEqual(sampled_edges[eats_edge][tfgnn.TARGET_NAME][2].shape[0], 5)
    self.assertEqual(sampled_edges[eats_edge][tfgnn.TARGET_NAME][2].shape[0], 5)

    self.assertTrue(tf.reduce_all(
        b'dog' == sampled_edges[eats_edge][tfgnn.SOURCE_NAME][0]))
    self.assertTrue(tf.reduce_all(
        b'cat' == sampled_edges[eats_edge][tfgnn.SOURCE_NAME][1]))
    self.assertEqual(3, tf.reduce_sum(
        tf.cast(
            sampled_edges[eats_edge][tfgnn.SOURCE_NAME][2] == b'cat',
            tf.int32)))
    self.assertEqual(2, tf.reduce_sum(
        tf.cast(
            sampled_edges[eats_edge][tfgnn.SOURCE_NAME][2] == b'dog',
            tf.int32)))

    # From there sampled foods, we sampled 2 animals that eat each. Since each
    # food can be eaten by at least 2 animals, we should have 2 edges per target
    # end point of first "eats" edge.
    self.assertEqual(
        sampled_edges[eaten_edge][tfgnn.SOURCE_NAME][0].shape[0], 4)
    self.assertEqual(
        sampled_edges[eaten_edge][tfgnn.TARGET_NAME][0].shape[0], 4)
    self.assertEqual(
        sampled_edges[eaten_edge][tfgnn.SOURCE_NAME][1].shape[0], 6)
    self.assertEqual(
        sampled_edges[eaten_edge][tfgnn.TARGET_NAME][1].shape[0], 6)

    sampled_foods = set()
    sampled_animals = set()

    # Verify all edges are true.
    for animal, food in zip(
        sampled_edges[eats_edge][tfgnn.SOURCE_NAME].values.numpy().tolist(),
        sampled_edges[eats_edge][tfgnn.TARGET_NAME].values.numpy().tolist()):
      self.assertIn(food.decode(), eats_edges[animal.decode()])
      sampled_foods.add(food)
      sampled_animals.add(animal)

    for animal, food in zip(
        sampled_edges[eaten_edge][tfgnn.TARGET_NAME].values.numpy().tolist(),
        sampled_edges[eaten_edge][tfgnn.SOURCE_NAME].values.numpy().tolist()):
      self.assertIn(food.decode(), eats_edges[animal.decode()])
      self.assertIn(food, sampled_foods)  # Second hop.
      sampled_animals.add(animal)

  def test_orphan_nodes(self):
    """Tests sampling subgraph from nodes without edges."""
    (unused_eats_edges, sampler, graph_schema,
     sampling_spec) = _get_test_edges_sampler_schema_spec()
    pipeline = subgraph_pipeline.SamplingPipeline(
        graph_schema, sampling_spec, sampler,
        node_features_accessor_factory=_test_create_feature_accessor)

    seed_nodes = tf.ragged.constant([
        ['unicorn'],  # Orphan node (no edges!)
        ['unicorn', 'dog'],  # Orphan node with a non-orphan node.
    ])
    graph_tensor = pipeline(seed_nodes)
    # Break into its subgraphs.
    graph_tensors = list(tf.data.Dataset.from_tensors(graph_tensor).unbatch())
    graph0 = graph_tensors[0]
    graph1 = graph_tensors[1]

    self.assertAllEqual(graph0.node_sets['animal']['#id'], [b'unicorn'])
    # No edges (unicorn needs no food).
    for edge_set0 in graph0.edge_sets.values():
      self.assertAllEqual(edge_set0.sizes, [0])

    # Dog eats 2 things (unicorn eats none):
    self.assertAllEqual(graph1.edge_sets['eats'].sizes, [2])

    # Make sure 'unicorn' appears in node set but without edges.
    bool_mask = graph1.node_sets['animal']['#id'] == b'unicorn'
    self.assertAllEqual(tf.reduce_sum(tf.cast(bool_mask, tf.int32)), 1)
    unicorn_index = tf.argmax(bool_mask, output_type=tf.int32)
    self.assertNotIn(unicorn_index, graph1.edge_sets['eats'].adjacency.source)

  def test_sampling_pipeline(self):
    (eats_edges, sampler, graph_schema,
     sampling_spec) = _get_test_edges_sampler_schema_spec()

    pipeline = subgraph_pipeline.SamplingPipeline(
        graph_schema, sampling_spec, sampler,
        node_features_accessor_factory=_test_create_feature_accessor)

    seed_nodes = tf.ragged.constant([['dog'], ['cat'], ['dog', 'cat']])
    graph_tensor = pipeline(seed_nodes)

    # 'cat' can reach 'spider_eater' and 'rat_eater' with 2 hops
    id_feature = graph_tensor.node_sets['animal']['#id']
    self.assertIn(b'spider_eater', id_feature[1])
    self.assertIn(b'spider_eater', id_feature[2])
    self.assertIn(b'rat_eater', id_feature[1])
    self.assertIn(b'rat_eater', id_feature[2])
    # But 'dogs' cannot reach them with 2 hops:
    self.assertNotIn(b'spider_eater', id_feature[0])
    self.assertNotIn(b'rat_eater', id_feature[0])

    graph_tensor = graph_tensor.merge_batch_to_components()

    self.assertAllEqual(
        tf.strings.upper(graph_tensor.node_sets['animal']['#id']),
        graph_tensor.node_sets['animal']['#ID'])

    # _TestNodeFeatureAccessor is programmed to not modify 'food' features.
    self.assertNotIn('#ID', graph_tensor.node_sets['food'].features)

    src_values = graph_tensor.edge_sets['eats'].adjacency.source
    src_values = tf.gather(graph_tensor.node_sets['animal']['#id'], src_values)
    tgt_values = graph_tensor.edge_sets['eats'].adjacency.target
    tgt_values = tf.gather(graph_tensor.node_sets['food']['#id'], tgt_values)

    self.assertSetEqual(
        set(src_values.numpy()), set(seed_nodes.values.numpy().reshape(-1)))

    for animal, food in zip(src_values.numpy(), tgt_values.numpy()):
      self.assertIn(food.decode(), eats_edges[animal.decode()])

    src_values = graph_tensor.edge_sets['rev_eats'].adjacency.source
    src_values = tf.gather(graph_tensor.node_sets['food']['#id'], src_values)
    tgt_values = graph_tensor.edge_sets['rev_eats'].adjacency.target
    tgt_values = tf.gather(graph_tensor.node_sets['animal']['#id'], tgt_values)

    for food, animal in zip(src_values.numpy(), tgt_values.numpy()):
      self.assertIn(food.decode(), eats_edges[animal.decode()])


class EdgeFeaturesTest(tf.test.TestCase):

  def test_homogeneous(self):
    graph_schema = tfgnn.GraphSchema()
    graph_schema.node_sets['a'].description = 'test node set'
    graph_schema.edge_sets['a->a'].source = 'a'
    graph_schema.edge_sets['a->a'].target = 'a'
    graph_schema.edge_sets['a->a'].features['f'].dtype = 1

    sampling_spec = sampling_spec_pb2.SamplingSpec()
    sampling_spec.seed_op.op_name = 'seed'
    sampling_spec.seed_op.node_set_name = 'a'
    sampling_spec.sampling_ops.add(
        op_name='hop1', edge_set_name='a->a', sample_size=100
    ).input_op_names.append('seed')

    def edge_sampler_factory(sampling_op):
      self.assertEqual(sampling_op.edge_set_name, 'a->a')
      return core.InMemUniformEdgesSampler(
          num_source_nodes=3,
          source=tf.constant([2, 0], tf.int32),
          target=tf.constant([0, 1], tf.int32),
          edge_features={'f': [2.0, 0.0]},
          seed=42,
          sample_size=sampling_op.sample_size,
          name=sampling_op.edge_set_name,
      )

    sampling_model = subgraph_pipeline.create_sampling_model_from_spec(
        graph_schema,
        sampling_spec,
        edge_sampler_factory,
        seed_node_dtype=tf.int32,
    )
    result = sampling_model(tf.ragged.constant([[2], [0]]))
    self.assertIn('a', result.node_sets)
    self.assertIn('a->a', result.edge_sets)
    edge_features = result.edge_sets['a->a'].get_features_dict()
    self.assertIn('f', edge_features)
    self.assertAllEqual(edge_features['f'], tf.ragged.constant([[2.0], [0.0]]))


class EvalDagTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([tf.string, tf.int32, tf.int64])
  def test(self, ids_dtype: tf.DType):
    graph_schema = tfgnn.GraphSchema()
    graph_schema.node_sets['a'].description = 'node set a'
    graph_schema.node_sets['a'].description = 'node set b'
    graph_schema.edge_sets['a->b'].source = 'a'
    graph_schema.edge_sets['a->b'].target = 'b'
    graph_schema.edge_sets['a->a'].features['weights'].dtype = 1

    sampling_spec = sampling_spec_pb2.SamplingSpec()
    sampling_spec.seed_op.op_name = 'seed'
    sampling_spec.seed_op.node_set_name = 'a'
    sampling_spec.sampling_ops.add(
        op_name='hop1', edge_set_name='a->b', sample_size=100
    ).input_op_names.append('seed')

    def edge_sampler_factory(sampling_op):
      self.assertEqual(sampling_op.edge_set_name, 'a->b')
      if ids_dtype == tf.string:
        accessor = core.InMemStringKeyToBytesAccessor(
            keys_to_values={b'a': b''}
        )
      else:
        accessor = core.InMemIntegerKeyToBytesAccessor(keys_to_values={0: b''})
      return core.UniformEdgesSampler(
          sample_size=sampling_op.sample_size,
          outgoing_edges_accessor=core.KeyToTfExampleAccessor(
              accessor,
              features_spec={
                  tfgnn.TARGET_NAME: tf.TensorSpec([None], tf.string),
                  'weights': tf.TensorSpec([None], tf.float32),
              },
          ),
      )

    sampling_model = subgraph_pipeline.create_sampling_model_from_spec(
        graph_schema,
        sampling_spec,
        edge_sampler_factory,
        seed_node_dtype=ids_dtype,
    )
    program, _ = eval_dag.create_program(sampling_model)
    self.assertNotEmpty(program.eval_dag.stages)
    self.assertNotEmpty(program.layers)
    self.assertIn('Input', program.layers)
    input_layer = program.layers['Input']
    self.assertLen(input_layer.outputs, 1)
    self.assertEqual(
        input_layer.outputs[0].ragged_tensor.dtype, ids_dtype.as_datatype_enum
    )

    self.assertIn('uniform_edges_sampler', program.layers)
    sampler_layer = program.layers['uniform_edges_sampler']
    self.assertLen(sampler_layer.inputs, 1)
    self.assertEqual(
        sampler_layer.inputs[0].ragged_tensor.dtype, ids_dtype.as_datatype_enum
    )


def _get_test_link_edges_sampler_schema_spec():
  reviews_edges = {
      'mike': ['alexa'],
      'alexa': ['mike', 'sam'],
      'arne': ['alexa', 'mike', 'sam'],
      'sam': ['bob'],
      'bob': []
  }
  sampler = StringIdsSampler({'reviews': reviews_edges})

  graph_schema = tfgnn.GraphSchema()
  graph_schema.node_sets['_readout'].features['label'].dtype = 1
  graph_schema.edge_sets['reviews'].source = 'authors'
  graph_schema.edge_sets['reviews'].target = 'authors'
  graph_schema.edge_sets['_readout/source'].target = '_readout'
  graph_schema.edge_sets['_readout/source'].source = 'authors'
  graph_schema.edge_sets['_readout/target'].target = '_readout'
  graph_schema.edge_sets['_readout/target'].source = 'authors'

  sampling_spec = sampling_spec_pb2.SamplingSpec()
  sampling_spec.symmetric_link_seed_op.op_name = 'seed'
  sampling_spec.sampling_ops.add(
      op_name='A', edge_set_name='reviews',
      sample_size=2).input_op_names.append('seed')

  return reviews_edges, sampler, graph_schema, sampling_spec


class LinkSamplingPipelineTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    tfgnn.enable_graph_tensor_validation_at_runtime()

  # TODO: b/301427603 - Test this parametrically on int64 and int32 seeds.
  def test_link_sampling_pipeline(self):
    (_, sampler, graph_schema,
     sampling_spec) = _get_test_link_edges_sampler_schema_spec()

    source_seeds = tf.ragged.constant([['alexa'], ['sam'], ['sam']])
    target_seeds = tf.ragged.constant([['mike'], ['mike'], ['arne']])
    seed_labels = tf.ragged.constant([[0.0], [1.0], [1.0]], dtype=tf.float32)

    inputs = {
        tfgnn.SOURCE_NAME: source_seeds,
        tfgnn.TARGET_NAME: target_seeds,
        'label': seed_labels,
    }
    pipeline = subgraph_pipeline.LinkSamplingPipeline(
        graph_schema, sampling_spec, sampler
    )
    graph_tensor = pipeline(inputs)
    dummy_node_features = tf.ragged.constant(
        [np.eye(3, 3), np.eye(4, 3), np.eye(4, 3)]
    )
    graph_tensor = graph_tensor.replace_features(
        node_sets={'authors': {tfgnn.HIDDEN_STATE: dummy_node_features}}
    )

    self.assertAllEqual(
        graph_tensor.node_sets['_readout'].sizes, [[1], [1], [1]]
    )
    self.assertAllEqual(
        graph_tensor.edge_sets['reviews'].sizes, [[3], [2], [3]]
    )
    self.assertAllEqual(
        graph_tensor.node_sets['authors'].sizes, [[3], [4], [4]]
    )
    self.assertAllEqual(
        graph_tensor.node_sets['_readout'].features['label'],
        [[0.0], [1.0], [1.0]],
    )
    merged_tensor = graph_tensor.merge_batch_to_components()
    self.assertAllEqual(
        merged_tensor.edge_sets['_readout/source'].adjacency.source, [0, 3, 7]
    )
    self.assertAllEqual(
        tfgnn.structured_readout(
            merged_tensor, 'source', feature_name=tfgnn.HIDDEN_STATE
        ),
        [[1.0, 0.0, 0.0]] * 3,
    )
    self.assertAllEqual(
        tfgnn.structured_readout(
            merged_tensor, 'target', feature_name=tfgnn.HIDDEN_STATE
        ),
        [[0.0, 1.0, 0.0]] * 3,
    )
    self.assertAllEqual(
        merged_tensor.node_sets['_readout'].features['label'], [0.0, 1.0, 1.0]
    )

  def test_seeds_with_no_edges(self):
    (_, sampler, graph_schema,
     sampling_spec) = _get_test_link_edges_sampler_schema_spec()

    source_seeds = tf.ragged.constant([['bob'], ['sam'], ['sam'], ['bob']])
    target_seeds = tf.ragged.constant([['sam'], ['bob'], ['sam'], ['bob']])
    labels = tf.ragged.constant([[0.0], [1.0], [2.0], [3.0]], dtype=tf.float32)
    inputs = {
        tfgnn.SOURCE_NAME: source_seeds,
        tfgnn.TARGET_NAME: target_seeds,
        'label': labels,
    }
    pipeline = subgraph_pipeline.LinkSamplingPipeline(
        graph_schema, sampling_spec, sampler
    )
    graph_tensor = pipeline(inputs)
    self.assertAllEqual(
        graph_tensor.node_sets['_readout'].sizes, [[1], [1], [1], [1]]
    )
    self.assertAllEqual(
        graph_tensor.node_sets['_readout'].features['label'],
        labels,
    )
    graph_tensor = graph_tensor.merge_batch_to_components()
    self.assertAllEqual(
        tfgnn.structured_readout(graph_tensor, 'source', feature_name='#id'),
        source_seeds.values,
    )
    self.assertAllEqual(
        tfgnn.structured_readout(graph_tensor, 'target', feature_name='#id'),
        target_seeds.values,
    )

  def test_multiple_seeds(self):
    (_, sampler, graph_schema, sampling_spec) = (
        _get_test_link_edges_sampler_schema_spec()
    )

    source_seeds = tf.ragged.constant([['bob', 'sam'], ['arne', 'mike']])
    target_seeds = tf.ragged.constant([['sam', 'bob'], ['mike', 'arne']])
    labels = tf.ragged.constant([[0.0, 1.0], [2.0, 3.0]], dtype=tf.float32)
    inputs = {
        tfgnn.SOURCE_NAME: source_seeds,
        tfgnn.TARGET_NAME: target_seeds,
        'label': labels,
    }
    pipeline = subgraph_pipeline.LinkSamplingPipeline(
        graph_schema, sampling_spec, sampler
    )
    graph_tensor = pipeline(inputs)
    self.assertAllEqual(
        graph_tensor.node_sets['_readout'].sizes, [[2], [2]]
    )
    self.assertAllEqual(
        graph_tensor.node_sets['_readout'].features['label'],
        labels,
    )
    graph_tensor = graph_tensor.merge_batch_to_components()
    self.assertAllEqual(
        tfgnn.structured_readout(graph_tensor, 'source', feature_name='#id'),
        source_seeds.values,
    )
    self.assertAllEqual(
        tfgnn.structured_readout(graph_tensor, 'target', feature_name='#id'),
        target_seeds.values,
    )


if __name__ == '__main__':
  tf.test.main()
