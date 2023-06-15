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
"""Tests for int_arithmetics_sampler."""

import collections
import functools
import os
from typing import Mapping, MutableMapping, Tuple

from absl.testing import parameterized
import numpy as np
import scipy.sparse
import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn.experimental.in_memory import datasets
from tensorflow_gnn.experimental.in_memory import int_arithmetic_sampler as ia_sampler
from tensorflow_gnn.experimental.sampler import subgraph_pipeline
from tensorflow_gnn.sampler import sampling_spec_builder
from tensorflow_gnn.sampler import sampling_spec_pb2

SamplingStrategy = sampling_spec_pb2.SamplingStrategy


class ToyDataset(datasets.InMemoryGraphData):

  def __init__(self):
    super().__init__()
    self.eats = {
        'dog': ['beef', 'water'],
        'cat': ['spider', 'rat', 'water'],
        'monkey': ['banana', 'water'],
        'cow': ['banana', 'water'],
        'unicorn': [],
    }

    food_set = set()
    for animal_food in self.eats.values():
      food_set.update(animal_food)
    id_animal = list(enumerate(self.eats.keys()))
    id_food = list(enumerate(food_set))

    self.id2food = [v for _, v in id_food]
    self.id2animal = [v for _, v in id_animal]
    self.animal2id = {v: i for i, v in id_animal}
    self.food2id = {v: i for i, v in id_food}

    edge_list = []
    for animal, foods in self.eats.items():
      for food in foods:
        edge_list.append((self.animal2id[animal], self.food2id[food]))

    edge_list = np.array(edge_list)
    self.eats_edgelist = tf.convert_to_tensor(edge_list.T)

  def node_features_dicts(self, add_id=True) -> Mapping[
      tfgnn.NodeSetName, MutableMapping[str, tf.Tensor]]:
    return {
        'animal': {
            '#id': tf.range(len(self.id2animal), dtype=tf.int32),
            'names': tf.convert_to_tensor(np.array(self.id2animal)),
        },
        'food': {
            '#id': tf.range(len(self.id2food), dtype=tf.int32),
            'names': tf.convert_to_tensor(np.array(self.id2food)),
        }
    }

  def node_counts(self) -> Mapping[tfgnn.NodeSetName, int]:
    return {'food': len(self.id2food), 'animal': len(self.id2animal)}

  def edge_lists(self) -> Mapping[Tuple[str, str, str], tf.Tensor]:
    return {
        ('animal', 'eats', 'food'): self.eats_edgelist,
    }


class IntArithmeticSamplerTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('WithReplacement', ia_sampler.EdgeSampling.WITH_REPLACEMENT),
      ('WithoutReplacement', ia_sampler.EdgeSampling.WITHOUT_REPLACEMENT))
  def test_sample_one_hop(self, strategy):
    toy_dataset = ToyDataset()
    sampler = ia_sampler.GraphSampler(toy_dataset, sampling_mode=strategy)
    source_node_names = ['dog', 'monkey']
    source_node_ids = [toy_dataset.animal2id[name]
                       for name in source_node_names]
    source_node_ids = np.array(source_node_ids)

    sample_size = 100
    next_hop = sampler.sample_one_hop(
        tf.convert_to_tensor(source_node_ids), 'eats', sample_size=sample_size,
        validate=False)

    self.assertEqual(next_hop.shape[0], source_node_ids.shape[0])
    self.assertEqual(next_hop.shape[1], sample_size)

    for i, src_name in enumerate(source_node_names):
      actual_neighbor_names = toy_dataset.eats[src_name]
      actual_neighbors = [toy_dataset.food2id[name]
                          for name in actual_neighbor_names]

      sampled_neighbors = next_hop[i]

      sampled_neighbors_set = set(sampled_neighbors.numpy())

      # Sampled neighbors are actual neighbors.
      self.assertEqual(
          sampled_neighbors_set,
          sampled_neighbors_set.intersection(actual_neighbors))

      sampled_counts = collections.Counter(sampled_neighbors.numpy())

      # Each node has 2 neighbors. Make sure that nothing is "sampled too much"
      self.assertGreater(min(sampled_counts.values()), sample_size // 10)

      if strategy == ia_sampler.EdgeSampling.WITHOUT_REPLACEMENT:
        # WITHOUT_REPLACEMENT is fair. It will pick each item once, then
        # re-iterate (in another random order).
        self.assertEqual(min(sampled_counts.values()), sample_size // 2)

  @parameterized.named_parameters(
      ('WithReplacement', ia_sampler.EdgeSampling.WITH_REPLACEMENT),
      ('WithoutReplacement', ia_sampler.EdgeSampling.WITHOUT_REPLACEMENT))
  def test_sample_two_hops(self, strategy):
    toy_dataset = ToyDataset()
    sampler = ia_sampler.GraphSampler(toy_dataset, sampling_mode=strategy)
    source_node_names = ['dog', 'monkey']
    source_node_ids = [toy_dataset.animal2id[name]
                       for name in source_node_names]
    source_node_ids = np.array(source_node_ids)

    batch_size = source_node_ids.shape[0]
    hop1_size = 30
    hop1 = sampler.sample_one_hop(
        tf.convert_to_tensor(source_node_ids), 'eats', sample_size=hop1_size,
        validate=False)
    self.assertIsInstance(hop1, tf.Tensor)
    assert isinstance(hop1, tf.Tensor)  # Assert needed to access `.shape`.

    hop2_size = 20
    hop2 = sampler.sample_one_hop(
        hop1, 'rev_eats', sample_size=hop2_size, validate=False)
    self.assertIsInstance(hop2, tf.Tensor)
    assert isinstance(hop2, tf.Tensor)  # Assert needed to access `.shape`.

    self.assertEqual(hop1.shape, (batch_size, hop1_size))
    self.assertEqual(hop2.shape, (batch_size, hop1_size, hop2_size))
    np_edgelist = toy_dataset.eats_edgelist.numpy()

    rev_eats = scipy.sparse.csr_matrix(
        (np.ones([np_edgelist.shape[-1]], dtype=bool),
         (np_edgelist[1], np_edgelist[0])))
    for i, src_name in enumerate(source_node_names):
      actual_neighbor_names = toy_dataset.eats[src_name]
      actual_neighbors = [toy_dataset.food2id[name]
                          for name in actual_neighbor_names]

      sampled_neighbors = hop1[i]
      sampled_neighbors_set = set(sampled_neighbors.numpy())

      # Sampled neighbors are actual neighbors.
      self.assertEqual(
          sampled_neighbors_set,
          sampled_neighbors_set.intersection(actual_neighbors))

      sampled_counts = collections.Counter(sampled_neighbors.numpy())

      # Each node has 2 neighbors. Make sure that nothing is "sampled too much"
      self.assertGreater(min(sampled_counts.values()), hop1_size // 10)

      if strategy == ia_sampler.EdgeSampling.WITHOUT_REPLACEMENT:
        # WITHOUT_REPLACEMENT is fair. It will pick each item once, then
        # re-iterate (in another random order).
        self.assertEqual(min(sampled_counts.values()), hop1_size // 2)

      for j in range(hop1_size):
        actual_edges = rev_eats[int(hop1[i, j].numpy())].nonzero()[1]
        sampled_hop2_counts = collections.Counter(hop2[i, j].numpy())
        actual_edge_set = set(actual_edges)
        # All sampled edges are True edges:
        self.assertEqual(
            set(sampled_hop2_counts),
            actual_edge_set.intersection(sampled_hop2_counts.keys()))
        self.assertEmpty(
            set(sampled_hop2_counts.keys()).difference(actual_edge_set))

        if strategy == ia_sampler.EdgeSampling.WITHOUT_REPLACEMENT:
          max_count = max(sampled_hop2_counts.values())
          min_count = min(sampled_hop2_counts.values())
          self.assertLessEqual(max_count - min_count, 1)  # Fair sampling.

  @parameterized.named_parameters(
      ('WithReplacement', ia_sampler.EdgeSampling.WITH_REPLACEMENT),
      ('WithoutReplacement', ia_sampler.EdgeSampling.WITHOUT_REPLACEMENT))
  def test_sample_walk_tree_with_validation(self, strategy):
    toy_dataset = ToyDataset()
    sampler = ia_sampler.GraphSampler(toy_dataset, sampling_mode=strategy)
    source_node_names = ['dog', 'unicorn']
    source_node_ids = [toy_dataset.animal2id[name]
                       for name in source_node_names]
    source_node_ids = np.array(source_node_ids)
    source_node_ids = tf.convert_to_tensor(source_node_ids)

    toy_graph_schema = toy_dataset.graph_schema()

    hop1_samples = hop2_samples = 10
    spec = sampling_spec_builder.SamplingSpecBuilder(
        toy_graph_schema,
        default_strategy=sampling_spec_builder.SamplingStrategy.RANDOM_UNIFORM)
    spec = (spec.seed('animal').sample(hop1_samples, 'eats')
            .sample(hop2_samples, 'rev_eats').build())
    walk_tree = sampler.sample_walk_tree(source_node_ids, spec)

    # Root node contains source nodes, all of which are valid.
    self.assertAllEqual(walk_tree.nodes, source_node_ids)
    self.assertAllEqual(walk_tree.valid_mask,
                        tf.ones(shape=source_node_ids.shape, dtype=tf.bool))
    self.assertLen(walk_tree.next_steps, 1)  # Sampled one edge from root.
    self.assertEqual(walk_tree.next_steps[0][0], 'eats')  # Sampled edge 'eats'.
    hop1 = walk_tree.next_steps[0][1]

    self.assertLen(hop1.next_steps, 1)  # Sampled one edge from hop1.
    self.assertEqual(hop1.next_steps[0][0], 'rev_eats')
    hop2 = hop1.next_steps[0][1]

    if strategy == ia_sampler.EdgeSampling.WITH_REPLACEMENT:
      self.assertTrue(np.all(hop1.valid_mask[0]))   # dog eats some things.
      self.assertFalse(np.any(hop1.valid_mask[1]))  # unicorn eats nothing.
      # Validity should be propagated.
      self.assertFalse(np.any(hop2.valid_mask[1]))  # ERROR
    elif strategy == ia_sampler.EdgeSampling.WITHOUT_REPLACEMENT:
      self.assertTrue(np.all(hop1.valid_mask[0, :2]))   # dog eats 2 things.
      self.assertFalse(np.any(hop1.valid_mask[0, 2:]))  # dog eats 2 things.
      self.assertFalse(np.any(hop1.valid_mask[1]))  # unicorn eats nothing
      # Validity should be propagated.
      self.assertFalse(np.any(hop2.valid_mask[0, 2:]))   # ERROR
      self.assertFalse(np.any(hop2.valid_mask[1]))

  def test_as_graph_tensor_on_orphan_nodes_graph(self):
    strategy = ia_sampler.EdgeSampling.WITHOUT_REPLACEMENT
    toy_dataset = ToyDataset()
    sampler = ia_sampler.GraphSampler(toy_dataset, sampling_mode=strategy)
    source_node_names = ['dog', 'unicorn']
    source_node_ids = [toy_dataset.animal2id[name]
                       for name in source_node_names]
    source_node_ids = np.array(source_node_ids)
    source_node_ids = tf.convert_to_tensor(source_node_ids)

    toy_graph_schema = toy_dataset.graph_schema()

    hop1_samples = hop2_samples = 10
    spec = sampling_spec_builder.SamplingSpecBuilder(
        toy_graph_schema,
        default_strategy=sampling_spec_builder.SamplingStrategy.RANDOM_UNIFORM)
    spec = (spec.seed('animal').sample(hop1_samples, 'eats')
            .sample(hop2_samples, 'rev_eats').build())
    walk_tree = sampler.sample_walk_tree(source_node_ids, spec)

    def node_features_fn(node_set_name, node_ids) -> Mapping[str, tf.Tensor]:
      del node_set_name
      return {'myfeat': node_ids}

    graph_tensor = walk_tree.as_graph_tensor(node_features_fn)

    # Animal.
    eats_src = graph_tensor.edge_sets['eats'].adjacency.source
    eats_src = tf.gather(graph_tensor.node_sets['animal']['myfeat'], eats_src)

    # Foods.
    eats_tgt = graph_tensor.edge_sets['eats'].adjacency.target
    eats_tgt = tf.gather(graph_tensor.node_sets['food']['myfeat'], eats_tgt)

    def are_all_edges_valid(eats_src, eats_tgt):
      eats_str_list = [toy_dataset.id2animal[i] for i in eats_src.numpy()]
      food_str_list = [toy_dataset.id2food[i] for i in eats_tgt.numpy()]
      for animal, food in zip(eats_str_list, food_str_list):
        if food not in toy_dataset.eats[animal]:
          return False
      return True

    self.assertTrue(are_all_edges_valid(eats_src, eats_tgt))

  @parameterized.named_parameters(
      ('WithEagerMode', 'Layer'),
      ('TFLoadSavedModel', 'TFLoadModel'),
      ('KerasLoadSavedModel', 'KerasLoadModel'),
      ('AsKerasFunctional', 'KerasFunctionalModel'))
  def test_edge_sampler_layer(self, operation_mode):
    strategy = ia_sampler.EdgeSampling.WITHOUT_REPLACEMENT
    toy_dataset = ToyDataset()
    sampler = ia_sampler.GraphSampler(toy_dataset, sampling_mode=strategy)
    edge_layer = sampler.make_edge_sampler(
        sampling_spec_pb2.SamplingOp(
            edge_set_name='eats', sample_size=3,
            strategy=SamplingStrategy.RANDOM_UNIFORM))
    seeds = tf.keras.layers.Input(shape=[None], dtype=tf.int32, ragged=True)
    if operation_mode == 'Layer':
      edge_model = edge_layer
    elif operation_mode == 'KerasFunctionalModel':
      edge_model = tf.keras.Model(inputs=seeds, outputs=edge_layer(seeds))
    elif operation_mode == 'TFLoadModel':
      edge_model = tf.keras.Model(inputs=seeds, outputs=edge_layer(seeds))
      export_dir = os.path.join(self.get_temp_dir(), 'sampler')
      edge_model.save(export_dir, include_optimizer=False)
      edge_model = tf.saved_model.load(export_dir)
    elif operation_mode == 'KerasLoadModel':
      edge_model = tf.keras.Model(inputs=seeds, outputs=edge_layer(seeds))
      export_dir = os.path.join(self.get_temp_dir(), 'sampler')
      edge_model.save(export_dir, include_optimizer=False)
      edge_model = tf.keras.models.load_model(export_dir)
    else:
      raise ValueError('unknown operation_mode %s' % operation_mode)

    cat_id = toy_dataset.animal2id['cat']
    dog_id = toy_dataset.animal2id['dog']
    monkey_id = toy_dataset.animal2id['monkey']
    ragged_ids = tf.ragged.constant([[cat_id, dog_id], [monkey_id]])
    edge_dict = edge_model(ragged_ids)
    self.assertEqual(tuple(sorted(edge_dict.keys())), ('#source', '#target'))

    # Cats eat 3 things while dogs and monkeys eat only 2.
    self.assertAllClose(
        edge_dict['#source'],
        tf.ragged.constant([[cat_id, cat_id, cat_id, dog_id, dog_id],
                            [monkey_id, monkey_id]]))
    cat_eats = edge_dict['#target'][0][:3].numpy()
    dog_eats = edge_dict['#target'][0][3:].numpy()
    monkey_eats = edge_dict['#target'][1].numpy()
    cat_eats = [toy_dataset.id2food[f] for f in cat_eats]
    dog_eats = [toy_dataset.id2food[f] for f in dog_eats]
    monkey_eats = [toy_dataset.id2food[f] for f in monkey_eats]
    self.assertSetEqual(set(cat_eats), set(toy_dataset.eats['cat']))
    self.assertSetEqual(set(dog_eats), set(toy_dataset.eats['dog']))
    self.assertSetEqual(set(monkey_eats), set(toy_dataset.eats['monkey']))

  def test_edge_sampler_layer_invalid_input_raises_exceptions(self):
    strategy = ia_sampler.EdgeSampling.WITHOUT_REPLACEMENT
    toy_dataset = ToyDataset()
    sampler = ia_sampler.GraphSampler(toy_dataset, sampling_mode=strategy)
    self.assertRaises(
        ValueError,
        functools.partial(
            sampler.make_edge_sampler,
            sampling_spec_pb2.SamplingOp(
                edge_set_name='invalid_name', sample_size=3,
                strategy=SamplingStrategy.RANDOM_UNIFORM)))

  def test_graph_sampler_constructor_without_edgesets_raises_exception(self):
    strategy = ia_sampler.EdgeSampling.WITHOUT_REPLACEMENT
    toy_dataset = ToyDataset()
    # Mock the edge_sets() to return empty edge-sets:
    toy_dataset.edge_sets = lambda: {}
    self.assertRaises(ValueError, functools.partial(ia_sampler.GraphSampler,
                                                    toy_dataset,
                                                    sampling_mode=strategy))

  @parameterized.named_parameters(
      ('TFLoadSavedModel', 'TFLoadModel'),
      ('KerasLoadSavedGModel', 'KerasLoadModel'),
      ('KerasModel', 'KerasModel')
  )
  def test_smoke_integration_sampling_pipeline_model(self, operation_mode):
    """Smoke integration test of int-arithmetic running with `SamplingPipline`.

    Args:
      operation_mode: Determines if `keras.Model` should be saved-then-loaded
        before invoking sampling.
    """
    toy_dataset = ToyDataset()
    strategy = ia_sampler.EdgeSampling.WITHOUT_REPLACEMENT
    sampler = ia_sampler.GraphSampler(toy_dataset, sampling_mode=strategy)

    hop1_samples = hop2_samples = 10
    spec = (
        sampling_spec_builder.SamplingSpecBuilder(
            toy_dataset.graph_schema(),
            default_strategy=SamplingStrategy.RANDOM_UNIFORM)
        .seed('animal').sample(hop1_samples, 'eats')
        .sample(hop2_samples, 'rev_eats')
        .build())

    sampling_model = subgraph_pipeline.create_sampling_model_from_spec(
        seed_node_dtype=tf.int32,
        graph_schema=toy_dataset.graph_schema(), sampling_spec=spec,
        edge_sampler_factory=sampler.make_edge_sampler)

    if operation_mode == 'TFLoadModel':
      export_dir = os.path.join(self.get_temp_dir(), 'sampler')
      sampling_model.save(export_dir, include_optimizer=False)
      sampling_model = tf.saved_model.load(export_dir)
    elif operation_mode == 'KerasLoadModel':
      export_dir = os.path.join(self.get_temp_dir(), 'sampler')
      sampling_model.save(export_dir, include_optimizer=False)
      sampling_model = tf.keras.models.load_model(export_dir)

    source_node_names = [['dog', 'monkey'], ['dog'], ['monkey']]
    node_ids = tf.nest.map_structure(
        toy_dataset.animal2id.get, source_node_names)
    node_ids = tf.ragged.constant(node_ids)

    sample_graph = sampling_model(node_ids)
    # A couple of checks only. This is an integration smoke test (assumes
    # `SamplingPipeline` is tested in its own file).
    self.assertIsInstance(sample_graph, tfgnn.GraphTensor)
    self.assertAllEqual(
        sample_graph.edge_sets['eats'].sizes,
        tf.convert_to_tensor([[4], [2], [2]], dtype=tf.int64))


if __name__ == '__main__':
  tf.test.main()
