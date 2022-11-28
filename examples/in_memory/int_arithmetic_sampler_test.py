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
from typing import Mapping, MutableMapping, Tuple

from absl.testing import parameterized
import numpy as np
import scipy.sparse
import tensorflow as tf
import tensorflow_gnn as tfgnn

import datasets
from tensorflow_gnn.examples.in_memory import int_arithmetic_sampler as ia_sampler


class ToyDataset(datasets.NodeClassificationDatasetWrapper):

  def __init__(self):
    self.eats = {
        'dog': ['beef', 'water'],
        'cat': ['spider', 'rat', 'water'],
        'monkey': ['banana', 'water'],
        'cow': ['banana', 'water'],
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
        'animals': {
            '#id': tf.range(len(self.id2animal), dtype=tf.int32),
            'names': tf.convert_to_tensor(np.array(self.id2animal)),
        },
        'food': {
            '#id': tf.range(len(self.id2food), dtype=tf.int32),
            'names': tf.convert_to_tensor(np.array(self.id2food)),
        }
    }

  def node_counts(self) -> Mapping[tfgnn.NodeSetName, int]:
    return {'food': len(self.id2food), 'animals': len(self.id2animal)}

  def edge_lists(self) -> Mapping[Tuple[str, str, str], tf.Tensor]:
    return {
        ('animals', 'eats', 'food'): self.eats_edgelist,
    }


class IntArithmeticSamplerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('WithReplacement', ia_sampler.EdgeSampling.WITH_REPLACEMENT),
      ('WithoutReplacement', ia_sampler.EdgeSampling.WITHOUT_REPLACEMENT))
  def test_sample_one_hop(self, strategy):
    toy_dataset = ToyDataset()
    sampler = ia_sampler.GraphSampler(toy_dataset)
    source_node_names = ['dog', 'monkey']
    source_node_ids = [toy_dataset.animal2id[name]
                       for name in source_node_names]
    source_node_ids = np.array(source_node_ids)

    sample_size = 100
    next_hop = sampler.sample_one_hop(
        tf.convert_to_tensor(source_node_ids), 'eats', sample_size=sample_size,
        sampling=strategy)

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
    sampler = ia_sampler.GraphSampler(toy_dataset)
    source_node_names = ['dog', 'monkey']
    source_node_ids = [toy_dataset.animal2id[name]
                       for name in source_node_names]
    source_node_ids = np.array(source_node_ids)

    batch_size = source_node_ids.shape[0]
    hop1_size = 30
    hop1 = sampler.sample_one_hop(
        tf.convert_to_tensor(source_node_ids), 'eats', sample_size=hop1_size,
        sampling=strategy)

    hop2_size = 20
    hop2 = sampler.sample_one_hop(
        hop1, 'rev_eats', sample_size=hop2_size, sampling=strategy)

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


if __name__ == '__main__':
  tf.test.main()
