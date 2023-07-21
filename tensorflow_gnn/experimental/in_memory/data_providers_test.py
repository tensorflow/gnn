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

import functools
from typing import Mapping, MutableMapping, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.experimental.in_memory import data_providers
from tensorflow_gnn.experimental.in_memory import datasets


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


class ToyLinkPredictionData(ToyDataset, datasets.LinkPredictionGraphData):

  def edge_split(self) -> datasets.EdgeSplit:
    dtype = self.eats_edgelist.dtype
    return datasets.EdgeSplit(
        train_edges=self.eats_edgelist,
        validation_edges=tf.constant([
            # More animals loves banana's nowadays!
            [self.animal2id['dog'], self.animal2id['cat']],
            [self.food2id['banana'], self.food2id['banana']]], dtype=dtype),
        test_edges=tf.zeros([2, 0], dtype=dtype),
        negative_validation_edges=tf.zeros([2, 0], dtype=dtype),
        negative_test_edges=tf.zeros([2, 0], dtype=dtype))

  @property
  def target_edgeset(self) -> tfgnn.EdgeSetName:
    return 'eats'


class ToyNodeClassificationData(
    ToyDataset, datasets.NodeClassificationGraphData):

  def node_split(self) -> datasets.NodeSplit:
    return datasets.NodeSplit(
        train=tf.constant(
            [self.animal2id['dog'], self.animal2id['cat']], dtype=tf.int32),
        validation=tf.constant(
            [self.animal2id['monkey'], self.animal2id['cow']], dtype=tf.int32),
        test=tf.constant(
            [self.animal2id['unicorn']], dtype=tf.int32),)

  def num_classes(self) -> int:
    return 3

  def labels(self) -> tf.Tensor:
    return tf.ones([len(self.eats)], dtype=tf.int32)

  def test_labels(self) -> tf.Tensor:
    return self.labels()

  @property
  def labeled_nodeset(self):
    return 'animal'

  def node_features_dicts_without_labels(self):
    return super().node_features_dicts()


class DataProvidersTest(tf.test.TestCase):

  def test_sample_negative_edges(self):
    edge = tf.constant([55, 220], dtype=tf.int32)
    sources, targets, labels = data_providers._sample_negative_edges(
        edge[0], edge[1],
        negative_links_sampling=data_providers.NegativeLinksSampling(
            negative_sources_fn=functools.partial(
                data_providers._uniform_random_int_negatives,
                output_dtype=tf.int32,
                max_node_id=100),
            negative_targets_fn=functools.partial(
                data_providers._uniform_random_int_negatives,
                output_dtype=tf.int32,
                max_node_id=300),
            num_negatives_per_source=10, num_negatives_per_target=4)
        )
    self.assertAllEqual(labels[0], 1.0)
    self.assertAllEqual(labels[1:], tf.zeros_like(labels[1:]))
    self.assertAllEqual([sources[0], targets[0]], edge)
    self.assertAllEqual(
        edge[0] * tf.ones([10], dtype=tf.int32), sources[1:11])
    self.assertAllEqual(
        edge[1] * tf.ones([4], dtype=tf.int32), targets[11:])

  def test_provide_link_prediction_data(self):
    graph_data = ToyLinkPredictionData()
    link_pred_data = data_providers.provide_link_prediction_data(
        graph_data=graph_data)

    # Check on cardinality.
    num_eats_edges = sum(map(len, graph_data.eats.values()))
    self.assertEqual(num_eats_edges, link_pred_data.provider.cardinality)

    ds = link_pred_data.provider.get_dataset(tf.distribute.InputContext())
    records = list(ds)

    # 4 negative samples per positive.
    self.assertLen(records, num_eats_edges * 5)

    for i, record in enumerate(records):
      if i % 5 == 0:
        # Positive.
        self.assertEqual(record.node_sets['_readout']['label'], [1.0])
        source_node = tf.gather(
            record.node_sets['animal']['names'],
            record.edge_sets['_readout/source'].adjacency.source)
        target_node = tf.gather(
            record.node_sets['food']['names'],
            record.edge_sets['_readout/target'].adjacency.source)
        for s, t in zip(source_node.numpy(), target_node.numpy()):
          s = s.decode()
          t = t.decode()
          self.assertIn(s, graph_data.eats)
          self.assertIn(t, graph_data.eats[s])
      else:
        # Negative.
        self.assertEqual(record.node_sets['_readout']['label'], [0.0])

    # Task must be a link-prediction one.
    self.assertIsInstance(
        link_pred_data.task,
        (
            data_providers.runner.DotProductLinkPrediction,
            data_providers.runner.HadamardProductLinkPrediction))

  def test_provide_node_classification_data(self):
    graph_data = ToyNodeClassificationData()
    node_classification = data_providers.provide_node_classification_data(
        graph_data=graph_data)

    # Only 2 labeled nodes
    self.assertEqual(2, node_classification.provider.cardinality)

    records = list(
        node_classification.provider.get_dataset(tf.distribute.InputContext()))
    self.assertLen(records, 2)

    # The labeled nodes are 'dog' and 'cat'
    labeled_name0 = tf.gather(
        records[0].node_sets['animal']['names'],
        records[0].edge_sets['_readout/seed'].adjacency.source)
    labeled_name1 = tf.gather(
        records[1].node_sets['animal']['names'],
        records[1].edge_sets['_readout/seed'].adjacency.source)
    self.assertSetEqual(
        set(tf.concat([labeled_name0, labeled_name1], axis=0).numpy()),
        set((b'cat', b'dog')))

    # The label is 1 for both.
    self.assertEqual(records[0].node_sets['_readout']['label'], [1])
    self.assertEqual(records[1].node_sets['_readout']['label'], [1])


if __name__ == '__main__':
  tf.test.main()
