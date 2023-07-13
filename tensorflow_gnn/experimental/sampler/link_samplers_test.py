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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.experimental.in_memory import datasets
from tensorflow_gnn.experimental.in_memory import int_arithmetic_sampler as ia_sampler
from tensorflow_gnn.experimental.sampler import link_samplers
from tensorflow_gnn.sampler import sampling_spec_builder


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

  def node_features_dicts(self, add_id=True) -> dict[
      tfgnn.NodeSetName, dict[str, tf.Tensor]]:
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

  def node_counts(self) -> dict[tfgnn.NodeSetName, int]:
    return {'food': len(self.id2food), 'animal': len(self.id2animal)}

  def edge_lists(self) -> dict[tuple[str, str, str], tf.Tensor]:
    return {
        ('animal', 'eats', 'food'): self.eats_edgelist,
    }


def _range_mod(mod, num_items=20):
  return tf.range(num_items, dtype=tf.int32) % mod


# Full module name that is under test.
_module = 'tensorflow_gnn.experimental.sampler.link_samplers'


class LinkSamplersIntegrationTest(tf.test.TestCase, parameterized.TestCase):

  def test_create_link_prediction_sampling_model(self):
    toy_data = ToyDataset()
    sampler = ia_sampler.GraphSampler(toy_data)

    # sampler.make_edge_sampler
    source_spec = sampling_spec_builder.make_sampling_spec_tree(
        toy_data.graph_schema(), 'animal', sample_sizes=[2, 3])
    target_spec = sampling_spec_builder.make_sampling_spec_tree(
        toy_data.graph_schema(), 'food', sample_sizes=[1, 1])
    lp_sampler = link_samplers.create_link_prediction_sampling_model(
        toy_data.graph_schema(),
        source_sampling_spec=source_spec,
        target_sampling_spec=target_spec,
        source_edge_sampler_factory=sampler.make_edge_sampler,
        target_edge_sampler_factory=sampler.make_edge_sampler,
        node_features_accessor_factory=(
            toy_data.create_node_features_lookup_factory()))

    # Test via dataset.
    animal_names = ['cat', 'cow', 'unicorn']
    food_names = ['water', 'water', 'beef']
    animals = tf.convert_to_tensor(
        [toy_data.animal2id[animal] for animal in animal_names],
        dtype=tf.int32)
    foods = tf.convert_to_tensor(
        [toy_data.food2id[food] for food in food_names],
        dtype=tf.int32)
    dataset = (
        tf.data.Dataset.range(3)
        .map(lambda i: (tf.gather(animals, i), tf.gather(foods, i)))
        .batch(3)
        .map(lambda sources, targets: lp_sampler((sources, targets))))

    sampled_subgraph = next(iter(dataset))
    sampled_subgraph = sampled_subgraph.merge_batch_to_components()
    self.assertAllEqual(
        # 3 components (batch size of 3)
        sampled_subgraph.context.sizes, [1, 1, 1])

    # Verify that the _readout is correctly mounted at nodes in batch.
    self.assertAllEqual(
        tf.gather(
            # 'names' feature should be available, since we passed:
            # `toy_data.create_node_features_lookup_factory`.
            sampled_subgraph.node_sets['animal']['names'],
            sampled_subgraph.edge_sets['_readout/source'].adjacency.source),
        tf.convert_to_tensor(animal_names))
    self.assertAllEqual(
        tf.gather(
            sampled_subgraph.node_sets['food']['names'],
            sampled_subgraph.edge_sets['_readout/target'].adjacency.source),
        tf.convert_to_tensor(food_names))

    # Verify that all sampled edges are true.
    for edge_set_name, edge_set in sampled_subgraph.edge_sets.items():
      if edge_set_name.startswith('_'):
        continue
      source_name = edge_set.adjacency.source_name
      source_nodes = tf.gather(
          sampled_subgraph.node_sets[source_name]['names'],
          edge_set.adjacency.source)
      target_name = edge_set.adjacency.target_name
      target_nodes = tf.gather(
          sampled_subgraph.node_sets[target_name]['names'],
          edge_set.adjacency.target)
      if source_name == 'animal':
        animals = source_nodes
        foods = target_nodes
      elif source_name == 'food':
        foods = source_nodes
        animals = target_nodes

      for animal, food in zip(animals.numpy().tolist(), foods.numpy().tolist()):
        animal = animal.decode()
        food = food.decode()
        self.assertIn(food, toy_data.eats[animal])


class LinkSamplersUnitTest(tf.test.TestCase, parameterized.TestCase):

  def test_uniqify_featureless_nodes(self):
    graph_tensor = _make_graph_tensor(
        node_sets={
            'A': {'#id': tf.constant([1, 2, 3, 1, 2, 1])},
            'B': {'#id': tf.constant([101, 102, 103, 102])},
            'C': {'#id': tf.constant([1001, 1002])},
        }, edge_sets={
            ('A', 'B'): (_range_mod(6), _range_mod(4)),
            ('A', 'A'): (_range_mod(6), _range_mod(5)),
            ('C', 'B'): (_range_mod(2), _range_mod(4)),
        })
    uniq_graph = link_samplers.uniqify_featureless_nodes(graph_tensor)

    for edge_set_name in ('A->B', 'A->A', 'C->B'):
      input_adj = graph_tensor.edge_sets[edge_set_name].adjacency
      uniq_adj = uniq_graph.edge_sets[edge_set_name].adjacency
      src_ns_name, tgt_ns_name = edge_set_name.split('->', 1)
      self.assertAllEqual(
          tf.gather(
              graph_tensor.node_sets[src_ns_name]['#id'], input_adj.source),
          tf.gather(
              uniq_graph.node_sets[src_ns_name]['#id'], uniq_adj.source))
      self.assertAllEqual(
          tf.gather(
              graph_tensor.node_sets[tgt_ns_name]['#id'], input_adj.target),
          tf.gather(
              uniq_graph.node_sets[tgt_ns_name]['#id'], uniq_adj.target))

  def test_add_readout(self):
    node_sets = {
        'A': {'#id': tf.constant([1, 2, 3]),
              'feat': tf.constant([14.0, 2.0, -1.0])},
    }
    edge_sets = {
        ('A', 'A'): (tf.constant([0, 0, 1, 2]), tf.constant([0, 2, 1, 1])),
    }
    graph = _make_graph_tensor(node_sets=node_sets, edge_sets=edge_sets)
    graph = link_samplers.add_readout(
        graph,
        source_id=tf.constant(1), target_id=tf.constant(3),
        source_node_set_name='A', target_node_set_name='A')

    src_features = tfgnn.structured_readout(
        graph, key='source', feature_name='feat')
    tgt_features = tfgnn.structured_readout(
        graph, key='target', feature_name='feat')
    self.assertAllEqual(tf.constant([14.0]), src_features)
    self.assertAllEqual(tf.constant([-1.0]), tgt_features)


def _make_graph_tensor(
        node_sets: dict[tfgnn.NodeSetName, tfgnn.Fields],
        edge_sets: dict[tuple[tfgnn.NodeSetName, tfgnn.NodeSetName],
                        tuple[tf.Tensor, tf.Tensor]]) -> tfgnn.GraphTensor:
  graph_node_sets = (
      {ns_name: tfgnn.NodeSet.from_fields(features=f, sizes=f['#id'].shape)
       for ns_name, f in node_sets.items()})
  graph_edge_sets = (
      {f'{src_n}->{tgt_n}': tfgnn.EdgeSet.from_fields(
          sizes=src.shape, adjacency=tfgnn.Adjacency.from_indices(
              source=(src_n, src), target=(tgt_n, tgt)))
       for (src_n, tgt_n), (src, tgt) in edge_sets.items()})
  return tfgnn.GraphTensor.from_pieces(
      node_sets=graph_node_sets, edge_sets=graph_edge_sets)


class MergeGraphsIntoOneComponentTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.contexts = [
        tf.constant([[1, 2, 3]]),
        tf.constant([[10, 20, 30]]),
        tf.constant([[100, 200, 300]]),
    ]

    # List of (nset1 features, nset2 features)
    self.nodes_features = [
        (tf.range(5) + 1, tf.range(3) + 20),
        (tf.range(2) * -1, tf.range(100) * -1),
        (tf.range(3) * -20, tf.range(2) * -11),
    ]  # node sets for 3 graph tensors.

    # List of (eset1, eset2)
    # eset1 connects nset1-nset1 and eset2 connects nset1-nset2
    self.adjacency_info = [
        (((0, 4), (1, 2)), ((1, 2, 4), (0, 2, 1))),
        ([[], []], [[], []]),  # no edges.
        (((0, 0), (0, 1)), ((0, 2), (1, 0))),
    ]

    uniform = tf.random.uniform  # for short.
    self.edge_features = (
        [(uniform([len(e1[0]), 5, 5]), uniform([len(e2[0]), 3]))
         for e1, e2 in self.adjacency_info])

    self.graph_tensors = []
    for node_feats, (adj1, adj2), edge_feats, context in zip(
        self.nodes_features, self.adjacency_info, self.edge_features,
        self.contexts):
      ns1_features, ns2_features = node_feats
      es1_features, es2_features = edge_feats

      self.graph_tensors.append(tfgnn.GraphTensor.from_pieces(
          context=tfgnn.Context.from_fields(features={'cf': context}),
          node_sets={
              'ns1': tfgnn.NodeSet.from_fields(
                  sizes=ns1_features.shape[:1], features={'f': ns1_features}),
              'ns2': tfgnn.NodeSet.from_fields(
                  sizes=ns2_features.shape[:1], features={'f2': ns2_features})
          },
          edge_sets={
              'es1': tfgnn.EdgeSet.from_fields(
                  sizes=es1_features.shape[:1],
                  adjacency=tfgnn.Adjacency.from_indices(
                      source=('ns1', tf.constant(adj1[0], dtype=tf.int32)),
                      target=('ns1', tf.constant(adj1[1], dtype=tf.int32)),
                  ),
                  features={'f': es1_features}),
              'es2': tfgnn.EdgeSet.from_fields(
                  sizes=es2_features.shape[:1],
                  adjacency=tfgnn.HyperAdjacency.from_indices({
                      0: ('ns1', tf.constant(adj2[0], dtype=tf.int32)),
                      1: ('ns2', tf.constant(adj2[1], dtype=tf.int32)),
                      # Repeats 0 to test HyperAdjacency.
                      2: ('ns1', tf.constant(adj2[0], dtype=tf.int32)),
                  }),
                  features={'f2': es2_features}),
          }))

  def test_merge_heterogeneous(self):
    # Method under test.
    gt = link_samplers.merge_graphs_into_one_component(
        self.graph_tensors)

    # Node and edge set names.
    self.assertSetEqual({'ns1', 'ns2'}, set(gt.node_sets.keys()))
    self.assertSetEqual({'es1', 'es2'}, set(gt.edge_sets.keys()))

    # Feature names.
    self.assertSetEqual({'cf'}, set(gt.context.features.keys()))
    self.assertSetEqual({'f'}, set(gt.node_sets['ns1'].features.keys()))
    self.assertSetEqual({'f2'}, set(gt.node_sets['ns2'].features.keys()))
    self.assertSetEqual({'f'}, set(gt.edge_sets['es1'].features.keys()))
    self.assertSetEqual({'f2'}, set(gt.edge_sets['es2'].features.keys()))

    # One Component.
    self.assertAllEqual(tf.constant([1]), gt.context.sizes)

    sizes_ns1, sizes_ns2 = zip(
        *[(f1.shape[0], f2.shape[0]) for f1, f2 in self.nodes_features])
    self.assertAllEqual(
        tf.reduce_sum(sizes_ns1, keepdims=1), gt.node_sets['ns1'].sizes)
    self.assertAllEqual(
        tf.reduce_sum(sizes_ns2, keepdims=1), gt.node_sets['ns2'].sizes)

    sizes_es1, sizes_es2 = zip(
        *[(len(es1[0]), len(es2[0])) for es1, es2 in self.adjacency_info])
    self.assertAllEqual(tf.reduce_sum(sizes_es1, keepdims=True),
                        gt.edge_sets['es1'].sizes)
    self.assertAllEqual(tf.reduce_sum(sizes_es2, keepdims=True),
                        gt.edge_sets['es2'].sizes)

    # Features.
    self.assertAllEqual(gt.context['cf'], tf.concat(self.contexts, axis=0))

    node_feats1 = tf.concat([nf[0] for nf in self.nodes_features], axis=0)
    node_feats2 = tf.concat([nf[1] for nf in self.nodes_features], axis=0)
    self.assertAllEqual(node_feats1, gt.node_sets['ns1']['f'])
    self.assertAllEqual(node_feats2, gt.node_sets['ns2']['f2'])

    edge_feats1 = tf.concat([ef[0] for ef in self.edge_features], axis=0)
    edge_feats2 = tf.concat([ef[1] for ef in self.edge_features], axis=0)
    self.assertAllEqual(edge_feats1, gt.edge_sets['es1']['f'])
    self.assertAllEqual(edge_feats2, gt.edge_sets['es2']['f2'])

    # Adjacencies
    self.assertEqual('ns1', gt.edge_sets['es1'].adjacency.source_name)
    self.assertEqual('ns1', gt.edge_sets['es1'].adjacency.target_name)
    self.assertEqual('ns1', gt.edge_sets['es2'].adjacency.node_set_name(0))
    self.assertEqual('ns2', gt.edge_sets['es2'].adjacency.node_set_name(1))

    #
    actual_es1_src = gt.edge_sets['es1'].adjacency.source
    actual_es1_tgt = gt.edge_sets['es1'].adjacency.target
    actual_es2_src = gt.edge_sets['es2'].adjacency[0]
    actual_es2_tgt = gt.edge_sets['es2'].adjacency[1]

    offset_ns1 = sum(sizes_ns1[:2])  # Offset for 3rd graph tensor
    offset_ns2 = sum(sizes_ns2[:2])  # Offset for 3rd graph tensor

    # in setUp, self.adjacency_info constructed as:
    # self.adjacency_info = [
    #     (((0, 4), (1, 2)), ((1, 2, 4), (0, 2, 1))),
    #     ([[], []], [[], []]),  # no edges.
    #     (((0, 0), (0, 1)), ((0, 2), (1, 0))),
    # ]
    self.assertAllEqual([0, 4, 0+offset_ns1, 0+offset_ns1], actual_es1_src)
    self.assertAllEqual([1, 2, 0+offset_ns1, 1+offset_ns1], actual_es1_tgt)
    self.assertAllEqual([1, 2, 4, 0+offset_ns1, 2+offset_ns1], actual_es2_src)
    self.assertAllEqual([0, 2, 1, 1+offset_ns2, 0+offset_ns2], actual_es2_tgt)


if __name__ == '__main__':
  tf.test.main()
