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
"""Tests for gt.GraphTensor extension type (go/tf-gnn-api)."""
import collections
from typing import Mapping, Union

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import broadcast_ops
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import graph_tensor_ops as ops
from tensorflow_gnn.graph import pool_ops
from tensorflow_gnn.graph import readout

as_tensor = tf.convert_to_tensor
as_ragged = tf.ragged.constant

GraphPiece = Union[gt.Context, gt.NodeSet, gt.EdgeSet]


class FirstNodeOpsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for operations on first nodes per component (e.g., root nodes)."""

  @parameterized.parameters([
      dict(
          description='1 component',
          node_set=gt.NodeSet.from_fields(
              sizes=as_tensor([3]),
              features={
                  'scalar': as_tensor([1., 2., 3]),
                  'vector': as_tensor([[1., 3.], [2., 2.], [3., 1.]]),
                  'matrix': as_tensor([[[1.]], [[2.]], [[3.]]]),
                  'ragged': as_ragged([[1, 2], [3], []])
              }),
          expected_fields={
              'scalar': as_tensor([1.]),
              'vector': as_tensor([[1., 3.]]),
              'matrix': as_tensor([[[1.]]]),
              'ragged': as_ragged([[1, 2]])
          }),
      dict(
          description='2 components',
          node_set=gt.NodeSet.from_fields(
              sizes=as_tensor([2, 1]),
              features={
                  'scalar': as_tensor([1., 2., 3]),
                  'vector': as_tensor([[1., 3.], [2., 2.], [3., 1.]]),
                  'matrix': as_tensor([[[1.]], [[2.]], [[3.]]]),
                  'ragged': as_ragged([[1, 2], [3], []])
              }),
          expected_fields={
              'scalar': as_tensor([1., 3.]),
              'vector': as_tensor([[1., 3.], [3., 1.]]),
              'matrix': as_tensor([[[1.]], [[3.]]]),
              'ragged': as_ragged([[1, 2], []])
          })
  ])
  def testGatherFirstNode(self, description: str, node_set: gt.NodeSet,
                          expected_fields: Mapping[str, const.Field]):
    del description
    graph = gt.GraphTensor.from_pieces(node_sets={'node': node_set})

    for fname, expected in expected_fields.items():
      self.assertAllEqual(
          expected, ops.gather_first_node(graph, 'node', feature_name=fname))

  def testGatherFirstNodeFails(self):
    graph = gt.GraphTensor.from_pieces(node_sets={
        'node': gt.NodeSet.from_fields(
            sizes=as_tensor([2, 0, 1]),
            features={'scalar': as_tensor([1., 2., 3])})})

    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                r'gather_first_node.* no nodes'):
      _ = ops.gather_first_node(graph, 'node', feature_name='scalar')


class ShuffleOpsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([
      dict(
          description='scalar',
          context=gt.Context.from_fields(
              features={
                  'scalar': as_tensor([1, 2, 3]),
              }
          ),
          node_set=gt.NodeSet.from_fields(
              sizes=as_tensor([2, 1]),
              features={
                  'scalar': as_tensor([1.0, 2.0, 3]),
              },
          ),
          edge_set=gt.EdgeSet.from_fields(
              sizes=as_tensor([2, 3]),
              adjacency=adj.HyperAdjacency.from_indices({
                  const.SOURCE: ('node', as_tensor([0, 0, 0, 2, 2])),
                  const.TARGET: ('node', as_tensor([2, 1, 0, 0, 0])),
              }),
              features={
                  'scalar': as_tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
              },
          ),
          expected_fields={
              gt.Context: {'scalar': [2, 1, 3]},
              gt.NodeSet: {'scalar': [2.0, 1.0, 3.0]},
              gt.EdgeSet: {'scalar': [5.0, 2.0, 3.0, 1.0, 4.0]},
          },
      ),
      dict(
          description='vector',
          context=gt.Context.from_fields(
              features={
                  'vector': as_tensor([[1], [2], [3]]),
              }
          ),
          node_set=gt.NodeSet.from_fields(
              sizes=as_tensor([2, 1]),
              features={
                  'vector': as_tensor([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]]),
              },
          ),
          edge_set=gt.EdgeSet.from_fields(
              sizes=as_tensor([2, 3]),
              adjacency=adj.HyperAdjacency.from_indices({
                  const.SOURCE: ('node', as_tensor([0, 0, 0, 2, 2])),
                  const.TARGET: ('node', as_tensor([2, 1, 0, 0, 0])),
              }),
              features={
                  'vector': as_tensor([
                      [1.0, 5.0],
                      [2.0, 4.0],
                      [3.0, 3.0],
                      [4.0, 2.0],
                      [5.0, 1.0],
                  ]),
              },
          ),
          expected_fields={
              gt.Context: {'vector': [[2], [1], [3]]},
              gt.NodeSet: {'vector': [[2.0, 2.0], [1.0, 3.0], [3.0, 1.0]]},
              gt.EdgeSet: {
                  'vector': [
                      [5.0, 1.0],
                      [2.0, 4.0],
                      [3.0, 3.0],
                      [1.0, 5.0],
                      [4.0, 2.0],
                  ]
              },
          },
      ),
      dict(
          description='matrix',
          context=gt.Context.from_fields(),
          node_set=gt.NodeSet.from_fields(
              sizes=as_tensor([2, 1]),
              features={
                  'matrix': as_tensor([[[1.0]], [[2.0]], [[3.0]]]),
              },
          ),
          edge_set=gt.EdgeSet.from_fields(
              sizes=as_tensor([2, 3]),
              adjacency=adj.HyperAdjacency.from_indices({
                  const.SOURCE: ('node', as_tensor([0, 0, 0, 2, 2])),
                  const.TARGET: ('node', as_tensor([2, 1, 0, 0, 0])),
              }),
              features={
                  'matrix': as_tensor(
                      [[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]]]
                  ),
              },
          ),
          expected_fields={
              gt.NodeSet: {'matrix': [[[2.0]], [[1.0]], [[3.0]]]},
              gt.EdgeSet: {
                  'matrix': [[[5.0]], [[2.0]], [[3.0]], [[1.0]], [[4.0]]]
              },
          },
      ),
      dict(
          description='ragged.1',
          context=gt.Context.from_fields(
              features={
                  'ragged.1': as_ragged([[[], [1], []], [[1], [3], [4], [2]]]),
              }
          ),
          node_set=gt.NodeSet.from_fields(
              sizes=as_tensor([2, 1]),
              features={
                  'ragged.1': as_ragged(
                      [[[1, 2], [4, 4], [5, 5]], [[3, 3]], []]
                  ),
              },
          ),
          edge_set=gt.EdgeSet.from_fields(
              sizes=as_tensor([2, 3]),
              adjacency=adj.HyperAdjacency.from_indices({
                  const.SOURCE: ('node', as_tensor([0, 0, 0, 2, 2])),
                  const.TARGET: ('node', as_tensor([2, 1, 0, 0, 0])),
              }),
              features={
                  'ragged.1': as_ragged([[[1, 2]], [], [[3, 4]], [], [[5, 5]]]),
              },
          ),
          expected_fields={
              gt.Context: {
                  'ragged.1': as_ragged([[[], [4], [2]], [[1], [], [1], [3]]])
              },
              gt.NodeSet: {
                  'ragged.1': as_ragged(
                      [[[4, 4], [5, 5], [3, 3]], [[1, 2]], []]
                  )
              },
              gt.EdgeSet: {
                  'ragged.1': as_ragged([[[5, 5]], [], [[1, 2]], [], [[3, 4]]])
              },
          },
      ),
      dict(
          description='ragged.2',
          context=gt.Context.from_fields(),
          node_set=gt.NodeSet.from_fields(
              sizes=as_tensor([2, 1]),
              features={'ragged.2': as_ragged([[[1, 2, 4], []], [[3]], [[5]]])},
          ),
          edge_set=gt.EdgeSet.from_fields(
              sizes=as_tensor([2, 3]),
              adjacency=adj.HyperAdjacency.from_indices({
                  const.SOURCE: ('node', as_tensor([0, 0, 0, 2, 2])),
                  const.TARGET: ('node', as_tensor([2, 1, 0, 0, 0])),
              }),
              features={
                  'ragged.2': as_ragged(
                      [[[1], [2], [4]], [], [[3], [5]], [], []]
                  )
              },
          ),
          expected_fields={
              gt.NodeSet: {
                  'ragged.2': as_ragged([[[1, 2, 4], [5]], [[3]], [[]]])
              },
              gt.EdgeSet: {
                  'ragged.2': as_ragged(
                      [[[2], [4], [3]], [], [[1], [5]], [], []]
                  )
              },
          },
      ),
  ])
  def testShuffleFeaturesGlobally(
      self,
      description: str,
      context: gt.Context,
      node_set: gt.NodeSet,
      edge_set: gt.EdgeSet,
      expected_fields: Mapping[GraphPiece, Mapping[str, const.Field]],
  ):
    del description
    graph = gt.GraphTensor.from_pieces(
        context,
        {'node': node_set},
        {'edge': edge_set})
    shuffled = ops.shuffle_features_globally(graph, seed=8191)

    for fname, expected in expected_fields.get(gt.Context, {}).items():
      self.assertAllEqual(expected, shuffled.context.features[fname])

    for fname, expected in expected_fields.get(gt.NodeSet, {}).items():
      self.assertAllEqual(expected, shuffled.node_sets['node'].features[fname])

    for fname, expected in expected_fields.get(gt.EdgeSet, {}).items():
      self.assertAllEqual(expected, shuffled.edge_sets['edge'].features[fname])

  def testTFLite(self):
    # TODO(b/277938756): Implement when tf.random.shuffle is supported.
    self.skipTest('Shuffling ops are unsupported in TFLite.')


def as_tensor_list(l):
  return [tf.convert_to_tensor(x) for x in l]


def as_ragged_list(l):
  return [tf.ragged.constant(x) for x in l]


class CombineFeaturesTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('Concat', as_tensor_list([[[1.]], [[2.]], [[3.]]]), 'concat',
       [[1., 2., 3.]]),
      ('ConcatRaggedLast',
       as_ragged_list([[[11., 12.], [21.]],
                       [[13.], [22., 23.]],
                       [[14., 15.], [24.]]]), 'concat',
       as_ragged([[11., 12., 13., 14., 15.],
                  [21., 22., 23., 24.]])),
      ('ConcatRaggedMiddle',
       as_ragged_list([
           [[[x+111., x+112.]],
            [[x+211., x+212.], [x+221., x+222.]]]
           for x in [1000., 2000.]]), 'concat',
       as_ragged(
           [[[1111., 1112., 2111., 2112.]],
            [[1211., 1212., 2211., 2212.], [1221., 1222., 2221., 2222.]]])),
      ('Sum', as_tensor_list([[[1.]], [[2.]], [[3.]]]), 'sum', [[6.]]),
      ('SumRaggedMiddle',
       as_ragged_list([[[[x+111., x+112.]],
                        [[x+211., x+212.], [x+221., x+222.]]]
                       for x in [1000., 2000.]]), 'sum',
       as_ragged([[[3222., 3224.]],
                  [[3422., 3424.], [3442., 3444.]]])))
  def test(self, inputs, combine_type, expected):
    actual = ops.combine_values(inputs, combine_type=combine_type)
    self.assertAllEqual(expected, actual)

  def testError(self):
    bad_inputs = [tf.ones((2, 2)), tf.ones((2, 3))]
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        r'combine_type="sum".*Please check'):
      ops.combine_values(bad_inputs, combine_type='sum')


class SelfLoopsTest(tf.test.TestCase, parameterized.TestCase):

  def testWithEmptyComponents(self):
    node_sizes = [0, 5]
    edge_sizes = [0, 0]

    graph = gt.GraphTensor.from_pieces(
        node_sets={
            'node': gt.NodeSet.from_fields(
                sizes=as_tensor(node_sizes),
                features={})
        },
        edge_sets={
            'edge': gt.EdgeSet.from_fields(
                sizes=tf.constant(as_tensor(edge_sizes), dtype=tf.int32),
                adjacency=adj.Adjacency.from_indices(
                    source=('node', tf.zeros([0], dtype=tf.int32)),
                    target=('node', tf.zeros([0], dtype=tf.int32)),
                ),
                features={}),
        })
    out_graph = ops.add_self_loops(graph, 'edge')
    self.assertLen(out_graph.node_sets, 1)
    self.assertAllEqual(node_sizes, out_graph.node_sets['node'].sizes)
    self.assertAllEqual(node_sizes, out_graph.edge_sets['edge'].sizes)
    self.assertAllEqual(tf.range(5, dtype=tf.int32),
                        out_graph.edge_sets['edge'].adjacency.source)
    self.assertAllEqual(tf.range(5, dtype=tf.int32),
                        out_graph.edge_sets['edge'].adjacency.target)

  def testSelfLoops(self):
    # pylint: disable=bad-whitespace
    component_edges = (
        # source nodes,                    target nodes
        ([],                               []),         # <-- Component 0 edges.
        ([1, 2, 3],                        [2, 3, 1]),  # <-- Component 1.
        ([4, 5, 6, 7, 8, 5],               [5, 6, 7, 8, 4, 7]),  # ...
        ([9, 10, 11, 12, 13, 14, 15, 10],  [10, 11, 12, 13, 14, 15, 9, 13]),
        ([],                               []),
        ([],                               []),
    )
    edge_sizes = []
    node_sizes = [1, 3, 5, 7, 3, 0]

    source_ids = []
    target_ids = []
    for component_source_ids, component_target_ids in component_edges:
      assert len(component_source_ids) == len(component_target_ids)
      edge_sizes.append(len(component_source_ids))
      source_ids.extend(component_source_ids)
      target_ids.extend(component_target_ids)

    total_edges = len(target_ids)
    edge_features = tf.random.uniform(shape=(total_edges, 5, 2))

    graph = gt.GraphTensor.from_pieces(
        node_sets={
            'node': gt.NodeSet.from_fields(
                sizes=as_tensor(node_sizes),
                features={})
        },
        edge_sets={
            'edge': gt.EdgeSet.from_fields(
                sizes=tf.constant(as_tensor(edge_sizes), dtype=tf.int32),
                adjacency=adj.Adjacency.from_indices(
                    source=('node', as_tensor(source_ids)),
                    target=('node', as_tensor(target_ids)),
                ),
                features={'feats': edge_features}),
        })
    out_graph = ops.add_self_loops(graph, 'edge')

    # Assert: Node sets are copied as-is
    self.assertLen(out_graph.node_sets, 1)
    self.assertIn('node', out_graph.node_sets)
    self.assertAllEqual(node_sizes, out_graph.node_sets['node'].sizes)

    # Assert: Edge counts are modified are added.
    expected_edge_sizes = [es + ns for es, ns in zip(edge_sizes, node_sizes)]
    self.assertAllEqual(expected_edge_sizes, out_graph.edge_sets['edge'].sizes)

    # Assert: each component has original edges with self-loops added.
    offset_edges = 0
    offset_new_edges = 0
    offset_nodes = 0
    all_edges = tf.stack([
        out_graph.edge_sets['edge'].adjacency.source,
        out_graph.edge_sets['edge'].adjacency.target,
    ], 1)
    out_features = out_graph.edge_sets['edge'].features['feats']
    for ns, es, (src, trgt) in zip(node_sizes, edge_sizes, component_edges):
      new_es = es + ns  # i.e., expected_edge_sizes[.]
      out_component_edges = (
          all_edges[offset_new_edges : offset_new_edges + new_es])
      expected_edges = list(zip(src, trgt))
      expected_edges.extend(
          [(i, i) for i in range(offset_nodes, offset_nodes + ns)])

      if expected_edges:
        # Assert that adjacency contains original and self-loop edges.
        self.assertAllEqual(expected_edges, out_component_edges)
      else:
        self.assertEqual(0, out_component_edges.shape[0])

      # Assert that features are copied correctly.
      out_component_features = (
          out_features[offset_new_edges : offset_new_edges + new_es])
      expected_component_features = (
          edge_features[offset_edges : offset_edges + es])
      expected_component_features = tf.concat([
          expected_component_features,
          tf.zeros_like(out_component_features[-ns:]),
      ], 0)
      self.assertAllEqual(out_component_features, expected_component_features)

      offset_new_edges += new_es
      offset_nodes += ns
      offset_edges += es

    self.assertAllEqual(offset_nodes, sum(node_sizes))
    self.assertAllEqual(offset_edges, sum(edge_sizes))
    self.assertAllEqual(offset_new_edges, sum(expected_edge_sizes))


class _ReorderNodes(tf.keras.layers.Layer):

  def call(self, graph, permutations):
    return ops.reorder_nodes(graph, permutations)


class ReorderNodesTest(tf.test.TestCase, parameterized.TestCase):
  _heterogeneous = gt.GraphTensor.from_pieces(
      node_sets={
          'A':
              gt.NodeSet.from_fields(
                  sizes=[2, 2], features={
                      's': ['a', 'b', 'c', 'd'],
                  }),
          'B':
              gt.NodeSet.from_fields(
                  sizes=[2, 1], features={
                      's': ['x', 'y', 'z'],
                  }),
      },
      edge_sets={
          'A->B':
              gt.EdgeSet.from_fields(
                  sizes=[2, 2],
                  adjacency=adj.Adjacency.from_indices(
                      source=('A', [0, 1, 2, 3]),
                      target=('B', [0, 1, 2, 0]),
                  )),
      })

  def testEmpty(self):
    graph = gt.GraphTensor.from_pieces(
        node_sets={'node': gt.NodeSet.from_fields(sizes=[0], features={})})

    result = ops.reorder_nodes(graph,
                               {'node': tf.convert_to_tensor([], tf.int32)})
    self.assertEqual(list(result.edge_sets.keys()), [])
    self.assertEqual(list(result.node_sets.keys()), ['node'])
    self.assertEmpty(result.node_sets['node'].features)
    self.assertEmpty(result.context.features)
    self.assertAllEqual(result.node_sets['node'].sizes, [0])

  @parameterized.named_parameters([('adjacency',
                                    adj.Adjacency.from_indices(
                                        source=('node', [0, 2, 1]),
                                        target=('node', [0, 1, 2]),
                                    )),
                                   ('hyper_adjacency',
                                    adj.HyperAdjacency.from_indices({
                                        const.SOURCE: ('node', [0, 2, 1]),
                                        const.TARGET: ('node', [0, 1, 2]),
                                    }))])
  def testFeaturesReorder(self, adjacency):
    graph = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(features={'s': ['x']}),
        node_sets={
            'node':
                gt.NodeSet.from_fields(
                    sizes=[3],
                    features={
                        's': ['a', 'b', 'c'],
                        'v': [[1, 2], [3, 4], [5, 6]],
                        'r': as_ragged([[], [1], [2, 3]]),
                    })
        },
        edge_sets={
            'edge':
                gt.EdgeSet.from_fields(
                    sizes=[3], adjacency=adjacency, features={'s': [1, 2, 3]}),
        })

    result = ops.reorder_nodes(graph, {'node': [1, 2, 0]})
    self.assertAllEqual(result.context['s'], ['x'])

    edge_set = result.edge_sets['edge']
    self.assertAllEqual(edge_set.sizes, [3])
    self.assertAllEqual(edge_set['s'], [1, 2, 3])
    self.assertAllEqual(edge_set.adjacency[const.SOURCE], [2, 1, 0])
    self.assertAllEqual(edge_set.adjacency[const.TARGET], [2, 0, 1])

    node_set = result.node_sets['node']
    self.assertAllEqual(node_set.sizes, [3])
    self.assertAllEqual(node_set['s'], ['b', 'c', 'a'])
    self.assertAllEqual(node_set['v'], [[3, 4], [5, 6], [1, 2]])
    self.assertAllEqual(node_set['r'], as_ragged([[1], [2, 3], []]))

  @parameterized.parameters([
      dict(permutations={
          'A': [0, 1, 2, 3],
          'B': [0, 1, 2]
      }),
      dict(permutations={'A': [0, 1, 2, 3]}),
      dict(permutations={'B': [0, 1, 2]}),
      dict(permutations={})
  ])
  def testNoPermutation(self, permutations):
    graph = self._heterogeneous
    result = ops.reorder_nodes(graph, permutations)
    self.assertAllEqual(result.node_sets['A']['s'], graph.node_sets['A']['s'])
    self.assertAllEqual(result.node_sets['B']['s'], graph.node_sets['B']['s'])
    self.assertAllEqual(result.edge_sets['A->B'].adjacency.source,
                        graph.edge_sets['A->B'].adjacency.source)
    self.assertAllEqual(result.edge_sets['A->B'].adjacency.target,
                        graph.edge_sets['A->B'].adjacency.target)

  def testPermutations(self):
    graph = self._heterogeneous
    result = ops.reorder_nodes(graph, {'A': [3, 2, 1, 0]})
    self.assertAllEqual(result.node_sets['A'].sizes, [2, 2])
    self.assertAllEqual(result.node_sets['A']['s'], ['d', 'c', 'b', 'a'])
    self.assertAllEqual(result.node_sets['B'].sizes, [2, 1])
    self.assertAllEqual(result.node_sets['B']['s'], ['x', 'y', 'z'])
    self.assertAllEqual(result.edge_sets['A->B'].sizes, [2, 2])
    self.assertAllEqual(result.edge_sets['A->B'].adjacency.source, [3, 2, 1, 0])
    self.assertAllEqual(result.edge_sets['A->B'].adjacency.target, [0, 1, 2, 0])

    result = ops.reorder_nodes(graph, {'B': [0, 2, 1]})
    self.assertAllEqual(result.node_sets['A']['s'], ['a', 'b', 'c', 'd'])
    self.assertAllEqual(result.node_sets['B']['s'], ['x', 'z', 'y'])
    self.assertAllEqual(result.edge_sets['A->B'].adjacency.source, [0, 1, 2, 3])
    self.assertAllEqual(result.edge_sets['A->B'].adjacency.target, [0, 2, 1, 0])

    result = ops.reorder_nodes(graph, {'A': [2, 3, 0, 1], 'B': [2, 0, 1]})
    self.assertAllEqual(result.node_sets['A']['s'], ['c', 'd', 'a', 'b'])
    self.assertAllEqual(result.node_sets['B']['s'], ['z', 'x', 'y'])
    self.assertAllEqual(result.edge_sets['A->B'].adjacency.source, [2, 3, 0, 1])
    self.assertAllEqual(result.edge_sets['A->B'].adjacency.target, [1, 2, 0, 1])

  def testTFLite(self):
    test_graph_dict = {'fa': tf.range(10)}
    inputs = {'fa': tf.keras.Input([1], None, 'fa', tf.int32)}
    graph_in = _MakeGraphTensor()(inputs)
    node_permuter = _ReorderNodes()
    graph_out = node_permuter(
        graph_in, {'a': tf.constant([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
                   'b': tf.constant([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])})
    outputs = tf.keras.layers.Layer(name='final_edge_adjacency')(
        graph_out.edge_sets['aa'].adjacency[const.SOURCE]
    )
    model = tf.keras.Model(inputs, outputs)
    expected = model(test_graph_dict).numpy()

    # TODO(b/276291104): Remove when TF 2.11+ is required by all of TFGNN
    if tf.__version__.startswith('2.10.'):
      self.skipTest('GNN models are unsupported in TFLite until TF 2.11 but '
                    f'got TF {tf.__version__}')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    model_content = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=model_content)
    signature_runner = interpreter.get_signature_runner('serving_default')
    obtained = signature_runner(**test_graph_dict)['final_edge_adjacency']
    self.assertAllEqual(expected, obtained)

_SEED = 42


class ShuffleNodesTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([('adjacency',
                                    adj.Adjacency.from_indices(
                                        source=('node', [1, 0, 0]),
                                        target=('node', [0, 1, 0]),
                                    )),
                                   ('hyper_adjacency',
                                    adj.HyperAdjacency.from_indices({
                                        const.SOURCE: ('node', [1, 0, 0]),
                                        const.TARGET: ('node', [0, 1, 0]),
                                    }))])
  def testSingletonHomogeneous(self, adjacency):
    graph = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(features={'s': ['x']}),
        node_sets={
            'node':
                gt.NodeSet.from_fields(
                    sizes=[2],
                    features={
                        's': ['a', 'b'],
                        'v': [[1, 2], [3, 4]],
                        'r': as_ragged([[1], [2, 3]]),
                    }),
            '_readout':  # To be ignored.
                gt.NodeSet.from_fields(
                    sizes=[2],
                    features={'label': [42, 105]}),
        },
        edge_sets={
            'edge':
                gt.EdgeSet.from_fields(
                    sizes=[3], adjacency=adjacency, features={'s': [1, 2, 3]}),
            '_readout/thingy':
                gt.EdgeSet.from_fields(
                    sizes=[2],
                    adjacency=adj.Adjacency.from_indices(
                        source=('node', [0, 1]),
                        target=('_readout', [0, 1]))),
        })
    a_first_count = 0
    for _ in range(30):
      result = ops.shuffle_nodes(graph, seed=_SEED)
      self.assertAllEqual(result.context['s'], ['x'])
      edge_set = result.edge_sets['edge']
      self.assertAllEqual(edge_set.features['s'], [1, 2, 3])

      features = result.node_sets['node'].features
      if features['s'][0] == 'a':
        a_first_count += 1
        self.assertAllEqual(features['s'], ['a', 'b'])
        self.assertAllEqual(features['v'], [[1, 2], [3, 4]])
        self.assertAllEqual(features['r'], as_ragged([[1], [2, 3]]))
        self.assertAllEqual(edge_set.adjacency[const.SOURCE], [1, 0, 0])
        self.assertAllEqual(edge_set.adjacency[const.TARGET], [0, 1, 0])
      else:
        self.assertAllEqual(features['s'], ['b', 'a'])
        self.assertAllEqual(features['v'], [[3, 4], [1, 2]])
        self.assertAllEqual(features['r'], as_ragged([[2, 3], [1]]))
        self.assertAllEqual(edge_set.adjacency[const.SOURCE], [0, 1, 1])
        self.assertAllEqual(edge_set.adjacency[const.TARGET], [1, 0, 1])

      # Readout is invariant to the shuffling of "all" node sets by default.
      self.assertAllEqual(result.node_sets['_readout']['label'], [42, 105])
      self.assertAllEqual(
          readout.structured_readout(result, 'thingy', feature_name='s'),
          ['a', 'b'])

    self.assertBetween(a_first_count, 1, 29)

  @parameterized.parameters([
      dict(node_sets=()),
      dict(node_sets=('1')),
      dict(node_sets=('2')),
      dict(node_sets=('1', '2')),
      dict(node_sets=None)
  ])
  def testSingletonHeterogeneous(self, node_sets):
    graph = gt.GraphTensor.from_pieces(
        node_sets={
            '1': gt.NodeSet.from_fields(sizes=[2], features={
                's': ['a', 'b'],
            }),
            '2': gt.NodeSet.from_fields(sizes=[2], features={
                's': ['x', 'y'],
            }),
        },
        edge_sets={
            'edge':
                gt.EdgeSet.from_fields(
                    sizes=[3],
                    adjacency=adj.Adjacency.from_indices(
                        source=('1', [1, 0, 0]),
                        target=('2', [0, 1, 0]),
                    )),
        })
    counts = collections.Counter()
    for _ in range(30):
      result = ops.shuffle_nodes(graph, node_sets=node_sets, seed=_SEED)
      case = (result.node_sets['1'].features['s'][0].numpy().decode(),
              result.node_sets['2'].features['s'][0].numpy().decode())
      counts[case] += 1

      adjacency = result.edge_sets['edge'].adjacency
      if case[0] == 'a':
        self.assertAllEqual(adjacency.source, [1, 0, 0])
      else:
        self.assertAllEqual(adjacency.source, [0, 1, 1])
      if case[1] == 'x':
        self.assertAllEqual(adjacency.target, [0, 1, 0])
      else:
        self.assertAllEqual(adjacency.target, [1, 0, 1])

    if node_sets is None:
      node_sets = ('1', '2')

    if '1' in node_sets:
      self.assertBetween(counts[('b', 'x')], 1, 29, msg=str(counts))
      self.assertBetween(counts[('a', 'x')], 1, 29)
    else:
      self.assertEqual(counts[('b', 'x')], 0)
      self.assertEqual(counts[('b', 'y')], 0)

    if '2' in node_sets:
      self.assertBetween(counts[('a', 'y')], 1, 29)
      self.assertBetween(counts[('a', 'x')], 1, 29)
    else:
      self.assertEqual(counts[('a', 'y')], 0)
      self.assertEqual(counts[('b', 'y')], 0)

  def testMultipleComponents(self):
    graph = gt.GraphTensor.from_pieces(
        node_sets={
            'node':
                gt.NodeSet.from_fields(
                    sizes=[2, 2], features={
                        's': [1, 2, 3, 4],
                    })
        },
        edge_sets={
            'edge':
                gt.EdgeSet.from_fields(
                    sizes=[2, 2],
                    adjacency=adj.Adjacency.from_indices(
                        source=('node', [1, 0, 2, 3]),
                        target=('node', [0, 1, 3, 2]),
                    ),
                    features={'s': [1, 2, 3]}),
        })
    unique_permutations = set()
    for _ in range(30):
      result = ops.shuffle_nodes(graph, seed=_SEED)
      permutation = tuple(result.node_sets['node'].features['s'].numpy())
      unique_permutations.add(permutation)
      adjacency = result.edge_sets['edge'].adjacency
      if permutation == (1, 2, 3, 4):
        self.assertAllEqual(adjacency.source, [1, 0, 2, 3])
        self.assertAllEqual(adjacency.target, [0, 1, 3, 2])
      elif permutation == (2, 1, 3, 4):
        self.assertAllEqual(adjacency.source, [0, 1, 2, 3])
        self.assertAllEqual(adjacency.target, [1, 0, 3, 2])
      elif permutation == (1, 2, 4, 3):
        self.assertAllEqual(adjacency.source, [1, 0, 3, 2])
        self.assertAllEqual(adjacency.target, [0, 1, 2, 3])
      elif permutation == (2, 1, 4, 3):
        self.assertAllEqual(adjacency.source, [0, 1, 3, 2])
        self.assertAllEqual(adjacency.target, [1, 0, 2, 3])
      else:
        assert False, permutation

    self.assertLen(unique_permutations, 4)

  def testTFLite(self):
    # TODO(b/277938756): Implement when tf.random.shuffle is supported.
    self.skipTest('Shuffling ops are unsupported in TFLite.')


class NodeDegreeTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for computing degree of each node w.r.t. one side of an edge set."""

  @parameterized.parameters([
      dict(
          description='varying degrees for receiver nodes',
          node_sets={
              'a': gt.NodeSet.from_fields(
                  sizes=as_tensor([4]),
                  features={'s': [1, 2, 3, 4]}),
              'b': gt.NodeSet.from_fields(
                  sizes=as_tensor([3]),
                  features={}),
              'c': gt.NodeSet.from_fields(
                  sizes=as_tensor([3]),
                  features={})
          },
          edge_sets={
              'a->b':
                  gt.EdgeSet.from_fields(
                      sizes=as_tensor([8]),
                      adjacency=adj.Adjacency.from_indices(
                          ('a', as_tensor([0, 0, 1, 1, 2, 2, 3, 3])),
                          ('b', as_tensor([0, 1, 0, 1, 0, 1, 1, 2])),
                      )),
              'a->c':
                  gt.EdgeSet.from_fields(
                      sizes=as_tensor([5]),
                      adjacency=adj.Adjacency.from_indices(
                          ('a', as_tensor([0, 0, 0, 2, 2])),
                          ('c', as_tensor([2, 1, 0, 0, 0])),
                      )),
              'b->c':
                  gt.EdgeSet.from_fields(
                      sizes=as_tensor([0]),
                      adjacency=adj.Adjacency.from_indices(
                          ('b', as_tensor([], dtype=tf.int32)),
                          ('c', as_tensor([], dtype=tf.int32)),
                      )),
          },
          expected_source_degree={'a->b': as_tensor([2, 2, 2, 2]),
                                  'a->c': as_tensor([3, 0, 2, 0]),
                                  'b->c': as_tensor([0, 0, 0])},
          expected_target_degree={'a->b': as_tensor([3, 4, 1]),
                                  'a->c': as_tensor([3, 1, 1]),
                                  'b->c': as_tensor([0, 0, 0])}),
  ])
  def test(
      self, description: str,
      node_sets: Mapping[str, gt.NodeSet],
      edge_sets: Mapping[str, gt.EdgeSet],
      expected_source_degree: Mapping[const.EdgeSetName, const.Field],
      expected_target_degree: Mapping[const.EdgeSetName, const.Field]):
    del description
    graph = gt.GraphTensor.from_pieces(node_sets=node_sets, edge_sets=edge_sets)

    for edge_set_name, expected in expected_source_degree.items():
      get = ops.node_degree(graph, edge_set_name, const.SOURCE)
      self.assertAllEqual(get, expected)

    for edge_set_name, expected in expected_target_degree.items():
      get = ops.node_degree(graph, edge_set_name, const.TARGET)
      self.assertAllEqual(get, expected)


class _MaskEdges(tf.keras.layers.Layer):

  def call(self, graph, edge_set_name, mask, masked_edge_set_name):
    return ops.mask_edges(graph, edge_set_name, mask, masked_edge_set_name)


class EdgeMaskingTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for edge-masking operations over a scalar GraphTensor."""

  @parameterized.named_parameters(
      dict(
          testcase_name='EdgeSetWVariousFeatures',
          graph=gt.GraphTensor.from_pieces(
              node_sets={
                  'a':
                      gt.NodeSet.from_fields(
                          features={'f': as_tensor([1., 2.])},
                          sizes=as_tensor([2])),
                  'b':
                      gt.NodeSet.from_fields(features={}, sizes=as_tensor([4])),
              },
              edge_sets={
                  'a->b':
                      gt.EdgeSet.from_fields(
                          features={
                              'f': as_tensor([1., 2., 3.]),
                              'r': as_ragged([[4., 5.], [], [1., 2., 3.]])
                          },
                          sizes=as_tensor([3]),
                          adjacency=adj.Adjacency.from_indices(
                              ('a', as_tensor([0, 1, 1])),
                              ('b', as_tensor([0, 1, 3])),
                          )),
              }),
          mask=tf.convert_to_tensor([True, True, False]),
          edge_set_name='a->b',
          masked_edge_set_name='masked_a->b',
          expected=gt.GraphTensor.from_pieces(
              node_sets={
                  'a':
                      gt.NodeSet.from_fields(
                          features={'f': as_tensor([1., 2.])},
                          sizes=as_tensor([2])),
                  'b':
                      gt.NodeSet.from_fields(features={}, sizes=as_tensor([4])),
              },
              edge_sets={
                  'a->b':
                      gt.EdgeSet.from_fields(
                          features={
                              'f': as_tensor([1., 2.]),
                              'r': as_ragged([[4., 5.], []])
                          },
                          sizes=as_tensor([2]),
                          adjacency=adj.Adjacency.from_indices(
                              ('a', as_tensor([0, 1])),
                              ('b', as_tensor([0, 1])),
                          )),
                  'masked_a->b':
                      gt.EdgeSet.from_fields(
                          features={
                              'f': as_tensor([3.]),
                              'r': as_ragged([[1., 2., 3.]])
                          },
                          sizes=as_tensor([1]),
                          adjacency=adj.Adjacency.from_indices(
                              ('a', as_tensor([1])),
                              ('b', as_tensor([3])),
                          ))
              })),
      dict(
          testcase_name='EdgeSetWMultiComponents',
          graph=gt.GraphTensor.from_pieces(
              node_sets={
                  'a':
                      gt.NodeSet.from_fields(
                          features={'f': as_tensor([1., 2., 7., 19., 13.])},
                          sizes=as_tensor([3, 2])),
                  'b':
                      gt.NodeSet.from_fields(
                          features={}, sizes=as_tensor([4, 2])),
              },
              edge_sets={
                  'a->b':
                      gt.EdgeSet.from_fields(
                          features={
                              'f':
                                  as_tensor([1., 2., 3., 4., 7.]),
                              'r':
                                  as_ragged([[4., 5.], [], [1., 2., 3.], [3.],
                                             [9., 0., 1.]])
                          },
                          sizes=as_tensor([3, 2]),
                          adjacency=adj.Adjacency.from_indices(
                              ('a', as_tensor([0, 1, 1, 3, 4])),
                              ('b', as_tensor([0, 1, 3, 4, 5])),
                          )),
              }),
          mask=tf.convert_to_tensor([True, True, False, True, False]),
          edge_set_name='a->b',
          masked_edge_set_name='masked_a->b',
          expected=gt.GraphTensor.from_pieces(
              node_sets={
                  'a':
                      gt.NodeSet.from_fields(
                          features={'f': as_tensor([1., 2., 7., 19., 13.])},
                          sizes=as_tensor([3, 2])),
                  'b':
                      gt.NodeSet.from_fields(
                          features={}, sizes=as_tensor([4, 2])),
              },
              edge_sets={
                  'a->b':
                      gt.EdgeSet.from_fields(
                          features={
                              'f': as_tensor([1., 2., 4.]),
                              'r': as_ragged([[4., 5.], [], [3.]])
                          },
                          sizes=as_tensor([2, 1]),
                          adjacency=adj.Adjacency.from_indices(
                              ('a', as_tensor([0, 1, 3])),
                              ('b', as_tensor([0, 1, 4])),
                          )),
                  'masked_a->b':
                      gt.EdgeSet.from_fields(
                          features={
                              'f': as_tensor([3., 7.]),
                              'r': as_ragged([[1., 2., 3.], [9., 0., 1.]])
                          },
                          sizes=as_tensor([1, 1]),
                          adjacency=adj.Adjacency.from_indices(
                              ('a', as_tensor([1, 4])),
                              ('b', as_tensor([3, 5])),
                          ))
              })),
      dict(
          testcase_name='ZeroEdgesMaskedOut',
          graph=gt.GraphTensor.from_pieces(
              node_sets={
                  'a':
                      gt.NodeSet.from_fields(
                          features={'f': as_tensor([1., 2.])},
                          sizes=as_tensor([2])),
                  'b':
                      gt.NodeSet.from_fields(features={}, sizes=as_tensor([4])),
              },
              edge_sets={
                  'a->b':
                      gt.EdgeSet.from_fields(
                          features={
                              'f': as_tensor([1., 2., 3.]),
                              'r': as_ragged([[4., 5.], [], [1., 2., 3.]])
                          },
                          sizes=as_tensor([3]),
                          adjacency=adj.Adjacency.from_indices(
                              ('a', as_tensor([0, 1, 1])),
                              ('b', as_tensor([0, 1, 3])),
                          )),
              }),
          mask=tf.convert_to_tensor([True, True, True]),
          edge_set_name='a->b',
          masked_edge_set_name='masked_a->b',
          expected=gt.GraphTensor.from_pieces(
              node_sets={
                  'a':
                      gt.NodeSet.from_fields(
                          features={'f': as_tensor([1., 2.])},
                          sizes=as_tensor([2])),
                  'b':
                      gt.NodeSet.from_fields(features={}, sizes=as_tensor([4])),
              },
              edge_sets={
                  'a->b':
                      gt.EdgeSet.from_fields(
                          features={
                              'f': as_tensor([1., 2., 3.]),
                              'r': as_ragged([[4., 5.], [], [1., 2., 3.]])
                          },
                          sizes=as_tensor([3]),
                          adjacency=adj.Adjacency.from_indices(
                              ('a', as_tensor([0, 1, 1])),
                              ('b', as_tensor([0, 1, 3])),
                          )),
                  'masked_a->b':
                      gt.EdgeSet.from_fields(
                          features={
                              'f': as_tensor([]),
                              'r': as_ragged([])
                          },
                          sizes=as_tensor([0]),
                          adjacency=adj.Adjacency.from_indices(
                              ('a', as_tensor([], dtype=tf.int32)),
                              ('b', as_tensor([], dtype=tf.int32)),
                          ))
              })),
      dict(
          testcase_name='AllEdgesMaskedOut',
          graph=gt.GraphTensor.from_pieces(
              node_sets={
                  'a':
                      gt.NodeSet.from_fields(
                          features={'f': as_tensor([1., 2.])},
                          sizes=as_tensor([2])),
                  'b':
                      gt.NodeSet.from_fields(features={}, sizes=as_tensor([4])),
              },
              edge_sets={
                  'a->b':
                      gt.EdgeSet.from_fields(
                          features={
                              'f': as_tensor([1., 2., 3.]),
                              'r': as_ragged([[4., 5.], [], [1., 2., 3.]])
                          },
                          sizes=as_tensor([3]),
                          adjacency=adj.Adjacency.from_indices(
                              ('a', as_tensor([0, 1, 1])),
                              ('b', as_tensor([0, 1, 3])),
                          )),
              }),
          mask=tf.convert_to_tensor([False, False, False]),
          edge_set_name='a->b',
          masked_edge_set_name='masked_a->b',
          expected=gt.GraphTensor.from_pieces(
              node_sets={
                  'a':
                      gt.NodeSet.from_fields(
                          features={'f': as_tensor([1., 2.])},
                          sizes=as_tensor([2])),
                  'b':
                      gt.NodeSet.from_fields(features={}, sizes=as_tensor([4])),
              },
              edge_sets={
                  'a->b':
                      gt.EdgeSet.from_fields(
                          features={
                              'f': as_tensor([]),
                              'r': as_ragged([])
                          },
                          sizes=as_tensor([0]),
                          adjacency=adj.Adjacency.from_indices(
                              ('a', as_tensor([], dtype=tf.int32)),
                              ('b', as_tensor([], dtype=tf.int32)),
                          )),
                  'masked_a->b':
                      gt.EdgeSet.from_fields(
                          features={
                              'f': as_tensor([1., 2., 3.]),
                              'r': as_ragged([[4., 5.], [], [1., 2., 3.]])
                          },
                          sizes=as_tensor([3]),
                          adjacency=adj.Adjacency.from_indices(
                              ('a', as_tensor([0, 1, 1])),
                              ('b', as_tensor([0, 1, 3])),
                          ))
              })),
      dict(
          testcase_name='EmptyNodesAndMask',
          graph=gt.GraphTensor.from_pieces(
              node_sets={
                  'a':
                      gt.NodeSet.from_fields(features={}, sizes=as_tensor([0])),
                  'b':
                      gt.NodeSet.from_fields(features={}, sizes=as_tensor([0])),
              },
              edge_sets={
                  'a->b':
                      gt.EdgeSet.from_fields(
                          features={
                              'f': as_tensor([]),
                              'r': as_ragged([])
                          },
                          sizes=as_tensor([0]),
                          adjacency=adj.Adjacency.from_indices(
                              ('a', as_tensor([], dtype=tf.int32)),
                              ('b', as_tensor([], dtype=tf.int32)),
                          )),
              }),
          mask=tf.convert_to_tensor([], dtype=tf.bool),
          edge_set_name='a->b',
          masked_edge_set_name='masked_a->b',
          expected=gt.GraphTensor.from_pieces(
              node_sets={
                  'a':
                      gt.NodeSet.from_fields(features={}, sizes=as_tensor([0])),
                  'b':
                      gt.NodeSet.from_fields(features={}, sizes=as_tensor([0])),
              },
              edge_sets={
                  'a->b':
                      gt.EdgeSet.from_fields(
                          features={
                              'f': as_tensor([]),
                              'r': as_ragged([])
                          },
                          sizes=as_tensor([0]),
                          adjacency=adj.Adjacency.from_indices(
                              ('a', as_tensor([], dtype=tf.int32)),
                              ('b', as_tensor([], dtype=tf.int32)),
                          )),
                  'masked_a->b':
                      gt.EdgeSet.from_fields(
                          features={
                              'f': as_tensor([]),
                              'r': as_ragged([])
                          },
                          sizes=as_tensor([0]),
                          adjacency=adj.Adjacency.from_indices(
                              ('a', as_tensor([], dtype=tf.int32)),
                              ('b', as_tensor([], dtype=tf.int32)),
                          ))
              })))
  def test(self, graph, mask, edge_set_name, masked_edge_set_name, expected):
    actual = ops.mask_edges(graph, edge_set_name, mask, masked_edge_set_name)

    for node_set_name in [*expected.node_sets, *actual.node_sets]:
      self.assertAllEqual(actual.node_sets[node_set_name].sizes,
                          expected.node_sets[node_set_name].sizes)
      self.assertAllEqual(actual.node_sets[node_set_name].features,
                          expected.node_sets[node_set_name].features)
    for edge_set_name in [*expected.edge_sets, *actual.edge_sets]:
      tf.print(actual.edge_sets[edge_set_name])
      self.assertAllEqual(actual.edge_sets[edge_set_name].sizes,
                          expected.edge_sets[edge_set_name].sizes)
      self.assertAllEqual(actual.edge_sets[edge_set_name].features,
                          expected.edge_sets[edge_set_name].features)
      self.assertAllEqual(actual.edge_sets[edge_set_name].adjacency.source,
                          expected.edge_sets[edge_set_name].adjacency.source)
      self.assertAllEqual(actual.edge_sets[edge_set_name].adjacency.target,
                          expected.edge_sets[edge_set_name].adjacency.target)

  def testNonExistent(self):
    graph = gt.GraphTensor.from_pieces(
        node_sets={
            'nodes':
                gt.NodeSet.from_fields(
                    features={'f': as_tensor([1., 2.])}, sizes=as_tensor([2])),
        },
        edge_sets={
            'edges':
                gt.EdgeSet.from_fields(
                    features={
                        'f': as_tensor([1., 2., 3.]),
                        'r': as_ragged([[4., 5.], [], [1., 2., 3.]])
                    },
                    sizes=as_tensor([3]),
                    adjacency=adj.Adjacency.from_indices(
                        ('a', as_tensor([0, 1, 1])),
                        ('b', as_tensor([0, 1, 3])),
                    )),
        })
    with self.assertRaisesRegex(
        ValueError,
        r'Please ensure edge_set_name: a->b exists as an edge-set.*'):
      ops.mask_edges(graph, 'a->b', as_tensor([True, False]), 'edges')

  def testError(self):
    graph = gt.GraphTensor.from_pieces(
        node_sets={
            'nodes':
                gt.NodeSet.from_fields(
                    features={'f': as_tensor([1., 2.])}, sizes=as_tensor([2])),
        },
        edge_sets={
            'edges':
                gt.EdgeSet.from_fields(
                    features={
                        'f': as_tensor([1., 2., 3.]),
                        'r': as_ragged([[4., 5.], [], [1., 2., 3.]])
                    },
                    sizes=as_tensor([3]),
                    adjacency=adj.Adjacency.from_indices(
                        ('a', as_tensor([0, 1, 1])),
                        ('b', as_tensor([0, 1, 3])),
                    )),
        })
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        r'boolean_edge_mask should have the same shape with the adjacency.*'):
      ops.mask_edges(graph, 'edges', as_tensor([True, False]), 'masked')

  def testTFLite(self):
    test_graph_dict = {'fa': tf.range(10)}
    inputs = {'fa': tf.keras.Input([1], None, 'fa', tf.int32)}
    graph_in = _MakeGraphTensor()(inputs)
    edge_masker = _MaskEdges()
    graph_out = edge_masker(
        graph_in,
        'aa',
        tf.convert_to_tensor([True, False, True, True]),
        'masked_aa')
    outputs = tf.keras.layers.Layer(name='final_edge_adjacency')(
        graph_out.edge_sets['masked_aa'].adjacency[const.SOURCE]
    )
    model = tf.keras.Model(inputs, outputs)
    expected = model(test_graph_dict).numpy()

    # TODO(b/276291104): Remove when TF 2.11+ is required by all of TFGNN
    if tf.__version__.startswith('2.10.'):
      self.skipTest('GNN models are unsupported in TFLite until TF 2.11 but '
                    f'got TF {tf.__version__}')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    model_content = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=model_content)
    signature_runner = interpreter.get_signature_runner('serving_default')
    obtained = signature_runner(**test_graph_dict)['final_edge_adjacency']
    self.assertAllEqual(expected, obtained)


class _MakeLineGraphTargetToSource(tf.keras.layers.Layer):

  def call(self, graph):
    return ops.convert_to_line_graph(
        graph,
        connect_from=const.TARGET,
        connect_to=const.SOURCE,)


class LineGraphTest(tf.test.TestCase, parameterized.TestCase):

  def _make_test_graph(self, add_readout=False):
    return _MakeGraphTensor()(
        {'fa': tf.range(10)},
        add_readout=add_readout
    )

  def testContextFeatures(self):
    graph_tensor = self._make_test_graph()
    line_graph = ops.convert_to_line_graph(graph_tensor)
    for feature_name, feature in graph_tensor.context.features.items():
      self.assertAllEqual(line_graph.context[feature_name], feature)

  def testNodeFeatures(self):
    graph_tensor = self._make_test_graph()
    line_graph = ops.convert_to_line_graph(graph_tensor)
    for edge_set_name, edge_set in graph_tensor.edge_sets.items():
      for feature_name, feature in edge_set.features.items():
        self.assertAllEqual(
            line_graph.node_sets[edge_set_name][feature_name], feature
        )

  def testEdgeFeatures(self):
    graph_tensor = self._make_test_graph()
    line_graph = ops.convert_to_line_graph(
        graph_tensor,
        use_node_features_as_line_graph_edge_features=True
    )
    self.assertCountEqual(
        line_graph.edge_sets,
        ['aa=>aa', 'aa=>ab', 'ab=>ba', 'ba=>aa', 'ba=>ab'],
    )

    def _assert_features_equal(
        node_set_name, edge_set_name, node_used_for_edge
    ):
      node_set = graph_tensor.node_sets[node_set_name]
      edge_set = line_graph.edge_sets[edge_set_name]
      for feature_name, node_feature in node_set.features.items():
        edge_feature = edge_set[feature_name]
        if node_used_for_edge:
          self.assertAllEqual(
              edge_feature, tf.gather(node_feature, node_used_for_edge)
          )
        else:
          self.assertEqual(tf.size(edge_feature), 0)

    _assert_features_equal('a', 'aa=>aa', [0, 0])
    _assert_features_equal('a', 'aa=>ab', [0, 1])
    _assert_features_equal('a', 'ba=>aa', [8])
    _assert_features_equal('a', 'ba=>ab', [7, 8])
    _assert_features_equal('b', 'ab=>ba', [0, 0, 1, 1, 1, 1, 1, 1, 9])

  def testResultIsAdjacency(self):
    graph_tensor = self._make_test_graph()
    line_graph = ops.convert_to_line_graph(
        graph_tensor,
        connect_from=const.TARGET,
        connect_to=const.SOURCE,
    )
    for edge_set in line_graph.edge_sets.values():
      self.assertIsInstance(edge_set.adjacency, adj.Adjacency)

  @parameterized.named_parameters([('', True), ('NonBacktracking', False)])
  def testConnectivityTargetToSource(self, backtracking: bool):
    graph_tensor = self._make_test_graph()
    line_graph = ops.convert_to_line_graph(
        graph_tensor,
        connect_from=const.TARGET,
        connect_to=const.SOURCE,
        non_backtracking=not backtracking,
    )
    self.assertCountEqual(
        line_graph.edge_sets,
        ['aa=>aa', 'aa=>ab', 'ab=>ba', 'ba=>aa', 'ba=>ab'],
    )
    self.assertAllEqual(
        line_graph.edge_sets['aa=>aa'].adjacency[const.SOURCE],
        tf.constant([0] * backtracking + [0]),
    )
    self.assertAllEqual(
        line_graph.edge_sets['aa=>aa'].adjacency[const.TARGET],
        tf.constant([0] * backtracking + [1]),
    )
    self.assertAllEqual(
        line_graph.edge_sets['aa=>ab'].adjacency[const.SOURCE],
        tf.constant([0, 1]),
    )
    self.assertAllEqual(
        line_graph.edge_sets['aa=>ab'].adjacency[const.TARGET],
        tf.constant([0, 2]),
    )
    self.assertAllEqual(
        line_graph.edge_sets['ab=>ba'].adjacency[const.SOURCE],
        tf.constant([1, 1, 0, 0, 0, 2, 2, 2] + [4] * backtracking),
    )
    self.assertAllEqual(
        line_graph.edge_sets['ab=>ba'].adjacency[const.TARGET],
        tf.constant([0, 3, 1, 2, 4, 1, 2, 4] + [5] * backtracking),
    )
    self.assertAllEqual(
        line_graph.edge_sets['ba=>aa'].adjacency[const.SOURCE],
        tf.constant([5]),
    )
    self.assertAllEqual(
        line_graph.edge_sets['ba=>aa'].adjacency[const.TARGET],
        tf.constant([3]),
    )
    self.assertAllEqual(
        line_graph.edge_sets['ba=>ab'].adjacency[const.SOURCE],
        tf.constant([0] + [5] * backtracking),
    )
    self.assertAllEqual(
        line_graph.edge_sets['ba=>ab'].adjacency[const.TARGET],
        tf.constant([3] + [4] * backtracking),
    )

  @parameterized.named_parameters([('', True), ('NonBacktracking', False)])
  def testConnectivityTargetToTarget(self, backtracking: bool):
    graph_tensor = self._make_test_graph()
    line_graph = ops.convert_to_line_graph(
        graph_tensor,
        connect_from=const.TARGET,
        connect_to=const.TARGET,
        non_backtracking=not backtracking,
    )
    self.assertCountEqual(
        line_graph.edge_sets,
        ['aa=>aa', 'aa=>ba', 'ab=>ab', 'ba=>aa', 'ba=>ba'],
    )
    self.assertAllEqual(
        line_graph.edge_sets['aa=>aa'].adjacency[const.SOURCE],
        tf.constant([0, 1, 2, 3] * backtracking),
    )
    self.assertAllEqual(
        line_graph.edge_sets['aa=>aa'].adjacency[const.TARGET],
        tf.constant([0, 1, 2, 3] * backtracking),
    )
    self.assertAllEqual(
        line_graph.edge_sets['aa=>ba'].adjacency[const.SOURCE],
        tf.constant([]),
    )
    self.assertAllEqual(
        line_graph.edge_sets['aa=>ba'].adjacency[const.TARGET],
        tf.constant([]),
    )
    self.assertAllEqual(
        line_graph.edge_sets['ab=>ab'].adjacency[const.SOURCE],
        tf.constant([1, 0] * backtracking + [0, 2] + [2, 3, 4] * backtracking),
    )
    self.assertAllEqual(
        line_graph.edge_sets['ab=>ab'].adjacency[const.TARGET],
        tf.constant([1, 0] * backtracking + [2, 0] + [2, 3, 4] * backtracking),
    )
    self.assertAllEqual(
        line_graph.edge_sets['ba=>aa'].adjacency[const.SOURCE],
        tf.constant([]),
    )
    self.assertAllEqual(
        line_graph.edge_sets['ba=>aa'].adjacency[const.TARGET],
        tf.constant([]),
    )
    self.assertAllEqual(
        line_graph.edge_sets['ba=>ba'].adjacency[const.SOURCE],
        tf.constant(
            [4, 2] * backtracking + [2, 3] + [3, 1, 0, 5] * backtracking
        ),
    )
    self.assertAllEqual(
        line_graph.edge_sets['ba=>ba'].adjacency[const.TARGET],
        tf.constant(
            [4, 2] * backtracking + [3, 2] + [3, 1, 0, 5] * backtracking
        ),
    )

  def testConnectOriginalFeatures(self):
    graph_tensor = self._make_test_graph()
    line_graph = ops.convert_to_line_graph(
        graph_tensor,
        connect_with_original_nodes=True,
    )
    for node_set_name, node_set in graph_tensor.node_sets.items():
      for feature_name, feature in node_set.features.items():
        self.assertAllEqual(
            line_graph.node_sets[f'original/{node_set_name}'][feature_name],
            feature,
        )

  def testOriginalToLineGraph(self):
    """The original/to/... edges have the correct endpoints."""
    graph_tensor = self._make_test_graph()
    line_graph = ops.convert_to_line_graph(
        graph_tensor,
        connect_with_original_nodes=True,
    )
    bcast = broadcast_ops.broadcast_node_to_edges(
        line_graph,
        'original/to/aa',
        const.SOURCE,
        feature_value=2 ** tf.range(10),
    )
    summed_messages = pool_ops.pool_edges_to_node(
        line_graph,
        'original/to/aa',
        const.TARGET,
        'sum',
        feature_value=bcast,
    )
    # In node set 'aa' of the line graph, each node receives the unique value
    # from the source node of the original edge it represents.
    self.assertAllEqual(
        summed_messages,
        2 ** graph_tensor.edge_sets['aa'].adjacency[const.SOURCE]
    )

  def testSendLineGraphToOriginal(self):
    """The original/from/... edges have the correct endpoints."""
    graph_tensor = self._make_test_graph()
    line_graph = ops.convert_to_line_graph(
        graph_tensor,
        connect_with_original_nodes=True,
    )
    bcast = broadcast_ops.broadcast_node_to_edges(
        line_graph,
        'original/from/ba',
        const.TARGET,
        feature_value=2 ** tf.range(10),
    )
    summed_messages = pool_ops.pool_edges_to_node(
        line_graph,
        'original/from/ba',
        const.SOURCE,
        'sum',
        feature_value=bcast,
    )
    # In node set 'ba' of the line graph, each node receives the unique value
    # from the target node of the original edge it represents.
    self.assertAllEqual(
        summed_messages,
        2 ** graph_tensor.edge_sets['ba'].adjacency[const.TARGET]
    )

  def testConnectOriginalEdges(self):
    graph_tensor = self._make_test_graph()
    line_graph = ops.convert_to_line_graph(
        graph_tensor,
        connect_with_original_nodes=True,
    )
    self.assertAllEqual(
        line_graph.edge_sets['original/to/aa'].adjacency[const.SOURCE],
        graph_tensor.edge_sets['aa'].adjacency[const.SOURCE],
    )
    self.assertAllEqual(
        line_graph.edge_sets['original/from/aa'].adjacency[const.TARGET],
        graph_tensor.edge_sets['aa'].adjacency[const.TARGET],
    )
    self.assertAllEqual(
        line_graph.edge_sets['original/to/ab'].adjacency[const.SOURCE],
        graph_tensor.edge_sets['ab'].adjacency[const.SOURCE],
    )
    self.assertAllEqual(
        line_graph.edge_sets['original/from/ab'].adjacency[const.TARGET],
        graph_tensor.edge_sets['ab'].adjacency[const.TARGET],
    )
    self.assertAllEqual(
        line_graph.edge_sets['original/to/ba'].adjacency[const.SOURCE],
        graph_tensor.edge_sets['ba'].adjacency[const.SOURCE],
    )
    self.assertAllEqual(
        line_graph.edge_sets['original/from/ba'].adjacency[const.TARGET],
        graph_tensor.edge_sets['ba'].adjacency[const.TARGET],
    )

  def testAuxiliaryNodeSetError(self):
    graph_tensor = self._make_test_graph(add_readout=True)
    edge_sets = dict(graph_tensor.edge_sets)
    del edge_sets['_readout/testE']
    graph_tensor = gt.GraphTensor.from_pieces(
        context=graph_tensor.context,
        node_sets=graph_tensor.node_sets,
        edge_sets=edge_sets,
    )
    with self.assertRaises(ValueError):
      ops.convert_to_line_graph(graph_tensor)

  def testAuxiliaryEdgeSetError(self):
    graph_tensor = self._make_test_graph(add_readout=True)
    node_sets = dict(graph_tensor.node_sets)
    del node_sets['_readout']
    graph_tensor = gt.GraphTensor.from_pieces(
        context=graph_tensor.context,
        node_sets=node_sets,
        edge_sets=graph_tensor.edge_sets,
    )
    with self.assertRaises(ValueError):
      ops.convert_to_line_graph(graph_tensor)

  def testNoAuxiliaryLineGraph(self):
    graph_tensor = self._make_test_graph(add_readout=True)
    line_graph = ops.convert_to_line_graph(
        graph_tensor, connect_with_original_nodes=True
    )
    self.assertCountEqual(
        line_graph.node_sets,
        ['aa', 'ab', 'ba', 'original/a', 'original/b', '_readout'],
    )
    self.assertCountEqual(
        line_graph.edge_sets,
        [
            'original/from/aa',
            'original/to/aa',
            'original/from/ab',
            'original/to/ab',
            'original/from/ba',
            'original/to/ba',
            '_readout/testE',
            'aa=>aa',
            'aa=>ab',
            'ab=>ba',
            'ba=>aa',
            'ba=>ab',
        ],
    )

  def testNonBacktrackingAuxiliary(self):
    graph_tensor = self._make_test_graph(add_readout=True)
    line_graph = ops.convert_to_line_graph(
        graph_tensor, connect_with_original_nodes=True
    )
    line_graph_nb = ops.convert_to_line_graph(
        graph_tensor, connect_with_original_nodes=True, non_backtracking=True
    )
    for edge_set_name in line_graph_nb.edge_sets:
      if edge_set_name not in [
          'aa=>aa',
          'aa=>ab',
          'ab=>ba',
          'ba=>aa',
          'ba=>ab',
      ]:
        self.assertAllEqual(
            line_graph.edge_sets[edge_set_name].adjacency.source,
            line_graph_nb.edge_sets[edge_set_name].adjacency.source,
        )
        self.assertAllEqual(
            line_graph.edge_sets[edge_set_name].adjacency.target,
            line_graph_nb.edge_sets[edge_set_name].adjacency.target,
        )

  def testStructuredReadout(self):
    graph_tensor = self._make_test_graph(add_readout=True)
    line_graph = ops.convert_to_line_graph(
        graph_tensor, connect_with_original_nodes=True
    )
    readout_f2 = readout.structured_readout(line_graph, 'testE',
                                            feature_name='f2')
    self.assertAllEqual(readout_f2, [100, 102, 101, 109])

  def testTFLite(self):
    test_graph_dict = {'fa': tf.range(10)}
    inputs = {'fa': tf.keras.Input([1], None, 'fa', tf.int32)}
    graph_in = _MakeGraphTensor()(inputs)
    linegraph_maker = _MakeLineGraphTargetToSource()
    graph_out = linegraph_maker(graph_in)
    outputs = tf.keras.layers.Layer(name='final_edge_adjacency')(
        graph_out.edge_sets['ab=>ba'].adjacency.source
    )
    model = tf.keras.Model(inputs, outputs)
    expected = model(test_graph_dict).numpy()

    # TODO(b/276291104): Remove when TF 2.11+ is required by all of TFGNN
    if tf.__version__.startswith('2.10.'):
      self.skipTest('GNN models are unsupported in TFLite until TF 2.11 but '
                    f'got TF {tf.__version__}')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    model_content = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=model_content)
    signature_runner = interpreter.get_signature_runner('serving_default')
    obtained = signature_runner(**test_graph_dict)['final_edge_adjacency']
    self.assertAllEqual(expected, obtained)


# TODO(b/274779989): Replace this layer with a more standard representation
# of GraphTensor as a dict of plain Tensors.
class _MakeGraphTensor(tf.keras.layers.Layer):

  def call(self, inputs, *, add_readout=False):
    node_sets = {
        'a': gt.NodeSet.from_fields(
            features={'fa': inputs['fa'], 'f2': 100 + tf.range(10)},
            sizes=as_tensor([8, 2]),
        ),
        'b': gt.NodeSet.from_fields(
            features={'fb': tf.range(10), 'f2': 100 + tf.range(10)},
            sizes=as_tensor([8, 2]),
        ),
    }
    edge_sets = {
        'aa': gt.EdgeSet.from_fields(
            features={'faa': tf.range(4), 'f2': 100 + tf.range(4)},
            sizes=as_tensor([3, 1]),
            adjacency=adj.HyperAdjacency.from_indices({
                const.SOURCE: ('a', as_tensor([0, 0, 2, 8])),
                const.TARGET: ('a', as_tensor([0, 1, 3, 9])),
            }),
        ),
        'ab': gt.EdgeSet.from_fields(
            features={'fab': tf.range(5), 'f2': 100 + tf.range(5)},
            sizes=as_tensor([4, 1]),
            adjacency=adj.HyperAdjacency.from_indices({
                const.SOURCE: ('a', as_tensor([0, 2, 1, 7, 8])),
                const.TARGET: ('b', as_tensor([1, 0, 1, 2, 9])),
            }),
        ),
        'ba': gt.EdgeSet.from_fields(
            features={'fba': tf.range(6), 'f2': 100 + tf.range(6)},
            sizes=as_tensor([5, 1]),
            adjacency=adj.HyperAdjacency.from_indices({
                const.SOURCE: ('b', as_tensor([0, 1, 1, 0, 1, 9])),
                const.TARGET: ('a', as_tensor([7, 6, 5, 5, 4, 8])),
            }),
        ),
    }
    if add_readout:
      node_sets['_readout'] = gt.NodeSet.from_fields(
          features={'f1': 100 + tf.range(4)},
          sizes=as_tensor([3, 1]),
      )
      edge_sets['_readout/testE'] = gt.EdgeSet.from_fields(
          features={'f1': 100 + tf.range(4)},
          sizes=as_tensor([3, 1]),
          adjacency=adj.HyperAdjacency.from_indices({
              const.SOURCE: ('b', as_tensor([0, 2, 1, 9])),
              const.TARGET: ('_readout', as_tensor([0, 1, 2, 3])),
          }),
      )
    return gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(
            features={'fc': as_tensor([10, 11]), 'f2': as_tensor([20, 21])}
        ),
        node_sets=node_sets,
        edge_sets=edge_sets,
    )


if __name__ == '__main__':
  tf.test.main()
