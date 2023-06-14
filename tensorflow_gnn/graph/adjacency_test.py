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
"""Tests for adjacency."""

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.graph import adjacency
from tensorflow_gnn.graph import graph_constants as const

as_tensor = tf.convert_to_tensor


class HyperAdjacencyTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([
      dict(
          description='rank-0, simple graph',
          indices={
              const.SOURCE: ('node', as_tensor([0, 1])),
              const.TARGET: ('node', as_tensor([1, 2])),
          },
          expected_shape=[]),
      dict(
          description='rank-0, hypergraph',
          indices={
              0: ('node', as_tensor([0, 1, 2])),
          },
          expected_shape=[]),
      dict(
          description='rank-1, variable size',
          indices={
              const.SOURCE: ('node.a', tf.ragged.constant([[0, 1], [0]])),
              const.TARGET: ('node.b', tf.ragged.constant([[1, 2], [1]])),
          },
          expected_shape=[2]),
      dict(
          description='rank-1, fixed size',
          indices={
              const.SOURCE: ('node.a', as_tensor([[0], [1], [2]])),
              const.TARGET: ('node.b', as_tensor([[0], [1], [2]])),
          },
          expected_shape=[3]),
  ])
  def testShapeResolution(self, description: str, indices: adjacency.Indices,
                          expected_shape: tf.TensorShape):
    result = adjacency.HyperAdjacency.from_indices(indices)
    self.assertEqual(result.shape.as_list(), expected_shape)

  @parameterized.parameters([
      dict(
          description='rank-0, sizes missmatch',
          indices={
              const.SOURCE: ('a', as_tensor([0, 1])),
              const.TARGET: ('b', as_tensor([1])),
          }),
      dict(
          description='rank-0, sizes missmatch for hyper-graph',
          indices={
              0: ('a', as_tensor([0, 1])),
              2: ('c', as_tensor([0, 1])),
              1: ('b', as_tensor([1])),
          }),
      dict(
          description='rank-1, dense',
          indices={
              const.SOURCE: ('a', as_tensor([[0, 1]])),
              const.TARGET: ('b', as_tensor([[0, 1], [2, 3]])),
          }),
      dict(
          description='rank-1, ragged value',
          indices={
              const.SOURCE: ('a', tf.ragged.constant([[0, 1], [0]])),
              const.TARGET: ('b', tf.ragged.constant([[1, 2], []])),
          }),
      dict(
          description='rank-1, ragged splits',
          indices={
              const.SOURCE: ('a', tf.ragged.constant([[0, 1], [0, 1]])),
              const.TARGET: ('b', tf.ragged.constant([[1, 2], [0], [1]])),
          }),
  ])
  def testRaisesOnIncompatibleIndices(self, description: str,
                                      indices: adjacency.Indices):
    self.assertRaisesRegex(
        Exception,
        r'Adjacency indices are not compatible: \(0, a\) and \(1, b\)',
        lambda: adjacency.HyperAdjacency.from_indices(indices))

  def testRaiseOnExtraPositionalArguments(self):
    self.assertRaisesRegex(
        TypeError, 'Positional arguments are not supported',
        lambda: adjacency.HyperAdjacency.from_indices((0, []), False))

  def testNodeSetName(self):
    adj = adjacency.HyperAdjacency.from_indices({
        const.SOURCE: ('node.a', tf.ragged.constant([[0, 1], [0]])),
        const.TARGET: ('node.b', tf.ragged.constant([[1, 2], [1]])),
    })
    self.assertAllEqual(adj.spec.node_set_name(const.SOURCE), 'node.a')
    self.assertAllEqual(adj.spec.node_set_name(const.TARGET), 'node.b')
    self.assertAllEqual(adj.node_set_name(const.SOURCE), 'node.a')
    self.assertAllEqual(adj.node_set_name(const.TARGET), 'node.b')

  def testGetIndices(self):
    indices = {
        const.SOURCE: ('node.a', as_tensor([0, 1])),
        const.TARGET: ('node.b', as_tensor([1, 0])),
    }
    adj = adjacency.HyperAdjacency.from_indices(indices)
    indices_dict = adj.get_indices_dict()
    for tag in (const.SOURCE, const.TARGET):
      self.assertAllEqual(indices[tag][0], indices_dict[tag][0])
      self.assertAllEqual(indices[tag][1], indices_dict[tag][1])

  def testIndices(self):
    adj = adjacency.HyperAdjacency.from_indices({
        const.SOURCE: ('node.a', as_tensor([0, 1, 2])),
        const.TARGET: ('node.b', as_tensor([2, 1, 0]))
    })
    self.assertAllEqual(adj[const.SOURCE], [0, 1, 2])
    self.assertAllEqual(adj[const.TARGET], [2, 1, 0])

  @parameterized.parameters([True, False])
  def testValidateArgSupport(self, validate):
    adj = adjacency.HyperAdjacency.from_indices(
        {
            const.SOURCE: ('node.a', as_tensor([0, 1, 2])),
            const.TARGET: ('node.b', as_tensor([2, 1, 0]))
        },
        validate=validate)
    self.assertAllEqual(adj[const.SOURCE], [0, 1, 2])
    self.assertAllEqual(adj[const.TARGET], [2, 1, 0])

  def testMergeFixedSizeBatchToComponents(self):
    adj = adjacency.HyperAdjacency.from_indices({
        const.SOURCE: ('node.a', as_tensor([[0, 1], [1, 0], [0, 1]])),
        const.TARGET: ('node.b', as_tensor([[1, 2], [1, 1], [0, 0]])),
    })
    result = adj._merge_batch_to_components(
        as_tensor([2, 2, 2]), {
            'node.a': as_tensor([2, 2, 2]),
            'node.b': as_tensor([3, 3, 3]),
            'node.c': as_tensor([4, 4, 4]),
        })
    self.assertAllEqual(result[const.SOURCE],
                        [0, 1, 1 + 2, 0 + 2, 0 + 4, 1 + 4])
    self.assertAllEqual(result[const.TARGET],
                        [1, 2, 1 + 3, 1 + 3, 0 + 6, 0 + 6])

  def testMergeRank1BatchToComponents(self):
    adj = adjacency.HyperAdjacency.from_indices({
        const.SOURCE: ('node.a', tf.ragged.constant([[0, 1], [1], [0]])),
        const.TARGET: ('node.b', tf.ragged.constant([[1, 2], [1], [1]])),
    })
    result = adj._merge_batch_to_components(
        as_tensor([2, 1, 1]), {
            'node.a': as_tensor([3, 2, 4]),
            'node.b': as_tensor([4, 3, 2]),
        })
    self.assertAllEqual(result[const.SOURCE], [0, 1, 1 + 3, 0 + 3 + 2])
    self.assertAllEqual(result[const.TARGET], [1, 2, 1 + 4, 1 + 4 + 3])

  def testMergeRank2BatchToComponents(self):
    adj = adjacency.HyperAdjacency.from_indices({
        0: ('node',
            tf.RaggedTensor.from_uniform_row_length(
                tf.ragged.constant([[0, 1], [1], [0], [0]]), 2)),
    })
    result = adj._merge_batch_to_components(
        as_tensor([2, 1, 1]), {
            'node': as_tensor([3, 2, 4, 1]),
        })
    self.assertAllEqual(result[0], [0, 1, 1 + 3, 0 + 3 + 2, 0 + 3 + 2 + 4])

  def testRelaxation(self):
    incident_node_sets = {0: 'a', 1: 'b', 2: 'c'}
    original = adjacency.HyperAdjacencySpec.from_incident_node_sets(
        incident_node_sets, index_spec=tf.TensorSpec([2], tf.int64))
    expected = adjacency.HyperAdjacencySpec.from_incident_node_sets(
        incident_node_sets, index_spec=tf.TensorSpec([None], tf.int64))
    self.assertEqual(original.relax(num_edges=True), expected)
    self.assertEqual(
        original.relax(num_edges=True).relax(num_edges=True), expected)


class AdjacencyTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([
      dict(
          description='rank-0, simple graph',
          source=('node', as_tensor([0, 1])),
          target=('node', as_tensor([1, 2])),
          expected_shape=[]),
      dict(
          description='rank-1, variable size',
          source=('node.a', tf.ragged.constant([[0, 1], [0]])),
          target=('node.b', tf.ragged.constant([[1, 2], [1]])),
          expected_shape=[2]),
      dict(
          description='rank-1, fixed size',
          source=('node.a', as_tensor([[0], [1], [2]])),
          target=('node.b', as_tensor([[0], [1], [2]])),
          expected_shape=[3]),
  ])
  def testShapeResolution(self, description: str, source: adjacency.Index,
                          target: adjacency.Index,
                          expected_shape: tf.TensorShape):
    result = adjacency.Adjacency.from_indices(source, target, validate=False)
    self.assertEqual(result.shape.as_list(), expected_shape)

  @parameterized.parameters([
      dict(
          description='rank-0, sizes missmatch',
          source=('a', as_tensor([0, 1])),
          target=('b', as_tensor([1]))),
      dict(
          description='rank-1, dense',
          source=('a', as_tensor([[0, 1]])),
          target=('b', as_tensor([[0, 1], [2, 3]]))),
      dict(
          description='rank-1, ragged value',
          source=('a', tf.ragged.constant([[0, 1], [0]])),
          target=('b', tf.ragged.constant([[1, 2], []]))),
      dict(
          description='rank-1, ragged splits',
          source=('a', tf.ragged.constant([[0, 1], [0, 1]])),
          target=('b', tf.ragged.constant([[1, 2], [0], [1]]))),
  ])
  def testRaisesOnIncompatibleIndices(self, description: str,
                                      source: adjacency.Index,
                                      target: adjacency.Index):
    self.assertRaisesRegex(
        Exception,
        r'Adjacency indices are not compatible: \(0, a\) and \(1, b\)',
        lambda: adjacency.Adjacency.from_indices(source, target))

  def testRaiseOnExtraPositionalArguments(self):
    self.assertRaisesRegex(
        TypeError, 'Positional arguments are not supported',
        lambda: adjacency.Adjacency.from_indices(('a', []), ('b', []), True))

  def testNodeSetName(self):
    adj = adjacency.Adjacency.from_indices(
        source=('node.a', tf.ragged.constant([[0, 1], [0]])),
        target=('node.b', tf.ragged.constant([[1, 2], [1]])))
    self.assertAllEqual(adj.spec.source_name, 'node.a')
    self.assertAllEqual(adj.spec.target_name, 'node.b')
    self.assertAllEqual(adj.source_name, 'node.a')
    self.assertAllEqual(adj.target_name, 'node.b')

  def testIndices(self):
    adj = adjacency.Adjacency.from_indices(
        source=('node.a', as_tensor([0, 1, 2])),
        target=('node.b', as_tensor([2, 1, 0])))
    self.assertAllEqual(adj.source, [0, 1, 2])
    self.assertAllEqual(adj.target, [2, 1, 0])

  @parameterized.parameters([True, False])
  def testValidateArgSupport(self, validate):
    adj = adjacency.Adjacency.from_indices(
        source=('node.a', as_tensor([0, 1, 2])),
        target=('node.b', as_tensor([2, 1, 0])),
        validate=validate)
    self.assertAllEqual(adj.source, [0, 1, 2])
    self.assertAllEqual(adj.target, [2, 1, 0])

  def testMergeFixedSizeBatchToComponents(self):
    adj = adjacency.Adjacency.from_indices(
        source=('node.a', as_tensor([[0, 1], [1, 0], [0, 1]])),
        target=('node.b', as_tensor([[1, 2], [1, 1], [0, 0]])))
    result = adj._merge_batch_to_components(
        as_tensor([2, 2, 2]), {
            'node.a': as_tensor([2, 2, 2]),
            'node.b': as_tensor([3, 3, 3]),
            'node.c': as_tensor([4, 4, 4]),
        })
    self.assertAllEqual(result.source, [0, 1, 1 + 2, 0 + 2, 0 + 4, 1 + 4])
    self.assertAllEqual(result.target, [1, 2, 1 + 3, 1 + 3, 0 + 6, 0 + 6])

  def testMergeRank1BatchToComponents(self):
    adj = adjacency.Adjacency.from_indices(
        source=('node.a', tf.ragged.constant([[0, 1], [1], [0]])),
        target=('node.b', tf.ragged.constant([[1, 2], [1], [1]])))
    result = adj._merge_batch_to_components(
        as_tensor([2, 1, 1]), {
            'node.a': as_tensor([3, 2, 4]),
            'node.b': as_tensor([4, 3, 2]),
        })
    self.assertAllEqual(result.source, [0, 1, 1 + 3, 0 + 3 + 2])
    self.assertAllEqual(result.target, [1, 2, 1 + 4, 1 + 4 + 3])

  def testAdjacencyRepr(self):
    adj = adjacency.Adjacency.from_indices(
        source=('node.a', as_tensor([0, 1, 2])),
        target=('node.b', as_tensor([2, 1, 0])))
    self.assertEqual(
        "Adjacency("
        "source=('node.a', <tf.Tensor: shape=(3,), dtype=tf.int32>), "
        "target=('node.b', <tf.Tensor: shape=(3,), dtype=tf.int32>))",
        repr(adj))

  def testRelaxation(self):
    original = adjacency.AdjacencySpec.from_incident_node_sets(
        'a', 'b', index_spec=tf.TensorSpec([3], tf.int64)
    )
    expected = adjacency.AdjacencySpec.from_incident_node_sets(
        'a', 'b', index_spec=tf.TensorSpec([None], tf.int64)
    )
    self.assertEqual(original.relax(num_edges=True), expected)
    self.assertEqual(
        original.relax(num_edges=True).relax(num_edges=True), expected)

if __name__ == '__main__':
  tf.test.main()
