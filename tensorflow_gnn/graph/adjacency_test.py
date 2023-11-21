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

# Enables tests for graph pieces that are members of test classes.
const.enable_graph_tensor_validation_at_runtime()


class HyperAdjacencyTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    const.enable_graph_tensor_validation_at_runtime()

  @parameterized.named_parameters([
      dict(
          testcase_name='rank-0, simple graph',
          indices={
              const.SOURCE: ('node', as_tensor([0, 1])),
              const.TARGET: ('node', as_tensor([1, 2])),
          },
          expected_shape=[]),
      dict(
          testcase_name='rank-0, hypergraph',
          indices={
              0: ('node', as_tensor([0, 1, 2])),
          },
          expected_shape=[]),
      dict(
          testcase_name='rank-1, variable size',
          indices={
              const.SOURCE: (
                  'node.a',
                  tf.ragged.constant([[0, 1], [0]], row_splits_dtype=tf.int32),
              ),
              const.TARGET: (
                  'node.b',
                  tf.ragged.constant([[1, 2], [1]], row_splits_dtype=tf.int32),
              ),
          },
          expected_shape=[2]),
      dict(
          testcase_name='rank-1, fixed size',
          indices={
              const.SOURCE: ('node.a', as_tensor([[0], [1], [2]])),
              const.TARGET: ('node.b', as_tensor([[0], [1], [2]])),
          },
          expected_shape=[3]),
  ])
  def testShapeResolution(
      self,
      indices: adjacency.Indices,
      expected_shape: tf.TensorShape,
  ):
    result = adjacency.HyperAdjacency.from_indices(indices)
    self.assertEqual(result.shape.as_list(), expected_shape)

  @parameterized.named_parameters([
      dict(
          testcase_name='rank-0, sizes missmatch',
          indices={
              const.SOURCE: ('a', as_tensor([0, 1])),
              const.TARGET: ('b', as_tensor([1])),
          }),
      dict(
          testcase_name='rank-0, sizes missmatch for hyper-graph',
          indices={
              0: ('a', as_tensor([0, 1])),
              2: ('c', as_tensor([0, 1])),
              1: ('b', as_tensor([1])),
          }),
      dict(
          testcase_name='rank-1, dense',
          indices={
              const.SOURCE: ('a', as_tensor([[0, 1]])),
              const.TARGET: ('b', as_tensor([[0, 1], [2, 3]])),
          }),
      dict(
          testcase_name='rank-1, ragged value',
          indices={
              const.SOURCE: (
                  'a',
                  tf.ragged.constant([[0, 1], [0]], row_splits_dtype=tf.int32),
              ),
              const.TARGET: (
                  'b',
                  tf.ragged.constant([[1, 2], []], row_splits_dtype=tf.int32),
              ),
          }),
      dict(
          testcase_name='rank-1, ragged splits',
          indices={
              const.SOURCE: (
                  'a',
                  tf.ragged.constant(
                      [[0, 1], [0, 1]], row_splits_dtype=tf.int32
                  ),
              ),
              const.TARGET: (
                  'b',
                  tf.ragged.constant(
                      [[1, 2], [0], [1]], row_splits_dtype=tf.int32
                  ),
              ),
          }),
  ])
  def testRaisesOnIncompatibleIndices(self, indices: adjacency.Indices):
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

  @parameterized.named_parameters([
      dict(
          testcase_name='rank-0', index=as_tensor([0, 1, 2]), expected_result=3
      ),
      dict(
          testcase_name='rank-1',
          index=tf.ragged.constant([[0, 1], [0], []]),
          expected_result=[2, 1, 0],
      ),
      dict(
          testcase_name='rank-2',
          index=tf.RaggedTensor.from_uniform_row_length(
              tf.ragged.constant([[0], [0, 1], [], [0], [], []]), 3
          ),
          expected_result=[[1, 2, 0], [1, 0, 0]],
      ),
  ])
  def testNumItems(self, index, expected_result):
    adj = adjacency.HyperAdjacency.from_indices(
        {0: ('a', index), 1: ('b', index), 2: ('c', index)}
    )
    self.assertAllEqual(adj._get_num_items(), expected_result)
    # Test caching:
    self.assertAllEqual(adj._get_num_items(), expected_result)

  @parameterized.named_parameters([
      dict(
          testcase_name='Empty',
          index=as_tensor([], dtype=tf.int64),
          expected_result=tf.int64.min,
      ),
      dict(
          testcase_name='Rank0',
          index=as_tensor([2, 1]),
          expected_result=2,
      ),
      dict(
          testcase_name='Rank1',
          index=tf.ragged.constant([[0, 1], [0]]),
          expected_result=[1, 0],
      ),
      dict(
          testcase_name='Rank2',
          index=tf.RaggedTensor.from_uniform_row_length(
              tf.ragged.constant([[0], [0, 3], [], []]), 2
          ),
          expected_result=[[0, 3], [tf.int32.min, tf.int32.min]],
      ),
  ])
  def testMaxIndex(self, index, expected_result):
    adj = adjacency.HyperAdjacency.from_indices({0: ('a', index)})
    self.assertAllEqual(adj._get_max_index('a'), expected_result)
    # Test caching:
    self.assertAllEqual(adj._get_max_index('a'), expected_result)


class AdjacencyTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    const.enable_graph_tensor_validation_at_runtime()

  @parameterized.named_parameters([
      dict(
          testcase_name='rank-0, simple graph',
          source=('node', as_tensor([0, 1])),
          target=('node', as_tensor([1, 2])),
          expected_shape=[]),
      dict(
          testcase_name='rank-1, variable size',
          source=('node.a', tf.ragged.constant([[0, 1], [0]])),
          target=('node.b', tf.ragged.constant([[1, 2], [1]])),
          expected_shape=[2]),
      dict(
          testcase_name='rank-1, fixed size',
          source=('node.a', as_tensor([[0], [1], [2]])),
          target=('node.b', as_tensor([[0], [1], [2]])),
          expected_shape=[3]),
  ])
  def testShapeResolution(
      self,
      source: adjacency.Index,
      target: adjacency.Index,
      expected_shape: tf.TensorShape,
  ):
    result = adjacency.Adjacency.from_indices(source, target, validate=False)
    self.assertEqual(result.shape.as_list(), expected_shape)

  @parameterized.named_parameters([
      dict(
          testcase_name='rank-0, sizes missmatch',
          source=('a', as_tensor([0, 1])),
          target=('b', as_tensor([1]))),
      dict(
          testcase_name='rank-1, dense',
          source=('a', as_tensor([[0, 1]])),
          target=('b', as_tensor([[0, 1], [2, 3]]))),
      dict(
          testcase_name='rank-1, ragged value',
          source=('a', tf.ragged.constant([[0, 1], [0]])),
          target=('b', tf.ragged.constant([[1, 2], []]))),
      dict(
          testcase_name='rank-1, ragged splits',
          source=('a', tf.ragged.constant([[0, 1], [0, 1]])),
          target=('b', tf.ragged.constant([[1, 2], [0], [1]]))),
  ])
  def testRaisesOnIncompatibleIndices(
      self, source: adjacency.Index, target: adjacency.Index
  ):
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

  @parameterized.named_parameters([
      dict(testcase_name='Rank0', index=as_tensor([0, 1]), expected_result=2),
      dict(
          testcase_name='Rank1',
          index=tf.ragged.constant([[0, 1], [0]]),
          expected_result=[2, 1],
      ),
      dict(
          testcase_name='Rank2',
          index=tf.RaggedTensor.from_uniform_row_length(
              tf.ragged.constant([[0], [0, 1], [], [0]]), 2
          ),
          expected_result=[[1, 2], [0, 1]],
      ),
  ])
  def testNumItems(self, index, expected_result):
    adj = adjacency.Adjacency.from_indices(
        source=('a', index), target=('b', index)
    )
    self.assertAllEqual(adj._get_num_items(), expected_result)
    # Checks with caching.
    self.assertAllEqual(adj._get_num_items(), expected_result)

  @parameterized.named_parameters([
      dict(
          testcase_name='Empty',
          source=as_tensor([], dtype=tf.int64),
          expected_source=tf.int64.min,
          target=as_tensor([], dtype=tf.int64),
          expected_target=tf.int64.min,
      ),
      dict(
          testcase_name='Rank0',
          source=as_tensor([0, 1]),
          expected_source=1,
          target=as_tensor([2, 1]),
          expected_target=2,
      ),
      dict(
          testcase_name='Rank1',
          source=tf.ragged.constant([[0, 1], [0]]),
          expected_source=[1, 0],
          target=tf.ragged.constant([[3, 0], [1]]),
          expected_target=[3, 1],
      ),
      dict(
          testcase_name='Rank2',
          source=tf.RaggedTensor.from_uniform_row_length(
              tf.ragged.constant([[0], [0, 1], [], [0]]), 2
          ),
          expected_source=[[0, 1], [tf.int32.min, 0]],
          target=tf.RaggedTensor.from_uniform_row_length(
              tf.ragged.constant([[3], [1, 0], [], [1]]), 2
          ),
          expected_target=[[3, 1], [tf.int32.min, 1]],
      ),
      dict(
          testcase_name='Rank3',
          source=tf.RaggedTensor.from_uniform_row_length(
              tf.RaggedTensor.from_uniform_row_length(
                  tf.ragged.constant([[], [0, 1]]), 2
              ),
              1,
          ),
          expected_source=[[[tf.int32.min, 1]]],
          target=tf.RaggedTensor.from_uniform_row_length(
              tf.RaggedTensor.from_uniform_row_length(
                  tf.ragged.constant([[], [1, 0]]), 2
              ),
              1,
          ),
          expected_target=[[[tf.int32.min, 1]]],
      ),
  ])
  def testMaxIndex(self, source, target, expected_source, expected_target):
    adj = adjacency.Adjacency.from_indices(
        source=('a', source), target=('b', target)
    )
    self.assertAllEqual(adj._get_max_index('a'), expected_source)
    self.assertAllEqual(adj._get_max_index('b'), expected_target)
    # Checks with caching:
    self.assertAllEqual(adj._get_max_index('a'), expected_source)
    self.assertAllEqual(adj._get_max_index('b'), expected_target)

if __name__ == '__main__':
  tf.test.main()
