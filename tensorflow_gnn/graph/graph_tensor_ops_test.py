"""Tests for gt.GraphTensor extension type (go/tf-gnn-api)."""

import functools
from typing import Mapping

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import graph_tensor_ops as ops

partial = functools.partial

as_tensor = tf.convert_to_tensor
as_ragged = tf.ragged.constant


class PoolingTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for pooling operations."""

  @parameterized.parameters([
      dict(
          description='max pooling of edge features to source and targed nodes',
          pooling='max',
          node_set=gt.NodeSet.from_fields(sizes=as_tensor([1, 2]), features={}),
          edge_set=gt.EdgeSet.from_fields(
              sizes=as_tensor([2, 2]),
              adjacency=adj.HyperAdjacency.from_indices({
                  const.SOURCE: ('node', as_tensor([0, 0, 1, 2])),
                  const.TARGET: ('node', as_tensor([0, 0, 2, 1]))
              }),
              features={
                  'scalar':
                      as_tensor([1., 2., 3., 4.]),
                  'ragged':
                      as_ragged(
                          [[[1, 8], [2, 7]], [[3, 6]], [[4, 5]], [[5, 4]]],
                          ragged_rank=1)
              }),
          expected_source_fields={
              'scalar':
                  as_tensor([2., 3., 4.]),
              'ragged':
                  as_ragged([[[3, 8], [2, 7]], [[4, 5]], [[5, 4]]],
                            ragged_rank=1),
          },
          expected_target_fields={
              'scalar':
                  as_tensor([2., 4., 3.]),
              'ragged':
                  as_ragged([[[3, 8], [2, 7]], [[5, 4]], [[4, 5]]],
                            ragged_rank=1),
          }),
      dict(
          description='sum pooling of edge features to source and targed nodes',
          pooling='sum',
          node_set=gt.NodeSet.from_fields(sizes=as_tensor([1, 2]), features={}),
          edge_set=gt.EdgeSet.from_fields(
              sizes=as_tensor([2, 3]),
              adjacency=adj.HyperAdjacency.from_indices({
                  const.SOURCE: ('node', as_tensor([0, 0, 0, 2, 2])),
                  const.TARGET: ('node', as_tensor([2, 1, 0, 0, 0]))
              }),
              features={
                  'scalar':
                      as_tensor([1., 2., 3., 4., 5.]),
                  'vector':
                      as_tensor([[1., 5.], [2., 4.], [3., 3.], [4., 2.],
                                 [5., 1.]]),
                  'matrix':
                      as_tensor([[[1.]], [[2.]], [[3.]], [[4.]], [[5.]]]),
                  'ragged.1':
                      as_ragged([[1, 2], [], [3, 4], [], [5]]),
                  'ragged.2':
                      as_ragged([[[1], [2]], [], [[3]], [], []])
              }),
          expected_source_fields={
              'scalar':
                  as_tensor([1. + 2. + 3., 0., 4. + 5.]),
              'vector':
                  as_tensor([[1. + 2. + 3., 5. + 4. + 3.], [0., 0.],
                             [4. + 5., 2. + 1.]]),
              'matrix':
                  as_tensor([[[1. + 2. + 3.]], [[0.]], [[4. + 5.]]]),
              'ragged.1':
                  as_ragged([[1 + 3, 2 + 4], [], [5]]),
              'ragged.2':
                  as_ragged([[[1 + 3], [2]], [], []]),
          },
          expected_target_fields={
              'scalar':
                  as_tensor([3. + 4. + 5., 2., 1.]),
              'vector':
                  as_tensor([[3. + 4. + 5., 3. + 2. + 1.], [2., 4.], [1., 5.]]),
              'matrix':
                  as_tensor([[[3. + 4. + 5.]], [[2.]], [[1.]]]),
              'ragged.1':
                  as_ragged([[3 + 5, 4], [], [1, 2]]),
              'ragged.2':
                  as_ragged([[[3]], [], [[1], [2]]]),
          })
  ])
  def testEdgeFieldToNode(self, description: str, pooling: str,
                          node_set: gt.NodeSet, edge_set: gt.EdgeSet,
                          expected_source_fields: Mapping[str, const.Field],
                          expected_target_fields: Mapping[str, const.Field]):
    graph = gt.GraphTensor.from_pieces(
        node_sets={'node': node_set}, edge_sets={'edge': edge_set})

    for fname, expected in expected_source_fields.items():
      self.assertAllEqual(
          expected,
          ops.pool_edges_to_node(
              graph, 'edge', const.SOURCE, pooling, feature_name=fname))
    for fname, expected in expected_target_fields.items():
      self.assertAllEqual(
          expected,
          ops.pool_edges_to_node(
              graph, 'edge', const.TARGET, pooling, feature_name=fname))

  @parameterized.parameters([
      dict(
          description='sum pooling of node features to context, 1 component',
          pooling='sum',
          node_set=gt.NodeSet.from_fields(
              sizes=as_tensor([3]),
              features={
                  'scalar': as_tensor([1., 2., 3]),
                  'vector': as_tensor([[1., 0.], [2., 0.], [3., 0.]]),
                  'matrix': as_tensor([[[1.]], [[2.]], [[3.]]]),
                  'ragged.1': as_ragged([[1, 2], [3], []])
              }),
          expected_context_fields={
              'scalar': as_tensor([1. + 2. + 3.]),
              'vector': as_tensor([[1. + 2. + 3., 0. + 0. + 0.]]),
              'matrix': as_tensor([[[1. + 2. + 3.]]]),
              'ragged.1': as_ragged([[1 + 3, 2]])
          }),
      dict(
          description='sum pooling of node features to context, 2 components',
          pooling='sum',
          node_set=gt.NodeSet.from_fields(
              sizes=as_tensor([2, 1]),
              features={
                  'scalar': as_tensor([1., 2., 3]),
                  'vector': as_tensor([[1., 0.], [2., 0.], [3., 0.]]),
                  'matrix': as_tensor([[[1.]], [[2.]], [[3.]]]),
                  'ragged.1': as_ragged([[1, 2], [3], []]),
                  'ragged.2': as_ragged([[[1, 2], []], [[3]], []])
              }),
          expected_context_fields={
              'scalar': as_tensor([1. + 2., 3.]),
              'vector': as_tensor([[1. + 2., 0. + 0.], [3., 0.]]),
              'matrix': as_tensor([[[1. + 2.]], [[3.]]]),
              'ragged.1': as_ragged([[1 + 3, 2], []]),
              'ragged.2': as_ragged([[[1 + 3, 2], []], []])
          })
  ])
  def testNodeFieldToContext(self, description: str, pooling: str,
                             node_set: gt.NodeSet,
                             expected_context_fields: Mapping[str,
                                                              const.Field]):
    graph = gt.GraphTensor.from_pieces(node_sets={'node': node_set})

    for fname, expected in expected_context_fields.items():
      self.assertAllEqual(
          expected,
          ops.pool_nodes_to_context(graph, 'node', pooling, feature_name=fname))

  @parameterized.parameters([
      dict(
          description='max pooling of edge features to graph context',
          pooling='max',
          edge_set=gt.EdgeSet.from_fields(
              sizes=as_tensor([3]),
              adjacency=adj.HyperAdjacency.from_indices({
                  0: ('node', as_tensor([0, 0, 0])),
              }),
              features={
                  'scalar': as_tensor([1., 2., 3]),
                  'vector': as_tensor([[1., 0.], [2., 0.], [3., 0.]]),
                  'matrix': as_tensor([[[1.]], [[2.]], [[3.]]]),
                  'ragged.1': as_ragged([[1, 2], [3], []])
              }),
          expected_context_fields={
              'scalar': as_tensor([3.]),
              'vector': as_tensor([[3., 0.]]),
              'matrix': as_tensor([[[3.]]]),
              'ragged.1': as_ragged([[3, 2]])
          }),
      dict(
          description='min pooling of node features to graph context',
          pooling='min',
          edge_set=gt.EdgeSet.from_fields(
              sizes=as_tensor([2, 1]),
              adjacency=adj.HyperAdjacency.from_indices({
                  0: ('node', as_tensor([0, 0, 0])),
              }),
              features={
                  'scalar': as_tensor([1., 2., 3]),
                  'vector': as_tensor([[1., 0.], [2., 0.], [3., 0.]]),
                  'matrix': as_tensor([[[1.]], [[2.]], [[3.]]]),
                  'ragged.1': as_ragged([[1, 2], [3], []]),
                  'ragged.2': as_ragged([[[1, 2], []], [[3]], []])
              }),
          expected_context_fields={
              'scalar': as_tensor([1., 3.]),
              'vector': as_tensor([[1., 0.], [3., 0.]]),
              'matrix': as_tensor([[[1.]], [[3.]]]),
              'ragged.1': as_ragged([[1, 2], []]),
              'ragged.2': as_ragged([[[1, 2], []], []])
          })
  ])
  def testEdgeFieldToContext(self, description: str, pooling: str,
                             edge_set: gt.EdgeSet,
                             expected_context_fields: Mapping[str,
                                                              const.Field]):
    graph = gt.GraphTensor.from_pieces(edge_sets={'edge': edge_set})

    for fname, expected in expected_context_fields.items():
      self.assertAllEqual(
          expected,
          ops.pool_edges_to_context(graph, 'edge', pooling, feature_name=fname))


class BroadcastingTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for broadcasting operations."""

  @parameterized.parameters([
      dict(
          description='source and target node features to edges broadcasting',
          node_set=gt.NodeSet.from_fields(
              sizes=as_tensor([3]),
              features={
                  'scalar': as_tensor([1., 2., 3]),
                  'vector': as_tensor([[1., 3.], [2., 2.], [3., 1.]]),
                  'matrix': as_tensor([[[1.]], [[2.]], [[3.]]]),
                  'ragged': as_ragged([[1, 2], [3], []])
              }),
          edge_set=gt.EdgeSet.from_fields(
              sizes=as_tensor([2, 2]),
              adjacency=adj.HyperAdjacency.from_indices({
                  const.SOURCE: ('node', as_tensor([0, 0, 0, 2, 2])),
                  const.TARGET: ('node', as_tensor([2, 1, 0, 0, 0]))
              }),
              features={}),
          expected_source_fields={
              'scalar':
                  as_tensor([1., 1., 1., 3., 3.]),
              'vector':
                  as_tensor([[1., 3.], [1., 3.], [1., 3.], [3., 1.], [3., 1.]]),
              'matrix':
                  as_tensor([[[1.]], [[1.]], [[1.]], [[3.]], [[3.]]]),
              'ragged':
                  as_ragged([[1, 2], [1, 2], [1, 2], [], []])
          },
          expected_target_fields={
              'scalar':
                  as_tensor([3., 2., 1., 1., 1.]),
              'vector':
                  as_tensor([[3., 1.], [2., 2.], [1., 3.], [1., 3.], [1., 3.]]),
              'matrix':
                  as_tensor([[[3.]], [[2.]], [[1.]], [[1.]], [[1.]]]),
              'ragged':
                  as_ragged([[], [3], [1, 2], [1, 2], [1, 2]])
          })
  ])
  def testEdgeFieldFromNode(self, description: str, node_set: gt.NodeSet,
                            edge_set: gt.EdgeSet,
                            expected_source_fields: Mapping[str, const.Field],
                            expected_target_fields: Mapping[str, const.Field]):
    graph = gt.GraphTensor.from_pieces(
        node_sets={'node': node_set}, edge_sets={'edge': edge_set})

    for fname, expected in expected_source_fields.items():
      self.assertAllEqual(
          expected,
          ops.broadcast_node_to_edges(
              graph, 'edge', const.SOURCE, feature_name=fname))
    for fname, expected in expected_target_fields.items():
      self.assertAllEqual(
          expected,
          ops.broadcast_node_to_edges(
              graph, 'edge', const.TARGET, feature_name=fname))

  @parameterized.parameters([
      dict(
          description='context features to nodes broadcasting, 1 component',
          context=gt.Context.from_fields(features={
              'scalar': as_tensor([1]),
              'vector': as_tensor([[1., 2.]]),
              'matrix': as_tensor([[[1., 2., 3.], [4., 5., 6.]]]),
              'ragged': as_ragged([[[], [1], [], [2, 3]]]),
          }),
          node_set=gt.NodeSet.from_fields(sizes=as_tensor([3]), features={}),
          expected_node_fields={
              'scalar':
                  as_tensor([1] * 3),
              'vector':
                  as_tensor([[1., 2.]] * 3),
              'matrix':
                  as_tensor([[[1., 2., 3.], [4., 5., 6.]]] * 3),
              'ragged':
                  as_ragged([[[], [1], [], [2, 3]], [[], [1], [], [2, 3]],
                             [[], [1], [], [2, 3]]]),
          }),
      dict(
          description='context features to nodes broadcasting, 2 components',
          context=gt.Context.from_fields(features={
              'scalar': as_tensor([1, 2]),
              'vector': as_tensor([[1.], [2.]]),
              'ragged': as_ragged([[[], [1], []], [[1], [], [2]]]),
          }),
          node_set=gt.NodeSet.from_fields(sizes=as_tensor([3, 2]), features={}),
          expected_node_fields={
              'scalar':
                  as_tensor([1, 1, 1, 2, 2]),
              'vector':
                  as_tensor([[1.], [1.], [1.], [2.], [2.]]),
              'ragged':
                  as_ragged([[[], [1], []], [[], [1], []], [[], [1], []],
                             [[1], [], [2]], [[1], [], [2]]]),
          })
  ])
  def testNodeFieldFromContext(self, description: str, context: gt.Context,
                               node_set: gt.NodeSet,
                               expected_node_fields: Mapping[str, const.Field]):
    graph = gt.GraphTensor.from_pieces(
        context=context, node_sets={'node': node_set})

    for fname, expected in expected_node_fields.items():
      self.assertAllEqual(
          expected,
          ops.broadcast_context_to_nodes(graph, 'node', feature_name=fname))

  @parameterized.parameters([
      dict(
          description='context features to edges broadcasting, 1 component',
          context=gt.Context.from_fields(features={
              'scalar': as_tensor([1]),
              'vector': as_tensor([[1., 2.]]),
              'matrix': as_tensor([[[1., 2., 3.], [4., 5., 6.]]]),
              'ragged': as_ragged([[[], [1], [], [2, 3]]]),
          }),
          edge_set=gt.EdgeSet.from_fields(
              sizes=as_tensor([3]),
              adjacency=adj.HyperAdjacency.from_indices({
                  0: ('node', as_tensor([0, 0, 0])),
              }),
              features={}),
          expected_edge_fields={
              'scalar':
                  as_tensor([1] * 3),
              'vector':
                  as_tensor([[1., 2.]] * 3),
              'matrix':
                  as_tensor([[[1., 2., 3.], [4., 5., 6.]]] * 3),
              'ragged':
                  as_ragged([[[], [1], [], [2, 3]], [[], [1], [], [2, 3]],
                             [[], [1], [], [2, 3]]]),
          }),
      dict(
          description='context features to nodes broadcasting, 2 components',
          context=gt.Context.from_fields(features={
              'scalar': as_tensor([1, 2]),
              'vector': as_tensor([[1.], [2.]]),
              'ragged': as_ragged([[[], [1], []], [[1], [], [2]]]),
          }),
          edge_set=gt.EdgeSet.from_fields(
              sizes=as_tensor([3, 2]),
              adjacency=adj.HyperAdjacency.from_indices({
                  0: ('node', as_tensor([0, 0, 0, 0, 0])),
              }),
              features={}),
          expected_edge_fields={
              'scalar':
                  as_tensor([1, 1, 1, 2, 2]),
              'vector':
                  as_tensor([[1.], [1.], [1.], [2.], [2.]]),
              'ragged':
                  as_ragged([[[], [1], []], [[], [1], []], [[], [1], []],
                             [[1], [], [2]], [[1], [], [2]]]),
          })
  ])
  def testEdgeFieldFromContext(self, description: str, context: gt.Context,
                               edge_set: gt.EdgeSet,
                               expected_edge_fields: Mapping[str, const.Field]):
    graph = gt.GraphTensor.from_pieces(
        context=context, edge_sets={'edge': edge_set})

    for fname, expected in expected_edge_fields.items():
      self.assertAllEqual(
          expected,
          ops.broadcast_context_to_edges(graph, 'edge', feature_name=fname))


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


if __name__ == '__main__':
  tf.test.main()
