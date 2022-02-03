"""Tests for gt.GraphTensor extension type (go/tf-gnn-api)."""

import functools
from typing import Mapping, Union

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import graph_tensor_ops as ops

partial = functools.partial

as_tensor = tf.convert_to_tensor
as_ragged = tf.ragged.constant

GraphPiece = Union[gt.Context, gt.NodeSet, gt.EdgeSet]


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
          description='max_no_inf pooling of edge features to connected nodes',
          pooling='max_no_inf',
          node_set=gt.NodeSet.from_fields(sizes=as_tensor([3]), features={}),
          edge_set=gt.EdgeSet.from_fields(
              sizes=as_tensor([4]),
              adjacency=adj.HyperAdjacency.from_indices({
                  const.SOURCE: ('node', as_tensor([1, 1, 2, 2])),
                  const.TARGET: ('node', as_tensor([0, 0, 1, 1]))
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
                  as_tensor([0., 2., 4.]),
              'ragged':
                  as_ragged([[], [[3, 8], [2, 7]], [[5, 5]]], ragged_rank=1),
          },
          expected_target_fields={
              'scalar':
                  as_tensor([2., 4., 0.]),
              'ragged':
                  as_ragged([[[3, 8], [2, 7]], [[5, 5]], []], ragged_rank=1),
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
    del description
    graph = gt.GraphTensor.from_pieces(
        node_sets={'node': node_set}, edge_sets={'edge': edge_set})

    for fname, expected in expected_source_fields.items():
      self.assertAllEqual(
          expected,
          ops.pool_edges_to_node(
              graph, 'edge', const.SOURCE, pooling, feature_name=fname))
      self.assertAllEqual(
          expected,
          ops.pool(graph, const.SOURCE, edge_set_name='edge',
                   reduce_type=pooling, feature_name=fname))
    for fname, expected in expected_target_fields.items():
      self.assertAllEqual(
          expected,
          ops.pool_edges_to_node(
              graph, 'edge', const.TARGET, pooling, feature_name=fname))
      self.assertAllEqual(
          expected,
          ops.pool(graph, const.TARGET, edge_set_name='edge',
                   reduce_type=pooling, feature_name=fname))

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
    del description
    graph = gt.GraphTensor.from_pieces(node_sets={'node': node_set})

    for fname, expected in expected_context_fields.items():
      self.assertAllEqual(
          expected,
          ops.pool_nodes_to_context(graph, 'node', pooling, feature_name=fname))
      self.assertAllEqual(
          expected,
          ops.pool(graph, const.CONTEXT, node_set_name='node',
                   reduce_type=pooling, feature_name=fname))

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
    del description
    graph = gt.GraphTensor.from_pieces(edge_sets={'edge': edge_set})

    for fname, expected in expected_context_fields.items():
      self.assertAllEqual(
          expected,
          ops.pool_edges_to_context(graph, 'edge', pooling, feature_name=fname))
      self.assertAllEqual(
          expected,
          ops.pool(graph, const.CONTEXT, edge_set_name='edge',
                   reduce_type=pooling, feature_name=fname))


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
    del description
    graph = gt.GraphTensor.from_pieces(
        node_sets={'node': node_set}, edge_sets={'edge': edge_set})

    for fname, expected in expected_source_fields.items():
      self.assertAllEqual(
          expected,
          ops.broadcast_node_to_edges(
              graph, 'edge', const.SOURCE, feature_name=fname))
      self.assertAllEqual(
          expected,
          ops.broadcast(
              graph, const.SOURCE, edge_set_name='edge', feature_name=fname))
    for fname, expected in expected_target_fields.items():
      self.assertAllEqual(
          expected,
          ops.broadcast_node_to_edges(
              graph, 'edge', const.TARGET, feature_name=fname))
      self.assertAllEqual(
          expected,
          ops.broadcast(
              graph, const.TARGET, edge_set_name='edge', feature_name=fname))

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
    del description
    graph = gt.GraphTensor.from_pieces(
        context=context, node_sets={'node': node_set})

    for fname, expected in expected_node_fields.items():
      self.assertAllEqual(
          expected,
          ops.broadcast_context_to_nodes(graph, 'node', feature_name=fname))
      self.assertAllEqual(
          expected,
          ops.broadcast(
              graph, const.CONTEXT, node_set_name='node', feature_name=fname))

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
    del description
    graph = gt.GraphTensor.from_pieces(
        context=context, edge_sets={'edge': edge_set})

    for fname, expected in expected_edge_fields.items():
      self.assertAllEqual(
          expected,
          ops.broadcast_context_to_edges(graph, 'edge', feature_name=fname))
      self.assertAllEqual(
          expected,
          ops.broadcast(
              graph, const.CONTEXT, edge_set_name='edge', feature_name=fname))


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
          context=gt.Context.from_fields(features={
              'scalar': as_tensor([1, 2, 3]),
          }),
          node_set=gt.NodeSet.from_fields(
              sizes=as_tensor([2, 1]),
              features={
                  'scalar': as_tensor([1., 2., 3]),
              }),
          edge_set=gt.EdgeSet.from_fields(
              sizes=as_tensor([2, 3]),
              adjacency=adj.HyperAdjacency.from_indices({
                  const.SOURCE: ('node', as_tensor([0, 0, 0, 2, 2])),
                  const.TARGET: ('node', as_tensor([2, 1, 0, 0, 0]))
              }),
              features={
                  'scalar': as_tensor([1., 2., 3., 4., 5.]),
              }),
          expected_fields={
              gt.Context: {
                  'scalar': [2, 1, 3]
              },
              gt.NodeSet: {
                  'scalar': [2., 1., 3.]
              },
              gt.EdgeSet: {
                  'scalar': [5., 2., 3., 1., 4.]
              },
          }),
      dict(
          description='vector',
          context=gt.Context.from_fields(features={
              'vector': as_tensor([[1], [2], [3]]),
          }),
          node_set=gt.NodeSet.from_fields(
              sizes=as_tensor([2, 1]),
              features={
                  'vector': as_tensor([[1., 3.], [2., 2.], [3., 1.]]),
              }),
          edge_set=gt.EdgeSet.from_fields(
              sizes=as_tensor([2, 3]),
              adjacency=adj.HyperAdjacency.from_indices({
                  const.SOURCE: ('node', as_tensor([0, 0, 0, 2, 2])),
                  const.TARGET: ('node', as_tensor([2, 1, 0, 0, 0]))
              }),
              features={
                  'vector':
                      as_tensor([[1., 5.], [2., 4.], [3., 3.], [4., 2.],
                                 [5., 1.]]),
              }),
          expected_fields={
              gt.Context: {
                  'vector': [[2], [1], [3]]
              },
              gt.NodeSet: {
                  'vector': [[2., 2.], [1., 3.], [3., 1.]]
              },
              gt.EdgeSet: {
                  'vector': [[5., 1.], [2., 4.], [3., 3.], [1., 5.], [4., 2.]]
              },
          }),
      dict(
          description='matrix',
          context=gt.Context.from_fields(),
          node_set=gt.NodeSet.from_fields(
              sizes=as_tensor([2, 1]),
              features={
                  'matrix': as_tensor([[[1.]], [[2.]], [[3.]]]),
              }),
          edge_set=gt.EdgeSet.from_fields(
              sizes=as_tensor([2, 3]),
              adjacency=adj.HyperAdjacency.from_indices({
                  const.SOURCE: ('node', as_tensor([0, 0, 0, 2, 2])),
                  const.TARGET: ('node', as_tensor([2, 1, 0, 0, 0]))
              }),
              features={
                  'matrix': as_tensor([[[1.]], [[2.]], [[3.]], [[4.]], [[5.]]]),
              }),
          expected_fields={
              gt.NodeSet: {
                  'matrix': [[[2.]], [[1.]], [[3.]]]
              },
              gt.EdgeSet: {
                  'matrix': [[[5.]], [[2.]], [[3.]], [[1.]], [[4.]]]
              },
          }),
      dict(
          description='ragged.1',
          context=gt.Context.from_fields(features={
              'ragged.1': as_ragged([[[], [1], []], [[1], [3], [4], [2]]]),
          }),
          node_set=gt.NodeSet.from_fields(
              sizes=as_tensor([2, 1]),
              features={
                  'ragged.1':
                      as_ragged([[[1, 2], [4, 4], [5, 5]], [[3, 3]], []]),
              }),
          edge_set=gt.EdgeSet.from_fields(
              sizes=as_tensor([2, 3]),
              adjacency=adj.HyperAdjacency.from_indices({
                  const.SOURCE: ('node', as_tensor([0, 0, 0, 2, 2])),
                  const.TARGET: ('node', as_tensor([2, 1, 0, 0, 0]))
              }),
              features={
                  'ragged.1': as_ragged([[[1, 2]], [], [[3, 4]], [], [[5, 5]]]),
              }),
          expected_fields={
              gt.Context: {
                  'ragged.1': [[[], [4], [2]], [[1], [], [1], [3]]]
              },
              gt.NodeSet: {
                  'ragged.1': [[[4, 4], [5, 5], [3, 3]], [[1, 2]], []]
              },
              gt.EdgeSet: {
                  'ragged.1': [[[5, 5]], [], [[1, 2]], [], [[3, 4]]]
              },
          }),
      dict(
          description='ragged.2',
          context=gt.Context.from_fields(),
          node_set=gt.NodeSet.from_fields(
              sizes=as_tensor([2, 1]),
              features={'ragged.2': as_ragged([[[1, 2, 4], []], [[3]],
                                               [[5]]])}),
          edge_set=gt.EdgeSet.from_fields(
              sizes=as_tensor([2, 3]),
              adjacency=adj.HyperAdjacency.from_indices({
                  const.SOURCE: ('node', as_tensor([0, 0, 0, 2, 2])),
                  const.TARGET: ('node', as_tensor([2, 1, 0, 0, 0]))
              }),
              features={
                  'ragged.2':
                      as_ragged([[[1], [2], [4]], [], [[3], [5]], [], []])
              }),
          expected_fields={
              gt.NodeSet: {
                  'ragged.2': [[[1, 2, 4], [5]], [[3]], [[]]]
              },
              gt.EdgeSet: {
                  'ragged.2': [[[2], [4], [3]], [], [[1], [5]], [], []]
              },
          }),
  ])
  def testShuffleScalarComponents(
      self,
      description: str,
      context: gt.Context,
      node_set: gt.NodeSet,
      edge_set: gt.EdgeSet,
      expected_fields: Mapping[GraphPiece, Mapping[str, const.Field]]):
    del description
    graph = gt.GraphTensor.from_pieces(
        context,
        {'node': node_set},
        {'edge': edge_set})
    shuffled = ops.shuffle_scalar_components(graph, seed=8191)

    for fname, expected in expected_fields.get(gt.Context, {}).items():
      self.assertAllEqual(expected, shuffled.context.features[fname])

    for fname, expected in expected_fields.get(gt.NodeSet, {}).items():
      self.assertAllEqual(expected, shuffled.node_sets['node'].features[fname])

    for fname, expected in expected_fields.get(gt.EdgeSet, {}).items():
      self.assertAllEqual(expected, shuffled.edge_sets['edge'].features[fname])


class ReduceOpsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([
      dict(
          reduce_op=tf.math.unsorted_segment_max,
          values=as_tensor([tf.float32.min, tf.float32.max,
                            float('nan')]),
          segment_ids=as_tensor([0, 1, 2]),
          num_segments=3,
          empty_set_value=-1,
          expected_result=as_tensor(
              [tf.float32.min, tf.float32.max,
               float('nan')])),
      dict(
          reduce_op=tf.math.unsorted_segment_min,
          values=as_tensor([tf.float32.min, tf.float32.max]),
          segment_ids=as_tensor([0, 1]),
          num_segments=3,
          empty_set_value=-1,
          expected_result=as_tensor([tf.float32.min, tf.float32.max, -1.])),
      dict(
          reduce_op=tf.math.unsorted_segment_max,
          values=as_tensor([1., 2.]),
          segment_ids=as_tensor([1, 1]),
          num_segments=3,
          empty_set_value=-1,
          expected_result=as_tensor([-1., 2., -1.])),
      dict(
          reduce_op=tf.math.unsorted_segment_min,
          values=as_tensor([[1., 2.], [3., 4.]]),
          segment_ids=as_tensor([0, 2]),
          num_segments=3,
          empty_set_value=0,
          expected_result=as_tensor([[1., 2.], [0., 0.], [3., 4.]])),
      dict(
          reduce_op=tf.math.unsorted_segment_max,
          values=as_ragged([[1., 2., 3.], [2., 1.], [1.]]),
          segment_ids=as_tensor([0, 0, 2]),
          num_segments=3,
          empty_set_value=0,
          expected_result=as_ragged([[2., 2., 3.], [], [1.]])),
      dict(
          reduce_op=tf.math.unsorted_segment_min,
          values=as_ragged([[[1., 2.], [3., 4.]], [], [[5., 6.]]],
                           ragged_rank=1),
          segment_ids=as_tensor([1, 2, 1]),
          num_segments=3,
          empty_set_value=0,
          expected_result=as_ragged([[], [[1., 2.], [3., 4.]], []],
                                    ragged_rank=1)),
      dict(
          reduce_op=tf.math.unsorted_segment_min,
          values=tf.constant([], tf.int32, shape=[0, 2]),
          segment_ids=tf.constant([], tf.int32),
          num_segments=3,
          empty_set_value=0,
          expected_result=as_tensor([[0, 0], [0, 0], [0, 0]])),
  ])
  def testWithEmptySetValue(self, reduce_op: ops.UnsortedReduceOp,
                            values: const.Field, segment_ids: tf.Tensor,
                            num_segments: tf.Tensor, empty_set_value,
                            expected_result):

    reduce_op = ops.with_empty_set_value(reduce_op, empty_set_value)
    self.assertAllEqual(
        reduce_op(values, segment_ids, num_segments), expected_result)

  @parameterized.parameters([
      dict(
          values=as_tensor([tf.float32.min, tf.float32.max, float('nan')]),
          segment_ids=as_tensor([0, 1, 2]),
          num_segments=3,
          replacement_value=-1,
          expected_result=as_tensor([-1., tf.float32.max, float('nan')])),
      dict(
          values=as_tensor([1., 2.]),
          segment_ids=as_tensor([1, 1]),
          num_segments=3,
          replacement_value=-1,
          expected_result=as_tensor([-1., 2., -1.])),
      dict(
          values=as_tensor([[1., 2.], [3., 4.]]),
          segment_ids=as_tensor([0, 2]),
          num_segments=3,
          replacement_value=0,
          expected_result=as_tensor([[1., 2.], [0., 0.], [3., 4.]])),
      dict(
          values=as_ragged([[1., 2., 3.], [2., 1.], [tf.float32.min]]),
          segment_ids=as_tensor([0, 0, 2]),
          num_segments=3,
          replacement_value=1,
          expected_result=as_ragged([[2., 2., 3.], [], [1.]])),
      dict(
          values=as_ragged([[[1., 2.], [3., 4.]], [], [[5., 6.]]],
                           ragged_rank=1),
          segment_ids=as_tensor([1, 2, 1]),
          num_segments=3,
          replacement_value=0,
          expected_result=as_ragged([[], [[5., 6.], [3., 4.]], []],
                                    ragged_rank=1)),
      dict(
          values=tf.constant([], tf.int32, shape=[0, 2]),
          segment_ids=tf.constant([], tf.int32),
          num_segments=3,
          replacement_value=0,
          expected_result=as_tensor([[0, 0], [0, 0], [0, 0]])),
  ])
  def testWithMinusInfReplaced(self,
                               values: const.Field, segment_ids: tf.Tensor,
                               num_segments: tf.Tensor, replacement_value,
                               expected_result):

    reduce_op = ops.with_minus_inf_replaced(tf.math.unsorted_segment_max,
                                            replacement_value)
    self.assertAllEqual(
        reduce_op(values, segment_ids, num_segments), expected_result)

  @parameterized.parameters([
      dict(
          values=as_tensor([tf.float32.min, tf.float32.max, float('nan')]),
          segment_ids=as_tensor([0, 1, 2]),
          num_segments=3,
          replacement_value=0,
          expected_result=as_tensor([tf.float32.min, 0., float('nan')])),
      dict(
          values=as_tensor([1., 2.]),
          segment_ids=as_tensor([1, 1]),
          num_segments=3,
          replacement_value=-1,
          expected_result=as_tensor([-1., 1., -1.])),
      dict(
          values=as_tensor([[1., 2.], [3., 4.]]),
          segment_ids=as_tensor([0, 2]),
          num_segments=3,
          replacement_value=0,
          expected_result=as_tensor([[1., 2.], [0., 0.], [3., 4.]])),
      dict(
          values=as_ragged([[1., 2., 3.], [2., 1.], [tf.float32.max]]),
          segment_ids=as_tensor([0, 0, 2]),
          num_segments=3,
          replacement_value=1,
          expected_result=as_ragged([[1., 1., 3.], [], [1.]])),
      dict(
          values=as_ragged([[[1., 2.], [3., 4.]], [], [[5., 6.]]],
                           ragged_rank=1),
          segment_ids=as_tensor([1, 2, 1]),
          num_segments=3,
          replacement_value=0,
          expected_result=as_ragged([[], [[1., 2.], [3., 4.]], []],
                                    ragged_rank=1)),
      dict(
          values=tf.constant([], tf.int32, shape=[0, 2]),
          segment_ids=tf.constant([], tf.int32),
          num_segments=3,
          replacement_value=0,
          expected_result=as_tensor([[0, 0], [0, 0], [0, 0]])),
  ])
  def testWithPlusInfReplaced(self,
                              values: const.Field, segment_ids: tf.Tensor,
                              num_segments: tf.Tensor, replacement_value,
                              expected_result):

    reduce_op = ops.with_plus_inf_replaced(tf.math.unsorted_segment_min,
                                           replacement_value)
    self.assertAllEqual(
        reduce_op(values, segment_ids, num_segments), expected_result)


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


if __name__ == '__main__':
  tf.test.main()
