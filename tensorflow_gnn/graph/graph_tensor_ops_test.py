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
import functools
from typing import Mapping, Union

from absl.testing import parameterized
# TODO(b/266817638): Remove when fixed
from packaging import version
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
  def testShuffleFeaturesGlobally(
      self,
      description: str,
      context: gt.Context,
      node_set: gt.NodeSet,
      edge_set: gt.EdgeSet,
      expected_fields: Mapping[GraphPiece, Mapping[str, const.Field]],
  ):

    # TODO(b/266817638): Remove when fixed
    if version.parse(tf.__version__) < version.parse(
        '2.11.0'
    ) and description in {'ragged.1', 'ragged.2'}:
      self.skipTest('Bad Test')

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
                    })
        },
        edge_sets={
            'edge':
                gt.EdgeSet.from_fields(
                    sizes=[3], adjacency=adjacency, features={'s': [1, 2, 3]}),
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


if __name__ == '__main__':
  tf.test.main()
