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
from typing import Mapping

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import pool_ops_v1

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
          pool_ops_v1.pool_edges_to_node(
              graph, 'edge', const.SOURCE, pooling, feature_name=fname))
      self.assertAllEqual(
          expected,
          pool_ops_v1.pool_v1(graph, const.SOURCE, edge_set_name='edge',
                              reduce_type=pooling, feature_name=fname))
    for fname, expected in expected_target_fields.items():
      self.assertAllEqual(
          expected,
          pool_ops_v1.pool_edges_to_node(
              graph, 'edge', const.TARGET, pooling, feature_name=fname))
      self.assertAllEqual(
          expected,
          pool_ops_v1.pool_v1(graph, const.TARGET, edge_set_name='edge',
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
          pool_ops_v1.pool_nodes_to_context(graph, 'node', pooling,
                                            feature_name=fname))
      self.assertAllEqual(
          expected,
          pool_ops_v1.pool_v1(graph, const.CONTEXT, node_set_name='node',
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
          pool_ops_v1.pool_edges_to_context(graph, 'edge', pooling,
                                            feature_name=fname))
      self.assertAllEqual(
          expected,
          pool_ops_v1.pool_v1(graph, const.CONTEXT, edge_set_name='edge',
                              reduce_type=pooling, feature_name=fname))


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
  def testWithEmptySetValue(self, reduce_op: pool_ops_v1.UnsortedReduceOp,
                            values: const.Field, segment_ids: tf.Tensor,
                            num_segments: tf.Tensor, empty_set_value,
                            expected_result):

    reduce_op = pool_ops_v1.with_empty_set_value(reduce_op, empty_set_value)
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

    reduce_op = pool_ops_v1.with_minus_inf_replaced(
        tf.math.unsorted_segment_max, replacement_value)
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

    reduce_op = pool_ops_v1.with_plus_inf_replaced(tf.math.unsorted_segment_min,
                                                   replacement_value)
    self.assertAllEqual(
        reduce_op(values, segment_ids, num_segments), expected_result)


if __name__ == '__main__':
  tf.test.main()
