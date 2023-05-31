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
"""Tests for broadcast_ops.py."""
from typing import Mapping

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import broadcast_ops
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt

as_tensor = tf.convert_to_tensor
as_ragged = tf.ragged.constant


# NOTE: Testing with TFLite requires a Keras wrapper and therefore is delegated
# to tensorflow_gnn/keras/layers/graph_ops_test.py.
class BroadcastXToYTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for basic broadcasting operations broadcast_*_to_*().

  For consistency, some tests run the corresponging call to the generic
  broadcast_v2() function as well, but see BroadcastV2Test for more on that.
  """

  @parameterized.named_parameters(
      ("WithAdjacency", False),
      ("WithHyperAdjacency", True))
  def testEdgeFieldFromNode(self, use_hyper_adjacency=False):
    node_set = gt.NodeSet.from_fields(
        sizes=as_tensor([3]),
        features={
            "scalar": as_tensor([1., 2., 3]),
            "vector": as_tensor([[1., 3.], [2., 2.], [3., 1.]]),
            "matrix": as_tensor([[[1.]], [[2.]], [[3.]]]),
            "ragged": as_ragged([[1, 2], [3], []])
        })
    edge_source = ("node", as_tensor([0, 0, 0, 2, 2]))
    edge_target = ("node", as_tensor([2, 1, 0, 0, 0]))
    if use_hyper_adjacency:
      adjacency = adj.HyperAdjacency.from_indices({const.SOURCE: edge_source,
                                                   const.TARGET: edge_target})
    else:
      adjacency = adj.Adjacency.from_indices(edge_source, edge_target)
    edge_set = gt.EdgeSet.from_fields(
        sizes=as_tensor([2, 2]), adjacency=adjacency, features={})
    expected_source_fields = {
        "scalar": as_tensor([1., 1., 1., 3., 3.]),
        "vector": as_tensor([[1., 3.], [1., 3.], [1., 3.], [3., 1.], [3., 1.]]),
        "matrix": as_tensor([[[1.]], [[1.]], [[1.]], [[3.]], [[3.]]]),
        "ragged": as_ragged([[1, 2], [1, 2], [1, 2], [], []])}
    expected_target_fields = {
        "scalar": as_tensor([3., 2., 1., 1., 1.]),
        "vector": as_tensor([[3., 1.], [2., 2.], [1., 3.], [1., 3.], [1., 3.]]),
        "matrix": as_tensor([[[3.]], [[2.]], [[1.]], [[1.]], [[1.]]]),
        "ragged": as_ragged([[], [3], [1, 2], [1, 2], [1, 2]])}

    graph = gt.GraphTensor.from_pieces(
        node_sets={"node": node_set}, edge_sets={"edge": edge_set})

    for fname, expected in expected_source_fields.items():
      self.assertAllEqual(
          expected,
          broadcast_ops.broadcast_node_to_edges(
              graph, "edge", const.SOURCE, feature_name=fname))
      self.assertAllEqual(
          expected,
          broadcast_ops.broadcast_v2(
              graph, const.SOURCE, edge_set_name="edge", feature_name=fname))
    for fname, expected in expected_target_fields.items():
      self.assertAllEqual(
          expected,
          broadcast_ops.broadcast_node_to_edges(
              graph, "edge", const.TARGET, feature_name=fname))
      self.assertAllEqual(
          expected,
          broadcast_ops.broadcast_v2(
              graph, const.TARGET, edge_set_name="edge", feature_name=fname))

  @parameterized.parameters([
      dict(
          description="context features to nodes broadcasting, 1 component",
          context=gt.Context.from_fields(features={
              "scalar": as_tensor([1]),
              "vector": as_tensor([[1., 2.]]),
              "matrix": as_tensor([[[1., 2., 3.], [4., 5., 6.]]]),
              "ragged": as_ragged([[[], [1], [], [2, 3]]]),
          }),
          node_set=gt.NodeSet.from_fields(sizes=as_tensor([3]), features={}),
          expected_node_fields={
              "scalar":
                  as_tensor([1] * 3),
              "vector":
                  as_tensor([[1., 2.]] * 3),
              "matrix":
                  as_tensor([[[1., 2., 3.], [4., 5., 6.]]] * 3),
              "ragged":
                  as_ragged([[[], [1], [], [2, 3]], [[], [1], [], [2, 3]],
                             [[], [1], [], [2, 3]]]),
          }),
      dict(
          description="context features to nodes broadcasting, 2 components",
          context=gt.Context.from_fields(features={
              "scalar": as_tensor([1, 2]),
              "vector": as_tensor([[1.], [2.]]),
              "ragged": as_ragged([[[], [1], []], [[1], [], [2]]]),
          }),
          node_set=gt.NodeSet.from_fields(sizes=as_tensor([3, 2]), features={}),
          expected_node_fields={
              "scalar":
                  as_tensor([1, 1, 1, 2, 2]),
              "vector":
                  as_tensor([[1.], [1.], [1.], [2.], [2.]]),
              "ragged":
                  as_ragged([[[], [1], []], [[], [1], []], [[], [1], []],
                             [[1], [], [2]], [[1], [], [2]]]),
          })
  ])
  def testNodeFieldFromContext(self, description: str, context: gt.Context,
                               node_set: gt.NodeSet,
                               expected_node_fields: Mapping[str, const.Field]):
    del description
    graph = gt.GraphTensor.from_pieces(
        context=context, node_sets={"node": node_set})

    for fname, expected in expected_node_fields.items():
      self.assertAllEqual(
          expected,
          broadcast_ops.broadcast_context_to_nodes(
              graph, "node", feature_name=fname))
      self.assertAllEqual(
          expected,
          broadcast_ops.broadcast_v2(
              graph, const.CONTEXT, node_set_name="node", feature_name=fname))

  @parameterized.parameters([
      dict(
          description="context features to edges broadcasting, 1 component",
          context=gt.Context.from_fields(features={
              "scalar": as_tensor([1]),
              "vector": as_tensor([[1., 2.]]),
              "matrix": as_tensor([[[1., 2., 3.], [4., 5., 6.]]]),
              "ragged": as_ragged([[[], [1], [], [2, 3]]]),
          }),
          edge_set=gt.EdgeSet.from_fields(
              sizes=as_tensor([3]),
              adjacency=adj.HyperAdjacency.from_indices({
                  0: ("node", as_tensor([0, 0, 0])),
              }),
              features={}),
          expected_edge_fields={
              "scalar":
                  as_tensor([1] * 3),
              "vector":
                  as_tensor([[1., 2.]] * 3),
              "matrix":
                  as_tensor([[[1., 2., 3.], [4., 5., 6.]]] * 3),
              "ragged":
                  as_ragged([[[], [1], [], [2, 3]], [[], [1], [], [2, 3]],
                             [[], [1], [], [2, 3]]]),
          }),
      dict(
          description="context features to nodes broadcasting, 2 components",
          context=gt.Context.from_fields(features={
              "scalar": as_tensor([1, 2]),
              "vector": as_tensor([[1.], [2.]]),
              "ragged": as_ragged([[[], [1], []], [[1], [], [2]]]),
          }),
          edge_set=gt.EdgeSet.from_fields(
              sizes=as_tensor([3, 2]),
              adjacency=adj.HyperAdjacency.from_indices({
                  0: ("node", as_tensor([0, 0, 0, 0, 0])),
              }),
              features={}),
          expected_edge_fields={
              "scalar":
                  as_tensor([1, 1, 1, 2, 2]),
              "vector":
                  as_tensor([[1.], [1.], [1.], [2.], [2.]]),
              "ragged":
                  as_ragged([[[], [1], []], [[], [1], []], [[], [1], []],
                             [[1], [], [2]], [[1], [], [2]]]),
          })
  ])
  def testEdgeFieldFromContext(self, description: str, context: gt.Context,
                               edge_set: gt.EdgeSet,
                               expected_edge_fields: Mapping[str, const.Field]):
    del description
    graph = gt.GraphTensor.from_pieces(
        context=context, edge_sets={"edge": edge_set})

    for fname, expected in expected_edge_fields.items():
      self.assertAllEqual(
          expected,
          broadcast_ops.broadcast_context_to_edges(
              graph, "edge", feature_name=fname))
      self.assertAllEqual(
          expected,
          broadcast_ops.broadcast_v2(
              graph, const.CONTEXT, edge_set_name="edge", feature_name=fname))


class BroadcastV2Test(tf.test.TestCase, parameterized.TestCase):
  """Tests for generic broadcast_v2() wrapper.

  These tests assume correctness of the underlying broadcast_*_to_*() ops;
  see BroadcastXtoYTest for these.
  """

  def testOneEdgeSetFromTag(self):
    input_graph = _get_test_graph_broadcast()
    def call_broadcast(from_tag):
      return broadcast_ops.broadcast_v2(
          input_graph, from_tag, edge_set_name="e",
          feature_value=tf.constant([[1., 2.], [3., 4.], [5., 6.]]))
    self.assertAllClose(np.array([[1., 2.], [3., 4.], [1., 2.], [5., 6.]]),
                        call_broadcast(const.SOURCE).numpy())
    self.assertAllClose(np.array([[3., 4.], [3., 4.], [1., 2.], [5., 6.]]),
                        call_broadcast(const.TARGET).numpy())

  @parameterized.named_parameters(
      ("List", list),
      ("Tuple", tuple))
  def testOneEdgeSetSequenceType(self, sequence_cls):
    input_graph = _get_test_graph_broadcast()
    edge_set_name = sequence_cls(x for x in ["e"])
    actual = broadcast_ops.broadcast_v2(
        input_graph, const.SOURCE,
        edge_set_name=edge_set_name,
        feature_value=tf.constant([[1., 2.], [3., 4.], [5., 6.]]))
    self.assertIsInstance(actual, list)
    self.assertLen(actual, 1)
    self.assertAllClose(np.array([[1., 2.], [3., 4.], [1., 2.], [5., 6.]]),
                        actual[0].numpy())

  def testOneEdgeSetFeatureName(self):
    input_graph = _get_test_graph_broadcast()
    actual = broadcast_ops.broadcast_v2(
        input_graph, const.SOURCE,
        edge_set_name="e", feature_name="feat")
    self.assertAllClose(
        np.array([[10., 11.], [20., 21.], [10., 11.], [30., 31.]]),
        actual.numpy())

  def testTwoEdgeSets(self):
    input_graph = _get_test_graph_broadcast()
    actual = broadcast_ops.broadcast_v2(
        input_graph, const.SOURCE,
        edge_set_name=["e", "f"],
        feature_value=tf.constant([[1., 2.], [3., 4.], [5., 6.]]))
    self.assertLen(actual, 2)
    self.assertAllClose(np.array([[1., 2.], [3., 4.], [1., 2.], [5., 6.]]),
                        actual[0].numpy())
    self.assertAllClose(np.array([[5., 6.], [5., 6.]]),
                        actual[1].numpy())

  def testTwoEdgeSetsRagged(self):
    input_graph = _get_test_graph_broadcast()
    actual = broadcast_ops.broadcast_v2(
        input_graph, const.SOURCE,
        edge_set_name=["e", "f"],
        feature_value=tf.ragged.constant([[1.], [2., 3.], [4., 5., 6.]]))
    self.assertLen(actual, 2)
    self.assertAllClose(
        tf.ragged.constant([[1.], [2., 3.], [1.], [4., 5., 6.]]),
        actual[0])
    self.assertAllClose(
        tf.ragged.constant([[4., 5., 6.], [4., 5., 6.]]),
        actual[1])

  def testNodeSetsFromContext(self):
    input_graph = _get_test_graph_broadcast()
    actual = broadcast_ops.broadcast_v2(
        input_graph, const.CONTEXT,
        node_set_name=["a", "b"],
        feature_value=tf.constant([[1., 2.], [3., 4.]]))
    self.assertLen(actual, 2)
    self.assertAllClose(np.array([[1., 2.], [1., 2.], [3., 4.]]),
                        actual[0].numpy())
    self.assertAllClose(np.array([[1., 2.], [3., 4.], [3., 4.]]),
                        actual[1].numpy())

  def testNodeSetsFromContextFeatureName(self):
    input_graph = _get_test_graph_broadcast()
    actual = broadcast_ops.broadcast_v2(
        input_graph, const.CONTEXT,
        node_set_name=["a", "b"],
        feature_name="feat")
    self.assertLen(actual, 2)
    self.assertAllClose(np.array([[80., 81.], [80., 81.], [90., 91.]]),
                        actual[0].numpy())
    self.assertAllClose(np.array([[80., 81.], [90., 91.], [90., 91.]]),
                        actual[1].numpy())


def _get_test_graph_broadcast():
  return gt.GraphTensor.from_pieces(
      node_sets={
          "a": gt.NodeSet.from_fields(
              sizes=tf.constant([2, 1]),
              features={"feat": tf.constant(
                  [[10., 11.],
                   [20., 21.],
                   [30., 31.]])}),
          "b": gt.NodeSet.from_fields(
              sizes=tf.constant([1, 2])),
          "c": gt.NodeSet.from_fields(
              sizes=tf.constant([1, 1])),
      },
      edge_sets={
          "e": gt.EdgeSet.from_fields(
              sizes=tf.constant([3, 1]),
              adjacency=adj.Adjacency.from_indices(
                  ("a", tf.constant([0, 1, 0, 2])),
                  ("a", tf.constant([1, 1, 0, 2])))),
          "f": gt.EdgeSet.from_fields(
              sizes=tf.constant([0, 2]),
              adjacency=adj.Adjacency.from_indices(
                  ("a", tf.constant([2, 2])),
                  ("b", tf.constant([2, 1])))),
          "g": gt.EdgeSet.from_fields(
              sizes=tf.constant([1, 0]),
              adjacency=adj.Adjacency.from_indices(
                  ("a", tf.constant([0])),
                  ("c", tf.constant([0])))),
      },
      context=gt.Context.from_fields(
          features={"feat": tf.constant(
              [[80., 81.],
               [90., 91.]])})
    )


if __name__ == "__main__":
  tf.test.main()
