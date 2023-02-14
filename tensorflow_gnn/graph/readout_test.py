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
"""Tests for readout_named() and associated functions."""

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import graph_tensor_ops as ops
from tensorflow_gnn.graph import readout


class ReadoutTest(tf.test.TestCase, parameterized.TestCase):
  """Tests readout_named() and supporting validation functions."""

  def testHomNodeClassification(self):
    test_graph = gt.GraphTensor.from_pieces(
        node_sets={
            "objects": gt.NodeSet.from_fields(
                sizes=tf.constant([3, 2]),
                features={const.HIDDEN_STATE: tf.constant(
                    [[1., 1.],  # Read out as "seed" 0.
                     [1., 2.],
                     [1., 3.],
                     [2., 4.],  # Read out as "seed" 1.
                     [2., 5.]])}),
            "_readout": gt.NodeSet.from_fields(
                # There is one readout per component.
                sizes=tf.constant([1, 1]),
                # Unrelated to the redaout op, this is how to store labels.
                features={"labels": tf.constant([[101], [201]])})},
        edge_sets={
            "relations": gt.EdgeSet.from_fields(
                sizes=tf.constant([2, 1]),
                adjacency=adj.Adjacency.from_indices(
                    ("objects", tf.constant([1, 2, 4])),
                    ("objects", tf.constant([0, 0, 3])))),
            "_readout/seed": gt.EdgeSet.from_fields(
                sizes=tf.constant([1, 1]),
                adjacency=adj.Adjacency.from_indices(
                    # The nodes for readout are defined here.
                    ("objects", tf.constant([0, 3])),
                    ("_readout", tf.constant([0, 1]))))})

    self.assertAllEqual(
        [[101], [201]],
        test_graph.node_sets["_readout"]["labels"])
    readout.validate_graph_tensor_spec_for_readout(test_graph.spec)
    readout.validate_graph_tensor_for_readout(test_graph, ["seed"])
    expected = tf.constant([[1., 1.], [2., 4.]])
    self.assertAllEqual(
        expected,
        readout.readout_named(test_graph, "seed",
                              feature_name=const.HIDDEN_STATE))
    # Same result as with the older method of ad-hoc readout.
    self.assertAllEqual(
        expected,
        ops.gather_first_node(test_graph, "objects",
                              feature_name=const.HIDDEN_STATE))

  def testEmptyReadout(self):
    test_graph = gt.GraphTensor.from_pieces(
        node_sets={
            "objects": gt.NodeSet.from_fields(
                sizes=tf.constant([1, 1]),
                features={const.HIDDEN_STATE: tf.constant(
                    [[1., 1.], [2., 2.]])}),
            "_readout": gt.NodeSet.from_fields(
                sizes=tf.constant([0, 0]))},
        edge_sets={
            "_readout/seed": gt.EdgeSet.from_fields(
                sizes=tf.constant([0, 0]),
                adjacency=adj.Adjacency.from_indices(
                    ("objects", tf.constant([], tf.int32)),
                    ("_readout", tf.constant([], tf.int32))))})

    readout.validate_graph_tensor_spec_for_readout(test_graph.spec)
    readout.validate_graph_tensor_for_readout(test_graph, ["seed"])
    self.assertAllEqual(
        tf.zeros([0, 2]),
        readout.readout_named(test_graph, "seed",
                              feature_name=const.HIDDEN_STATE))

  def testHomEdgeRegression(self):
    test_graph = gt.GraphTensor.from_pieces(
        node_sets={
            "items": gt.NodeSet.from_fields(
                sizes=tf.constant([3, 2]),
                features={const.HIDDEN_STATE: tf.constant(
                    [[1., 1.], [1., 2.], [1., 3.], [2., 1.], [2., 2.]])}),
            "_readout": gt.NodeSet.from_fields(
                # There is one readout per component.
                sizes=tf.constant([1, 1])),
            "_shadow/links": gt.NodeSet.from_fields(
                # A shadow node set, indexed like edge set "links".
                sizes=tf.constant([2, 2]))},
        edge_sets={
            "links": gt.EdgeSet.from_fields(
                sizes=tf.constant([2, 2]),
                features={const.HIDDEN_STATE: tf.constant(
                    [[1., 11.],
                     [1., 12.],  # Read out as "seed_edge" 0.
                     [2., 22.],  # Read out as "seed_edge" 1.
                     [2., 23.]])},
                adjacency=adj.Adjacency.from_indices(
                    ("items", tf.constant([0, 1, 3])),
                    ("items", tf.constant([1, 2, 4])))),
            "_readout/seed_edge": gt.EdgeSet.from_fields(
                sizes=tf.constant([1, 1]),
                adjacency=adj.Adjacency.from_indices(
                    # The edges for readout are defined here,
                    # notionally as shadow nodes of corresponding indices.
                    ("_shadow/links", tf.constant([1, 2])),
                    ("_readout", tf.constant([0, 1]))))})

    readout.validate_graph_tensor_spec_for_readout(test_graph.spec)
    readout.validate_graph_tensor_for_readout(test_graph, ["seed_edge"])
    self.assertAllEqual(
        [[1., 12.], [2., 22.]],
        readout.readout_named(test_graph, "seed_edge",
                              feature_name=const.HIDDEN_STATE))

  def testMultiTargetLinkPrediction(self):
    # Node states are [x, y, z] where
    # x in {1, 2} encodes users or items,
    # y counts the graph component (one-based, to make it less trivial),
    # z counts the nodes per component (one-based).
    test_graph = gt.GraphTensor.from_pieces(
        node_sets={
            "users": gt.NodeSet.from_fields(
                sizes=tf.constant([3, 2, 2, 2]),
                features={const.HIDDEN_STATE: tf.constant(
                    [[1., 1., 1.],
                     [1., 1., 2.],  # Read out as "target" 0.
                     [1., 1., 3.],  # Read out as "source" 0.
                     [1., 2., 1.],
                     [1., 2., 2.],  # Read out as "source" 1.
                     [1., 3., 1.],  # Read out as "target" 2.
                     [1., 3., 2.],  # Read out as "source" 2.
                     [1., 4., 1.],  # Read out as "source" 3.
                     [1., 4., 2.]])}),
            "items": gt.NodeSet.from_fields(
                sizes=tf.constant([2, 1, 1, 2]),
                features={const.HIDDEN_STATE: tf.constant(
                    [[2., 1., 1.],
                     [2., 1., 2.],
                     [2., 2., 1.],  # Read out as "target" 1.
                     [2., 3., 1.],
                     [2., 4., 1.],  # Read out as "target" 3.
                     [2., 4., 2.]])}),
            "_readout": gt.NodeSet.from_fields(
                sizes=tf.constant([1, 1, 1, 1]),
                features={"labels": tf.constant([0, 0, 1, 1])})},
        edge_sets={
            "is_friend_of": gt.EdgeSet.from_fields(
                sizes=tf.constant([1, 1, 0, 0]),
                adjacency=adj.Adjacency.from_indices(
                    ("users", tf.constant([1, 3])),
                    ("users", tf.constant([2, 4])))),
            "has_purchased": gt.EdgeSet.from_fields(
                sizes=tf.constant([1, 0, 1, 0]),
                adjacency=adj.Adjacency.from_indices(
                    ("users", tf.constant([1, 5])),
                    ("items", tf.constant([0, 3])))),
            "_readout/source/1": gt.EdgeSet.from_fields(
                sizes=tf.constant([1, 1, 1, 1]),
                adjacency=adj.Adjacency.from_indices(
                    # The "source" users are defined here.
                    ("users", tf.constant([2, 4, 6, 7])),
                    ("_readout", tf.constant([0, 1, 2, 3])))),
            "_readout/target/1": gt.EdgeSet.from_fields(
                sizes=tf.constant([1, 0, 1, 0]),
                adjacency=adj.Adjacency.from_indices(
                    # The "target" users are defined here.
                    ("users", tf.constant([1, 5])),
                    ("_readout", tf.constant([0, 2])))),
            "_readout/target/2": gt.EdgeSet.from_fields(
                sizes=tf.constant([0, 1, 0, 1]),
                adjacency=adj.Adjacency.from_indices(
                    # The "target" items are defined here.
                    ("items", tf.constant([2, 4])),
                    ("_readout", tf.constant([1, 3]))))})

    self.assertAllEqual(
        [0, 0, 1, 1],
        test_graph.node_sets["_readout"]["labels"])
    readout.validate_graph_tensor_spec_for_readout(test_graph.spec)
    readout.validate_graph_tensor_for_readout(test_graph, ["source", "target"])
    self.assertAllEqual(
        [[1., 1., 3.],
         [1., 2., 2.],
         [1., 3., 2.],
         [1., 4., 1.]],
        readout.readout_named(test_graph, "source",
                              feature_name=const.HIDDEN_STATE))
    self.assertAllEqual(
        [[1., 1., 2.],
         [2., 2., 1.],
         [1., 3., 1.],
         [2., 4., 1.]],
        readout.readout_named(test_graph, "target",
                              feature_name=const.HIDDEN_STATE))

  def testBadReadoutIndices(self):
    test_graph = gt.GraphTensor.from_pieces(
        node_sets={
            "objects": gt.NodeSet.from_fields(
                sizes=tf.constant([1, 1]),
                features={const.HIDDEN_STATE: tf.constant(
                    [[1., 1.], [2., 2.]])}),
            "_readout": gt.NodeSet.from_fields(
                sizes=tf.constant([1, 1]))},
        edge_sets={
            "_readout/seed": gt.EdgeSet.from_fields(
                sizes=tf.constant([1, 1]),
                adjacency=adj.Adjacency.from_indices(
                    ("objects", tf.constant([0, 1], tf.int32)),
                    ("_readout", tf.constant([1, 0], tf.int32))))})
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                "Not strictly sorted by target"):
      _ = readout.readout_named(test_graph, "seed",
                                feature_name=const.HIDDEN_STATE)

# TODO(b/269076334): Test error detection more completely: all cases,
# also from within Dataset.map().
# TODO(b/269076334): Test alternative values of readout_node_set.


if __name__ == "__main__":
  tf.test.main()
