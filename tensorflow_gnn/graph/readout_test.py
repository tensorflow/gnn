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
"""Tests for structured_readout() and associated functions."""

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import graph_tensor_ops as ops
from tensorflow_gnn.graph import readout


class StructuredReadoutTest(tf.test.TestCase):
  """Tests structured_readout() and supporting validation functions.

  TFLite integration is tested with tfgnn.keras.layers.StructuredReadout.
  """

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
        readout.structured_readout(test_graph, "seed",
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
        readout.structured_readout(test_graph, "seed",
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
        readout.structured_readout(test_graph, "seed_edge",
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
        readout.structured_readout(test_graph, "source",
                                   feature_name=const.HIDDEN_STATE))
    self.assertAllEqual(
        [[1., 1., 2.],
         [2., 2., 1.],
         [1., 3., 1.],
         [2., 4., 1.]],
        readout.structured_readout(test_graph, "target",
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
      _ = readout.structured_readout(test_graph, "seed",
                                     feature_name=const.HIDDEN_STATE)

# TODO(b/269076334): Test error detection more completely: all cases,
# also from within Dataset.map().
# TODO(b/269076334): Test alternative values of readout_node_set.


class StructuredReadoutIntoFeatureTest(tf.test.TestCase,
                                       parameterized.TestCase):
  """Tests structured_readout_into_feature().

  TFLite integration is tested with
  tfgnn.keras.layers.StructuredReadoutIntoFeature.
  """

  @parameterized.named_parameters(
      ("KeepInput", False),
      ("RemoveInput", True))
  def test(self, remove_input_feature):
    test_graph = gt.GraphTensor.from_pieces(
        node_sets={
            "objects": gt.NodeSet.from_fields(
                sizes=tf.constant([2]),
                features={"labels": tf.constant([1, 2]),
                          "zeros": tf.zeros([2])}),
            "unrelated": gt.NodeSet.from_fields(
                sizes=tf.constant([1]),
                features={"labels": tf.constant([9, 9]),
                          "stuff": tf.constant([[3.14, 2.71]])}),
            "_readout": gt.NodeSet.from_fields(
                sizes=tf.constant([2]),
                features={"other": tf.constant([9., 9.])}),
            "_shadow/links": gt.NodeSet.from_fields(
                sizes=tf.constant([2]))},
        edge_sets={
            "links": gt.EdgeSet.from_fields(
                sizes=tf.constant([2]),
                adjacency=adj.Adjacency.from_indices(
                    ("objects", tf.constant([0, 1])),
                    ("objects", tf.constant([1, 0]))),
                features={"labels": tf.constant([3, 4]),
                          "ones": tf.ones([2, 1])}),
            "_readout/the_key/1": gt.EdgeSet.from_fields(
                sizes=tf.constant([1]),
                adjacency=adj.Adjacency.from_indices(
                    ("objects", tf.constant([1])),
                    ("_readout", tf.constant([0])))),
            "_readout/the_key/2": gt.EdgeSet.from_fields(
                sizes=tf.constant([1]),
                adjacency=adj.Adjacency.from_indices(
                    ("_shadow/links", tf.constant([1])),
                    ("_readout", tf.constant([1]))))})

    graph = readout.structured_readout_into_feature(
        test_graph, "the_key", feature_name="labels", new_feature_name="target",
        remove_input_feature=remove_input_feature)

    # Check read-out features.
    self.assertCountEqual(["target", "other"],
                          graph.node_sets["_readout"].features.keys())
    self.assertAllEqual([2, 4], graph.node_sets["_readout"]["target"])

    # Check features on non-aux node sets and edge sets.
    self.assertCountEqual(["zeros"] + ["labels"] * (not remove_input_feature),
                          graph.node_sets["objects"].features.keys())
    self.assertCountEqual(["ones"] + ["labels"] * (not remove_input_feature),
                          graph.edge_sets["links"].features.keys())
    self.assertCountEqual(["stuff", "labels"],
                          graph.node_sets["unrelated"].features.keys())

  def testOverwrite(self):
    test_graph = gt.GraphTensor.from_pieces(
        node_sets={
            "objects": gt.NodeSet.from_fields(
                sizes=tf.constant([2]),
                features={"labels": tf.constant([1, 2]),
                          "zeros": tf.zeros([2])}),
            "_readout": gt.NodeSet.from_fields(
                sizes=tf.constant([2]),
                features={"target": tf.constant([9, 9]),
                          "other": tf.constant([9., 9.])})},
        edge_sets={
            "_readout/the_key": gt.EdgeSet.from_fields(
                sizes=tf.constant([1]),
                adjacency=adj.Adjacency.from_indices(
                    ("objects", tf.constant([1, 0])),
                    ("_readout", tf.constant([0, 1]))))})

    with self.assertRaisesRegex(ValueError, r"already exists"):
      _ = readout.structured_readout_into_feature(
          test_graph, "the_key", feature_name="labels",
          new_feature_name="target")

    graph = readout.structured_readout_into_feature(
        test_graph, "the_key", feature_name="labels",
        new_feature_name="target", overwrite=True)
    self.assertAllEqual([2, 1], graph.node_sets["_readout"]["target"])


class AddReadoutFromFirstNodeTest(tf.test.TestCase):
  """Tests add_readout_from_first_node().

  TFLite integration is tested with tfgnn.keras.layers.AddReadoutFromFirstNode.
  """

  def test(self):
    test_graph = gt.GraphTensor.from_pieces(
        node_sets={
            "objects": gt.NodeSet.from_fields(
                sizes=tf.constant([2, 3]),
                features={"elevens": tf.constant([11, 22, 33, 44, 55])}),
            "unrelated": gt.NodeSet.from_fields(
                sizes=tf.constant([1, 1]),
                features={"elevens": tf.constant([99, 99])})},
        edge_sets={
            "links": gt.EdgeSet.from_fields(
                sizes=tf.constant([1, 1]),
                adjacency=adj.Adjacency.from_indices(
                    ("objects", tf.constant([1, 3])),
                    ("unrelated", tf.constant([0, 1]))))})

    graph = readout.add_readout_from_first_node(
        test_graph, "my_key", node_set_name="objects")
    self.assertAllEqual(
        [11, 33],
        readout.structured_readout(graph, "my_key", feature_name="elevens"))


class ContextReadoutIntoFeatureTest(tf.test.TestCase, parameterized.TestCase):
  """Tests context_readout_into_feature()."""

  @parameterized.named_parameters(
      ("KeepInput", False),
      ("RemoveInput", True))
  def test(self, remove_input_feature):
    test_graph = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(
            features={"labels": tf.constant([1, 2]),
                      "zeros": tf.zeros([2])}),
        node_sets={
            "unrelated": gt.NodeSet.from_fields(
                sizes=tf.constant([1]),
                features={"labels": tf.constant([9, 9]),
                          "stuff": tf.constant([[3.14, 2.71]])})})

    graph = readout.context_readout_into_feature(
        test_graph, feature_name="labels", new_feature_name="target",
        remove_input_feature=remove_input_feature)

    # Check read-out features.
    self.assertAllEqual([1, 2], graph.node_sets["_readout"]["target"])

    # Check features on non-aux node sets and edge sets.
    self.assertCountEqual(["zeros"] + ["labels"] * (not remove_input_feature),
                          graph.context.features.keys())
    self.assertCountEqual(["stuff", "labels"],
                          graph.node_sets["unrelated"].features.keys())

  def testExistingReadout(self):
    test_graph = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(
            features={"labels": tf.constant([1, 2]),
                      "zeros": tf.zeros([2])}),
        node_sets={
            "objects": gt.NodeSet.from_fields(
                sizes=tf.constant([2, 3]),
                features={"labels": tf.constant([11, 22, 33, 44, 55]),
                          "ones": tf.ones([5, 1])}),
            "unrelated": gt.NodeSet.from_fields(
                sizes=tf.constant([1]),
                features={"labels": tf.constant([9, 9]),
                          "stuff": tf.constant([[3.14, 2.71]])})})
    # Put a readout node set like a multi-task training pipeline with
    # context and seed node features would.
    test_graph = readout.add_readout_from_first_node(test_graph, "seed",
                                                     node_set_name="objects")
    self.assertCountEqual(["objects", "unrelated", "_readout"],
                          test_graph.node_sets.keys())
    test_graph = readout.structured_readout_into_feature(
        test_graph, "seed", feature_name="labels",
        new_feature_name="seed_labels")
    self.assertAllEqual([11, 33],
                        test_graph.node_sets["_readout"]["seed_labels"])

    graph = readout.context_readout_into_feature(
        test_graph, feature_name="labels", new_feature_name="graph_labels")

    # Check read-out features.
    self.assertAllEqual([1, 2], graph.node_sets["_readout"]["graph_labels"])
    self.assertAllEqual([11, 33], graph.node_sets["_readout"]["seed_labels"])

  def testOverwrite(self):
    test_graph = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(
            features={"labels": tf.constant([1, 2]),
                      "zeros": tf.zeros([2])}),
        node_sets={
            "_readout": gt.NodeSet.from_fields(
                sizes=tf.constant([1, 1]),
                features={"target": tf.constant([[9], [9]]),
                          "other": tf.constant([9., 9.])})})

    with self.assertRaisesRegex(ValueError, r"already exists"):
      _ = readout.context_readout_into_feature(
          test_graph, feature_name="labels", new_feature_name="target")

    graph = readout.context_readout_into_feature(
        test_graph, feature_name="labels", new_feature_name="target",
        overwrite=True)
    self.assertAllEqual([1, 2], graph.node_sets["_readout"]["target"])


if __name__ == "__main__":
  tf.test.main()
