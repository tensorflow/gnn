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
"""Tests for NARS feature aggregation utils."""
from typing import List, Sequence
from absl.testing import parameterized

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models.nars import utils


class GenerateRelationalSubgraphFeatureTest(tf.test.TestCase,
                                            parameterized.TestCase):

  def test_generate_relational_subgraph(self):
    input_graph = _make_test_graph_abcuv()
    expected = _make_test_graph_abu()
    relational_subset = ["u"]
    actual = utils.generate_relational_subgraph(
        input_graph, relational_subset)
    assert actual.node_sets.keys() == expected.node_sets.keys()
    assert actual.edge_sets.keys() == expected.edge_sets.keys()

  def test_initialize_subgraph_features(self):
    relational_subgraph = _make_test_graph_abu()
    actual = utils.initialize_subgraph_features(relational_subgraph)

    tf.assert_equal(actual["a"]["norm"],
                    tf.constant([[1 / 2], [1 / 2], [1.0], [0.0]]))
    tf.assert_equal(actual["a"]["hop_0"],
                    relational_subgraph.node_sets["a"][tfgnn.HIDDEN_STATE])
    tf.assert_equal(actual["b"]["norm"],
                    tf.constant([[1 / 3], [1 / 2], [0.0]]))
    tf.assert_equal(actual["b"]["hop_0"],
                    relational_subgraph.node_sets["b"][tfgnn.HIDDEN_STATE])

  def test_one_hop_feature_aggregation(self):
    relational_subgraph = _make_test_graph_abu()
    subgraph_feature_dict = utils.initialize_subgraph_features(
        relational_subgraph)
    edge_set_name = "u"
    feature_name = tfgnn.HIDDEN_STATE
    # initialize node_type_to_feat to zeros
    node_type_to_feat = {
        node_set_name: tf.zeros_like(node_set[feature_name])
        for node_set_name, node_set in relational_subgraph.node_sets.items()
    }
    hop = 1
    receiver_tag = tfgnn.SOURCE  # node set "a"
    node_type_to_feat = utils.one_hop_feature_aggregation(
        relational_subgraph, edge_set_name, subgraph_feature_dict,
        node_type_to_feat, hop, receiver_tag)

    tf.assert_equal(node_type_to_feat["a"],
                    tf.constant([[3.0], [3.0], [1.0], [0.0]]))

    receiver_tag = tfgnn.TARGET  # node set "b"
    node_type_to_feat = utils.one_hop_feature_aggregation(
        relational_subgraph, edge_set_name, subgraph_feature_dict,
        node_type_to_feat, hop, receiver_tag)

    tf.assert_equal(node_type_to_feat["a"],
                    tf.constant([[3.0], [3.0], [1.0], [0.0]]))
    tf.assert_equal(node_type_to_feat["b"],
                    tf.constant([[6.0], [3.0], [0.0]]))

  @parameterized.named_parameters(
      {
          "testcase_name": "OneHopOneEdgeSetForNodeSetA",
          "relational_subset": ["u"],
          "num_hops": 1,
          "root_node_set_name": "a",
          "expected": [
              tf.constant([[1.0], [2.0], [3.0], [4.0]]),
              tf.constant([[1.5], [1.5], [1.0], [0.0]])]
      },
      {
          "testcase_name": "OneHopOneEdgeSetForNodeSetB",
          "relational_subset": ["u"],
          "num_hops": 1,
          "root_node_set_name": "b",
          "expected": [
              tf.constant([[1.0], [2.0], [3.0]]),
              tf.constant([[2.0], [1.5], [0.0]])]
      },
      {
          "testcase_name": "TwoHopOneEdgeSetForNodeSetA",
          "relational_subset": ["u"],
          "num_hops": 2,
          "root_node_set_name": "a",
          "expected": [
              tf.constant([[1.0], [2.0], [3.0], [4.0]]),
              tf.constant([[1.5], [1.5], [1.0], [0.0]]),
              tf.constant([[1.75], [1.75], [2.0], [0.0]])]
      },
      {
          "testcase_name": "TwoHopOneEdgeSetForNodeSetB",
          "relational_subset": ["u"],
          "num_hops": 2,
          "root_node_set_name": "b",
          "expected": [
              tf.constant([[1.0], [2.0], [3.0]]),
              tf.constant([[2.0], [1.5], [0.0]]),
              tf.constant([[4/3], [1.5], [0.0]])]
      },
      {
          "testcase_name": "OneHopTwoEdgeSetForNodeSetA",
          "relational_subset": ["u", "v"],
          "num_hops": 1,
          "root_node_set_name": "a",
          "expected": [
              tf.constant([[1.0], [2.0], [3.0], [4.0]]),
              tf.constant([[7/3], [1.5], [1.0], [0.0]])]
      },
      {
          "testcase_name": "TwoHopTwoEdgeSetForNodeSetA",
          "relational_subset": ["u", "v"],
          "num_hops": 2,
          "root_node_set_name": "a",
          "expected": [
              tf.constant([[1.0], [2.0], [3.0], [4.0]]),
              tf.constant([[7/3], [1.5], [1.0], [0.0]]),
              tf.constant([[1.5], [1.75], [2.0], [0.0]])]
      },
  )
  def test_generate_relational_subgraph_features(
      self, relational_subset: Sequence[str], num_hops: int,
      root_node_set_name: str, expected: List[tfgnn.Field]):
    input_graph = _make_test_graph_abcuv()
    relational_subgraph = utils.generate_relational_subgraph(
        input_graph, relational_subset
    )
    actual = utils.generate_relational_subgraph_features(
        relational_subgraph, num_hops=num_hops,
        root_node_set_name=root_node_set_name)

    # Calculation for one edge set "u"
    # "norm_a": tf.constant([[1 / 2], [1 / 2], [1.0], [0.0]])
    # "norm_b": tf.constant([[1 / 3], [1 / 2], [0.0]])

    # Round Zero
    # "a" :  tf.constant([[1.0], [2.0], [3.0], [4.0]])
    # "b" :  tf.constant([[1.0], [2.0], [3.0]])

    # Round One (normalized aggregated features)
    # "a" :  tf.constant([[1.5], [1.5], [1.0], [0.0]])
    # "b" :  tf.constant([[2.0], [1.5], [0.0]])

    # Round Two (normalized aggregated features)
    # "a" :  tf.constant([[1.75], [1.75], [2.0], [0.0]])
    # "b" :  tf.constant([[4/3], [1.5], [0.0]])

    assert len(actual) == num_hops + 1
    for i in range(num_hops + 1):
      # need to choose allclose as (sum / node_degree) is giving slight
      # numerical differences given the order of operations.
      tf.experimental.numpy.allclose(actual[i], expected[i])

  def test_preprocess_features(self):
    input_graph = _make_test_graph_abcuv()
    relational_subsets = [["u"], ["u", "v"]]
    num_hops = 2
    root_node_set_name = "a"
    actual = utils.preprocess_features(
        input_graph,
        relational_subsets,
        num_hops=num_hops,
        root_node_set_name=root_node_set_name,
    )

    # Combining the two cases of previous test
    # relational_subset = ["u"] and ["u", "v"] for node set "a"
    # Arranged with outer dimension being number of hops
    expected = [
        # Hop 0
        [
            tf.constant([[[1.0], [1.0]],
                         [[2.0], [2.0]],
                         [[3.0], [3.0]],
                         [[4.0], [4.0]]])
        ],
        # Hop 1
        [
            tf.constant([[[1.5], [7 / 3]],
                         [[1.5], [1.5]],
                         [[1.0], [1.0]],
                         [[0.0], [0.0]]]),

        ],
        # Hop 2
        [
            tf.constant([[[1.75], [1.5]],
                         [[1.75], [1.75]],
                         [[2.0], [2.0]],
                         [[0.0], [0.0]]]),
        ],
    ]
    for i in range(num_hops + 1):
      # need to choose allclose as (sum / node_degree) is giving slight
      # numerical differences given the order of operations.
      tf.experimental.numpy.allclose(actual[i], expected[i])


def _make_test_graph_abcuv():
  return tfgnn.GraphTensor.from_pieces(
      node_sets={
          "a": tfgnn.NodeSet.from_fields(
              sizes=tf.constant([4]),
              features={
                  tfgnn.HIDDEN_STATE: tf.constant([[1.0], [2.0], [3.0], [4.0]])
              },
          ),
          "b": tfgnn.NodeSet.from_fields(
              sizes=tf.constant([3]),
              features={tfgnn.HIDDEN_STATE: tf.constant([[1.0], [2.0], [3.0]])},
          ),
          "c": tfgnn.NodeSet.from_fields(
              sizes=tf.constant([3]),
              features={tfgnn.HIDDEN_STATE: tf.constant([[4.0], [5.0], [6.0]])},
          ),
      },
      edge_sets={
          "u": tfgnn.EdgeSet.from_fields(
              sizes=tf.constant([1]),
              adjacency=tfgnn.Adjacency.from_indices(
                  ("a", tf.constant([0, 0, 1, 1, 2])),
                  ("b", tf.constant([0, 1, 0, 1, 0]))
              ),
          ),
          "v": tfgnn.EdgeSet.from_fields(
              sizes=tf.constant([1]),
              adjacency=tfgnn.Adjacency.from_indices(
                  ("a", tf.constant([0])), ("c", tf.constant([0]))
              ),
          ),
      },
      context=tfgnn.Context.from_fields(
          sizes=tf.constant([1]), features={"cf": tf.constant([[2.0]])}
      ),
  )


def _make_test_graph_abu():
  return tfgnn.GraphTensor.from_pieces(
      node_sets={
          "a": tfgnn.NodeSet.from_fields(
              sizes=tf.constant([4]),
              features={
                  tfgnn.HIDDEN_STATE: tf.constant([[1.0], [2.0], [3.0], [4.0]])
              },
          ),
          "b": tfgnn.NodeSet.from_fields(
              sizes=tf.constant([3]),
              features={tfgnn.HIDDEN_STATE: tf.constant([[1.0], [2.0], [3.0]])},
          ),
      },
      edge_sets={
          "u": tfgnn.EdgeSet.from_fields(
              sizes=tf.constant([1]),
              features={tfgnn.HIDDEN_STATE: tf.constant([[1., 0.]])},
              adjacency=tfgnn.Adjacency.from_indices(
                  ("a", tf.constant([0, 0, 1, 1, 2])),
                  ("b", tf.constant([0, 1, 0, 1, 0])))),
      },
      context=tfgnn.Context.from_fields(
          sizes=tf.constant([1]),
          features={"cf": tf.constant([[2.]])}))


if __name__ == "__main__":
  tf.test.main()
