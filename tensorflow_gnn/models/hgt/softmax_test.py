# Copyright 2022 The TensorFlow GNN Authors. All Rights Reserved.
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
"""Tests for hgt."""
import math

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models.hgt import softmax

ct = tf.constant
# Graph structure for the tests in this suite
gt = tfgnn.GraphTensor.from_pieces(
    node_sets={
        "air": tfgnn.NodeSet.from_fields(sizes=[3],),
        "ground": tfgnn.NodeSet.from_fields(sizes=[2],)
    },
    edge_sets={
        "aa":
            tfgnn.EdgeSet.from_fields(
                adjacency=tfgnn.Adjacency.from_indices(
                    ("air", [0, 1, 2, 1]),
                    ("air", [1, 2, 1, 0]),
                ),
                sizes=[4],
            ),
        "ga":
            tfgnn.EdgeSet.from_fields(
                adjacency=tfgnn.Adjacency.from_indices(
                    ("ground", [0, 1, 0]),
                    ("air", [2, 0, 2]),
                ),
                sizes=[3],
            ),
    },
)


class HgtSoftmaxTest(tf.test.TestCase, parameterized.TestCase):

  def test_hgt_softmax_different_node_sets(self):
    feature_values = {
        "aa": ct([50., 10000, 0, 0]),
        "ga": ct([10., 50, 0]),
    }
    test_f = softmax.global_segmented_softmax_edges_per_node
    self.assertRaisesRegex(
        ValueError,
        "all edge sets in feature_values need the same source node set",
        lambda: test_f(gt, tfgnn.SOURCE, feature_values))

  def test_hgt_softmax_single_edge_set(self):
    hom_gt = tfgnn.homogeneous(
        source=ct([0, 1, 2, 3]),
        target=ct([1, 2, 3, 1]),
        node_features=ct([1., 1, 1, 1]),
    )
    casted = tfgnn.broadcast_node_to_edges(
        hom_gt,
        tfgnn.EDGES,
        tfgnn.TARGET,
        feature_value=hom_gt.node_sets[tfgnn.NODES][tfgnn.HIDDEN_STATE])
    got = softmax.global_segmented_softmax_edges_per_node(
        hom_gt, tfgnn.TARGET, {tfgnn.EDGES: casted})
    want = tfgnn.softmax_edges_per_node(
        hom_gt, tfgnn.EDGES, tfgnn.TARGET, feature_value=casted)
    # Node 1 is target for 0 and 3
    # Nodes 2 and 3 are targets for 1 and 2 respectively
    want_numeric = ct([0.5, 1., 1., 0.5])
    self.assertAllClose(got[tfgnn.EDGES], want)
    self.assertAllClose(got[tfgnn.EDGES], want_numeric)
    source_casted = tfgnn.broadcast_node_to_edges(
        hom_gt,
        tfgnn.EDGES,
        tfgnn.SOURCE,
        feature_value=hom_gt.node_sets[tfgnn.NODES][tfgnn.HIDDEN_STATE])
    got_source = softmax.global_segmented_softmax_edges_per_node(
        hom_gt, tfgnn.SOURCE, {tfgnn.EDGES: source_casted})
    want_source_numeric = ct([1., 1., 1., 1.])
    self.assertAllClose(got_source[tfgnn.EDGES], want_source_numeric)

  def test_hgt_softmax_multiple_edge_sets(self):
    # Node 0 is target in edge set "aa" at position 3 and "ga" at position 1
    # Node 1 is target in edge set "aa" at positions 0,2
    # Node 2 is target in edge set "aa" at position 1 and "ga" at positions 0,2
    # The feature values are the logs of desired relative proportions.
    # Weighting for nodes are 80/20 for node 0, 60/40 for node 1, 50/25/25 for
    # node 3
    feature_values = {
        "aa": tf.math.log(ct([60., 50, 40, 20])),
        "ga": tf.math.log(ct([25., 80, 25])),
    }
    got = softmax.global_segmented_softmax_edges_per_node(
        gt, tfgnn.TARGET, feature_values
    )
    want = {
        "aa": ct([0.6, 0.5, 0.4, 0.2]),
        "ga": ct([0.25, 0.8, 0.25]),
    }
    for edge_set_name in gt.edge_sets:
      self.assertAllClose(got[edge_set_name], want[edge_set_name])

  def test_hgt_softmax_per_receiver_max(self):
    # Softmax implementation involves subtracting the max to prevent overflow
    # We want to ensure that large values for one receiver don't cause
    # underflow for other nodes. `math.exp(88)` is barely under the max
    # for a float32 (`math.exp(89)` overflows). If we subtracted 88
    # (the global max) from all values here, there would be underflow
    # for position 1 in aa and positions 0,2 in ga
    # and the result at those positions would be all `nan`.
    feature_values = {
        "aa": ct([math.log(60), -87, math.log(40), 0]),
        "ga": ct([-2., 88, -2]),
    }
    got = softmax.global_segmented_softmax_edges_per_node(
        gt, tfgnn.TARGET, feature_values
    )
    want = {
        "aa": ct([0.6, 0, 0.4, 0]),
        "ga": ct([0.5, 1, 0.5]),
    }
    for edge_set_name in gt.edge_sets:
      self.assertAllClose(got[edge_set_name], want[edge_set_name])

if __name__ == "__main__":
  tf.test.main()
