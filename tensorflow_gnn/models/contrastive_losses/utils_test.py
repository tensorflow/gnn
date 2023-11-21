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
"""Tests for utils."""
from typing import Mapping

import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn.models.contrastive_losses import utils


class UtilsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    tfgnn.enable_graph_tensor_validation_at_runtime()

  def assertFieldsEqual(self, actual: tfgnn.Fields, expected: tfgnn.Fields):
    self.assertIsInstance(actual, Mapping)
    self.assertAllEqual(actual.keys(), expected.keys())
    for key in actual.keys():
      self.assertAllEqual(actual[key], expected[key], msg=f"feature={key}")

  def assertNodeSetsEqual(
      self, actual: tfgnn.GraphTensor, expected: tfgnn.GraphTensor):
    for node_set_name, node_set in expected.node_sets.items():
      assert(
          node_set_name in actual.node_sets.keys()
      ), f"node_set_name={node_set_name} not in actual."
      self.assertFieldsEqual(actual.node_sets[node_set_name].features,
                             node_set.features)

  def test_slice_graph_tensor(self):
    self._slice_graph_tensor = utils.SliceNodeSetFeatures()
    stacked_graph_tensor = _get_x_y_z_stacked()
    expected_x = _get_x()
    expected_y = _get_y()
    expected_z = _get_z()

    got_x = self._slice_graph_tensor(stacked_graph_tensor, index=0)
    self.assertNodeSetsEqual(got_x, expected_x)

    got_y = self._slice_graph_tensor(stacked_graph_tensor, index=1)
    self.assertNodeSetsEqual(got_y, expected_y)

    got_z = self._slice_graph_tensor(stacked_graph_tensor, index=2)
    self.assertNodeSetsEqual(got_z, expected_z)


def _get_x():
  node_sets = {
      "a": tfgnn.NodeSet.from_fields(
          sizes=tf.constant([2]),
          features={
              "scalar": tf.constant([[1], [4]]),
              "vector": tf.constant([[1, 1], [4, 4]]),
          },
      ),
      "b": tfgnn.NodeSet.from_fields(
          sizes=tf.constant([1]),
          features={
              "2d_vector": tf.constant([
                  [[1, 1], [1, 1]],
              ])
          },
      ),
  }
  edge_sets = {
      "a->b": tfgnn.EdgeSet.from_fields(
          sizes=tf.constant([2]),
          adjacency=tfgnn.Adjacency.from_indices(
              source=("a", [0, 1]),
              target=("b", [0, 0]),
          ),
      ),
  }
  return tfgnn.GraphTensor.from_pieces(node_sets=node_sets, edge_sets=edge_sets)


def _get_y():
  node_sets = {
      "a": tfgnn.NodeSet.from_fields(
          sizes=tf.constant([2]),
          features={
              "scalar": tf.constant([[2], [5]]),
              "vector": tf.constant([[2, 2], [5, 5]]),
          },
      ),
      "b": tfgnn.NodeSet.from_fields(
          sizes=tf.constant([1]),
          features={
              "2d_vector": tf.constant([
                  [[2, 2], [2, 2]],
              ])
          },
      ),
  }
  edge_sets = {
      "a->b": tfgnn.EdgeSet.from_fields(
          sizes=tf.constant([2]),
          adjacency=tfgnn.Adjacency.from_indices(
              source=("a", [0, 1]),
              target=("b", [0, 0]),
          ),
      ),
  }
  return tfgnn.GraphTensor.from_pieces(node_sets=node_sets, edge_sets=edge_sets)


def _get_z():
  node_sets = {
      "a": tfgnn.NodeSet.from_fields(
          sizes=tf.constant([2]),
          features={
              "scalar": tf.constant([[3], [6]]),
              "vector": tf.constant([[3, 3], [6, 6]]),
          },
      ),
      "b": tfgnn.NodeSet.from_fields(
          sizes=tf.constant([1]),
          features={
              "2d_vector": tf.constant([
                  [[3, 3], [3, 3]],
              ])
          },
      ),
  }
  edge_sets = {
      "a->b": tfgnn.EdgeSet.from_fields(
          sizes=tf.constant([2]),
          adjacency=tfgnn.Adjacency.from_indices(
              source=("a", [0, 1]),
              target=("b", [0, 0]),
          ),
      ),
  }
  return tfgnn.GraphTensor.from_pieces(node_sets=node_sets, edge_sets=edge_sets)


def _get_x_y_z_stacked():
  node_sets = {
      "a": tfgnn.NodeSet.from_fields(
          sizes=tf.constant([2]),
          features={
              "scalar": tf.constant([[[1], [2], [3]], [[4], [5], [6]]]),
              "vector": tf.constant(
                  [[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]]
              ),
          },
      ),
      "b": tfgnn.NodeSet.from_fields(
          sizes=tf.constant([1]),
          features={
              "2d_vector": tf.constant(
                  [[[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]]]
              )
          },
      ),
  }
  edge_sets = {
      "a->b": tfgnn.EdgeSet.from_fields(
          sizes=tf.constant([2]),
          adjacency=tfgnn.Adjacency.from_indices(
              source=("a", [0, 1]),
              target=("b", [0, 0]),
          ),
      ),
  }
  return tfgnn.GraphTensor.from_pieces(node_sets=node_sets, edge_sets=edge_sets)


if __name__ == "__main__":
  tf.test.main()
