"""Tests for triples."""

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
import tensorflow as tf
from tensorflow_gnn.converters import triples


class TriplesTest(tf.test.TestCase):

  def test_triples_from_array(self):
    spos = [["gnns", "are", "awesome"], ["tfgnn", "is", "awesome"],
            ["nns", "are", "awesome"]]

    gt = triples.triple_to_graphtensor(triples=spos)
    is_source = int(gt.edge_sets["is"].adjacency.source)
    is_target = int(gt.edge_sets["is"].adjacency.target)

    self.assertLen(gt.node_sets, 1)
    self.assertLen(gt.edge_sets, 2)

    self.assertEqual(gt.node_sets["nodes"].sizes, 4)
    self.assertEqual(gt.node_sets["nodes"].features["#id"].shape, 4)

    self.assertEqual(gt.node_sets["nodes"].features["#id"][is_source], "tfgnn")
    self.assertEqual(gt.node_sets["nodes"].features["#id"][is_target],
                     "awesome")


if __name__ == "__main__":
  tf.test.main()
