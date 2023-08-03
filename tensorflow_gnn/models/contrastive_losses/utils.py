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
"""Utils for contrastive losses."""

import tensorflow as tf
import tensorflow_gnn as tfgnn


class SliceNodeSetFeatures(tf.keras.layers.Layer):
  """A class that slices node set features at a particular index.

  Given a graph tensor comprising of node set features of N graph tensors
  stacked together to give features of shape (batch, N, **feature_dims) for each
  node set, outputs a graph tensor constructed only of the feature at index
  `index` (specified during the call) for each node set. The edge sets and
  context remain unchanged. For example:

  ```
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
  }
  edge_sets = {
      "a->a": tfgnn.EdgeSet.from_fields(
          sizes=tf.constant([2]),
          adjacency=tfgnn.Adjacency.from_indices(
              source=("a", [0, 1]),
              target=("a", [0, 0]),
          ),
      ),
  }
  inp = tfgnn.GraphTensor.from_pieces(node_sets=node_sets, edge_sets=edge_sets)
  out = SliceNodeSetFeatures()(inp, index=0)

  output_node_sets = {
      "a": tfgnn.NodeSet.from_fields(
          sizes=tf.constant([2]),
          features={
              "scalar": tf.constant([[1], [4]]),
              "vector": tf.constant([[1, 1], [4, 4]]),
          },
      ),
  }
  output_edge_sets = {
      "a->a": tfgnn.EdgeSet.from_fields(
          sizes=tf.constant([2]),
          adjacency=tfgnn.Adjacency.from_indices(
              source=("a", [0, 1]),
              target=("a", [0, 0]),
          ),
      ),
  }
  ```
  """

  def call(self, inputs: tfgnn.GraphTensor, index: int) -> tfgnn.GraphTensor:
    new_node_sets = {}
    for node_set_name, node_set in inputs.node_sets.items():
      new_node_set_feats = {}
      features = node_set.get_features_dict()
      for feature_name, feature_value in features.items():
        feat = feature_value[:, index]
        new_node_set_feats[feature_name] = feat
      new_node_sets[node_set_name] = tfgnn.NodeSet.from_fields(
          sizes=node_set.sizes,
          features=new_node_set_feats)

    return tfgnn.GraphTensor.from_pieces(
        node_sets=new_node_sets,
        edge_sets=inputs.edge_sets,
        context=inputs.context)
