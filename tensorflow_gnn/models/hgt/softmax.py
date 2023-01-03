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
"""Contains the softmax implementation for Heterogeneous Graph Transformer.
"""
from __future__ import annotations

from typing import Mapping

import tensorflow as tf
import tensorflow_gnn as tfgnn


def _correct_node_set_at_tag(
    gt, edge_set_name, node_set_name, node_tag):

  return gt.edge_sets[edge_set_name].adjacency.node_set_name(
      node_tag) == node_set_name


def global_segmented_softmax_edges_per_node(
    graph: tfgnn.GraphTensor,
    node_tag: tfgnn.IncidentNodeTag,
    feature_values: Mapping[tfgnn.EdgeSetName, tfgnn.Field],
) -> tfgnn.Fields:
  """softmax() over edges where `node_set_name` is the `node_tag` node."""

  # Ensure that all the edge sets have the same node_set at the desired node_tag
  if len(
      set(graph.edge_sets[edge_set_name].adjacency.node_set_name(node_tag)
          for edge_set_name in feature_values)) != 1:
    msg_tag = "target" if node_tag == 1 else "source"
    raise ValueError(
        f"all edge sets in feature_values need the same {msg_tag} node set")

  maxes = []
  exps_by_edge = {}
  sumexps = []
  softmax_segments = {}
  # First get the maxes to subtract out
  for edge_set_name, value in feature_values.items():
    maxes.append(
        tfgnn.pool_edges_to_node(
            graph,
            edge_set_name,
            node_tag,
            reduce_type="max",
            feature_value=value))

  max_by_node = tf.keras.layers.maximum(maxes)

  # Next get all the offset exp sums, saving the per-edge exp sums
  for edge_set_name, value in feature_values.items():
    max_cast = tfgnn.broadcast_node_to_edges(
        graph,
        edge_set_name,
        node_tag,
        feature_value=max_by_node,
    )
    exps_by_edge[edge_set_name] = tf.exp(value - max_cast)
    sumexps.append(
        tfgnn.pool_edges_to_node(
            graph,
            edge_set_name,
            node_tag,
            reduce_type="sum",
            feature_value=exps_by_edge[edge_set_name],
        ))
  sum_by_node = tf.math.add_n(sumexps)

  # Finally compute the softmax segments per edge set
  for edge_set_name, exps in exps_by_edge.items():
    softmax_segments[edge_set_name] = (
        exps / tfgnn.broadcast_node_to_edges(
            graph, edge_set_name, node_tag, feature_value=sum_by_node))
  return softmax_segments
