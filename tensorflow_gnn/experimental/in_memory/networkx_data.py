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
"""Wraps a homogeneous (simple) NetworkX graph for use with GraphTensor."""

import collections
from typing import Dict, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.experimental.in_memory import datasets

Example = tf.train.Example


class NetworkXDiGraphData(datasets.InMemoryGraphData):
  """Implementation of in-memory dataset loader for networkx format.

  This class only wraps a homogeneous (simple) NetworkX graph (networkx.Graph
  or networkx.DiGraph) for use with GraphTensor.

  The schema used for this graph has one node set ("nodes") and two edge sets
  {"edges", "rev_edges"} containing the forward and reverse directed edges
  respectively.

  Note: This converter currently ignores node and edge features.
  """

  def __init__(
      self,
      nx_graph,
      graph_schema: Optional[tfgnn.GraphSchema] = None,
  ):
    """Constructor to represent a NetworkX graph in memory.

    Args:
      nx_graph: A networkx DiGraph.
      graph_schema: A tfgnn.GraphSchema protobuf message.
    """
    super().__init__()
    self._graph_schema = graph_schema

    # Make "compression map":
    #   String Node ID ->  auto-incrementing int node id
    self.graph = nx_graph
    self.compression_map = {}
    self.reverse_compression_map = {}
    for idx, node_id in enumerate(sorted(nx_graph.nodes())):
      self.compression_map[node_id] = idx
      self.reverse_compression_map[idx] = node_id

    # We'll also create a version of the graph with compressed ids.
    src = []
    dst = []
    weight = []
    for _, node_id in enumerate(sorted(nx_graph.nodes())):
      for dst_id in nx_graph[node_id]:
        src.append(self.compression_map[node_id])
        dst.append(self.compression_map[dst_id])
        if "weight" in nx_graph[node_id][dst_id]:
          weight.append(nx_graph[node_id][dst_id]["weight"])
        else:
          weight.append(1.0)

    self.flat_adjacency = sp.coo_array(
        (weight, (src, dst)),
        shape=(len(self.compression_map), len(self.compression_map)),
        dtype="float",
    )

    # Add node feature support (this constructor would take a list of
    # node features to copy from the networkx graph)
    self.node_features = {}

  def get_adjacency_list(self) -> Dict[tfgnn.EdgeSetName, Dict[str, Example]]:
    """Returns weighted edges as an adjacency list of nested dictionaries.

    This function is useful for testing for fast access to edge features based
    on (source, target) IDs. Beware, this function will create an object that
    may increase memory usage.
    """
    adjacency_sets = collections.defaultdict(dict)
    adjacency_sets["edges"] = collections.defaultdict(dict)
    for source, target, weight in zip(
        self.flat_adjacency.row,
        self.flat_adjacency.col,
        self.flat_adjacency.data,
    ):
      e = tf.train.Example(
          features=tf.train.Features(
              feature={
                  "weight": tf.train.Feature(
                      float_list=tf.train.FloatList(value=[weight])
                  )
              }
          )
      )
      adjacency_sets["edges"][source][target] = e

    return adjacency_sets

  def node_features_dicts(
      self,
  ) -> Mapping[tfgnn.NodeSetName, MutableMapping[tfgnn.FieldName, tf.Tensor]]:
    ret = {}
    ret["nodes"] = self.node_features
    return ret

  def node_counts(self) -> Mapping[tfgnn.NodeSetName, int]:
    """Returns total number of graph nodes per node set."""
    return {"nodes": len(self.compression_map)}

  def edge_lists(
      self,
  ) -> Mapping[
      Tuple[tfgnn.NodeSetName, tfgnn.EdgeSetName, tfgnn.NodeSetName], tf.Tensor
  ]:
    """Returns dict from "edge type tuple" to int array of shape (2, num_edges).

    "edge type tuple" string-tuple: (src_node_set, edge_set, target_node_set).
    """
    return {
        ("nodes", "edges", "nodes"): np.vstack(
            [self.flat_adjacency.row, self.flat_adjacency.col],
        )
    }


if __name__ == "__main__":
  tf.test.main()
