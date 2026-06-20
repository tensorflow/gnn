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
"""Utils for feature aggregation in NARS."""
import collections
from typing import Dict, List, Sequence

import tensorflow as tf
import tensorflow_gnn as tfgnn


def generate_relational_subgraph(
    graph_tensor: tfgnn.GraphTensor, relational_subset: Sequence[str]
) -> tfgnn.GraphTensor:
  """Generate subgraph consisting of only edges belonging to the subset.

  Args:
    graph_tensor: The input scalar graph tensor.
    relational_subset: The relational subset comprising of the sequence of edges
      that make up the relational subgraph.

  Returns:
    A scalar graph tensor with filtered node set and edge sets dictated by
    edge_subset
  """
  filtered_edge_sets = {
      edge_set_name: edge_set
      for edge_set_name, edge_set in graph_tensor.edge_sets.items()
      if edge_set_name in relational_subset
  }
  filtered_node_set_names = set()
  for edge_set_name in relational_subset:
    adj = graph_tensor.edge_sets[edge_set_name].adjacency
    source_node_set_name = adj.source_name
    target_node_set_name = adj.target_name
    filtered_node_set_names.add(source_node_set_name)
    filtered_node_set_names.add(target_node_set_name)
  filtered_node_sets = {
      node_set_name: node_set
      for node_set_name, node_set in graph_tensor.node_sets.items()
      if node_set_name in filtered_node_set_names
  }

  return tfgnn.GraphTensor.from_pieces(
      node_sets=filtered_node_sets,
      edge_sets=filtered_edge_sets,
      context=graph_tensor.context,
  )


def initialize_subgraph_features(
    relational_subgraph: tfgnn.GraphTensor,
    feature_name: str = tfgnn.HIDDEN_STATE,
) -> Dict[str, Dict[str, tfgnn.Field]]:
  """Initialize the subgraph features for the aggregation step.

  Args:
    relational_subgraph: The input scalar relational subgraph tensor.
    feature_name: The name of feature to accumulate for L hops.

  Returns:
    A Dict from node set name to its corresponding intialized feature dict.
  """

  subgraph_feature_dict: Dict[str, Dict[str, tfgnn.Field]] = {}
  # Generate node features and calculate degrees
  for node_set_name, node_set in relational_subgraph.node_sets.items():
    subgraph_feature_dict[node_set_name] = {}
    subgraph_feature_dict[node_set_name]["hop_0"] = node_set[feature_name]
    deg = 0
    for edge_set_name, edge_set in relational_subgraph.edge_sets.items():
      adj = edge_set.adjacency
      if node_set_name == adj.source_name:
        deg = deg + tfgnn.node_degree(
            relational_subgraph, edge_set_name, tfgnn.SOURCE
        )
      if node_set_name == adj.target_name:
        deg = deg + tfgnn.node_degree(
            relational_subgraph, edge_set_name, tfgnn.TARGET
        )

    norm = 1.0 / tf.cast(deg, tf.float32)
    norm = tf.where(tf.math.is_inf(norm), tf.zeros_like(norm), norm)
    subgraph_feature_dict[node_set_name]["norm"] = tf.reshape(norm, [-1, 1])
  return subgraph_feature_dict


def one_hop_feature_aggregation(
    relational_subgraph: tfgnn.GraphTensor,
    edge_set_name: str,
    subgraph_feature_dict: Dict[str, Dict[str, tfgnn.Field]],
    node_type_to_feat: Dict[str, tfgnn.Field],
    hop: int,
    receiver_tag: tfgnn.IncidentNodeTag
) -> Dict[str, tfgnn.Field]:
  """Performs one hop of feature aggregation for a particular edge set.

  Args:
    relational_subgraph: The input scalar relational subgraph tensor.
    edge_set_name: The edge set name for which feature aggregation is performed.
    subgraph_feature_dict: Dict from node set names to dict of aggregated
      features from previous hops.
    node_type_to_feat: Dict from node set name to aggregated features in
      current hop. The node_type_to_feat gets updated in each call to this
      function.
    hop: The index of the current hop.
    receiver_tag: The incident node tag at which features are aggregated.

  Returns:
    The updated node_type_to_feat dict from node set name
    to aggregated features in the current hop.
  """

  adj = relational_subgraph.edge_sets[edge_set_name].adjacency
  if receiver_tag == tfgnn.SOURCE:
    sender_name = adj.target_name
    receiver_name = adj.source_name
  else:
    sender_name = adj.source_name
    receiver_name = adj.target_name

  feature_value = subgraph_feature_dict[sender_name][f"hop_{hop-1}"]
  edge_state = tfgnn.broadcast_node_to_edges(
      relational_subgraph,
      edge_set_name,
      tfgnn.reverse_tag(receiver_tag),
      feature_value=feature_value,
  )

  node_type_to_feat[receiver_name] += tfgnn.pool_edges_to_node(
      relational_subgraph,
      edge_set_name,
      receiver_tag,
      feature_value=edge_state,
  )

  return node_type_to_feat


def generate_relational_subgraph_features(
    relational_graph: tfgnn.GraphTensor,
    *,
    num_hops: int = 1,
    feature_name: str = tfgnn.HIDDEN_STATE,
    root_node_set_name: str = "paper",
) -> List[tfgnn.Field]:
  """Generate aggregated features corresponding to relational subset.

  Args:
    relational_graph: The input scalar relational graph tensor.
    num_hops: Number of hops L for which to accumulate features
    feature_name: The name of feature to accumulate for L hops.
    root_node_set_name: The node set name for which features are to be
      aggregated for L hops.

  Returns:
    The accumulated features for L hops for `root_node_set_name` node set
    corresponding to the provided relational subset. Output is a list where each
    element has shape `[num_root_nodes, feat_size]` where `num_root_nodes`
    correspond to the number of root nodes in the graph and `feat_size` denotes
    the feature dimension.
  """
  subgraph_feature_dict = initialize_subgraph_features(
      relational_graph, feature_name
  )

  accumulated_root_feats = []
  # compute L-hop aggregated features
  for hop in range(1, num_hops + 1):
    # initialize node_type_to_feat
    node_type_to_feat = {
        node_set_name: tf.zeros_like(node_set[feature_name])
        for node_set_name, node_set in relational_graph.node_sets.items()
    }
    # Aggregate features for source and target nodes for each edge set
    for edge_set_name, _ in relational_graph.edge_sets.items():
      # accumulate at source nodes
      node_type_to_feat = one_hop_feature_aggregation(
          relational_graph,
          edge_set_name,
          subgraph_feature_dict,
          node_type_to_feat,
          hop,
          tfgnn.SOURCE,
      )

      # accumulate at target nodes
      node_type_to_feat = one_hop_feature_aggregation(
          relational_graph,
          edge_set_name,
          subgraph_feature_dict,
          node_type_to_feat,
          hop,
          tfgnn.TARGET,
      )

    # Update the aggregated feature dict for each node set
    for node_set_name, _ in relational_graph.node_sets.items():
      feat_dict = subgraph_feature_dict[node_set_name]
      old_feat = feat_dict.pop(f"hop_{hop-1}")
      # save accumulated paper node features
      if node_set_name == root_node_set_name:
        accumulated_root_feats.append(old_feat)
      subgraph_feature_dict[node_set_name][f"hop_{hop}"] = (
          node_type_to_feat.pop(node_set_name) * feat_dict["norm"]
      )

  accumulated_root_feats.append(
      subgraph_feature_dict[root_node_set_name].pop(f"hop_{num_hops}")
  )
  return accumulated_root_feats


def preprocess_features(
    graph_tensor: tfgnn.GraphTensor,
    relational_subsets: List[Sequence[str]],
    *,
    num_hops: int = 1,
    feature_name: str = tfgnn.HIDDEN_STATE,
    root_node_set_name: str = "paper",
) -> List[tfgnn.Field]:
  """Generate aggregated features corresponding to all relational subsets.

  Args:
    graph_tensor: The input scalar graph tensor.
    relational_subsets: The list of relational subsets where each relational
      subset comprises of the sequence of edge set names that make up the
      relational subgraph.
    num_hops: Number of hops L for which to accumulate features
    feature_name: The name of feature to accumulate for L hops.
    root_node_set_name: The node set name for which features are to be
      aggregated for L hops.

  Returns:
    The accumulated features for L hops for `root_node_set_name` node set for
    all the relational subgraphs. Output is a list where each element has shape
    `[num_root_nodes, len(relational_subsets), feat_size]` where
    `num_root_nodes` correspond to the number of root nodes in the graph and
    `feat_size` denotes the feature dimension.
  """

  # Outer list corresponds to num_hops; inner list corresponds to subset_ids
  feats_list = collections.defaultdict(list)
  print("Start generating features for each sub-metagraph:")
  for _, relational_subset in enumerate(relational_subsets):
    relational_graph = generate_relational_subgraph(
        graph_tensor, relational_subset
    )
    feats = generate_relational_subgraph_features(
        relational_graph,
        num_hops=num_hops,
        feature_name=feature_name,
        root_node_set_name=root_node_set_name,
    )
    # Append to feature list
    for hop_id in range(num_hops + 1):
      feats_list[hop_id].append(feats[hop_id])

  # Stack the relational subset feats together
  preprocessed_features = []
  for hop_id in range(num_hops + 1):
    preprocessed_features.append(tf.stack(feats_list[hop_id], axis=1))

  return preprocessed_features
