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
"""Convert a `GraphTensor` to a `networkx.MultiDiGraph` object."""

import collections

from absl import logging


def to_networkx_graph(tf_gnn_graph):
  """Convert a `GraphTensor` to a `networkx.MultiDiGraph` object.

  Note: This is a prototype converter whose interface may be changed over
  time. Currently this only converts graph structure, and not node features,
  edge features, or context.

  Args:
    tf_gnn_graph:  A `tensorflow_gnn.GraphTensor` object containing a
      heterogeneous directed graph.

  Returns:
    A network.MultiDiGraph containing the adjancency information in
  the GraphTensor.
  """
  try:
    import networkx as nx  # pylint: disable=g-import-not-at-top
  except ImportError:
    logging.error('networkx not present!')
    return

  nodes, edges = to_edge_list(tf_gnn_graph)
  nx_graph = nx.MultiDiGraph()

  for node in nodes:
    node_id = node['#global_id']
    nx_graph.add_node(node_id)
    nx_graph.nodes[node_id]['#type'] = node['#type']
    nx_graph.nodes[node_id]['#id'] = node['#id']
    nx_graph.nodes[node_id]['#local_id'] = node['#local_id']

  assert len(nodes) == nx_graph.number_of_nodes(), 'duplicate node ids present'

  for edge in edges:
    print(edge['source'], edge['target'])
    nx_graph.add_edge(edge['source'], edge['target'])

  assert (
      len(nodes) == nx_graph.number_of_nodes()
  ), 'edges pointing to non-existant nodes'

  return nx_graph


def to_edge_list(tf_gnn_graph):
  """Convert a `GraphTensor` to lists of nodes and edges."""

  nodes = []
  edges = []

  local_id_to_global_id = collections.defaultdict(dict)
  global_id_to_localid = {}
  global_id_to_str = {}
  node_cnt = 0

  for node_set_name, node_set in sorted(tf_gnn_graph.node_sets.items()):
    if len(node_set.sizes) > 1:
      size = int(node_set.sizes[0])
    else:
      size = int(node_set.sizes)
    if '#id' in node_set.features and list(node_set.features['#id']):
      id_range = [str(x.numpy()).encode('utf-8') for x in node_set['#id']]
    else:
      id_range = [str(x).encode('utf-8') for x in range(size)]

    for idx, str_id in enumerate(id_range):
      local_id_to_global_id[node_set_name][idx] = node_cnt
      global_id_to_localid[node_cnt] = (idx, node_set_name)
      global_id_to_str[node_cnt] = str_id
      nodes.append({
          '#global_id': node_cnt,
          '#local_id': idx,
          '#id': str_id.decode('utf-8'),
          '#type': node_set_name,
      })
      node_cnt += 1

  for name, edge_set in tf_gnn_graph.edge_sets.items():
    source_nodeset = edge_set.adjacency.source_name
    target_nodeset = edge_set.adjacency.target_name

    for src, dst in zip(
        edge_set.adjacency.source.numpy(), edge_set.adjacency.target.numpy()
    ):
      edges.append({
          'source': local_id_to_global_id[source_nodeset][src],
          'target': local_id_to_global_id[target_nodeset][dst],
          '#src_id': global_id_to_str[src].decode('utf-8'),
          '#target_id': global_id_to_str[dst].decode('utf-8'),
          '#src_type': source_nodeset,
          '#target_type': target_nodeset,
          '#type': name,
      })

  return nodes, edges
