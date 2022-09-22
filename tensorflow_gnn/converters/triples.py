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
"""Converts semantic triples (RDF triples) to GraphTensor objects."""

import collections
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_tensor as gt


def triple_to_graphtensor(triples):
  """Converts a set of semantic triples (ie RDF triples) into GraphTensor.

  The assumed graph schema is as follows:
    - There is one node set ("nodes") which contains the entites (subjects
    and objects) of the triples.
    - There are k edge sets, for each of the k unique kinds of relations
    (predicates) used in `triples`.

  For further processing, a complete graph schema (which enumerates all
  possible relations in the dataset) will be required for TF-GNN.

  Limitations:  This is an *in-memory* converter which works on small
  datasets.
  For datasets which do not fit in memory, the TF-GNN sampling tools offer
  much more capable and full-featured ways to create `GraphTensor` objects
  from structured data.

  Args:
    triples: an enumerable set of string semantic triples in (subject,
      predicate, object) format.

  Returns:
    A GraphTensor which contains the graph representation of triples.
  """

  # Create one node set for all entites.
  nodes = {}
  reverse_index_nodes = {}
  current_node = 0

  # Group triples by predicate.
  predicates = collections.defaultdict(list)
  for spo in triples:
    s, p, o = spo

    # Add edge to edge set.
    predicates[p].append(spo)

    # Add nodes to graph.
    if s not in nodes:
      nodes[s] = current_node
      reverse_index_nodes[current_node] = s
      current_node += 1
    if o not in nodes:
      nodes[o] = current_node
      reverse_index_nodes[current_node] = o
      current_node += 1

  # Linearize node ids.
  sorted_node_ids = [
      x[0] for x in sorted(nodes.items(), key=lambda item: item[1])
  ]

  # Make node set.
  node_sets = {
      "nodes":
          gt.NodeSet.from_fields(
              sizes=tf.constant([len(sorted_node_ids)]),
              features={"#id": tf.constant(sorted_node_ids)})
  }

  # Create adjacency matrices.
  edge_sets = {}
  for pred, spos in predicates.items():
    sources = []
    targets = []

    for spo in spos:
      sources.append(nodes[spo[0]])
      targets.append(nodes[spo[2]])

    edge_sets[pred] = gt.EdgeSet.from_fields(
        sizes=tf.constant([len(sources)]),
        adjacency=adj.Adjacency.from_indices(
            source=("nodes", sources),
            target=("nodes", targets),
        ))

  graph = gt.GraphTensor.from_pieces(node_sets=node_sets, edge_sets=edge_sets)

  return graph
