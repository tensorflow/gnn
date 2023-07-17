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
"""Functions to read from unigraph format into that accepted by sampler_v2."""

from typing import Dict, List, Tuple

import apache_beam as beam
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn import sampler as sampler_lib
from tensorflow_gnn.data import unigraph


PCollection = beam.pvalue.PCollection


def _create_seeds(node_id: bytes) -> Tuple[bytes, List[List[np.ndarray]]]:
  values = [
      np.array([node_id], dtype=np.object_),
      np.array([1], dtype=np.int64),
  ]
  return bytes(f'S{node_id}', 'utf-8'), [values]


def read_seeds(root: beam.Pipeline, data_path: str) -> PCollection:
  """Reads a seed node set from a file and emits sampler-compatible seeds.

  Args:
    root: The root Beam Pipeline.
    data_path: The file path for the input node set.
    
  Returns:
    PCollection of sampler-compatible seeds.
  """
  seed_nodes = unigraph.read_node_set(root, data_path, 'seeds')
  return (
      seed_nodes
      | 'SeedKeys' >> beam.Keys()
      | 'MakeSeeds' >> beam.Map(_create_seeds)
  )


def seeds_from_graph_dict(
    graph_dict: Dict[str, PCollection],
    sampling_spec: sampler_lib.SamplingSpec) -> PCollection:
  """Emits sampler-compatible seeds from a collection of graph data and a sampling spec.
  
  Args:
    graph_dict: A dict of graph data represented as PCollections.
    sampling_spec: The sampling spec with the node set used for seeding.
    
  Returns:
    PCollection of sampler-compatible seeds.
  """
  seed_nodes = graph_dict[f'nodes/{sampling_spec.seed_op.node_set_name}']
  return (seed_nodes
          | 'SeedKeys' >> beam.Keys()
          | 'MakeSeeds' >> beam.Map(_create_seeds))


def _create_node_features(
    node_id: bytes, example: tf.train.Example
) -> Tuple[bytes, bytes]:
  return node_id, example.SerializeToString()


def _create_edge(
    source_id: bytes,
    target_id: bytes,
    example: tf.train.Example) -> Tuple[bytes, bytes]:
  for key in example.features.feature.keys():
    if key not in ('#source', '#target'):
      raise NotImplementedError('Edge features are not currently supported')
  return source_id, target_id


class ReadAndConvertUnigraph(beam.PTransform):
  """Converts the unigraph data representation to that accepted by the sampler."""

  def __init__(self, graph_schema: tfgnn.GraphSchema, data_path: str):
    """Constructor for ReadAndConvertUnigraph PTransform.
    
    Args:
      graph_schema: tfgnn.GraphSchema for the input graph.
      data_path: A file path for the graph data in accepted file formats.
    """
    self._graph_schema = graph_schema
    self._data_path = data_path

  def expand(self,
             rcoll: PCollection
            ) -> Dict[str, PCollection]:
    graph_data = unigraph.read_graph(self._graph_schema, self._data_path, rcoll)
    result_dict = {}
    for node_set_name in graph_data['nodes'].keys():
      result_dict[f'nodes/{node_set_name}'] = (
          graph_data['nodes'][node_set_name]
          | f'ExtractNodeFeatures/{node_set_name}'
          >> beam.MapTuple(_create_node_features)
      )
    for edge_set_name in graph_data['edges'].keys():
      result_dict[f'edges/{edge_set_name}'] = (
          graph_data['edges'][edge_set_name]
          | f'ExtractEdges/{edge_set_name}' >> beam.MapTuple(_create_edge))
    return result_dict
