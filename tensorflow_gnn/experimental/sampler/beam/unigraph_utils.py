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

from __future__ import annotations

from typing import Dict, List, Tuple

import apache_beam as beam
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn import sampler as sampler_lib
from tensorflow_gnn.data import unigraph
from tensorflow_gnn.experimental.sampler.beam import executor_lib


PCollection = beam.pvalue.PCollection
Values = executor_lib.Values


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


def _make_seed_feature(
    example: tf.train.Example, feat_name: str
) -> tuple[bytes, Values]:
  """Formats a particular feature from a tf.train.Example into a seed format.

  Args:
    example: tf.train.Example with the seed features.
    feat_name: The feature to extract in this call

  Returns:
    bytes key and a list/np.array representation of a ragged tensor

  Raises:
    ValueError: on a malformed Example without the given feature present
  """

  seed_source = example.features.feature[tfgnn.SOURCE_NAME].bytes_list.value[0]
  seed_target = example.features.feature[tfgnn.TARGET_NAME].bytes_list.value[0]
  key = bytes(
      f'S{seed_source.decode("utf-8")}:T{seed_target.decode("utf-8")}', 'utf-8'
  )
  if example.features.feature[feat_name].HasField('bytes_list'):
    bytes_value = example.features.feature[feat_name].bytes_list.value
    value = [[
        np.array(bytes_value, dtype=np.object_),
        np.array([1], dtype=np.int64),
    ]]
  elif example.features.feature[feat_name].HasField('float_list'):
    float_value = example.features.feature[feat_name].float_list.value
    value = [[
        np.array(float_value, dtype=np.float32),
        np.array([1], dtype=np.int64),
    ]]
  elif example.features.feature[feat_name].HasField('int64_list'):
    int64_value = example.features.feature[feat_name].int64_list.value
    value = [[
        np.array(int64_value, dtype=np.float32),
        np.array([1], dtype=np.int64),
    ]]
  else:
    raise ValueError(f'Feature {feat_name} is not present in this example')
  return (key, value)


class ReadLinkSeeds(beam.PTransform):
  """Reads seeds for link prediction into PCollections for each seed feature."""

  def __init__(self, graph_schema: tfgnn.GraphSchema, data_path: str):
    """Constructor for ReadLinkSeeds PTransform.

    Args:
      graph_schema: tfgnn.GraphSchema for the input graph.
      data_path: A file path for the seed data in accepted file formats.
    """
    self._graph_schema = graph_schema
    self._data_path = data_path
    self._readout_feature_names = [
        key
        for key in graph_schema.node_sets['_readout'].features.keys()
        if key not in [tfgnn.SOURCE_NAME, tfgnn.TARGET_NAME]
    ]

  def expand(self, rcoll: PCollection) -> Dict[str, PCollection]:
    seed_table = rcoll | 'ReadSeedTable' >> unigraph.ReadTable(
        self._data_path,
        converters=unigraph.build_converter_from_schema(
            self._graph_schema.node_sets['_readout'].features
        ),
    )
    pcolls_out = {}
    pcolls_out['SeedSource'] = seed_table | 'MakeSeedSource' >> beam.Map(
        _make_seed_feature, tfgnn.SOURCE_NAME
    )
    pcolls_out['SeedTarget'] = seed_table | 'MakeSeedTarget' >> beam.Map(
        _make_seed_feature, tfgnn.TARGET_NAME
    )
    for feature in self._readout_feature_names:
      pcolls_out[f'Seed/{feature}'] = (
          seed_table
          | f'MakeSeed/{feature}' >> beam.Map(_make_seed_feature, feature)
      )
    return pcolls_out


def seeds_from_graph_dict(
    graph_dict: Dict[str, PCollection], sampling_spec: sampler_lib.SamplingSpec
) -> PCollection:
  """Emits sampler-compatible seeds from a collection of graph data and a sampling spec.

  Args:
    graph_dict: A dict of graph data represented as PCollections.
    sampling_spec: The sampling spec with the node set used for seeding.

  Returns:
    PCollection of sampler-compatible seeds.
  """
  seed_nodes = graph_dict[f'nodes/{sampling_spec.seed_op.node_set_name}']
  return (
      seed_nodes
      | 'SeedKeys' >> beam.Keys()
      | 'MakeSeeds' >> beam.Map(_create_seeds)
  )


def _create_node_features(
    node_id: bytes, example: tf.train.Example
) -> Tuple[bytes, bytes]:
  return node_id, example.SerializeToString()


def _create_edge(
    source_id: bytes,
    target_id: bytes,
    example: tf.train.Example,
) -> tf.train.Example:
  """Creates input for edge set sampling stages."""
  for name, value in (
      (tfgnn.SOURCE_NAME, source_id),
      (tfgnn.TARGET_NAME, target_id),
  ):
    example.features.feature[name].CopyFrom(
        tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    )
  return example


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

  def expand(self, rcoll: PCollection) -> Dict[str, PCollection]:
    graph_data = unigraph.read_graph(self._graph_schema, self._data_path, rcoll)
    result_dict = {}
    for node_set_name in graph_data['nodes'].keys():
      result_dict[f'nodes/{node_set_name}'] = graph_data['nodes'][
          node_set_name
      ] | f'ExtractNodeFeatures/{node_set_name}' >> beam.MapTuple(
          _create_node_features
      )
    for edge_set_name in graph_data['edges'].keys():
      result_dict[f'edges/{edge_set_name}'] = graph_data['edges'][
          edge_set_name
      ] | f'ExtractEdges/{edge_set_name}' >> beam.MapTuple(_create_edge)
    return result_dict
