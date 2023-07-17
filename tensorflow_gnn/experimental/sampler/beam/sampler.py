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
"""Runs sampling pipeline defined by the GraphSchema and SamplingSpec.

Closely follows V1.
"""
from __future__ import annotations

import functools
import os
from typing import Optional

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.data import unigraph
from tensorflow_gnn.experimental import sampler
from tensorflow_gnn.experimental.sampler.beam import accessors  # pylint: disable=unused-import
from tensorflow_gnn.experimental.sampler.beam import edge_samplers  # pylint: disable=unused-import
from tensorflow_gnn.experimental.sampler.beam import executor_lib
from tensorflow_gnn.experimental.sampler.beam import unigraph_utils
from tensorflow_gnn.proto import graph_schema_pb2
import tensorflow_gnn.sampler as sampler_lib

from google.protobuf import text_format


_DIRECT_RUNNER = 'DirectRunner'
_DATAFLOW_RUNNER = 'DataflowRunner'


def _get_shape(feature: graph_schema_pb2.Feature) -> tf.TensorShape:
  dim_fn = lambda dim: (None if dim.size == -1 else dim.size)
  dims = [dim_fn(dim) for dim in feature.shape.dim]
  return tf.TensorShape(dims)


def get_sampling_model(
    graph_schema: tfgnn.GraphSchema,
    sampling_spec: sampler_lib.SamplingSpec,
) -> tuple[tf.keras.Model, dict[str, str]]:
  """Constructs sampling model from schema and sampling spec.

  Args:
    graph_schema: Attribute `edge_sets` identifies end-point node set names.
    sampling_spec: The number of nodes sampled from edge set. The spec defines
      the structure of the sampled subgraphs, that look like rooted trees,
      possibly densified adding all pairwise edges between sampled nodes.

  Returns:
    A Keras model for sampling and 
    a mapping from the layer's name to the corresponding edge set.
  """
  layer_name_to_edge_set = {}
  def edge_sampler_factory(
      op: sampler_lib.SamplingOp,
      *,
      counter: dict[str, int],
  ) -> sampler.UniformEdgesSampler:
    accessor = sampler.KeyToTfExampleAccessor(
        sampler.InMemStringKeyToBytesAccessor(
            keys_to_values={'b': b'b'}),
        features_spec={
            '#target': tf.TensorSpec([None], tf.string),
        },
    )
    edge_set_count = counter.setdefault(op.edge_set_name, 0)
    counter[op.edge_set_name] += 1
    layer_name = f'edges/{op.edge_set_name}_{edge_set_count}'
    sample_size = op.sample_size
    edge_target_feature_name = '#target'

    result = sampler.UniformEdgesSampler(
        outgoing_edges_accessor=accessor,
        sample_size=sample_size,
        edge_target_feature_name=edge_target_feature_name,
        name=layer_name,
    )
    assert (
        layer_name not in layer_name_to_edge_set
    ), f'Duplicate layer name: {layer_name}'
    layer_name_to_edge_set[layer_name] = f'edges/{op.edge_set_name}'
    return result

  def node_features_accessor_factory(
      node_set_name: tfgnn.NodeSetName,
  ) -> sampler.KeyToTfExampleAccessor:
    if not graph_schema.node_sets[node_set_name].features:
      return None
    node_features = graph_schema.node_sets[node_set_name].features
    features_spec = {}
    for name, feature in node_features.items():
      shape = _get_shape(feature)
      dtype = tf.dtypes.as_dtype(feature.dtype)
      features_spec[name] = tf.TensorSpec(shape, dtype)
    accessor = sampler.KeyToTfExampleAccessor(
        sampler.InMemStringKeyToBytesAccessor(
            keys_to_values={'b': b'b'},
            name=f'nodes/{node_set_name}'),
        features_spec=features_spec,
    )
    return accessor

  counter = {}
  return sampler.create_sampling_model_from_spec(
      graph_schema,
      sampling_spec,
      edge_sampler_factory=functools.partial(
          edge_sampler_factory, counter=counter),
      node_features_accessor_factory=node_features_accessor_factory,
  ), layer_name_to_edge_set


def _create_beam_runner(
    runner_name: Optional[str],
) -> beam.runners.PipelineRunner:
  """Creates appropriate runner."""
  if runner_name == _DIRECT_RUNNER:
    runner = beam.runners.DirectRunner()
  elif runner_name == _DATAFLOW_RUNNER:
    runner = beam.runners.DataflowRunner()
  else:
    runner = None
  return runner


def save_artifacts(artifacts: sampler.Artifacts, artifacts_path: str) -> None:
  for layer_id, model in artifacts.models.items():
    path = os.path.join(artifacts_path, layer_id)
    tf.io.gfile.makedirs(path)
    sampler.save_model(model, path)


def define_flags():
  """Creates commandline flags."""

  flags.DEFINE_string(
      'graph_schema',
      None,
      'Path to a text-formatted GraphSchema proto file or directory '
      'containing one for a graph in Universal Graph Format. This '
      'defines the input graph to be sampled.',
  )

  flags.DEFINE_string(
      'data_path',
      None,
      'Path to data files for node and edge sets. Defaults to the directory '
      'containing graph_schema.',
  )

  flags.DEFINE_string(
      'input_seeds',
      None,
      'Path to an input file with the seed node ids to restrict sampling over. '
      'The file can be in any of the supported unigraph table formats, and as '
      "for node sets, the 'id' column will be used. If the seeds aren't "
      'specified, the full set of nodes from the graph will be used '
      '(optional).',
  )

  flags.DEFINE_string(
      'sampling_spec',
      None,
      'An input file with a text-formatted SamplingSpec proto to use. This is '
      "a required input and to some extent may mirror some of the schema's "
      'structure. See `sampling_spec.proto` for details on the configuration.',
  )

  flags.DEFINE_string(
      'output_samples',
      None,
      'Output file with serialized graph tensor Example protos.',
  )

  runner_choices = [_DIRECT_RUNNER, _DATAFLOW_RUNNER]
  runner_choices.append('flume')
  flags.DEFINE_enum(
      'runner',
      None,
      runner_choices,
      'The underlying runner; if not specified, use the default runner.',
  )

  flags.mark_flags_as_required(
      ['graph_schema', 'sampling_spec', 'output_samples']
  )


def app_main(argv) -> None:
  """Main sampler entrypoint.

  Args:
    argv: List of arguments passed by flags parser.
  """
  FLAGS = flags.FLAGS  # pylint: disable=invalid-name
  pipeline_args = argv[1:]
  graph_schema: tfgnn.GraphSchema = unigraph.read_schema(FLAGS.graph_schema)

  with tf.io.gfile.GFile(FLAGS.sampling_spec, 'r') as f:
    sampling_spec = text_format.Parse(
        f.read(), sampler_lib.SamplingSpec()
    )
  # we have graph schema which defines Graph...
  # and sampling spec which defines how to sample in V1 format.
  # 1. Let's define sampling model as TF keras model.
  # Example:
  #  model, layers_mapping = get_sampling_model(
  # mag_graph_schema, mag_sampling_spec)
  #  model(tf.ragged.constant([[0], [1]]))
  #  # returns GraphTensor for seed papers 0 and 1.

  model, layers_mapping = get_sampling_model(graph_schema, sampling_spec)
  # layers_mapping: Dict[layer name, nodes/{node_set_nam}|edges/{edge_set_name}]
  # Export sampling model as a "sampling program".
  program_pb, artifacts = sampler.create_program(model)
  # here `eval_dag` defines Beam stages to run, artifacts are TF models
  # for some Beam stages.

  if not FLAGS.data_path:
    data_path = os.path.dirname(FLAGS.graph_schema)
  else:
    data_path = FLAGS.data_path

  output_dir = os.path.dirname(FLAGS.output_samples)
  artifacts_path = os.path.join(output_dir, 'artifacts')
  if tf.io.gfile.exists(artifacts_path):
    raise ValueError(f'{artifacts_path} already exists.')

  tf.io.gfile.makedirs(artifacts_path)
  save_artifacts(artifacts, artifacts_path)

  pipeline_options = PipelineOptions(pipeline_args)
  pipeline_options.view_as(SetupOptions).save_main_session = True

  with beam.Pipeline(
      runner=_create_beam_runner(FLAGS.runner), options=pipeline_options
  ) as root:
    feeds_unique = root | unigraph_utils.ReadAndConvertUnigraph(
        graph_schema, data_path
    )
    feeds = feeds_unique
    feeds.update({
        layer_name: feeds_unique[layers_mapping[layer_name]]
        for layer_name in layers_mapping
    })
    if FLAGS.input_seeds:
      seeds = unigraph_utils.read_seeds(root, FLAGS.input_seeds)
    else:
      seeds = unigraph_utils.seeds_from_graph_dict(feeds, sampling_spec)
    inputs = {
        'Input': seeds,
    }
    examples = executor_lib.execute(
        program_pb,
        inputs,
        feeds=feeds,
        artifacts_path=artifacts_path
    )
    # results are tuple: example_id to tf.Example with graph tensors.
    coder = beam.coders.ProtoCoder(tf.train.Example)
    _ = (
        examples
        | 'DropExampleId' >> beam.Values()
        | 'WriteToTFRecord'
        >> beam.io.WriteToTFRecord(
            os.path.join(output_dir, 'examples.tfrecord'), coder=coder
        )
    )
    logging.info('Pipeline complete')


def main():
  define_flags()
  app.run(
      app_main, flags_parser=lambda argv: flags.FLAGS(argv, known_only=True)
  )

if __name__ == '__main__':
  main()
