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
"""Compute stats on a file of encoded graph tensors.

This script should not require a schema.
"""

import re
from typing import Any, Dict, Iterator, Optional, Tuple

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from apache_beam.runners.direct import direct_runner
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.tools import sampled_stats_pb2 as sspb


def define_flags():
  """Define commandline flags."""

  flags.DEFINE_string(
      "graph_schema", None,
      ("Path to a text-formatted GraphSchema proto file describing the "
       "features."))

  flags.DEFINE_string(
      "input_pattern", None,
      ("File pattern of tensorflow.Example values encoded as features for "
       "graph tensors. The format of the container is specified with --format"))
  flags.DEFINE_string(
      "input_format", "tfrecord",
      "Container of --input_graph_tensor_pattern files.")

  flags.DEFINE_integer(
      "num_quantiles", 10,
      "Number of quantiles to compute.")

  flags.DEFINE_string(
      "output_filename", None,
      "Stats output text file.")

  runner_choices = ["direct"]
  # Placeholder for Google-internal pipeline runner
  flags.DEFINE_enum(
      "runner", None, runner_choices,
      "The underlying runner; if not specified, use the default runner.")

  flags.mark_flags_as_required(["graph_schema", "input_pattern",
                                "output_filename"])

  return flags.FLAGS


class ReadExamples(beam.PTransform):
  """Read a stream of tf.train.Example, dispatch between formats."""

  def __init__(self, input_pattern: str, input_format: str):
    super(ReadExamples, self).__init__()
    self.input_pattern = input_pattern
    self.input_format = input_format

  def expand(self, pcoll: beam.PCollection) -> beam.PCollection:
    if self.input_format in {"tfrecord", "tfrecords"}:
      return pcoll | beam.io.tfrecordio.ReadFromTFRecord(self.input_pattern)
    # Placeholder for Google-internal input file formats
    else:
      raise NotImplementedError("Format: {}".format(self.input_format))


def create_beam_runner(
    runner_name: Optional[str]) -> beam.runners.PipelineRunner:
  """Create appropriate runner."""
  if runner_name == "direct":
    runner = direct_runner.DirectRunner()
  # Placeholder for creating Google-internal pipeline runner
  else:
    runner = None
  return runner


def run_pipeline(input_pattern: str,
                 input_format: str,
                 schema: tfgnn.GraphSchema,
                 num_quantiles: int,
                 output_filename: str,
                 runner_name: Optional[str] = None):
  """Main pipeline definition."""
  with beam.Pipeline(runner=create_beam_runner(runner_name)) as root:
    stats = (root
             | "ReadExample" >> ReadExamples(input_pattern, input_format)
             | "ProcessExample" >> (beam.FlatMap(process_example, schema)
                                    .with_outputs()))

    # The list of keys from the schema is statically defined and thus allows us
    # to avoid processing the entire set of features together segregating them
    # by key in a second stage, for the quantile aggregation. This is more
    # efficient as we already have the sets separated out.
    quantiles = {}
    for key, _ in iter_stats_schema(schema):
      quantiles[key] = (
          stats[key]
          | f"{key}.q" >> beam.ApproximateQuantiles.Globally(num_quantiles + 1)
          | f"{key}.m" >> beam.Map(lambda q, k=key: (k, q)))

    _ = (quantiles.values()
         | beam.Flatten()
         | "CombineStats" >> beam.combiners.ToDict()
         | beam.Map(make_stats_proto)
         | WriteTextProto(output_filename))


def iter_stats_schema(schema: tfgnn.GraphSchema) -> Iterator[Tuple[str, Any]]:
  """Iterator over all numerical features. Produces (key, set-object)."""

  for set_type, set_name, set_obj in tfgnn.iter_sets(schema):
    if set_type != tfgnn.CONTEXT:
      # Output a feature for the size of the set.
      key = f"{set_type}/{set_name}/{tfgnn.SIZE_NAME}"
      yield key, set_obj

    # Output the values for each feature.
    for feature_name, feature in set_obj.features.items():
      if tf.dtypes.as_dtype(feature.dtype) == tf.string:
        continue
      key = f"{set_type}/{set_name}/{feature_name}"
      yield key, feature


def iter_stats_graph(graph: tfgnn.GraphTensor) -> Iterator[Tuple[str,
                                                                 tfgnn.Field]]:
  """Iterator over all numerical features. Produces (key, tensor)."""

  for set_type, set_name, set_obj in tfgnn.iter_sets(graph):
    if set_type != tfgnn.CONTEXT:
      # Output a feature for the size of the set.
      key = f"{set_type}/{set_name}/{tfgnn.SIZE_NAME}"
      yield key, set_obj.sizes

    # Output the values for each feature.
    for feature_name, tensor in set_obj.features.items():
      if tensor.dtype == tf.string:
        continue
      key = f"{set_type}/{set_name}/{feature_name}"
      yield key, tensor


def process_example(example_string: tf.train.Example,
                    schema: tfgnn.GraphSchema):
  """Process a single tf.train.Example string and emit stats."""
  spec = tfgnn.create_graph_spec_from_schema_pb(schema)
  graph = tfgnn.parse_single_example(spec, example_string)

  # Note: the output tags cannot be structured; they must be single string
  # objects.
  for key, tensor in iter_stats_graph(graph):
    if isinstance(tensor, tf.RaggedTensor):
      tensor = tensor.flat_values
    for value in tensor.numpy().flat:
      yield beam.pvalue.TaggedOutput(key, value)


def make_stats_proto(stats_dict: Dict[str, Any]) -> sspb.GraphTensorStats:
  """Read a combined dict of stats and fill in a stats proto."""

  result = sspb.GraphTensorStats()
  for name, quantiles in stats_dict.items():
    match = re.match(r"(nodes|edges|context)/(.*)/(.*)", name)
    if not match:
      logging.error("Invalid stats key format: %s", name)
      continue

    # Add an output slot.
    feature_stats = result.feature_stats.add()
    (feature_stats.set_type,
     feature_stats.set_name,
     feature_stats.feature_name) = match.groups()

    # Fill with distributional data.
    st = feature_stats.stats
    st.min = quantiles[0]
    st.max = quantiles[-1]
    assert len(quantiles) % 2 == 1
    st.median = quantiles[int(len(quantiles)//2)]
    # TODO(blais): Compute and merge the mean.
    st.quantiles.extend(quantiles)

  return result


class WriteTextProto(beam.PTransform):
  """Convert a proto to text and write to a single text file."""

  def __init__(self, filename: str):
    self.filename = filename

  def expand(self, pcoll: beam.PCollection) -> beam.PCollection:
    return (pcoll
            | beam.Map(str)
            | beam.io.textio.WriteToText(
                self.filename, num_shards=None, shard_name_template=""))


def app_main(unused_argv):
  FLAGS = flags.FLAGS  # pylint: disable=invalid-name
  run_pipeline(FLAGS.input_pattern,
               FLAGS.input_format,
               tfgnn.read_schema(FLAGS.graph_schema),
               FLAGS.num_quantiles,
               FLAGS.output_filename,
               FLAGS.runner)


def main():
  define_flags()
  app.run(app_main)


if __name__ == "__main__":
  main()
