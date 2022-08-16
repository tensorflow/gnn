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
"""An e2e training example for OGBN-MAG."""
from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn import runner
from tensorflow_gnn.models import vanilla_mpnn

FLAGS = flags.FLAGS

# TODO(b/196880966): Update flag return values to _UPPER_SNAKE_CASE.
_samples = flags.DEFINE_string(
    "samples",
    None,
    "A file pattern for a TFRecord file of GraphTensors of sampled subgraphs.",
    required=True)

_graph_schema = flags.DEFINE_string(
    "graph_schema",
    None,
    "A filepath for the GraphSchema of the --samples.",
    required=True)

_base_dir = flags.DEFINE_string(
    "base_dir",
    None,
    "The training and export base directory "
    "(`runner.incrementing_model_dir(...)` is used to generate the model "
    "directory).",
    required=True)

_tpu_address = flags.DEFINE_string(
    "tpu_address",
    None,
    "An optional TPU address "
    "(see: `tf.distribute.cluster_resolver.TPUClusterResolver`), if empty "
    "string: TensorFlow will try to automatically resolve the Cloud TPU; if "
    "`None`: `MirroredStrategy` is used.")

_paper_dim = flags.DEFINE_integer(
    "paper_dim", 512,
    "Dimensionality of dense layer applied to paper features. "
    "Set to 'None' for no dense transform."
)


# The following helper lets us filter a single input dataset by OGBN-MAG's
# specific rule for the test/validation/train split before parsing the full
# GraphTensor. (Models for production systems should probably use separate
# datasets.)
def _is_in_split(split_name: str):
  """Returns a `filter_fn` for OGBN-MAG's dataset splitting."""
  def filter_fn(serialized_example):
    features = {
        "years": tf.io.RaggedFeature(tf.int64, value_key="nodes/paper.year")
    }
    years = tf.io.parse_single_example(serialized_example, features)["years"]
    year = years[0]  # By convention, the root node is the first node
    if split_name == "train":
      return year <= 2017  # 629,571 examples
    elif split_name == "validation":
      return year == 2018  # 64,879 examples
    elif split_name == "test":
      return year == 2019  # 41,939 examples
    else:
      raise ValueError(f"Unknown split_name: '{split_name}'")
  return filter_fn


class _SplitDatasetProvider:
  """Splits a `delegate` for OGBN-MAG.

  The OGBN-MAG datasets splits test/validation/train by paper year. This class
  filters a `delegate` with the entire OGBN-MAG dataset by the split name
  (test/validation/train).
  """

  def __init__(self, delegate: runner.DatasetProvider, split_name: str):
    self._delegate = delegate
    self._split_name = split_name

  def get_dataset(self, context: tf.distribute.InputContext) -> tf.data.Dataset:
    dataset = self._delegate.get_dataset(context)
    return dataset.filter(_is_in_split(self._split_name))


def main(_) -> None:
  ds_provider = runner.TFRecordDatasetProvider(_samples.value)
  train_ds_provider = _SplitDatasetProvider(ds_provider, "train")
  valid_ds_provider = _SplitDatasetProvider(ds_provider, "validation")

  graph_schema = tfgnn.read_schema(_graph_schema.value)
  gtspec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

  def extract_labels(graphtensor: tfgnn.GraphTensor):
    labels = tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name="paper",
        feature_name="labels")(graphtensor)
    return graphtensor, labels

  def drop_all_features(_, **unused_kwargs):
    return {}

  def  process_node_features(node_set: tfgnn.NodeSet, node_set_name: str):
    if node_set_name == "field_of_study":
      return {"hashed_id": tf.keras.layers.Hashing(50_000)(node_set["#id"])}
    if node_set_name == "institution":
      return {"hashed_id": tf.keras.layers.Hashing(6_500)(node_set["#id"])}
    if node_set_name == "paper":
      return {"feat": node_set["feat"]}
    if node_set_name == "author":
      return {"empty_state": tfgnn.keras.layers.MakeEmptyFeature()(node_set)}
    raise KeyError(f"Unexpected node_set_name='{node_set_name}'")

  def set_initial_node_states(node_set: tfgnn.NodeSet, node_set_name: str):
    if node_set_name == "field_of_study":
      return tf.keras.layers.Embedding(50_000, 32)(node_set["hashed_id"])
    if node_set_name == "institution":
      return tf.keras.layers.Embedding(6_500, 16)(node_set["hashed_id"])
    if node_set_name == "paper":
      if _paper_dim.value is None:
        logging.info("Skipping dense layer for paper.")
        return node_set["feat"]
      logging.info("Applying dense layer %d to paper.", _paper_dim.value)
      return tf.keras.layers.Dense(_paper_dim.value)(node_set["feat"])
    if node_set_name == "author":
      return node_set["empty_state"]
    raise KeyError(f"Unexpected node_set_name='{node_set_name}'")

  def model_fn(gtspec: tfgnn.GraphTensorSpec):
    graph = inputs = tf.keras.layers.Input(type_spec=gtspec)
    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn=set_initial_node_states)(graph)
    for _ in range(4):
      graph = vanilla_mpnn.VanillaMPNNGraphUpdate(
          units=128,
          message_dim=128,
          receiver_tag=tfgnn.SOURCE,
          l2_regularization=5e-4,
          dropout_rate=0.1,
      )(graph)
    return tf.keras.Model(inputs, graph)

  task = runner.RootNodeMulticlassClassification(
      node_set_name="paper",
      num_classes=349)

  epochs = 5
  global_batch_size = 128
  validation_batch_size = 32
  steps_per_epoch = 629_571 // global_batch_size  # len(train) == 629,571

  # len(validation) == 64,879
  validation_steps = 64_879 // validation_batch_size

  if _tpu_address.value is not None:
    strategy = runner.TPUStrategy(_tpu_address.value)
    train_padding = runner.FitOrSkipPadding(gtspec, train_ds_provider)
    valid_padding = runner.TightPadding(gtspec, train_ds_provider)
  else:
    strategy = tf.distribute.MirroredStrategy()
    train_padding = None
    valid_padding = None

  trainer = runner.KerasTrainer(
      strategy=strategy,
      model_dir=runner.incrementing_model_dir(_base_dir.value),
      steps_per_epoch=steps_per_epoch,
      validation_per_epoch=4,
      validation_steps=validation_steps)

  runner.run(
      train_ds_provider=train_ds_provider,
      train_padding=train_padding,
      model_fn=model_fn,
      optimizer_fn=tf.keras.optimizers.Adam,
      epochs=epochs,
      trainer=trainer,
      task=task,
      gtspec=gtspec,
      global_batch_size=global_batch_size,
      feature_processors=[
          extract_labels,  # Extract any labels first!
          tfgnn.keras.layers.MapFeatures(
              context_fn=drop_all_features,
              node_sets_fn=process_node_features,
              edge_sets_fn=drop_all_features)
      ],
      valid_ds_provider=valid_ds_provider,
      valid_padding=valid_padding)


if __name__ == "__main__":
  app.run(main)
