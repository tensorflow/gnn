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
import functools
from typing import Any, Callable, Mapping, Optional, Sequence

from absl import app
from absl import flags
from absl import logging
from ml_collections import config_dict
from ml_collections import config_flags

import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn import runner
from tensorflow_gnn.models import multi_head_attention
from tensorflow_gnn.models import vanilla_mpnn

FLAGS = flags.FLAGS

_SAMPLES_FORMAT = flags.DEFINE_string(
    "samples_format",
    "tfrecord",
    "Indicates the file format for --samples.")

_SAMPLES = flags.DEFINE_string(
    "samples",
    None,
    "A file pattern for a sharded file of GraphTensors of sampled subgraphs. "
    "The file's container format must be as given by --samples_format, which "
    f"defaults to {_SAMPLES_FORMAT.default}",
    required=True)

_GRAPH_SCHEMA = flags.DEFINE_string(
    "graph_schema",
    None,
    "A filepath for the GraphSchema of the --samples.",
    required=True)

_BASE_DIR = flags.DEFINE_string(
    "base_dir",
    None,
    "The training and export base directory "
    "(`runner.incrementing_model_dir(...)` is used to generate the model "
    "directory).",
    required=True)

_TPU_ADDRESS = flags.DEFINE_string(
    "tpu_address",
    None,
    "An optional TPU address "
    "(see: `tf.distribute.cluster_resolver.TPUClusterResolver`), if empty "
    "string: TensorFlow will try to automatically resolve the Cloud TPU; if "
    "`None`: `MirroredStrategy` is used.")

_EPOCHS = flags.DEFINE_integer(
    "epochs", 5,
    "Training runs this many times over the dataset.")

_RESTORE_BEST_WEIGHTS = flags.DEFINE_boolean(
    "restore_best_weights",
    False,
    "By default, exports the trained model from the end of the training. "
    "If set, exports the trained model with the best validation result.")

_LEARNING_RATE_SCHEDULE = flags.DEFINE_enum(
    "lr_schedule", "cosine_decay", ["constant", "cosine_decay"],
    "The learning rate schedule for the optimizer.")

_LEARNING_RATE = flags.DEFINE_float(
    "learning_rate", 1e-3,
    "The initial learning rate of the learning rate schedule.")

_PAPER_DIM = flags.DEFINE_integer(
    "paper_dim", 512,
    "Dimensionality of dense layer applied to paper features. "
    "Set to '0' for no dense transform.")


# The GNN model used by this script is configured by a ConfigDict
# (see https://github.com/google/ml_collections).
# The following function defines the available configuration options
# and their default values. The subsequent config_flags definition
# allows users to override them from the command line, for example,
# --config.gnn.vanilla_mpnn.dropout_rate = 0.1
def get_config_dict() -> config_dict.ConfigDict:
  """The default config that users can override with --config.foo=bar."""
  cfg = config_dict.ConfigDict()
  cfg.gnn = config_dict.ConfigDict()
  cfg.gnn.type = "vanilla_mpnn"
  # For each supported gnn.type="foo", there is a config gnn.foo for that type's
  # GraphUpdate class, overridden with the defaults for this training.
  cfg.gnn.vanilla_mpnn = vanilla_mpnn.graph_update_get_config_dict()
  cfg.gnn.vanilla_mpnn.units = 128
  cfg.gnn.vanilla_mpnn.message_dim = 128
  cfg.gnn.vanilla_mpnn.receiver_tag = tfgnn.SOURCE
  cfg.gnn.vanilla_mpnn.dropout_rate = 0.2
  cfg.gnn.vanilla_mpnn.l2_regularization = 6e-6
  cfg.gnn.vanilla_mpnn.use_layer_normalization = True
  # Config for multi head attention
  cfg.gnn.multi_head_attention = (
      multi_head_attention.graph_update_get_config_dict())
  cfg.gnn.multi_head_attention.units = 128
  cfg.gnn.multi_head_attention.message_dim = 128
  cfg.gnn.multi_head_attention.num_heads = 4
  cfg.gnn.multi_head_attention.receiver_tag = tfgnn.SOURCE
  cfg.gnn.multi_head_attention.state_dropout_rate = 0.2
  cfg.gnn.multi_head_attention.l2_regularization = 6e-6
  cfg.gnn.multi_head_attention.edge_dropout_rate = 0.2
  cfg.lock()
  return cfg

_CONFIG = config_flags.DEFINE_config_dict("config", get_config_dict())


def _graph_update_from_config(
    cfg: config_dict.ConfigDict) -> tf.keras.layers.Layer:
  """Returns one instance of the configured GraphUpdate layer."""
  if cfg.gnn.type == "vanilla_mpnn":
    return vanilla_mpnn.graph_update_from_config_dict(cfg.gnn.vanilla_mpnn)
  elif cfg.gnn.type == "multi_head_attention":
    return multi_head_attention.graph_update_from_config_dict(
        cfg.gnn.multi_head_attention)
  else:
    raise ValueError(f"Unknown gnn.type: {cfg.gnn.type}")


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


class _SplitDatasetProvider(runner.DatasetProvider):
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

_DEFAULT_DATASET_FORMATS = {
    "tfrecord": runner.TFRecordDatasetProvider,
}


def main(
    args,
    *,
    extra_keras_callbacks:
        Optional[Sequence[tf.keras.callbacks.Callback]] = None,
    extra_dataset_formats: Optional[Mapping[str, Callable[[str], Any]]] = None,
) -> None:
  if len(args) > 1:
    raise app.UsageError("Too many command-line arguments.")

  dataset_formats = _DEFAULT_DATASET_FORMATS.copy()
  if extra_dataset_formats is not None:
    dataset_formats.update(extra_dataset_formats)
  ds_provider = dataset_formats[_SAMPLES_FORMAT.value](_SAMPLES.value)
  train_ds_provider = _SplitDatasetProvider(ds_provider, "train")
  valid_ds_provider = _SplitDatasetProvider(ds_provider, "validation")

  graph_schema = tfgnn.read_schema(_GRAPH_SCHEMA.value)
  gtspec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

  def extract_labels(graphtensor: tfgnn.GraphTensor):
    labels = tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name="paper",
        feature_name="labels")(graphtensor)
    graphtensor = graphtensor.remove_features(node_sets={"paper": ["labels"]})
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
      if not _PAPER_DIM.value:
        logging.info("Skipping dense layer for paper.")
        return node_set["feat"]
      logging.info("Applying dense layer %d to paper.", _PAPER_DIM.value)
      return tf.keras.layers.Dense(_PAPER_DIM.value)(node_set["feat"])
    if node_set_name == "author":
      return node_set["empty_state"]
    raise KeyError(f"Unexpected node_set_name='{node_set_name}'")

  def model_fn(gtspec: tfgnn.GraphTensorSpec):
    graph = inputs = tf.keras.layers.Input(type_spec=gtspec)
    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn=set_initial_node_states)(graph)
    for _ in range(4):
      graph_update = _graph_update_from_config(_CONFIG.value)
      graph = graph_update(graph)
    return tf.keras.Model(inputs, graph)

  task = runner.RootNodeMulticlassClassification(
      node_set_name="paper",
      num_classes=349)

  global_batch_size = 128
  validation_batch_size = 32
  steps_per_epoch = 629_571 // global_batch_size  # len(train) == 629,571

  # len(validation) == 64,879
  validation_steps = 64_879 // validation_batch_size

  # Determine learning rate schedule
  if _LEARNING_RATE_SCHEDULE.value == "cosine_decay":
    learning_rate = tf.keras.optimizers.schedules.CosineDecay(
        _LEARNING_RATE.value, steps_per_epoch*_EPOCHS.value)
  elif _LEARNING_RATE_SCHEDULE.value == "constant":
    learning_rate = _LEARNING_RATE.value
  else:
    raise ValueError(
        f"Learning rate schedule '{_LEARNING_RATE_SCHEDULE.value}' not defined")

  # Optimizer Function
  optimizer_fn = functools.partial(tf.keras.optimizers.Adam,
                                   learning_rate=learning_rate)

  if _TPU_ADDRESS.value is not None:
    strategy = runner.TPUStrategy(_TPU_ADDRESS.value)
    # Update `min_nodes_per_component.` Default requirement of at least one node
    # from each node set in input is sufficient but more than necessary.
    # The condition that each graph component must contain at least one "paper"
    # node is sufficient.
    min_nodes_per_component = {"paper": 1}
    train_padding = runner.FitOrSkipPadding(gtspec, train_ds_provider,
                                            min_nodes_per_component)
    valid_padding = runner.TightPadding(gtspec, valid_ds_provider,
                                        min_nodes_per_component)
  else:
    strategy = tf.distribute.MirroredStrategy()
    train_padding = None
    valid_padding = None

  trainer = runner.KerasTrainer(
      strategy=strategy,
      model_dir=runner.incrementing_model_dir(_BASE_DIR.value),
      callbacks=extra_keras_callbacks,
      steps_per_epoch=steps_per_epoch,
      validation_steps=validation_steps,
      restore_best_weights=_RESTORE_BEST_WEIGHTS.value,
      checkpoint_every_n_steps=("epoch" if _RESTORE_BEST_WEIGHTS.value
                                else "never"),
  )

  runner.run(
      train_ds_provider=train_ds_provider,
      train_padding=train_padding,
      model_fn=model_fn,
      optimizer_fn=optimizer_fn,
      epochs=_EPOCHS.value,
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
