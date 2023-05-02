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
"""Tests for distribution strategies."""
from collections.abc import Sequence
import os
from typing import Any

from absl.testing import parameterized
import tensorflow as tf
import tensorflow.__internal__.distribute as tfdistribute
import tensorflow.__internal__.test as tftest
import tensorflow_gnn as tfgnn
from tensorflow_gnn import runner
from tensorflow_gnn.models import vanilla_mpnn
from tensorflow_gnn.models.contrastive_losses import tasks

_CLASSES = tuple(range(32))
_SCHEMA = """
  context {
    features {
      key: "classes"
      value {
        dtype: DT_INT64
      }
    }
  }
  node_sets {
    key: "node"
    value {
      features {
        key: "features"
        value {
          dtype: DT_FLOAT
          shape { dim { size: 4 } }
        }
      }
    }
  }
  edge_sets {
    key: "edge"
    value {
      source: "node"
      target: "node"
    }
  }
"""


def _all_eager_strategy_and_task_combinations():
  strategies = [
      # default
      tfdistribute.combinations.default_strategy,
      # MirroredStrategy
      tfdistribute.combinations.mirrored_strategy_with_gpu_and_cpu,
      tfdistribute.combinations.mirrored_strategy_with_one_cpu,
      tfdistribute.combinations.mirrored_strategy_with_one_gpu,
      # MultiWorkerMirroredStrategy
      tfdistribute.combinations.multi_worker_mirrored_2x1_cpu,
      tfdistribute.combinations.multi_worker_mirrored_2x1_gpu,
      # ParameterServerStrategy
      tfdistribute.combinations.parameter_server_strategy_3worker_2ps_cpu,
      tfdistribute.combinations.parameter_server_strategy_3worker_2ps_1gpu,
      tfdistribute.combinations.parameter_server_strategy_1worker_2ps_cpu,
      tfdistribute.combinations.parameter_server_strategy_1worker_2ps_1gpu,
      # Temporarily disable TPU tests of CL (b/269648832, b/269249455).
      # tfdistribute.combinations.tpu_strategy,
      # tfdistribute.combinations.tpu_strategy_one_core,
      # tfdistribute.combinations.tpu_strategy_packed_var,
  ]
  tasklist = [
      tasks.DeepGraphInfomaxTask("node"),
      tasks.BarlowTwinsTask("node"),
      tasks.VicRegTask("node"),
      # Multi-task: supervised & self-supervised tasks.
      {
          "classification": runner.RootNodeMulticlassClassification(
              "node",
              num_classes=len(_CLASSES),
              label_feature_name="classes"),
          "dgi": tasks.DeepGraphInfomaxTask("node"),
      },
      # Multi-task: multiple self-supervised tasks.
      {
          "bt": tasks.BarlowTwinsTask("node", representations_layer_name="bt"),
          "vr": tasks.VicRegTask("node", representations_layer_name="vr"),
      },
  ]
  return tftest.combinations.combine(distribution=strategies, task=tasklist)


def with_readout(gt: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
  return tfgnn.experimental.context_readout_into_feature(
      gt,
      feature_name="classes",
      remove_input_feature=True)


def random_graph_tensor() -> tfgnn.GraphTensor:
  schema = tfgnn.parse_schema(_SCHEMA)
  gtspec = tfgnn.create_graph_spec_from_schema_pb(schema)
  sample_dict = {(tfgnn.CONTEXT, None, "classes"): _CLASSES}
  return with_readout(tfgnn.random_graph_tensor(gtspec, sample_dict))


class DatasetProviderFromTensors(runner.DatasetProvider):

  def __init__(self, elements: Sequence[Any]):
    self._elements = list(elements)

  def get_dataset(self, _: tf.distribute.InputContext) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices(self._elements)


# TODO(b/265776928): Consider deduplication of distribute testing boilerplate.
class DistributeTests(tf.test.TestCase, parameterized.TestCase):

  @tfdistribute.combinations.generate(
      _all_eager_strategy_and_task_combinations()
  )
  def test_run(self, distribution: tf.distribute.Strategy, task: runner.Task):
    gts = (random_graph_tensor(),) * 4
    serialized = [tfgnn.write_example(gt).SerializeToString() for gt in gts]
    ds_provider = DatasetProviderFromTensors(serialized)

    def model_fn(graph_tensor_spec):
      graph = inputs = tf.keras.layers.Input(type_spec=graph_tensor_spec)
      graph = tfgnn.keras.layers.MapFeatures(
          node_sets_fn=lambda node_set, node_set_name: node_set["features"]
      )(graph)
      graph = vanilla_mpnn.VanillaMPNNGraphUpdate(
          units=2,
          message_dim=3,
          receiver_tag=tfgnn.SOURCE,
          l2_regularization=5e-4,
          dropout_rate=0.1)(graph)
      return tf.keras.Model(inputs, graph)

    model_dir = self.create_tempdir()

    trainer = runner.KerasTrainer(
        strategy=distribution,
        model_dir=model_dir,
        steps_per_epoch=1,
        validation_steps=1,
        restore_best_weights=False)

    if isinstance(distribution, tf.distribute.TPUStrategy):
      train_padding = runner.FitOrSkipPadding(
          gts[0].spec,
          ds_provider,
          fit_or_skip_sample_sample_size=5,
          fit_or_skip_success_ratio=0.7)
      valid_padding = runner.TightPadding(gts[0].spec, ds_provider)
    else:
      train_padding = None
      valid_padding = None

    runner.run(
        train_ds_provider=ds_provider,
        train_padding=train_padding,
        model_fn=model_fn,
        optimizer_fn=tf.keras.optimizers.Adam,
        epochs=1,
        trainer=trainer,
        task=task,
        gtspec=gts[0].spec,
        global_batch_size=2,
        feature_processors=tuple(),
        valid_ds_provider=ds_provider,
        valid_padding=valid_padding,
    )

    dataset = ds_provider.get_dataset(tf.distribute.InputContext())
    kwargs = {"examples": next(iter(dataset.batch(2)))}

    saved_model = tf.saved_model.load(os.path.join(model_dir, "export"))

    results = saved_model.signatures["serving_default"](**kwargs)
    self.assertLen(results, len(tf.nest.flatten(task)))  # One output per task.

    # The above distribute test verifies *only* that the `Task` runs under many
    # distribution strategies. Verify here that run and export also produce
    # finite values.
    for result in results.values():
      self.assertAllInRange(result, result.dtype.min, result.dtype.max)


if __name__ == "__main__":
  tfdistribute.multi_process_runner.test_main()
