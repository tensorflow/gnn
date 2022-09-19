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
"""Tests for orchestraion."""
import os
from typing import Sequence

from absl.testing import parameterized
import tensorflow as tf
import tensorflow.__internal__.distribute as tfdistribute
import tensorflow.__internal__.test as tftest
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import vanilla_mpnn
from tensorflow_gnn.runner import orchestration
from tensorflow_gnn.runner.tasks import classification
from tensorflow_gnn.runner.trainers import keras_fit
from tensorflow_gnn.runner.utils import model_templates
from tensorflow_gnn.runner.utils import padding

SCHEMA = """
  context {
    features {
      key: "label"
      value {
        dtype: DT_INT32
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
  }"""


def _all_eager_strategy_combinations():
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
      # TPUStrategy
      tfdistribute.combinations.tpu_strategy,
      tfdistribute.combinations.tpu_strategy_one_core,
      tfdistribute.combinations.tpu_strategy_packed_var,
      # ParameterServerStrategy
      tfdistribute.combinations.parameter_server_strategy_3worker_2ps_cpu,
      tfdistribute.combinations.parameter_server_strategy_3worker_2ps_1gpu,
      tfdistribute.combinations.parameter_server_strategy_1worker_2ps_cpu,
      tfdistribute.combinations.parameter_server_strategy_1worker_2ps_1gpu,
  ]
  return tftest.combinations.combine(distribution=strategies)


class DatasetProvider:

  def __init__(self, examples: Sequence[bytes]):
    self._examples = list(examples)

  def get_dataset(self, _: tf.distribute.InputContext) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices(self._examples)


class OrchestrationTests(tf.test.TestCase, parameterized.TestCase):

  @tfdistribute.combinations.generate(_all_eager_strategy_combinations())
  def test_run(self, distribution: tf.distribute.Strategy):
    schema = tfgnn.parse_schema(SCHEMA)
    gtspec = tfgnn.create_graph_spec_from_schema_pb(schema)
    smaller_graph = tfgnn.random_graph_tensor(gtspec, row_lengths_range=[1, 2])
    larger_graph = tfgnn.random_graph_tensor(gtspec, row_lengths_range=[7, 19])
    ds_provider = DatasetProvider(
        [tfgnn.write_example(smaller_graph).SerializeToString()] * 4 +
        [tfgnn.write_example(larger_graph).SerializeToString()])

    def extract_labels(gt):
      return gt, gt.context["label"] % 10  # Ten labels

    def node_sets_fn(node_set, node_set_name):
      del node_set_name
      return node_set["features"]

    model_fn = model_templates.ModelFromInitAndUpdates(
        init=tfgnn.keras.layers.MapFeatures(node_sets_fn=node_sets_fn),
        updates=[vanilla_mpnn.VanillaMPNNGraphUpdate(
            units=1,
            message_dim=2,
            receiver_tag=tfgnn.SOURCE,
            l2_regularization=5e-4,
            dropout_rate=0.1)])

    task = classification.RootNodeMulticlassClassification(
        node_set_name="node",
        num_classes=10)  # Ten classes (like above)

    model_dir = self.create_tempdir()

    trainer = keras_fit.KerasTrainer(
        strategy=distribution,
        model_dir=model_dir,
        steps_per_epoch=1,
        validation_steps=1,
        restore_best_weights=False)

    if isinstance(distribution, tf.distribute.TPUStrategy):
      train_padding = padding.FitOrSkipPadding(
          gtspec,
          ds_provider,
          fit_or_skip_sample_sample_size=5,
          fit_or_skip_success_ratio=0.7)
      valid_padding = padding.TightPadding(gtspec, ds_provider)
    else:
      train_padding = None
      valid_padding = None

    orchestration.run(
        train_ds_provider=ds_provider,
        train_padding=train_padding,
        model_fn=model_fn,
        optimizer_fn=tf.keras.optimizers.Adam,
        epochs=1,
        trainer=trainer,
        task=task,
        gtspec=gtspec,
        drop_remainder=False,
        global_batch_size=4,
        feature_processors=[extract_labels],
        valid_ds_provider=ds_provider,
        valid_padding=valid_padding)

    dataset = ds_provider.get_dataset(tf.distribute.InputContext())
    kwargs = {"examples": next(iter(dataset.batch(2)))}

    saved_model = tf.saved_model.load(os.path.join(model_dir, "export"))
    output = saved_model.signatures["serving_default"](**kwargs)
    actual = next(iter(output.values()))

    # The model has one output
    self.assertLen(output, 1)

    # The expected shape is (batch size, num classes) or (2, 10)
    self.assertShapeEqual(actual, tf.random.uniform((2, 10)))


if __name__ == "__main__":
  tfdistribute.multi_process_runner.test_main()
