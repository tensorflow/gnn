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
"""Tests for orchestration."""
import os
from typing import Sequence

from absl.testing import parameterized
from immutabledict import immutabledict
import tensorflow as tf
import tensorflow.__internal__.distribute as tfdistribute
import tensorflow.__internal__.test as tftest
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import vanilla_mpnn
from tensorflow_gnn.runner import interfaces
from tensorflow_gnn.runner import orchestration
from tensorflow_gnn.runner.tasks import classification
from tensorflow_gnn.runner.tasks import regression
from tensorflow_gnn.runner.trainers import keras_fit
from tensorflow_gnn.runner.utils import label_fns
from tensorflow_gnn.runner.utils import padding

_LABELS = tuple(range(32))

_SAMPLE_DICT = immutabledict({(tfgnn.CONTEXT, None, "label"): _LABELS})

_SCHEMA = """
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
  }
"""


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


def _all_task_combinations():

  def extract_binary_labels(inputs):
    x, y = label_fns.ContextLabelFn("label")(inputs)
    return x, y % 2

  def extract_multiclass_labels(inputs):
    return label_fns.ContextLabelFn("label")(inputs)

  def extract_regression_labels(inputs):
    x, y = label_fns.ContextLabelFn("label")(inputs)
    return x, tf.ones_like(y, dtype=tf.float32)

  tasks = [
      # Root node classification
      classification.RootNodeBinaryClassification(
          "node",
          label_fn=extract_binary_labels),
      classification.RootNodeMulticlassClassification(
          "node",
          num_classes=len(_LABELS),
          label_fn=extract_multiclass_labels),
      # Graph classification
      classification.GraphBinaryClassification(
          "node",
          label_fn=extract_binary_labels),
      classification.GraphMulticlassClassification(
          "node",
          num_classes=len(_LABELS),
          label_fn=extract_multiclass_labels),
      # Root node regression
      regression.RootNodeMeanAbsoluteError(
          "node",
          label_fn=extract_regression_labels),
      regression.RootNodeMeanAbsolutePercentageError(
          "node",
          label_fn=extract_regression_labels),
      regression.RootNodeMeanSquaredError(
          "node",
          label_fn=extract_regression_labels),
      regression.RootNodeMeanSquaredLogarithmicError(
          "node",
          label_fn=extract_regression_labels),
      regression.RootNodeMeanSquaredLogScaledError(
          "node",
          label_fn=extract_regression_labels),
      # Graph regression
      regression.GraphMeanAbsoluteError(
          "node",
          label_fn=extract_regression_labels),
      regression.GraphMeanAbsolutePercentageError(
          "node",
          label_fn=extract_regression_labels),
      regression.GraphMeanSquaredError(
          "node",
          label_fn=extract_regression_labels),
      regression.GraphMeanSquaredLogarithmicError(
          "node",
          label_fn=extract_regression_labels),
      regression.GraphMeanSquaredLogScaledError(
          "node",
          label_fn=extract_regression_labels),
  ]
  return tftest.combinations.combine(task=tasks)


class DatasetProvider(interfaces.DatasetProvider):

  def __init__(self, examples: Sequence[bytes]):
    self._examples = list(examples)

  def get_dataset(self, _: tf.distribute.InputContext) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices(self._examples)


class OrchestrationTests(tf.test.TestCase, parameterized.TestCase):

  @tfdistribute.combinations.generate(
      tftest.combinations.times(
          _all_eager_strategy_combinations(),
          _all_task_combinations()
      )
  )
  def test_run(
      self,
      distribution: tf.distribute.Strategy,
      task: interfaces.Task):
    schema = tfgnn.parse_schema(_SCHEMA)
    gtspec = tfgnn.create_graph_spec_from_schema_pb(schema)
    gt = tfgnn.write_example(tfgnn.random_graph_tensor(
        gtspec,
        sample_dict=_SAMPLE_DICT))
    ds_provider = DatasetProvider((gt.SerializeToString(),) * 4)

    def model_fn(graph_tensor_spec):
      graph = inputs = tf.keras.layers.Input(type_spec=graph_tensor_spec)
      graph = tfgnn.keras.layers.MapFeatures(
          node_sets_fn=lambda node_set, node_set_name: node_set["features"]
      )(graph)
      graph = vanilla_mpnn.VanillaMPNNGraphUpdate(
          units=1,
          message_dim=2,
          receiver_tag=tfgnn.SOURCE,
          l2_regularization=5e-4,
          dropout_rate=0.1)(graph)
      return tf.keras.Model(inputs, graph)

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
        global_batch_size=2,
        valid_ds_provider=ds_provider,
        valid_padding=valid_padding)

    dataset = ds_provider.get_dataset(tf.distribute.InputContext())
    kwargs = {"examples": next(iter(dataset.batch(2)))}

    saved_model = tf.saved_model.load(os.path.join(model_dir, "export"))
    saved_model.signatures["serving_default"](**kwargs)

    results = saved_model.signatures["serving_default"](**kwargs)
    self.assertLen(results, 1)  # The task has a single output.

    [result] = results.values()
    # The above distribute test verifies *only* that the `Task` runs under many
    # distribution strategies. Verify here that run and export also produce
    # finite values.
    self.assertAllInRange(result, result.dtype.min, result.dtype.max)


if __name__ == "__main__":
  tfdistribute.multi_process_runner.test_main()
