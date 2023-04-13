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
import functools
import os
from typing import Any, Sequence, Union

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import vanilla_mpnn
from tensorflow_gnn.runner import orchestration
from tensorflow_gnn.runner.tasks import classification
from tensorflow_gnn.runner.trainers import keras_fit
from tensorflow_gnn.runner.utils import label_fns

_LABELS = tuple(range(32))
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


def graph_spec() -> tfgnn.GraphTensorSpec:
  schema = tfgnn.parse_schema(_SCHEMA)
  return tfgnn.create_graph_spec_from_schema_pb(schema)


def random_graph_tensor() -> tfgnn.GraphTensor:
  sample_dict = {(tfgnn.CONTEXT, None, "label"): _LABELS}
  return tfgnn.random_graph_tensor(graph_spec(), sample_dict=sample_dict)


def random_serialized_graph_tensor() -> tfgnn.GraphTensor:
  return tfgnn.write_example(random_graph_tensor()).SerializeToString()


class DatasetProvider(orchestration.DatasetProvider):

  def __init__(
      self,
      element: Union[tfgnn.GraphTensor, bytes, Any],
      cardinality: int = 8):
    self._ds = tf.data.Dataset.from_tensors(element).repeat(cardinality)

  def get_dataset(self, _: tf.distribute.InputContext) -> tf.data.Dataset:
    return self._ds


class OrchestrationTests(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="GraphTensors",
          gtspec=graph_spec(),
          ds_provider=DatasetProvider(random_graph_tensor()),
          examples=tf.constant((random_serialized_graph_tensor(),) * 2),
      ),
      dict(
          testcase_name="SerializedGraphTensors",
          gtspec=graph_spec(),
          ds_provider=DatasetProvider(random_serialized_graph_tensor()),
          examples=tf.constant((random_serialized_graph_tensor(),) * 2),
      ),
  ])
  def test_run(
      self,
      gtspec: tfgnn.GraphTensorSpec,
      ds_provider: orchestration.DatasetProvider,
      examples: Sequence[str]):

    @functools.lru_cache(None)
    def model():
      inputs = x = tf.keras.Input(type_spec=gtspec)
      def tail_fn(inputs, **unused_kwargs):
        return tf.keras.Sequential([
            tf.keras.layers.Concatenate(),
            tf.keras.layers.Dense(8),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(.25),
        ])(inputs.features.values())
      x = tfgnn.keras.layers.MapFeatures(node_sets_fn=tail_fn)(x)
      for _ in range(4):
        x = vanilla_mpnn.VanillaMPNNGraphUpdate(
            units=2,
            message_dim=3,
            receiver_tag=tfgnn.SOURCE,
            l2_regularization=5e-4,
            dropout_rate=0.1)(x)
      def head_fn(inputs, **unused_kwargs):
        return tf.keras.activations.tanh(inputs[tfgnn.HIDDEN_STATE])
      outputs = tfgnn.keras.layers.MapFeatures(node_sets_fn=head_fn)(x)
      return tf.keras.Model(inputs, outputs)

    def model_fn(_):
      return model()

    task = classification.RootNodeMulticlassClassification(
        "node",
        num_classes=len(_LABELS),
        label_fn=label_fns.ContextLabelFn("label"))

    model_dir = self.create_tempdir()

    trainer = keras_fit.KerasTrainer(
        strategy=tf.distribute.get_strategy(),
        model_dir=model_dir,
        steps_per_epoch=1,
        validation_steps=1,
        restore_best_weights=False)

    run_result = orchestration.run(
        train_ds_provider=ds_provider,
        model_fn=model_fn,
        optimizer_fn=tf.keras.optimizers.Adam,
        epochs=1,
        trainer=trainer,
        task=task,
        gtspec=gtspec,
        drop_remainder=False,
        global_batch_size=4,
        valid_ds_provider=ds_provider)

    # Check the base model.
    actual = run_result.base_model
    expected = model_fn(gtspec)

    # `actual_names` will be a superset because of the `_BASE_MODEL_TAG` layer.
    actual_names = tuple(sm.name for sm in actual.submodules)
    expected_names = tuple(sm.name for sm in expected.submodules)

    self.assertAllInSet(expected_names, actual_names)

    # Any computations are identical.
    inputs = tfgnn.random_graph_tensor(gtspec)
    actual_outputs = actual(inputs).node_sets["node"][tfgnn.HIDDEN_STATE]
    expected_outputs = expected(inputs).node_sets["node"][tfgnn.HIDDEN_STATE]

    self.assertAllClose(expected_outputs, actual_outputs)

    # Check the exported model.
    saved_model = tf.saved_model.load(os.path.join(model_dir, "export"))
    output = saved_model.signatures["serving_default"](examples=examples)

    # The model has one output
    self.assertLen(output, 1)

    # The expected shape is (batch size, num classes).
    actual = next(iter(output.values())).shape
    expected = (examples.shape[0], len(_LABELS))

    self.assertAllEqual(expected, actual)


if __name__ == "__main__":
  tf.test.main()
