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
from typing import Callable, Sequence, Tuple, Union

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import vanilla_mpnn
from tensorflow_gnn.runner import interfaces
from tensorflow_gnn.runner import orchestration
from tensorflow_gnn.runner.tasks import classification
from tensorflow_gnn.runner.trainers import keras_fit
from tensorflow_gnn.runner.utils import label_fns

_CLASSES = tuple(range(32))
_SCHEMA = """
  context {
    features {
      key: "classes"
      value {
        dtype: DT_INT32
      }
    }
    features {
      key: "values"
      value {
        dtype: DT_FLOAT
      }
    }
  }
  node_sets {
    key: "nodes"
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
    key: "edges"
    value {
      source: "nodes"
      target: "nodes"
    }
  }
"""


def gt_spec() -> tfgnn.GraphTensorSpec:
  schema = tfgnn.parse_schema(_SCHEMA)
  return tfgnn.create_graph_spec_from_schema_pb(schema)


def random_graph_tensor() -> tfgnn.GraphTensor:
  sample_dict = {(tfgnn.CONTEXT, None, "classes"): _CLASSES}
  return tfgnn.random_graph_tensor(gt_spec(), sample_dict=sample_dict)


def random_serialized_graph_tensor() -> tfgnn.GraphTensor:
  return tfgnn.write_example(random_graph_tensor()).SerializeToString()


def model_fn() -> tf.keras.Model:
  node_sets_fn = lambda node_set, node_set_name: node_set["features"]
  inputs = x = tf.keras.layers.Input(type_spec=gt_spec())
  x = tfgnn.keras.layers.MapFeatures(node_sets_fn=node_sets_fn)(x)
  outputs = vanilla_mpnn.VanillaMPNNGraphUpdate(
      units=1,
      message_dim=2,
      receiver_tag=tfgnn.SOURCE,
      l2_regularization=5e-4,
      dropout_rate=0.1)(x)
  return tf.keras.Model(inputs, outputs)


def product(gt: tfgnn.GraphTensor, x: Union[int, float]) -> tfgnn.GraphTensor:
  def fn(inputs, **unused_kwargs):
    return {k: v * x for k, v in inputs.features.items()}
  return tfgnn.keras.layers.MapFeatures(fn, fn, fn)(gt)


def with_readout(gt: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
  for feature_name in list(gt.context.features.keys()):
    gt = tfgnn.experimental.context_readout_into_feature(
        gt,
        feature_name=feature_name,
        remove_input_feature=True)
  return gt


class DatasetProvider(interfaces.DatasetProvider):

  def __init__(
      self,
      element: Union[tfgnn.GraphTensor, bytes],
      cardinality: int = 8):
    self._ds = tf.data.Dataset.from_tensors(element).repeat(cardinality)

  def get_dataset(self, _: tf.distribute.InputContext) -> tf.data.Dataset:
    return self._ds


class SentinelTask(interfaces.Task):
  """Sentinel `Task`.

  This task is for testing `Task.preprocess(...)` and `Task.predict(...)`
  coordination in orchestraion.py.

  `Task.preprocess(...)` returns `len(head_weights)` `GraphTensor` outputs where
  the features of each is multiplied by its 1-based numbering position in the
  output sequence. The first output is returned unmutated (instead of
  multiplication by 1). `Task.predict(...)` returns the sum of its `GraphTensor`
  inputs first node readout (with `node_set_name="nodes"` and
  `feature_name=tfgnn.HIDDEN_STATE`) after applying any `head_weights`.
  Importantly: the order of `Task.preprocess(...)` outputs and
  `Task.predict(...)` inputs is expected to match after any base GNN
  application. (As asserted in `test_multi_task`.)
  """

  def __init__(self, head_weights: Sequence[float]):
    self._head_weights = head_weights

  def preprocess(
      self,
      # TODO(b/274672364): make this tuple[...] in Python 3.9 style
      # when we drop py38 support.
      gt: tfgnn.GraphTensor) -> Tuple[Sequence[tfgnn.GraphTensor], tfgnn.Field]:
    x = [gt, *[product(gt, i + 2) for i in range(len(self._head_weights) - 1)]]
    y = tfgnn.keras.layers.Readout(
        feature_name="values",
        node_set_name="_readout")(gt)
    return x, y

  def predict(self, *gts: tfgnn.GraphTensor) -> tfgnn.Field:
    readout = tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name="nodes",
        feature_name=tfgnn.HIDDEN_STATE)
    logits = [
        readout(gt) * weight for gt, weight in zip(gts, self._head_weights)
    ]
    return tf.add_n(logits)

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (tf.keras.losses.MeanSquaredError(),)

  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return tuple()


class OrchestrationTests(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="GraphTensors",
          ds_provider=DatasetProvider(random_graph_tensor()),
      ),
      dict(
          testcase_name="SerializedGraphTensors",
          ds_provider=DatasetProvider(random_serialized_graph_tensor()),
      ),
  ])
  def test_run(
      self,
      ds_provider: orchestration.DatasetProvider):
    task = classification.RootNodeMulticlassClassification(
        "nodes",
        num_classes=len(_CLASSES),
        label_fn=label_fns.ContextLabelFn("classes"))
    model = model_fn()
    model_dir = self.create_tempdir()
    trainer = keras_fit.KerasTrainer(
        strategy=tf.distribute.get_strategy(),
        model_dir=model_dir,
        steps_per_epoch=1,
        restore_best_weights=False)

    run_result = orchestration.run(
        train_ds_provider=ds_provider,
        model_fn=lambda _: model,
        optimizer_fn=tf.keras.optimizers.Adam,
        epochs=1,
        trainer=trainer,
        task=task,
        gtspec=gt_spec(),
        global_batch_size=2)

    # Check the base model.
    actual = run_result.base_model
    expected = model

    # `actual_names` will be a superset because of the `_BASE_MODEL_TAG` layer.
    actual_names = tuple(sm.name for sm in actual.submodules)
    expected_names = tuple(sm.name for sm in expected.submodules)

    self.assertAllInSet(expected_names, actual_names)

    # Any computations are identical.
    inputs = tfgnn.random_graph_tensor(gt_spec())
    actual_outputs = actual(inputs).node_sets["nodes"][tfgnn.HIDDEN_STATE]
    expected_outputs = expected(inputs).node_sets["nodes"][tfgnn.HIDDEN_STATE]

    self.assertAllClose(expected_outputs, actual_outputs)

    # Check the exported model.
    saved_model = tf.saved_model.load(os.path.join(model_dir, "export"))

    examples = tf.constant((random_serialized_graph_tensor(),) * 2)
    output = saved_model.signatures["serving_default"](examples=examples)

    # The model has one output
    self.assertLen(output, 1)

    # The expected shape is (batch size, num classes).
    actual = next(iter(output.values())).shape
    expected = (examples.shape[0], len(_CLASSES))

    self.assertAllEqual(expected, actual)

  def test_multi_task(self):
    gt = with_readout(random_graph_tensor())
    tasks = {
        # One unmutated and one mutated `Task.preprocess(...)` output (both
        # used by `Task.predict(...)`).
        "s1": SentinelTask((1., 1.)),
        # One unmutated and one mutated `Task.preprocess(...)` output (only the
        # second mutated output is used by `Task.predict(...)`).
        "s2": SentinelTask((0., 1.)),
        # One unmutated `Task.preprocess(...)` output (used by
        # `Task.predict(...)`).
        "s3": SentinelTask((1.,)),
    }
    model_dir = self.create_tempdir()
    trainer = keras_fit.KerasTrainer(
        strategy=tf.distribute.get_strategy(),
        model_dir=model_dir,
        steps_per_epoch=1,
        restore_best_weights=False)

    run_result = orchestration.run(
        train_ds_provider=DatasetProvider(gt),
        model_fn=lambda _: model_fn(),
        optimizer_fn=tf.keras.optimizers.Adam,
        epochs=1,
        trainer=trainer,
        task=tasks,
        gtspec=gt.spec,
        global_batch_size=2)

    def serialize(inputs):
      return tf.constant((tfgnn.write_example(inputs).SerializeToString(),))

    xs, ys = run_result.preprocess_model(serialize(gt))

    # Two preprocessing outputs are deduplicated: there are a total of 3 unique
    # preprocessing outputs only...
    self.assertLen(xs, 3)
    # and labels match the structure of `tasks`.
    tf.nest.assert_same_structure(ys, tasks)

    actual = run_result.trained_model(xs)

    # `actual["s1"]` uses the sum of both `predict(...)` inputs.
    self.assertAllClose(actual["s1"], actual["s2"] + actual["s3"])

    # `actual["s2"]` uses the second `predict(...)` input only (i.e.: twice
    # the first input).
    xs, _ = run_result.preprocess_model(serialize(product(gt, 2)))
    self.assertAllClose(actual["s2"], run_result.trained_model(xs)["s3"])

    # `actual["s3"]` uses the first `predict(...)` input only (i.e., half the
    # second input).
    xs, _ = run_result.preprocess_model(serialize(product(gt, .5)))
    self.assertAllClose(actual["s3"], run_result.trained_model(xs)["s2"])

  def test_make_preprocessing_model(self):
    task_processorx = lambda gt: ((gt, product(gt, 2)), gt.context["classes"])
    task_processory = lambda gt: (gt, gt.context["classes"])
    task_processorz = lambda gt: ((product(gt, 4), gt), gt.context["classes"])

    task = {"x": task_processorx, "y": task_processory, "z": task_processorz}

    preprocess_model, oimap = orchestration._make_preprocessing_model(
        gt_spec(),
        tuple(),
        task)

    # `gt` is present in all outputs.
    self.assertEqual(oimap["x"][0], oimap["y"])
    self.assertEqual(oimap["y"], oimap["z"][1])

    # `product(gt, 2)` and `product(gt, 4)` do not match each other (or `gt`).
    self.assertNotEqual(oimap["x"][1], oimap["z"][0])
    self.assertNotEqual(oimap["x"][1], oimap["y"])
    self.assertNotEqual(oimap["z"][0], oimap["y"])

    # There exist 3 unique objects (`gt`, `product(gt, 2)` and
    # `product(gt, 4)`).
    indices = sorted(set([*oimap["x"], oimap["y"], *oimap["z"]]))
    self.assertAllEqual(indices, (0, 1, 2))

    xs, ys = preprocess_model.output

    # There exist 3 unique outputs (corresponding to the unique objects).
    self.assertLen(xs, 3)
    # There exists 1 label per `task` (i.e., the nested structure matches).
    tf.nest.assert_same_structure(ys, task)


if __name__ == "__main__":
  tf.test.main()
