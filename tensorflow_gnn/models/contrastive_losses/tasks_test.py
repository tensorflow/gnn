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
"""Tests for tasks."""
from __future__ import annotations
from collections.abc import Sequence
import functools
from typing import Any

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn import runner
from tensorflow_gnn.models import vanilla_mpnn
from tensorflow_gnn.models.contrastive_losses import tasks

SCHEMA = """
node_sets {
  key: "node"
  value {
    features {
      key: "%s"
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
""" % tfgnn.HIDDEN_STATE


@functools.lru_cache(None)
def gtspec() -> tfgnn.GraphTensorSpec:
  return tfgnn.create_graph_spec_from_schema_pb(tfgnn.parse_schema(SCHEMA))


@functools.lru_cache(None)
def random_graph_tensor() -> tfgnn.GraphTensor:
  return tfgnn.random_graph_tensor(gtspec())


@functools.lru_cache(None)
def fixed_e2e_graph_and_model() -> tuple[tfgnn.GraphTensor, tf.keras.Model]:
  graph = tfgnn.GraphTensor.from_pieces(
      node_sets={
          "node": tfgnn.NodeSet.from_fields(
              sizes=[2],
              features={
                  tfgnn.HIDDEN_STATE: tf.constant(
                      [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]
                  )
              },
          ),
      },
  )
  inputs = tf.keras.layers.Input(type_spec=graph.spec)
  hidden_state = inputs.node_sets["node"][tfgnn.HIDDEN_STATE]
  output = inputs.replace_features(
      node_sets={
          "node": {
              tfgnn.HIDDEN_STATE: tf.keras.layers.Dense(2, use_bias=False)(
                  hidden_state
              )
          }
      }
  )
  model = tf.keras.Model(inputs, output)
  return graph, model


def dnn() -> tf.keras.Model:
  """Builds a Deep Neural Network (tensor input and output)."""
  inputs = tf.keras.Input((4,))
  return tf.keras.Model(inputs, inputs * 4)


def gnn() -> tf.keras.Model:
  """Builds a Graph Neural Network (graph input and output)."""
  inputs = tf.keras.Input(type_spec=gtspec())
  return tf.keras.Model(inputs, inputs)


def gdnn() -> tf.keras.Model:
  """Builds a Graph Deep Neural Network (graph input but tensor output)."""
  inputs = tf.keras.Input(type_spec=gtspec())
  outputs = tfgnn.keras.layers.ReadoutFirstNode(
      node_set_name="node",
      feature_name=tfgnn.HIDDEN_STATE)(inputs)
  return tf.keras.Model(inputs, outputs)


def gnn_real() -> tf.keras.Model:
  """Builds a Graph Neural Network that does some actual transformations."""
  inputs = tf.keras.Input(type_spec=gtspec())
  layer = vanilla_mpnn.VanillaMPNNGraphUpdate(
      units=2,
      message_dim=3,
      receiver_tag=tfgnn.SOURCE,
      l2_regularization=5e-4,
      dropout_rate=0.1,
  )
  return tf.keras.Model(inputs, layer(inputs))


def all_tasks() -> Sequence[runner.Task]:
  return [
      tasks.DeepGraphInfomaxTask("node", seed=8191),
      tasks.VicRegTask("node", seed=8191),
      tasks.BarlowTwinsTask("node", seed=8191),
  ]


def all_tasks_inputs() -> Sequence[dict[str, Any]]:
  output = []
  for task in all_tasks():
    output.append(
        dict(
            testcase_name=task.__class__.__name__,
            task=task,
        )
    )
  return output


def bad_parameters_inputs() -> Sequence[dict[str, Any]]:
  output = []
  for task in all_tasks():
    output.append(
        dict(
            testcase_name="NoGraphTensorInput" + task.__class__.__name__,
            model=dnn(),
            task=task,
            expected_error=r"Expected a `GraphTensor` input \(got .*\)",
        )
    )
    output.append(
        dict(
            testcase_name="NoGraphTensorOutput" + task.__class__.__name__,
            model=gdnn(),
            task=task,
            expected_error=r"Expected a `GraphTensor` output \(got .*\)",
        )
    )
  return output


class ContrastiveTasksSharedTests(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(bad_parameters_inputs())
  def test_value_error(
      self, model: tf.keras.Model, task: runner.Task, expected_error: str
  ):
    """Verifies invalid input raises `ValueError`."""
    with self.assertRaisesRegex(ValueError, expected_error):
      _ = task.adapt(model)

  @parameterized.named_parameters(all_tasks_inputs())
  def test_fit(self, task: runner.Task):
    """Verifies an adapted model's fit."""
    ds = tf.data.Dataset.from_tensors(random_graph_tensor()).repeat()
    ds = ds.batch(2).map(tfgnn.GraphTensor.merge_batch_to_components)
    # `preprocess` performs no manipulations on `x`.
    ds = ds.map(lambda x: (x, task.preprocess(x)[1])).take(5)

    def get_loss():
      values = model.evaluate(ds)
      return values

    model = task.adapt(gnn_real())
    model.compile(loss=task.losses(), metrics=task.metrics())
    before = get_loss()
    model.fit(ds)
    after = get_loss()
    # NOTE: as much as I'd like the loss to fall, we can only reliably test
    # inequality here.
    self.assertNotEqual(before, after)

  @parameterized.named_parameters(all_tasks_inputs())
  def test_preprocess(self, task: runner.Task):
    """Verifies pseudo-labels for always outputting the input `graph_tensor`."""
    # See `test_pseudolabels` for tests of the second part of `preprocess` fn.
    expected = random_graph_tensor()
    actual, _ = task.preprocess(expected)

    self.assertIsNone(actual)

  @parameterized.named_parameters(all_tasks_inputs())
  def test_adapt(self, task: runner.Task):
    """Verifies an adapted model's output shape."""
    adapted = task.adapt(gnn())
    gt = random_graph_tensor()

    # Submodules (i.e., try to recover clean representations).
    submodule, *others = [
        m for m in adapted.submodules
        if m.name == "clean_representations"
    ]
    self.assertEmpty(others)

    # Clean representations (i.e., root node readout).
    expected = tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name="node",
        feature_name=tfgnn.HIDDEN_STATE)(gt)
    self.assertShapeEqual(submodule(gt), expected)


class DeepGraphInfomaxTaskTest(tf.test.TestCase):
  task = tasks.DeepGraphInfomaxTask("node", seed=8191)

  def test_output_shape(self):
    adapted = self.task.adapt(gnn())
    gt = random_graph_tensor()
    # Clean and corrupted logits (i.e., shape (1, 2)).
    logits = adapted(gt)
    self.assertEqual(logits.shape, (1, 2))

  def test_pseudolabels(self):
    # See `test_preprocess` for tests of the first part of `preprocess` fn.
    gt = random_graph_tensor()
    _, pseudolabels = self.task.preprocess(gt)
    self.assertAllEqual(pseudolabels, ((1.0, 0.0),))


class BarlowTwinsTaskTest(tf.test.TestCase):
  task = tasks.BarlowTwinsTask("node", seed=8191)

  def test_output_shape(self):
    adapted = self.task.adapt(gnn())
    gt = random_graph_tensor()
    # Clean and corrupted representations (shape (1, 4)) packed in one Tensor.
    output = adapted(gt)
    self.assertEqual(output.shape, (1, 2, 4))

  def test_pseudolabels(self):
    # See `test_preprocess` for tests of the first part of `preprocess` fn.
    gt = random_graph_tensor()
    _, pseudolabels = self.task.preprocess(gt)
    self.assertAllEqual(pseudolabels, ((),))

  def test_metrics(self):
    adapted = self.task.adapt(gnn())
    gt = random_graph_tensor()
    _, fake_y = self.task.preprocess(gt)
    logits = adapted(gt)
    metric_fns = self.task.metrics()
    self.assertLen(metric_fns, 1)
    for metric_fn in metric_fns:
      metric_value = metric_fn(fake_y, logits)
      self.assertEqual(metric_value.shape, ())

  def test_loss_e2e(self):
    # A separate task here to not have a trivial case of the loss with BN
    task = tasks.BarlowTwinsTask("node", seed=8191, normalize_batch=False)

    graph, model = fixed_e2e_graph_and_model()
    model = task.adapt(model)
    weights = {v.name: v for v in model.trainable_weights}
    self.assertLen(weights, 1)
    weights["dense/kernel:0"].assign(
        [[0.0, 1.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0]]
    )
    output = model(graph)
    # The result of matrix multiplication of the node features @ weights.
    # Note that we use readout_first_node here, so the second output is coming
    # from the shuffled node features.
    expected = tf.convert_to_tensor([[[0., 2.], [0., 0.]]], dtype=tf.float32)
    self.assertAllClose(output, expected)

    _, fake_y = task.preprocess(graph)
    losses = [loss_fn(fake_y, output) for loss_fn in task.losses()]
    # Loss matrix is 2D identity, hence both loss terms are 1.
    self.assertEqual(losses[0], 2.0)


class VicRegTaskTest(tf.test.TestCase):
  task = tasks.VicRegTask("node", seed=8191)

  def test_output_shape(self):
    adapted = self.task.adapt(gnn())
    gt = random_graph_tensor()
    # Clean and corrupted representations (shape (1, 4)) packed in one Tensor.
    output = adapted(gt)
    self.assertEqual(output.shape, (1, 2, 4))

  def test_pseudolabels(self):
    # See `test_preprocess` for tests of the first part of `preprocess` fn.
    gt = random_graph_tensor()
    _, pseudolabels = self.task.preprocess(gt)
    self.assertAllEqual(pseudolabels, ((),))

  def test_metrics(self):
    adapted = self.task.adapt(gnn())
    gt = random_graph_tensor()
    _, fake_y = self.task.preprocess(gt)
    logits = adapted(gt)
    metric_fns = self.task.metrics()
    self.assertLen(metric_fns, 1)
    for metric_fn in metric_fns:
      metric_value = metric_fn(fake_y, logits)
      self.assertEqual(metric_value.shape, ())

  def test_loss_e2e(self):
    # A separate task here to have an analytic solution to the loss function.
    task = tasks.VicRegTask("node", seed=8191, var_weight=0, cov_weight=0)

    graph, model = fixed_e2e_graph_and_model()
    model = task.adapt(model)
    weights = {v.name: v for v in model.trainable_weights}
    self.assertLen(weights, 1)
    weights["dense/kernel:0"].assign(
        [[0.0, 1.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0]]
    )
    output = model(graph)
    expected = tf.convert_to_tensor([[[0., 2.], [0., 0.]]], dtype=tf.float32)
    self.assertAllClose(output, expected)

    _, fake_y = task.preprocess(graph)
    losses = [loss_fn(fake_y, output) for loss_fn in task.losses()]
    self.assertLen(losses, 1)

    # Loss should be 2 * 25 (default `sim_weight`) = 50
    self.assertEqual(losses[0], 50.0)


if __name__ == "__main__":
  tf.test.main()
