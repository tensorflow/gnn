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
from typing import Any, Mapping

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn import runner
from tensorflow_gnn.models import vanilla_mpnn
from tensorflow_gnn.models.contrastive_losses import tasks


@functools.lru_cache(None)
def graph_tensor() -> tfgnn.GraphTensor:
  return tfgnn.GraphTensor.from_pieces(
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
      edge_sets={
          "edge": tfgnn.EdgeSet.from_fields(
              sizes=[2],
              adjacency=tfgnn.Adjacency.from_indices(
                  source=("node", [0, 1]),
                  target=("node", [1, 0]),
              ),
          )
      },
  )


def gnn_real(type_spec: tf.TypeSpec) -> tf.keras.Model:
  """Builds a Graph Neural Network that does some actual transformations."""
  inputs = tf.keras.Input(type_spec=type_spec)
  layer = vanilla_mpnn.VanillaMPNNGraphUpdate(
      units=2,
      message_dim=3,
      receiver_tag=tfgnn.SOURCE,
      l2_regularization=5e-4,
      dropout_rate=0.1,
  )
  return tf.keras.Model(inputs, layer(inputs))


def gnn_static(type_spec: tf.TypeSpec) -> tf.keras.Model:
  """Builds a static Graph Neural Network with known weights."""
  weights = [np.array([[0.0, 1.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0]])]
  def fn(node_set, *, node_set_name):
    del node_set_name
    dense = tf.keras.layers.Dense(2, use_bias=False, trainable=False)
    dense.build(node_set[tfgnn.HIDDEN_STATE].shape)
    dense.set_weights(weights)
    return dense(node_set[tfgnn.HIDDEN_STATE])
  inputs = tf.keras.Input(type_spec=type_spec)
  outputs = tfgnn.keras.layers.MapFeatures(node_sets_fn=fn)(inputs)
  return tf.keras.Model(inputs, outputs)


def all_tasks() -> Mapping[str, runner.Task]:
  return {
      "DGI": tasks.DeepGraphInfomaxTask("node", seed=8191),
      "VicReg": tasks.VicRegTask("node", seed=8191),
      "BarlowTwins": tasks.BarlowTwinsTask("node", seed=8191),
      "DGI_projected": tasks.DeepGraphInfomaxTask(
          "node", projector_units=[2], seed=8191
      ),
      **all_tasks_with_projector_as_inner_dimension([2]),
  }


def all_tasks_with_projector_as_inner_dimension(
    projector_units: Sequence[int],
) -> Mapping[str, runner.Task]:
  return {
      "VicReg_projected": tasks.VicRegTask(
          "node", projector_units=projector_units, seed=8191
      ),
      "BarlowTwins_projected": tasks.BarlowTwinsTask(
          "node", projector_units=projector_units, seed=8191
      ),
  }


def tasks_to_named_parameters(
    names_tasks: Mapping[str, runner.Task],
) -> Sequence[dict[str, Any]]:
  output = []
  for task_name, task in names_tasks.items():
    output.append(
        dict(
            testcase_name=task_name,
            task=task,
        )
    )
  return output


def bad_parameters_inputs() -> Sequence[dict[str, Any]]:
  output = []
  for task_name, task in all_tasks().items():
    output.append(
        dict(
            testcase_name="NoGraphTensorInputRight" + task_name,
            inputs=(graph_tensor(), tf.constant(range(8))),
            task=task,
            expected_error=r"Expected a `GraphTensor` input \(got .*\)",
        )
    )
    output.append(
        dict(
            testcase_name="NoGraphTensorInputLeft" + task_name,
            inputs=(tf.constant(range(8)), graph_tensor()),
            task=task,
            expected_error=r"Expected a `GraphTensor` input \(got .*\)",
        )
    )
  return output


class ContrastiveTasksSharedTests(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(bad_parameters_inputs())
  def test_bad_parameters(
      self,
      inputs: Sequence[Any],
      task: runner.Task,
      expected_error: str):
    with self.assertRaisesRegex(ValueError, expected_error):
      _ = task.predict(*inputs)

  @parameterized.named_parameters(tasks_to_named_parameters(all_tasks()))
  def test_fit(self, task: runner.Task):
    ds = tf.data.Dataset.from_tensors(graph_tensor()).repeat()
    ds = ds.batch(2).map(tfgnn.GraphTensor.merge_batch_to_components)
    ds = ds.map(task.preprocess).take(5)

    gts, _ = next(iter(ds))
    inputs = [tf.keras.Input(type_spec=gt.spec) for gt in gts]
    # All specs are the same (asserted in `test_preprocess`).
    model = gnn_real(inputs[0].spec)
    outputs = task.predict(*[model(i) for i in inputs])

    predicted = tf.keras.Model(inputs, outputs)
    predicted.compile(loss=task.losses(), metrics=task.metrics())

    before = predicted.evaluate(ds)
    predicted.fit(ds)

    self.assertLess(predicted.evaluate(ds), before)

  @parameterized.named_parameters(tasks_to_named_parameters(all_tasks()))
  def test_preprocess(self, task: runner.Task):
    # See `test_pseudolabels` for tests of the second part of `preprocess` fn.
    gt = graph_tensor()
    gts, _ = task.preprocess(gt)

    # Two `gts` with the same spec are returned (where the first matches the
    # input `gt` identity).
    self.assertLen(gts, 2)
    self.assertIs(gts[0], gt)
    self.assertEqual(gts[0].spec, gts[1].spec)

  @parameterized.named_parameters(
      tasks_to_named_parameters(
          all_tasks_with_projector_as_inner_dimension([7])
      )
  )
  def test_projector_shape(self, task: runner.Task):
    gts, _ = task.preprocess(graph_tensor())
    output = task.predict(*gts)
    self.assertEqual(output.shape, (1, 2, 7))

  @parameterized.named_parameters(tasks_to_named_parameters(all_tasks()))
  def test_predict(self, task: runner.Task):
    gts, _ = task.preprocess(graph_tensor())
    inputs = [tf.keras.Input(type_spec=gt.spec) for gt in gts]
    # All specs are the same (asserted in `test_preprocess`).
    model = gnn_real(inputs[0].spec)
    outputs = task.predict(*[model(i) for i in inputs])

    predicted = tf.keras.Model(inputs, outputs)

    # Recover the clean representations.
    layers = [
        l for l in predicted.layers if "clean_representations" == l.name
    ]
    self.assertLen(layers, 1)
    submodule = tf.keras.Model(predicted.input, layers[0].output)

    # Clean representations: root node readout.
    expected = tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name="node",
        feature_name=tfgnn.HIDDEN_STATE)(model(graph_tensor()))
    self.assertShapeEqual(submodule((graph_tensor(), graph_tensor())), expected)


class DeepGraphInfomaxTaskTest(tf.test.TestCase):
  task = tasks.DeepGraphInfomaxTask("node", seed=8191)

  def test_output_shape(self):
    inputs, _ = self.task.preprocess(graph_tensor())
    outputs = self.task.predict(*inputs)

    # Clean and corrupted logits (i.e., shape (1, 2)).
    self.assertEqual(outputs.shape, (1, 2))

  def test_pseudolabels(self):
    # See `test_preprocess` for tests of the first part of `preprocess` fn.
    _, pseudolabels = self.task.preprocess(graph_tensor())
    self.assertAllEqual(pseudolabels, ((1.0, 0.0),))


class BarlowTwinsTaskTest(tf.test.TestCase):
  task = tasks.BarlowTwinsTask("node", seed=8191)

  def test_output_shape(self):
    inputs, _ = self.task.preprocess(graph_tensor())
    outputs = self.task.predict(*inputs)

    # Clean and corrupted representations (shape (1, 4)) packed in one Tensor.
    self.assertEqual(outputs.shape, (1, 2, 4))

  def test_pseudolabels(self):
    # See `test_preprocess` for tests of the first part of `preprocess` fn.
    _, pseudolabels = self.task.preprocess(graph_tensor())
    self.assertAllEqual(pseudolabels, ((),))

  def test_metrics(self):
    # Clean and corrupted representations (shape (1, 4)) packed in one Tensor.
    y_pred = [[[0., 0., 0., 0.], [0., 0., 0., 0.]]]
    _, fake_y = self.task.preprocess(graph_tensor())

    self.assertLen(self.task.metrics(), 3)

    for metric_fn in self.task.metrics():
      self.assertEqual(metric_fn(fake_y, y_pred).shape, ())

  def test_loss_e2e(self):
    # A separate task here to not have a trivial case of the loss with BN.
    task = tasks.BarlowTwinsTask("node", seed=8191, normalize_batch=False)

    inputs, fake_y = task.preprocess(graph_tensor())
    model = gnn_static(inputs[0].spec)
    y_pred = task.predict(*[model(i) for i in inputs])
    losses = [loss_fn(fake_y, y_pred) for loss_fn in task.losses()]

    self.assertLen(losses, 1)
    # Loss matrix is 2D identity, hence both loss terms are 1.
    self.assertEqual(losses[0], 2.0)


class VicRegTaskTest(tf.test.TestCase):
  task = tasks.VicRegTask("node", seed=8191)

  def test_output_shape(self):
    inputs, _ = self.task.preprocess(graph_tensor())
    outputs = self.task.predict(*inputs)

    # Clean and corrupted representations (shape (1, 4)) packed in one Tensor.
    self.assertEqual(outputs.shape, (1, 2, 4))

  def test_pseudolabels(self):
    # See `test_preprocess` for tests of the first part of `preprocess` fn.
    _, pseudolabels = self.task.preprocess(graph_tensor())
    self.assertAllEqual(pseudolabels, ((),))

  def test_metrics(self):
    # Clean and corrupted representations (shape (1, 4)) packed in one Tensor.
    y_pred = [[[0., 0., 0., 0.], [0., 0., 0., 0.]]]
    _, fake_y = self.task.preprocess(graph_tensor())

    self.assertLen(self.task.metrics(), 3)

    for metric_fn in self.task.metrics():
      self.assertEqual(metric_fn(fake_y, y_pred).shape, ())

  def test_loss_e2e(self):
    # A separate task here to have an analytic solution to the loss function.
    task = tasks.VicRegTask("node", seed=8191, var_weight=0, cov_weight=0)

    inputs, fake_y = task.preprocess(graph_tensor())
    model = gnn_static(inputs[0].spec)
    y_pred = task.predict(*[model(i) for i in inputs])
    losses = [loss_fn(fake_y, y_pred) for loss_fn in task.losses()]

    self.assertLen(losses, 1)
    # Loss should be 2 * 25 (default `sim_weight`) = 50.
    self.assertEqual(losses[0], 50.0)


if __name__ == "__main__":
  tf.test.main()
