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
"""Tests for classification."""
from __future__ import annotations
from typing import Callable, Sequence, Type

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn.runner import interfaces
from tensorflow_gnn.runner.tasks import classification

GraphTensor = tfgnn.GraphTensor
Field = tfgnn.Field

TEST_GRAPH_TENSOR = GraphTensor.from_pieces(
    context=tfgnn.Context.from_fields(
        features={"labels": tf.constant((8, 1, 9, 1))}
    ),
    node_sets={
        "nodes": tfgnn.NodeSet.from_fields(
            sizes=tf.constant((2, 4, 8, 4)),
            features={
                tfgnn.HIDDEN_STATE: tf.random.uniform((18, 8)),
            },
        )
    },
)


def label_fn(num_labels: int) -> Callable[..., tuple[GraphTensor, Field]]:
  def fn(inputs):
    y = inputs.context["labels"]
    x = inputs.remove_features(context=("labels",))
    return x, y % num_labels
  return fn


def with_readout(num_labels: int, gt: GraphTensor) -> GraphTensor:
  context_fn = lambda inputs: {"labels": inputs["labels"] % num_labels}
  gt = tfgnn.keras.layers.MapFeatures(context_fn=context_fn)(gt)
  return tfgnn.experimental.context_readout_into_feature(
      gt,
      feature_name="labels",
      remove_input_feature=False)


class Classification(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="GraphBinaryClassificationLabelFn",
          task=classification.GraphBinaryClassification(
              "nodes",
              label_fn=label_fn(2)),
          inputs=TEST_GRAPH_TENSOR,
          expected_gt=TEST_GRAPH_TENSOR.remove_features(context=("labels",)),
          expected_labels=(0, 1, 1, 1)),
      dict(
          testcase_name="GraphBinaryClassificationReadout",
          task=classification.GraphBinaryClassification(
              "nodes",
              label_feature_name="labels"),
          inputs=with_readout(2, TEST_GRAPH_TENSOR),
          expected_gt=with_readout(2, TEST_GRAPH_TENSOR),
          expected_labels=(0, 1, 1, 1)),
      dict(
          testcase_name="GraphMulticlassClassificationLabelFn",
          task=classification.GraphMulticlassClassification(
              "nodes",
              num_classes=4,
              label_fn=label_fn(4)),
          inputs=TEST_GRAPH_TENSOR,
          expected_gt=TEST_GRAPH_TENSOR.remove_features(context=("labels",)),
          expected_labels=(0, 1, 1, 1)),
      dict(
          testcase_name="GraphMulticlassClassificationReadout",
          task=classification.GraphMulticlassClassification(
              "nodes",
              num_classes=4,
              label_feature_name="labels"),
          inputs=with_readout(4, TEST_GRAPH_TENSOR),
          expected_gt=with_readout(4, TEST_GRAPH_TENSOR),
          expected_labels=(0, 1, 1, 1)),
      dict(
          testcase_name="RootNodeBinaryClassificationLabelFn",
          task=classification.RootNodeBinaryClassification(
              "nodes",
              label_fn=label_fn(2)),
          inputs=TEST_GRAPH_TENSOR,
          expected_gt=TEST_GRAPH_TENSOR.remove_features(context=("labels",)),
          expected_labels=(0, 1, 1, 1)),
      dict(
          testcase_name="RootNodeBinaryClassificationReadout",
          task=classification.RootNodeBinaryClassification(
              "nodes",
              label_feature_name="labels"),
          inputs=with_readout(2, TEST_GRAPH_TENSOR),
          expected_gt=with_readout(2, TEST_GRAPH_TENSOR),
          expected_labels=(0, 1, 1, 1)),
      dict(
          testcase_name="RootNodeMulticlassClassificationLabelFn",
          task=classification.RootNodeMulticlassClassification(
              "nodes",
              num_classes=3,
              label_fn=label_fn(3)),
          inputs=TEST_GRAPH_TENSOR,
          expected_gt=TEST_GRAPH_TENSOR.remove_features(context=("labels",)),
          expected_labels=(2, 1, 0, 1)),
      dict(
          testcase_name="RootNodeMulticlassClassificationReadout",
          task=classification.RootNodeMulticlassClassification(
              "nodes",
              num_classes=3,
              label_feature_name="labels"),
          inputs=with_readout(3, TEST_GRAPH_TENSOR),
          expected_gt=with_readout(3, TEST_GRAPH_TENSOR),
          expected_labels=(2, 1, 0, 1)),
  ])
  def test_preprocess(
      self,
      task: interfaces.Task,
      inputs: GraphTensor,
      expected_gt: Sequence[int],
      expected_labels: Sequence[int]):
    xs, ys = task.preprocess(inputs)

    self.assertEqual(xs.spec, expected_gt.spec)  # Assert `GraphTensor` specs.
    self.assertAllEqual(ys, expected_labels)

  @parameterized.named_parameters([
      dict(
          testcase_name="GraphBinaryClassification",
          task=classification.GraphBinaryClassification(
              "nodes",
              label_fn=label_fn(2)),
          gt=TEST_GRAPH_TENSOR,
          expected_loss=tf.keras.losses.BinaryCrossentropy,
          expected_shape=tf.TensorShape((None, 1))),
      dict(
          testcase_name="GraphMulticlassClassification",
          task=classification.GraphMulticlassClassification(
              "nodes",
              num_classes=4,
              label_feature_name="labels"),
          gt=with_readout(4, TEST_GRAPH_TENSOR),
          expected_loss=tf.keras.losses.SparseCategoricalCrossentropy,
          expected_shape=tf.TensorShape((None, 4))),
      dict(
          testcase_name="RootNodeBinaryClassification",
          task=classification.RootNodeBinaryClassification(
              "nodes",
              label_fn=label_fn(2)),
          gt=TEST_GRAPH_TENSOR,
          expected_loss=tf.keras.losses.BinaryCrossentropy,
          expected_shape=tf.TensorShape((None, 1))),
      dict(
          testcase_name="RootNodeMulticlassClassification",
          task=classification.RootNodeMulticlassClassification(
              "nodes",
              num_classes=3,
              label_feature_name="labels"),
          gt=with_readout(3, TEST_GRAPH_TENSOR),
          expected_loss=tf.keras.losses.SparseCategoricalCrossentropy,
          expected_shape=tf.TensorShape((None, 3))),
  ])
  def test_predict(
      self,
      task: interfaces.Task,
      gt: GraphTensor,
      expected_loss: Type[tf.keras.losses.Loss],
      expected_shape: tf.TensorShape):
    # Assert head readout, activation and shape.
    inputs = tf.keras.layers.Input(type_spec=gt.spec)
    model = tf.keras.Model(inputs, task.predict(inputs))
    self.assertLen(model.layers, 3)
    self.assertIsInstance(model.layers[0], tf.keras.layers.InputLayer)
    self.assertIsInstance(
        model.layers[1],
        (tfgnn.keras.layers.ReadoutFirstNode, tfgnn.keras.layers.Pool))
    self.assertIsInstance(model.layers[2], tf.keras.layers.Dense)

    _, _, dense = model.layers
    self.assertEqual(dense.get_config()["activation"], "linear")
    self.assertTrue(expected_shape.is_compatible_with(dense.output_shape))

    # Assert losses.
    losses = task.losses()
    self.assertLen(losses, 1)

    [loss] = losses
    self.assertIsInstance(loss, expected_loss)
    self.assertTrue(loss.get_config()["from_logits"])

  @parameterized.named_parameters([
      dict(
          testcase_name="GraphBinaryClassification",
          task=classification.GraphBinaryClassification(
              "nodes",
              label_fn=label_fn(2)),
          gt=TEST_GRAPH_TENSOR,
          batch_size=1),
      dict(
          testcase_name="GraphMulticlassClassification",
          task=classification.GraphMulticlassClassification(
              "nodes",
              num_classes=4,
              label_feature_name="labels"),
          gt=with_readout(4, TEST_GRAPH_TENSOR),
          batch_size=1),
      dict(
          testcase_name="RootNodeBinaryClassification",
          task=classification.RootNodeBinaryClassification(
              "nodes",
              label_fn=label_fn(2)),
          gt=TEST_GRAPH_TENSOR,
          batch_size=1),
      dict(
          testcase_name="RootNodeMulticlassClassification",
          task=classification.RootNodeMulticlassClassification(
              "nodes",
              num_classes=3,
              label_feature_name="labels"),
          gt=with_readout(3, TEST_GRAPH_TENSOR),
          batch_size=1),
      dict(
          testcase_name="GraphBinaryClassificationBatchSize2",
          task=classification.GraphBinaryClassification(
              "nodes",
              label_fn=label_fn(2)),
          gt=TEST_GRAPH_TENSOR,
          batch_size=2),
      dict(
          testcase_name="GraphMulticlassClassificationBatchSize2",
          task=classification.GraphMulticlassClassification(
              "nodes",
              num_classes=4,
              label_feature_name="labels"),
          gt=with_readout(4, TEST_GRAPH_TENSOR),
          batch_size=2),
      dict(
          testcase_name="RootNodeBinaryClassificationBatchSize2",
          task=classification.RootNodeBinaryClassification(
              "nodes",
              label_fn=label_fn(2)),
          gt=TEST_GRAPH_TENSOR,
          batch_size=2),
      dict(
          testcase_name="RootNodeMulticlassClassificationBatchSize2",
          task=classification.RootNodeMulticlassClassification(
              "nodes",
              num_classes=3,
              label_feature_name="labels"),
          gt=with_readout(3, TEST_GRAPH_TENSOR),
          batch_size=2),
  ])
  def test_fit(
      self,
      task: interfaces.Task,
      gt: GraphTensor,
      batch_size: int):
    inputs = tf.keras.layers.Input(type_spec=gt.spec)
    outputs = task.predict(inputs)
    model = tf.keras.Model(inputs, outputs)

    ds = tf.data.Dataset.from_tensors(gt).repeat().batch(batch_size).take(1)
    ds = ds.map(GraphTensor.merge_batch_to_components).map(task.preprocess)

    model.compile(loss=task.losses(), metrics=task.metrics())
    model.fit(ds)

  def test_per_class_metrics_with_num_classes(self):
    task = classification.GraphMulticlassClassification(
        "nodes",
        num_classes=5,
        per_class_statistics=True,
        label_fn=label_fn(5))
    metric_names = [metric.name for metric in task.metrics()]
    self.assertContainsSubset(
        [
            "precision_for_class_0",
            "precision_for_class_1",
            "precision_for_class_2",
            "precision_for_class_3",
            "precision_for_class_4",
            "recall_for_class_0",
            "recall_for_class_1",
            "recall_for_class_2",
            "recall_for_class_3",
            "recall_for_class_4",
        ],
        metric_names)

  def test_per_class_metrics_with_class_names(self):
    task = classification.RootNodeMulticlassClassification(
        "nodes",
        per_class_statistics=True,
        class_names=["foo", "bar", "baz"],
        label_fn=label_fn(3))
    metric_names = [metric.name for metric in task.metrics()]
    self.assertContainsSubset(
        [
            "precision_for_foo",
            "precision_for_bar",
            "precision_for_baz",
            "recall_for_foo",
            "recall_for_bar",
            "recall_for_baz",
        ],
        metric_names)

  def test_invalid_both_num_classes_and_class_names(self):
    with self.assertRaisesRegex(
        ValueError,
        r"Exactly one of `num_classes` or `class_names` must be specified"):
      classification.GraphMulticlassClassification(
          "nodes",
          num_classes=5,
          class_names=["foo", "bar"],
          label_fn=label_fn(2))

  def test_invalid_no_num_classes_or_class_names(self):
    with self.assertRaisesRegex(
        ValueError,
        r"Exactly one of `num_classes` or `class_names` must be specified"):
      classification.GraphMulticlassClassification(
          "nodes",
          label_fn=label_fn(2))

if __name__ == "__main__":
  tf.test.main()
