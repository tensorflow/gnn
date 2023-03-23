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
from typing import Callable, Type

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn.runner import interfaces
from tensorflow_gnn.runner.tasks import classification

GT_SCHEMA = """
context {
features {
  key: "label"
  value {
    dtype: DT_INT64
      shape { dim { size: 1 } }
    }
  }
}
node_sets {
  key: "nodes"
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
  key: "edges"
  value {
    source: "nodes"
    target: "nodes"
  }
}
""" % tfgnn.HIDDEN_STATE

GT_SPEC = tfgnn.create_graph_spec_from_schema_pb(tfgnn.parse_schema(GT_SCHEMA))

as_tensor = tf.convert_to_tensor
as_ragged = tf.ragged.constant
merge_batch_to_components = tfgnn.GraphTensor.merge_batch_to_components

GraphTensor = tfgnn.GraphTensor
Field = tfgnn.Field


def label_fn(num_labels: int) -> Callable[..., tuple[GraphTensor, Field]]:
  def fn(inputs):
    y = inputs.context["label"]
    x = inputs.remove_features(context=("label",))
    return x, y % num_labels
  return fn


class Classification(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="GraphBinaryClassification",
          task=classification.GraphBinaryClassification(
              "nodes",
              label_fn=label_fn(2)),
          expected_loss=tf.keras.losses.BinaryCrossentropy,
          expected_shape=tf.TensorShape((None, 1))),
      dict(
          testcase_name="GraphMulticlassClassification",
          task=classification.GraphMulticlassClassification(
              "nodes",
              num_classes=4,
              label_fn=label_fn(4)),
          expected_loss=tf.keras.losses.SparseCategoricalCrossentropy,
          expected_shape=tf.TensorShape((None, 4))),
      dict(
          testcase_name="RootNodeBinaryClassification",
          task=classification.RootNodeBinaryClassification(
              "nodes",
              label_fn=label_fn(2)),
          expected_loss=tf.keras.losses.BinaryCrossentropy,
          expected_shape=tf.TensorShape((None, 1))),
      dict(
          testcase_name="RootNodeMulticlassClassification",
          task=classification.RootNodeMulticlassClassification(
              "nodes",
              num_classes=3,
              label_fn=label_fn(3)),
          expected_loss=tf.keras.losses.SparseCategoricalCrossentropy,
          expected_shape=tf.TensorShape((None, 3))),
  ])
  def test_predict(
      self,
      task: interfaces.Task,
      expected_loss: Type[tf.keras.losses.Loss],
      expected_shape: tf.TensorShape):
    # Assert head readout, activation and shape.
    inputs = tf.keras.layers.Input(type_spec=GT_SPEC)
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
          batch_size=1),
      dict(
          testcase_name="GraphMulticlassClassification",
          task=classification.GraphMulticlassClassification(
              "nodes",
              num_classes=4,
              label_fn=label_fn(4)),
          batch_size=1),
      dict(
          testcase_name="RootNodeBinaryClassification",
          task=classification.RootNodeBinaryClassification(
              "nodes",
              label_fn=label_fn(2)),
          batch_size=1),
      dict(
          testcase_name="RootNodeMulticlassClassification",
          task=classification.RootNodeMulticlassClassification(
              "nodes",
              num_classes=3,
              label_fn=label_fn(3)),
          batch_size=1),
      dict(
          testcase_name="GraphBinaryClassificationBatchSize2",
          task=classification.GraphBinaryClassification(
              "nodes",
              label_fn=label_fn(2)),
          batch_size=2),
      dict(
          testcase_name="GraphMulticlassClassificationBatchSize2",
          task=classification.GraphMulticlassClassification(
              "nodes",
              num_classes=4,
              label_fn=label_fn(4)),
          batch_size=2),
      dict(
          testcase_name="RootNodeBinaryClassificationBatchSize2",
          task=classification.RootNodeBinaryClassification(
              "nodes",
              label_fn=label_fn(2)),
          batch_size=2),
      dict(
          testcase_name="RootNodeMulticlassClassificationBatchSize2",
          task=classification.RootNodeMulticlassClassification(
              "nodes",
              num_classes=3,
              label_fn=label_fn(3)),
          batch_size=2),
  ])
  def test_fit(
      self,
      task: interfaces.Task,
      batch_size: int):
    inputs = tf.keras.layers.Input(type_spec=GT_SPEC)
    outputs = task.predict(inputs)
    model = tf.keras.Model(inputs, outputs)

    ds = tf.data.Dataset.from_tensors(tfgnn.random_graph_tensor(GT_SPEC))
    ds = ds.repeat().batch(batch_size).map(merge_batch_to_components).take(1)
    ds = ds.map(task.preprocess)

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
