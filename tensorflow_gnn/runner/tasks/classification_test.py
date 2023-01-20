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
from typing import Sequence

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn.runner.tasks import classification

as_tensor = tf.convert_to_tensor
as_ragged = tf.ragged.constant

SCHEMA = """
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


class Classification(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="GraphBinaryClassification",
          schema=SCHEMA,
          task=classification.GraphBinaryClassification(node_set_name="nodes"),
          y_true=[[1]],
          expected_y_pred=[[-0.4159315]],
          expected_loss=[0.9225837]),
      dict(
          testcase_name="GraphMulticlassClassification",
          schema=SCHEMA,
          task=classification.GraphMulticlassClassification(
              num_classes=4, node_set_name="nodes"),
          y_true=[3],
          expected_y_pred=[[0.35868323, -0.4112632, -0.23154753, 0.20909603]],
          expected_loss=[1.2067872]),
      dict(
          testcase_name="RootNodeBinaryClassification",
          schema=SCHEMA,
          task=classification.RootNodeBinaryClassification(
              node_set_name="nodes"),
          y_true=[[1]],
          expected_y_pred=[[-0.3450081]],
          expected_loss=[0.8804569]),
      dict(
          testcase_name="RootNodeMulticlassClassification",
          schema=SCHEMA,
          task=classification.RootNodeMulticlassClassification(
              num_classes=3, node_set_name="nodes"),
          y_true=[2],
          expected_y_pred=[[-0.4718209, 0.04619305, -0.5249821]],
          expected_loss=[1.3415444]),
  ])
  def test_adapt(self,
                 schema: str,
                 task: classification._Classification,
                 y_true: Sequence[float],
                 expected_y_pred: Sequence[float],
                 expected_loss: Sequence[float]):
    gtspec = tfgnn.create_graph_spec_from_schema_pb(tfgnn.parse_schema(schema))
    inputs = tf.keras.layers.Input(type_spec=gtspec)
    hidden_state = inputs.node_sets["nodes"][tfgnn.HIDDEN_STATE]
    output = inputs.replace_features(
        node_sets={"nodes": {
            tfgnn.HIDDEN_STATE: tf.keras.layers.Dense(16)(hidden_state)
        }})
    model = tf.keras.Model(inputs, output)
    model = task.adapt(model)

    self.assertIs(model.input, inputs)
    self.assertAllEqual(as_tensor(expected_y_pred).shape, model.output.shape)

    y_pred = model(tfgnn.random_graph_tensor(gtspec))
    self.assertAllClose(expected_y_pred, y_pred)

    loss = [loss_fn(y_true, y_pred) for loss_fn in task.losses()]
    self.assertAllClose(expected_loss, loss)

  @parameterized.named_parameters([
      dict(
          testcase_name="GraphBinaryClassification",
          schema=SCHEMA,
          task=classification.GraphBinaryClassification(node_set_name="nodes"),
          batch_size=1),
      dict(
          testcase_name="GraphMulticlassClassification",
          schema=SCHEMA,
          task=classification.GraphMulticlassClassification(
              num_classes=4, node_set_name="nodes"),
          batch_size=1),
      dict(
          testcase_name="RootNodeBinaryClassification",
          schema=SCHEMA,
          task=classification.RootNodeBinaryClassification(
              node_set_name="nodes"),
          batch_size=1),
      dict(
          testcase_name="RootNodeMulticlassClassification",
          schema=SCHEMA,
          task=classification.RootNodeMulticlassClassification(
              num_classes=3, node_set_name="nodes"),
          batch_size=1),
      dict(
          testcase_name="GraphBinaryClassificationBatchSize2",
          schema=SCHEMA,
          task=classification.GraphBinaryClassification(node_set_name="nodes"),
          batch_size=2),
      dict(
          testcase_name="GraphMulticlassClassificationBatchSize2",
          schema=SCHEMA,
          task=classification.GraphMulticlassClassification(
              num_classes=4, node_set_name="nodes"),
          batch_size=2),
      dict(
          testcase_name="RootNodeBinaryClassificationBatchSize2",
          schema=SCHEMA,
          task=classification.RootNodeBinaryClassification(
              node_set_name="nodes"),
          batch_size=2),
      dict(
          testcase_name="RootNodeMulticlassClassificationBatchSize2",
          schema=SCHEMA,
          task=classification.RootNodeMulticlassClassification(
              num_classes=3, node_set_name="nodes"),
          batch_size=2),
  ])
  def test_fit(self, schema: str, task: classification._Classification,
               batch_size: int):
    gtspec = tfgnn.create_graph_spec_from_schema_pb(tfgnn.parse_schema(schema))
    inputs = tf.keras.layers.Input(type_spec=gtspec)
    hidden_state = inputs.node_sets["nodes"][tfgnn.HIDDEN_STATE]
    output = inputs.replace_features(
        node_sets={"nodes": {
            tfgnn.HIDDEN_STATE: tf.keras.layers.Dense(16)(hidden_state)
        }})
    model = tf.keras.Model(inputs, output)
    model = task.adapt(model)

    examples = tf.data.Dataset.from_tensors(tfgnn.random_graph_tensor(gtspec))
    # In the batched case, prepare exactly 1 batch by copying the graph tensor.
    examples = examples.repeat(batch_size).batch(batch_size).map(
        tfgnn.GraphTensor.merge_batch_to_components)
    labels = tf.data.Dataset.from_tensors([1.] * batch_size)

    dataset = tf.data.Dataset.zip((examples.repeat(2), labels.repeat(2)))

    model.compile(loss=task.losses(), metrics=task.metrics(), run_eagerly=True)
    model.fit(dataset)

  def test_per_class_metrics_with_num_classes(self):
    task = classification.GraphMulticlassClassification(
        num_classes=5, node_set_name="nodes", per_class_statistics=True)
    metric_names = [metric.name for metric in task.metrics()]
    self.assertContainsSubset([
        "precision_for_class_0", "precision_for_class_1",
        "precision_for_class_2", "precision_for_class_3",
        "precision_for_class_4", "recall_for_class_0", "recall_for_class_1",
        "recall_for_class_2", "recall_for_class_3", "recall_for_class_4"
    ], metric_names)

  def test_per_class_metrics_with_class_names(self):
    task = classification.RootNodeMulticlassClassification(
        node_set_name="nodes",
        per_class_statistics=True,
        class_names=["foo", "bar", "baz"])
    metric_names = [metric.name for metric in task.metrics()]
    self.assertContainsSubset([
        "precision_for_foo", "precision_for_bar", "precision_for_baz",
        "recall_for_foo", "recall_for_bar", "recall_for_baz"
    ], metric_names)

  def test_invalid_both_num_classes_and_class_names(self):
    with self.assertRaisesRegex(
        ValueError,
        r"Exactly one of `num_classes` or `class_names` must be specified"):
      classification.GraphMulticlassClassification(
          num_classes=5, node_set_name="nodes", class_names=["foo", "bar"])

  def test_invalid_no_num_classes_or_class_names(self):
    with self.assertRaisesRegex(
        ValueError,
        r"Exactly one of `num_classes` or `class_names` must be specified"):
      classification.GraphMulticlassClassification(node_set_name="nodes")

if __name__ == "__main__":
  tf.test.main()
