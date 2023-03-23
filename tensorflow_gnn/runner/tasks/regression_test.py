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
"""Tests for node regression."""
from __future__ import annotations
from typing import Type

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn.runner import interfaces
from tensorflow_gnn.runner.tasks import regression

GT_SCHEMA = """
context {
features {
  key: "label"
  value {
    dtype: DT_FLOAT
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


def label_fn(inputs: GraphTensor) -> tuple[GraphTensor, Field]:
  y = inputs.context["label"]
  x = inputs.remove_features(context=("label",))
  return x, y


class Regression(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="GraphMeanAbsoluteError",
          task=regression.GraphMeanAbsoluteError(
              "nodes",
              label_fn=label_fn),
          expected_activation="linear",
          expected_loss=tf.keras.losses.MeanAbsoluteError,
          expected_shape=tf.TensorShape((None, 1))),
      dict(
          testcase_name="GraphMeanAbsolutePercentageError",
          task=regression.GraphMeanAbsolutePercentageError(
              "nodes",
              label_fn=label_fn),
          expected_activation="linear",
          expected_loss=tf.keras.losses.MeanAbsolutePercentageError,
          expected_shape=tf.TensorShape((None, 1))),
      dict(
          testcase_name="GraphMeanSquaredError",
          task=regression.GraphMeanSquaredError(
              "nodes",
              label_fn=label_fn),
          expected_activation="linear",
          expected_loss=tf.keras.losses.MeanSquaredError,
          expected_shape=tf.TensorShape((None, 1))),
      dict(
          testcase_name="GraphMeanSquaredLogarithmicError",
          task=regression.GraphMeanSquaredLogarithmicError(
              "nodes",
              units=3,
              label_fn=label_fn),
          expected_activation="linear",
          expected_loss=tf.keras.losses.MeanSquaredLogarithmicError,
          expected_shape=tf.TensorShape((None, 3))),
      dict(
          testcase_name="GraphMeanSquaredLogScaledError",
          task=regression.GraphMeanSquaredLogScaledError(
              "nodes",
              units=2,
              label_fn=label_fn),
          expected_activation="linear",
          expected_loss=regression.MeanSquaredLogScaledError,
          expected_shape=tf.TensorShape((None, 2))),
      dict(
          testcase_name="RootNodeMeanAbsoluteError",
          task=regression.RootNodeMeanAbsoluteError(
              "nodes",
              label_fn=label_fn),
          expected_activation="linear",
          expected_loss=tf.keras.losses.MeanAbsoluteError,
          expected_shape=tf.TensorShape((None, 1))),
      dict(
          testcase_name="RootNodeMeanAbsolutePercentageError",
          task=regression.RootNodeMeanAbsolutePercentageError(
              "nodes",
              label_fn=label_fn),
          expected_activation="linear",
          expected_loss=tf.keras.losses.MeanAbsolutePercentageError,
          expected_shape=tf.TensorShape((None, 1))),
      dict(
          testcase_name="RootNodeMeanSquaredError",
          task=regression.RootNodeMeanSquaredError(
              "nodes",
              label_fn=label_fn),
          expected_activation="linear",
          expected_loss=tf.keras.losses.MeanSquaredError,
          expected_shape=tf.TensorShape((None, 1))),
      dict(
          testcase_name="RootNodeMeanSquaredLogarithmicError",
          task=regression.RootNodeMeanSquaredLogarithmicError(
              "nodes",
              units=3,
              label_fn=label_fn),
          expected_activation="linear",
          expected_loss=tf.keras.losses.MeanSquaredLogarithmicError,
          expected_shape=tf.TensorShape((None, 3))),
      dict(
          testcase_name="RootNodeMeanSquaredLogScaledError",
          task=regression.RootNodeMeanSquaredLogScaledError(
              "nodes",
              units=2,
              label_fn=label_fn),
          expected_activation="linear",
          expected_loss=regression.MeanSquaredLogScaledError,
          expected_shape=tf.TensorShape((None, 2))),
      dict(
          testcase_name="RootNodeMeanAbsoluteLogarithmicError",
          task=regression.RootNodeMeanAbsoluteLogarithmicError(
              node_set_name="nodes",
              units=2,
              label_fn=label_fn,
          ),
          expected_activation="relu",
          expected_loss=regression.MeanAbsoluteLogarithmicErrorLoss,
          expected_shape=tf.TensorShape((None, 2))),
  ])
  def test_predict(
      self,
      task: interfaces.Task,
      expected_activation: str,
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
    self.assertEqual(dense.get_config()["activation"], expected_activation)
    self.assertTrue(expected_shape.is_compatible_with(dense.output_shape))

    # Assert losses.
    losses = task.losses()
    self.assertLen(losses, 1)

    [loss] = losses
    self.assertIsInstance(loss, expected_loss)

  @parameterized.named_parameters([
      dict(
          testcase_name="GraphMeanAbsoluteError",
          task=regression.GraphMeanAbsoluteError(
              "nodes",
              label_fn=label_fn)),
      dict(
          testcase_name="GraphMeanAbsolutePercentageError",
          task=regression.GraphMeanAbsolutePercentageError(
              "nodes",
              label_fn=label_fn)),
      dict(
          testcase_name="GraphMeanSquaredError",
          task=regression.GraphMeanSquaredError(
              "nodes",
              label_fn=label_fn)),
      dict(
          testcase_name="GraphMeanSquaredLogarithmicError",
          task=regression.GraphMeanSquaredLogarithmicError(
              "nodes",
              units=3,
              label_fn=label_fn)),
      dict(
          testcase_name="GraphMeanSquaredLogScaledError",
          task=regression.GraphMeanSquaredLogScaledError(
              "nodes",
              units=2,
              label_fn=label_fn)),
      dict(
          testcase_name="RootNodeMeanAbsoluteError",
          task=regression.RootNodeMeanAbsoluteError(
              "nodes",
              label_fn=label_fn)),
      dict(
          testcase_name="RootNodeMeanAbsolutePercentageError",
          task=regression.RootNodeMeanAbsolutePercentageError(
              "nodes",
              label_fn=label_fn)),
      dict(
          testcase_name="RootNodeMeanSquaredError",
          task=regression.RootNodeMeanSquaredError(
              "nodes",
              label_fn=label_fn)),
      dict(
          testcase_name="RootNodeMeanSquaredLogarithmicError",
          task=regression.RootNodeMeanSquaredLogarithmicError(
              "nodes",
              units=3,
              label_fn=label_fn)),
      dict(
          testcase_name="RootNodeMeanSquaredLogScaledError",
          task=regression.RootNodeMeanSquaredLogScaledError(
              "nodes",
              units=2,
              label_fn=label_fn)),
      dict(
          testcase_name="RootNodeMeanAbsoluteLogarithmicError",
          task=regression.RootNodeMeanAbsoluteLogarithmicError(
              node_set_name="nodes",
              units=2,
              label_fn=label_fn)),
  ])
  def test_fit(
      self,
      task: interfaces.Task):
    inputs = tf.keras.layers.Input(type_spec=GT_SPEC)
    outputs = task.predict(inputs)
    model = tf.keras.Model(inputs, outputs)

    ds = tf.data.Dataset.from_tensors(tfgnn.random_graph_tensor(GT_SPEC))
    ds = ds.repeat().batch(2).map(merge_batch_to_components).take(1)
    ds = ds.map(task.preprocess)

    model.compile(loss=task.losses(), metrics=task.metrics())
    model.fit(ds)

if __name__ == "__main__":
  tf.test.main()
