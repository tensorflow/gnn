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
import functools

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn import runner
from tensorflow_gnn.models.contrastive_losses.deep_graph_infomax import tasks

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


def task() -> runner.Task:
  return tasks.DeepGraphInfomaxTask("node", seed=8191)


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


class DeepGraphInfomaxTaskTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="NoGraphTensorInput",
          model=dnn(),
          expected_error=r"Expected a `GraphTensor` input \(got .*\)"
      ),
      dict(
          testcase_name="NoGraphTensorOutput",
          model=gdnn(),
          expected_error=r"Expected a `GraphTensor` output \(got .*\)"
      ),
  ])
  def test_value_error(
      self,
      model: tf.keras.Model,
      expected_error: str):
    """Verifies invalid input raises `ValueError`."""
    with self.assertRaisesRegex(ValueError, expected_error):
      _ = tasks.DeepGraphInfomaxTask("node").adapt(model)

  def test_adapt(self):
    """Verifies an adapted model's output shape."""
    adapted = task().adapt(gnn())
    gt = random_graph_tensor()

    # Clean and corrupted logits (i.e., shape (1, 2)).
    logits = adapted(gt)
    self.assertEqual(logits.shape, (1, 2))

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

  def test_fit(self):
    """Verifies an adapted model's fit."""
    dgi = task()

    ds = tf.data.Dataset.from_tensors(random_graph_tensor()).repeat()
    ds = ds.batch(4).map(tfgnn.GraphTensor.merge_batch_to_components)
    ds = ds.map(dgi.preprocess)

    model = dgi.adapt(gnn())
    model.compile(loss=dgi.losses(), metrics=dgi.metrics())
    model.fit(ds.take(1))

  def test_preprocess(self):
    """Verifies psuedo-labels."""
    expected = random_graph_tensor()
    actual, pseudolabels = task().preprocess(expected)

    self.assertEqual(actual, expected)
    self.assertAllEqual(pseudolabels, ((1., 0.),))


if __name__ == "__main__":
  tf.test.main()
