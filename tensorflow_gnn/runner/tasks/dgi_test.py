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
"""Tests for dgi."""
import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn.runner import orchestration
from tensorflow_gnn.runner.tasks import dgi

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


class DeepGraphInfomaxTest(tf.test.TestCase):

  gtspec = tfgnn.create_graph_spec_from_schema_pb(tfgnn.parse_schema(SCHEMA))
  task = dgi.DeepGraphInfomax("node", seed=8191)

  def build_model(self):
    graph = inputs = tf.keras.layers.Input(type_spec=self.gtspec)

    for _ in range(2):  # Message pass twice
      values = tfgnn.broadcast_node_to_edges(
          graph,
          "edge",
          tfgnn.TARGET,
          feature_name=tfgnn.HIDDEN_STATE)
      messages = tf.keras.layers.Dense(16)(values)

      pooled = tfgnn.pool_edges_to_node(
          graph,
          "edge",
          tfgnn.SOURCE,
          reduce_type="sum",
          feature_value=messages)
      h_old = graph.node_sets["node"].features[tfgnn.HIDDEN_STATE]

      h_next = tf.keras.layers.Concatenate()((pooled, h_old))
      h_next = tf.keras.layers.Dense(8)(h_next)

      graph = graph.replace_features(
          node_sets={"node": {
              tfgnn.HIDDEN_STATE: h_next
          }})

    return tf.keras.Model(inputs, graph)

  def test_adapt(self):
    model = self.build_model()
    adapted = self.task.adapt(model)

    gt = tfgnn.random_graph_tensor(self.gtspec)
    # Output should be y_clean (i.e., root node representation)
    expected = tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name="node",
        feature_name=tfgnn.HIDDEN_STATE)(model(gt))
    actual = adapted(gt)

    self.assertAllClose(actual, expected)

  def test_fit(self):
    gt = tfgnn.random_graph_tensor(self.gtspec)
    ds = tf.data.Dataset.from_tensors(gt).repeat(8)
    ds = ds.batch(2).map(tfgnn.GraphTensor.merge_batch_to_components)

    model = self.task.adapt(self.build_model())
    model.compile()

    def get_loss():
      values = model.evaluate(ds)
      return dict(zip(model.metrics_names, values))["loss"]

    before = get_loss()
    model.fit(ds)
    after = get_loss()

    self.assertAllClose(before, 250.42036, rtol=1e-04, atol=1e-04)
    self.assertAllClose(after, 13.18533, rtol=1e-04, atol=1e-04)

  def test_protocol(self):
    self.assertIsInstance(dgi.DeepGraphInfomax, orchestration.Task)


if __name__ == "__main__":
  tf.test.main()
