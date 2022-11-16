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

    return tf.keras.Model(inputs=inputs, outputs=graph)

  def test_head(self):
    model = self.task.adapt(self.build_model())
    logits = model(tfgnn.random_graph_tensor(self.gtspec))

    # Model output should have inner dim == 2: one logit for positives and
    # one logit for negatives
    self.assertAllEqual(logits.shape, (1, 2))

  def test_fit(self):
    gt = tfgnn.random_graph_tensor(self.gtspec)
    xs = tf.data.Dataset.from_tensors(gt).repeat(8)
    ys = tf.data.Dataset.from_tensors([1., 0.]).repeat()
    ds = tf.data.Dataset.zip((xs, ys))
    ds = ds.batch(4).map(lambda x, y: (x.merge_batch_to_components(), y))

    model = self.task.adapt(self.build_model())
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    model.fit(ds)

  def test_embeddings_submodule(self):
    model = self.task.adapt(self.build_model())
    dgi_embeddings_model = [
        m for m in model.submodules if m.name == "DeepGraphInfomaxEmbeddings"
    ]
    self.assertLen(dgi_embeddings_model, 1)
    embeddings = dgi_embeddings_model[0](tfgnn.random_graph_tensor(self.gtspec))
    self.assertAllEqual(embeddings.shape, (1, 8))

  def test_preprocessor(self):
    gt = tfgnn.random_graph_tensor(self.gtspec)
    ds = tf.data.Dataset.from_tensors(gt).repeat(8).map(self.task.preprocess)

    for x, y in ds:
      self.assertAllEqual(
          x.node_sets["node"].features[tfgnn.HIDDEN_STATE],
          gt.node_sets["node"].features[tfgnn.HIDDEN_STATE])
      self.assertAllEqual(
          y,
          tf.constant([[1, 0]], dtype=tf.int32))

  def test_protocol(self):
    self.assertIsInstance(dgi.DeepGraphInfomax, orchestration.Task)


if __name__ == "__main__":
  tf.test.main()
