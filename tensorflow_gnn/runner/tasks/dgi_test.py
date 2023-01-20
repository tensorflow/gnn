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
from absl.testing import parameterized
import tensorflow as tf
import tensorflow.__internal__.distribute as tfdistribute
import tensorflow.__internal__.test as tftest
import tensorflow_gnn as tfgnn

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


class DeepGraphInfomaxTest(tf.test.TestCase, parameterized.TestCase):

  gtspec = tfgnn.create_graph_spec_from_schema_pb(tfgnn.parse_schema(SCHEMA))
  task = dgi.DeepGraphInfomax("node", seed=8191)

  def build_model(self):
    graph = inputs = tf.keras.layers.Input(type_spec=self.gtspec)

    for _ in range(2):  # Message pass twice
      values = tfgnn.broadcast_node_to_edges(
          graph, "edge", tfgnn.TARGET, feature_name=tfgnn.HIDDEN_STATE)
      messages = tf.keras.layers.Dense(
          16, kernel_initializer=tf.constant_initializer(1.))(
              values)

      pooled = tfgnn.pool_edges_to_node(
          graph,
          "edge",
          tfgnn.SOURCE,
          reduce_type="sum",
          feature_value=messages)
      h_old = graph.node_sets["node"].features[tfgnn.HIDDEN_STATE]

      h_next = tf.keras.layers.Concatenate()((pooled, h_old))
      h_next = tf.keras.layers.Dense(
          8, kernel_initializer=tf.constant_initializer(1.))(
              h_next)

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

  @tfdistribute.combinations.generate(
      tftest.combinations.combine(distribution=[
          tfdistribute.combinations.mirrored_strategy_with_one_gpu,
          tfdistribute.combinations.multi_worker_mirrored_2x1_gpu,
      ]))
  def test_distributed(self, distribution):
    gt = tfgnn.GraphTensor.from_pieces(
        node_sets={
            "node":
                tfgnn.NodeSet.from_fields(
                    features={
                        tfgnn.HIDDEN_STATE:
                            tf.convert_to_tensor([[0.1, 0.2, 0.3, 0.4],
                                                  [0.11, 0.11, 0.11, 0.11],
                                                  [0.19, 0.19, 0.19, 0.19]])
                    },
                    sizes=tf.convert_to_tensor([3])),
        },
        edge_sets={
            "edge":
                tfgnn.EdgeSet.from_fields(
                    sizes=tf.convert_to_tensor([3]),
                    adjacency=tfgnn.Adjacency.from_indices(
                        ("node", tf.convert_to_tensor([0, 1, 1],
                                                      dtype=tf.int32)),
                        ("node", tf.convert_to_tensor([0, 0, 2],
                                                      dtype=tf.int32)),
                    )),
        })

    def dataset_fn(input_context=None, gt=gt):
      ds = tf.data.Dataset.from_tensors(gt).repeat(8)
      if input_context:
        ds = ds.shard(input_context.num_input_pipelines,
                      input_context.input_pipeline_id)
        batch_size = input_context.get_per_replica_batch_size(4)
      else:
        batch_size = 4
      ds = ds.batch(batch_size).map(tfgnn.GraphTensor.merge_batch_to_components)
      ds = ds.map(self.task.preprocess)
      return ds

    distributed_ds = distribution.distribute_datasets_from_function(dataset_fn)

    with distribution.scope():
      tf.random.set_seed(8191)
      model = self.task.adapt(self.build_model())
      model.compile(loss=self.task.losses(), metrics=self.task.metrics())

    def get_loss():
      tf.random.set_seed(8191)
      values = model.evaluate(distributed_ds, steps=2)
      return dict(zip(model.metrics_names, values))["loss"]

    before = get_loss()
    model.fit(distributed_ds, steps_per_epoch=2)
    after = get_loss()
    tf.print(f"before: {before}, after: {after}")
    self.assertAllClose(before, 1576777.75, rtol=1e-04, atol=1e-04)
    self.assertAllClose(after, 535825.25, rtol=1e-04, atol=1e-04)

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
      self.assertAllEqual(x.node_sets["node"].features[tfgnn.HIDDEN_STATE],
                          gt.node_sets["node"].features[tfgnn.HIDDEN_STATE])
      self.assertAllEqual(y, tf.constant([[1, 0]], dtype=tf.int32))


if __name__ == "__main__":
  tfdistribute.multi_process_runner.test_main()
