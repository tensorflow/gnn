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
import os

from absl.testing import parameterized
import tensorflow as tf
import tensorflow.__internal__.distribute as tfdistribute
import tensorflow.__internal__.test as tftest
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


def _all_eager_distributed_strategy_combinations():
  strategies = [
      # MirroredStrategy
      tfdistribute.combinations.mirrored_strategy_with_gpu_and_cpu,
      tfdistribute.combinations.mirrored_strategy_with_one_cpu,
      tfdistribute.combinations.mirrored_strategy_with_one_gpu,
      """    # MultiWorkerMirroredStrategy
      tfdistribute.combinations.multi_worker_mirrored_2x1_cpu,
      tfdistribute.combinations.multi_worker_mirrored_2x1_gpu,
      # TPUStrategy
      tfdistribute.combinations.tpu_strategy,
      tfdistribute.combinations.tpu_strategy_one_core,
      tfdistribute.combinations.tpu_strategy_packed_var,
      # ParameterServerStrategy
      tfdistribute.combinations.parameter_server_strategy_3worker_2ps_cpu,
      tfdistribute.combinations.parameter_server_strategy_3worker_2ps_1gpu,
      tfdistribute.combinations.parameter_server_strategy_1worker_2ps_cpu,
      tfdistribute.combinations.parameter_server_strategy_1worker_2ps_1gpu, """
  ]
  return tftest.combinations.combine(distribution=strategies)


class DeepGraphInfomaxTest(tf.test.TestCase, parameterized.TestCase):

  global_batch_size = 2
  gtspec = tfgnn.create_graph_spec_from_schema_pb(tfgnn.parse_schema(SCHEMA))
  seed = 8191
  task = dgi.DeepGraphInfomax(
      "node", global_batch_size=global_batch_size, seed=seed)

  def get_graph_tensor(self):
    gt = tfgnn.GraphTensor.from_pieces(
        node_sets={
            "node":
                tfgnn.NodeSet.from_fields(
                    features={
                        tfgnn.HIDDEN_STATE:
                            tf.convert_to_tensor([[1., 2., 3., 4.],
                                                  [11., 11., 11., 11.],
                                                  [19., 19., 19., 19.]])
                    },
                    sizes=tf.convert_to_tensor([3])),
        },
        edge_sets={
            "edge":
                tfgnn.EdgeSet.from_fields(
                    sizes=tf.convert_to_tensor([2]),
                    adjacency=tfgnn.Adjacency.from_indices(
                        ("node", tf.convert_to_tensor([0, 1], dtype=tf.int32)),
                        ("node", tf.convert_to_tensor([2, 0], dtype=tf.int32)),
                    )),
        })
    return gt

  def build_model(self):
    graph = inputs = tf.keras.layers.Input(type_spec=self.gtspec)

    for _ in range(2):  # Message pass twice
      values = tfgnn.broadcast_node_to_edges(
          graph,
          "edge",
          tfgnn.TARGET,
          feature_name=tfgnn.HIDDEN_STATE)
      messages = tf.keras.layers.Dense(
          8, kernel_initializer=tf.constant_initializer(1.))(
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
          4, kernel_initializer=tf.constant_initializer(1.))(
              h_next)

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

    self.assertAllClose(actual, expected, rtol=1e-04, atol=1e-04)

  def test_fit(self):
    ds = tf.data.Dataset.from_tensors(self.get_graph_tensor()).repeat(8)
    ds = ds.batch(self.global_batch_size).map(
        tfgnn.GraphTensor.merge_batch_to_components)

    tf.random.set_seed(self.seed)
    model = self.task.adapt(self.build_model())
    model.compile()

    def get_loss():
      tf.random.set_seed(self.seed)
      values = model.evaluate(ds)
      return dict(zip(model.metrics_names, values))["loss"]

    before = get_loss()
    model.fit(ds)
    after = get_loss()
    self.assertAllClose(before, 21754138.0, rtol=1e-04, atol=1e-04)
    self.assertAllClose(after, 16268301.0, rtol=1e-04, atol=1e-04)

  @tfdistribute.combinations.generate(
      tftest.combinations.combine(distribution=[
          tfdistribute.combinations.mirrored_strategy_with_one_gpu,
          tfdistribute.combinations.multi_worker_mirrored_2x1_gpu,
      ]))
  def test_distributed(self, distribution):
    gt = self.get_graph_tensor()

    def dataset_fn(input_context=None, gt=gt):
      ds = tf.data.Dataset.from_tensors(gt).repeat(8)
      if input_context:
        batch_size = input_context.get_per_replica_batch_size(
            self.global_batch_size)
      else:
        batch_size = self.global_batch_size
      ds = ds.batch(batch_size).map(tfgnn.GraphTensor.merge_batch_to_components)
      return ds

    with distribution.scope():
      tf.random.set_seed(self.seed)
      model = self.task.adapt(self.build_model())
      model.compile()

    def get_loss():
      tf.random.set_seed(self.seed)
      values = model.evaluate(
          distribution.distribute_datasets_from_function(dataset_fn), steps=4)
      return dict(zip(model.metrics_names, values))["loss"]

    before = get_loss()
    model.fit(
        distribution.distribute_datasets_from_function(dataset_fn),
        steps_per_epoch=4)
    after = get_loss()
    self.assertAllClose(before, 21754138.0, rtol=1e-04, atol=1e-04)
    self.assertAllClose(after, 16268301.0, rtol=1e-04, atol=1e-04)

    export_dir = os.path.join(self.get_temp_dir(), "dropout-model")
    model.save(export_dir)

  def test_protocol(self):
    self.assertIsInstance(dgi.DeepGraphInfomax, orchestration.Task)


if __name__ == "__main__":
  tfdistribute.multi_process_runner.test_main()
