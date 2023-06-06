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
"""Tests for attribution."""
import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn.runner import interfaces
from tensorflow_gnn.runner.utils import attribution

IntegratedGradientsExporter = attribution.IntegratedGradientsExporter


class AttributionTest(tf.test.TestCase):

  gt = tfgnn.GraphTensor.from_pieces(
      context=tfgnn.Context.from_fields(features={
          "h": tf.convert_to_tensor((.514, .433)),
          # An integer feature with uniform values.
          "labels": tf.convert_to_tensor((0,)),
      }),
      node_sets={
          "node":
              tfgnn.NodeSet.from_fields(
                  features={
                      "h": tf.convert_to_tensor((.8191, .9474, .1634)),
                  },
                  sizes=tf.constant((3,)),
              ),
      },
      edge_sets={
          "edge":
              tfgnn.EdgeSet.from_fields(
                  features={"weight": tf.convert_to_tensor((.153, .9))},
                  sizes=tf.constant((2,)),
                  adjacency=tfgnn.Adjacency.from_indices(
                      source=("node", (0, 1)), target=("node", (1, 2))),
              ),
      },
  )

  def test_counterfactual_random(self):
    counterfactual = attribution.counterfactual(self.gt, random=True, seed=8191)

    self.assertAllEqual(
        counterfactual.context.features["h"],
        tf.convert_to_tensor((0.49280962, 0.466383)))
    self.assertAllEqual(
        counterfactual.context.features["labels"],
        tf.convert_to_tensor((0,)))

    self.assertAllEqual(
        counterfactual.edge_sets["edge"].features["weight"],
        tf.convert_to_tensor((0.8038801, 0.45028666)))

    self.assertAllEqual(
        counterfactual.node_sets["node"].features["h"],
        tf.convert_to_tensor((0.6295455, 0.37102205, 0.51270497)))

  def test_counterfactual_zeros(self):
    counterfactual = attribution.counterfactual(self.gt, random=False)

    self.assertAllEqual(
        counterfactual.context.features["h"],
        tf.convert_to_tensor((0, 0)))
    self.assertAllEqual(
        counterfactual.context.features["labels"],
        tf.convert_to_tensor((0,)))

    self.assertAllEqual(
        counterfactual.edge_sets["edge"].features["weight"],
        tf.convert_to_tensor((0, 0)))

    self.assertAllEqual(
        counterfactual.node_sets["node"].features["h"],
        tf.convert_to_tensor((0, 0, 0)))

  def test_subtract_graph_features(self):
    deltas = attribution.subtract_graph_features(
        self.gt,
        self.gt.replace_features(
            context={
                "h": tf.convert_to_tensor((.4, .8)),
                "labels": tf.convert_to_tensor((1,))
            },
            node_sets={
                "node": {
                    "h": tf.convert_to_tensor((.1, .2, .3))
                }
            },
            edge_sets={
                "edge": {
                    "weight": tf.convert_to_tensor((.2, .1))
                }
            }))

    self.assertAllClose(
        deltas.context.features["h"],
        tf.convert_to_tensor((.514 - .4, .433 - .8)))
    self.assertAllClose(
        deltas.context.features["labels"],
        tf.convert_to_tensor((0 - 1,)))

    self.assertAllClose(
        deltas.edge_sets["edge"].features["weight"],
        tf.convert_to_tensor((.153 - .2, .9 - .1)))

    self.assertAllClose(
        deltas.node_sets["node"].features["h"],
        tf.convert_to_tensor((.8191 - .1, .9474 - .2, .1634 - .3),))

  def test_interpolate(self):
    counterfactual = attribution.counterfactual(self.gt, random=True, seed=8191)
    interpolations = attribution.interpolate_graph_features(
        self.gt,
        counterfactual,
        steps=4)

    self.assertLen(interpolations, 4)

    # Interpolation 0
    self.assertAllEqual(
        interpolations[0].context.features["h"],
        tf.convert_to_tensor((0.49280962, 0.466383)))

    self.assertAllEqual(
        interpolations[0].edge_sets["edge"].features["weight"],
        tf.convert_to_tensor((0.8038801, 0.45028666)))

    self.assertAllClose(
        interpolations[0].node_sets["node"].features["h"],
        tf.convert_to_tensor((0.6295455, 0.37102205, 0.51270497)))

    # Interpolation 1
    self.assertAllEqual(
        interpolations[1].context.features["h"],
        tf.convert_to_tensor((0.49280962 + (.514 - 0.49280962) * 1 / 3,
                              0.466383 + (.433 - 0.466383) * 1 / 3)))

    self.assertAllClose(
        interpolations[1].edge_sets["edge"].features["weight"],
        tf.convert_to_tensor((0.8038801 + (.153 - 0.8038801) * 1 / 3,
                              0.45028666 + (.9 - 0.45028666) * 1 / 3)))

    self.assertAllClose(
        interpolations[1].node_sets["node"].features["h"],
        tf.convert_to_tensor((0.6295455 + (.8191 - 0.6295455) * 1 / 3,
                              0.37102205 + (.9474 - 0.37102205) * 1 / 3,
                              0.51270497 + (.1634 - 0.51270497) * 1 / 3)))

    # Interpolation 2
    self.assertAllEqual(
        interpolations[2].context.features["h"],
        tf.convert_to_tensor((0.49280962 + (.514 - 0.49280962) * 2 / 3,
                              0.466383 + (.433 - 0.466383) * 2 / 3)))

    self.assertAllClose(
        interpolations[2].edge_sets["edge"].features["weight"],
        tf.convert_to_tensor((0.8038801 + (.153 - 0.8038801) * 2 / 3,
                              0.45028666 + (.9 - 0.45028666) * 2 / 3)))

    self.assertAllClose(
        interpolations[2].node_sets["node"].features["h"],
        tf.convert_to_tensor((0.6295455 + (.8191 - 0.6295455) * 2 / 3,
                              0.37102205 + (.9474 - 0.37102205) * 2 / 3,
                              0.51270497 + (.1634 - 0.51270497) * 2 / 3)))

    # Interpolation 3
    self.assertAllEqual(
        interpolations[3].context.features["h"],
        tf.convert_to_tensor((.514, .433)))

    self.assertAllEqual(
        interpolations[3].edge_sets["edge"].features["weight"],
        tf.convert_to_tensor((.153, .9)))

    self.assertAllEqual(
        interpolations[3].node_sets["node"].features["h"],
        tf.convert_to_tensor((.8191, .9474, .1634)))

  def test_sum_graph_features(self):
    summation = attribution.sum_graph_features((self.gt,) * 4)

    self.assertAllEqual(
        summation.context.features["h"],
        tf.convert_to_tensor((.514 * 4, .433 * 4)))

    self.assertAllEqual(
        summation.edge_sets["edge"].features["weight"],
        tf.convert_to_tensor((.153 * 4, .9 * 4)))

    self.assertAllEqual(
        summation.node_sets["node"].features["h"],
        tf.convert_to_tensor((.8191 * 4, .9474 * 4, .1634 * 4)))

  def test_integrated_gradients_exporter(self):
    # Preprocess model
    examples = tf.keras.Input(shape=(), dtype=tf.string, name="examples")
    parsed = tfgnn.keras.layers.ParseExample(self.gt.spec)(examples)
    parsed = parsed.merge_batch_to_components()
    label = tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name="node",
        feature_name="h")(parsed)
    label = tf.random.uniform(
        tf.shape(label),
        minval=0,
        maxval=9,
        dtype=tf.int32)  # 10 classes

    preprocess_model = tf.keras.Model(examples, ((parsed,), label))

    # Model
    inputs = graph = tf.keras.Input(type_spec=parsed.spec)
    values = tfgnn.broadcast_node_to_edges(
        graph,
        "edge",
        tfgnn.TARGET,
        feature_name="h")
    weights = graph.edge_sets["edge"].features["weight"]
    messages = (values[:, None], weights[:, None])
    messages = tf.keras.layers.Concatenate()(messages)
    messages = tf.keras.layers.Dense(16)(messages)
    pooled = tfgnn.pool_edges_to_node(
        graph,
        "edge",
        tfgnn.SOURCE,
        reduce_type="sum",
        feature_value=messages)
    h_old = graph.node_sets["node"].features["h"]
    h_next = tf.keras.layers.Concatenate()((pooled, h_old[:, None]))
    h_next = tf.keras.layers.Dense(16)(h_next)
    graph = graph.replace_features(node_sets={"node": {"h": h_next}})
    activations = tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name="node",
        feature_name="h")(graph)
    logits = tf.keras.layers.Dense(10)(activations)  # 10 classes

    model = tf.keras.Model(inputs, logits)

    # Dataset
    example = tfgnn.write_example(self.gt)
    ds = tf.data.Dataset.from_tensors(example.SerializeToString()).repeat(4)
    ds = ds.batch(1)

    # Compile and fit
    model.compile("adam", "sparse_categorical_crossentropy")
    model.fit(ds.map(preprocess_model), epochs=16)

    # Export
    export_dir = self.create_tempdir()
    exporter = attribution.IntegratedGradientsExporter("output", steps=3)

    run_result = interfaces.RunResult(preprocess_model, None, model)
    exporter.save(run_result, export_dir)

    saved_model = tf.saved_model.load(export_dir)

    # Integrated gradients
    kwargs = {
        "examples": next(iter(ds)),
    }
    outputs = saved_model.signatures["integrated_gradients"](**kwargs)
    gt = outputs["output"]

    # The above GNN passes a single message over the only edge type before
    # collecting a seed node for activations. The above graph is a line:
    # seed --weight 0--> node 1 --weight 1--> node 2.
    #
    # Information from weight 1 and node 2 never reaches the activations: they
    # should see no integrated gradients.
    self.assertAllClose(
        gt.node_sets["node"].features["h"],
        tf.convert_to_tensor((2.1929948, -1.0116194, 0.)), 1e-04, 1e-04)

    self.assertAllClose(
        gt.edge_sets["edge"].features["weight"],
        tf.convert_to_tensor((3.4002826, 0.)), 1e-04, 1e-04)


if __name__ == "__main__":
  tf.test.main()
