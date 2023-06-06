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
"""Tests for graph_sage."""

import enum
import math
import os

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn.models.graph_sage import layers as graph_sage

_FEATURE_NAME = "f"


def _get_test_graph():
  graph = tfgnn.GraphTensor.from_pieces(
      context=tfgnn.Context.from_fields(
          features={_FEATURE_NAME: tf.constant([0., 0.])}),
      node_sets={
          "topic":
              tfgnn.NodeSet.from_fields(
                  features={_FEATURE_NAME: tf.constant([[1.] * 30, [0.] * 30])},
                  sizes=tf.constant([1, 1])),
          "paper":
              tfgnn.NodeSet.from_fields(
                  features={
                      _FEATURE_NAME: tf.constant([[1., 2., 3.], [2., 1., 3.]])
                  },
                  sizes=tf.constant([1, 1])),
          "author":
              tfgnn.NodeSet.from_fields(
                  features={
                      _FEATURE_NAME: tf.constant([[1., 0.], [0., 2.]] * 2)
                  },
                  sizes=tf.constant([2, 2])),
          "institution":
              tfgnn.NodeSet.from_fields(
                  features={
                      _FEATURE_NAME:
                          tf.constant([[1., 2., 3., 0.], [2., 1., 3., 0.]])
                  },
                  sizes=tf.constant([1, 1])),
      },
      edge_sets={
          "written":
              tfgnn.EdgeSet.from_fields(
                  features={},
                  sizes=tf.constant([2, 1]),
                  adjacency=tfgnn.Adjacency.from_indices(
                      ("paper", tf.constant([0, 0, 1])),
                      ("author", tf.constant([1, 0, 3])),
                  )),
          "correlates":
              tfgnn.EdgeSet.from_fields(
                  features={},
                  sizes=tf.constant([1, 0]),
                  adjacency=tfgnn.Adjacency.from_indices(
                      ("topic", tf.constant([0])),
                      ("topic", tf.constant([1])),
                  )),
          "affiliated_with":
              tfgnn.EdgeSet.from_fields(
                  features={},
                  sizes=tf.constant([2, 2]),
                  adjacency=tfgnn.Adjacency.from_indices(
                      ("institution", tf.constant([0, 0, 1, 1])),
                      ("author", tf.constant([0, 1, 2, 3])),
                  )),
      },
  )
  return graph


class ReloadModel(int, enum.Enum):
  """Controls how to reload a model for further testing after saving."""
  SKIP = 0
  SAVED_MODEL = 1
  KERAS = 2


class GraphsageTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(("MaxPooling", "max"),
                                  ("MaxNoInfPooling", "max_no_inf"),
                                  ("MeanPooling", "mean"))
  def testPooling(self, reduce_type):
    graph = _get_test_graph()
    out_units = 1
    conv = graph_sage.GraphSAGEPoolingConv(
        receiver_tag=tfgnn.TARGET,
        sender_node_feature=_FEATURE_NAME,
        units=out_units,
        hidden_units=out_units,
        reduce_type=reduce_type)
    _ = conv(graph, edge_set_name="written")  # Build weights.
    weights = {v.name: v for v in conv.trainable_weights}
    self.assertLen(weights, 3)
    source_node_dims = 3
    weights["graph_sage_pooling_conv/dense/kernel:0"].assign([[1.]] *
                                                             source_node_dims)
    weights["graph_sage_pooling_conv/dense/bias:0"].assign([0.])
    weights["graph_sage_pooling_conv/dense_1/kernel:0"].assign([[1.]] *
                                                               out_units)
    actual = conv(graph, edge_set_name="written")
    expected_output_dict = {
        "max":
            tf.constant([
                [6.],
                [6.],
                [tf.float32.min],  # No neighbors.
                [6.]
            ]),
        "max_no_inf":
            tf.constant([
                [6.],
                [6.],
                [0.],  # No neighbors.
                [6.]
            ]),
        "mean":
            tf.constant([
                [6.],
                [6.],
                [0.],  # No neighbors.
                [6.]
            ])
    }
    self.assertAllEqual(expected_output_dict[reduce_type], actual)

  def testMeanAggregation(self):
    graph = _get_test_graph()
    out_units = 1
    conv = graph_sage.GraphSAGEAggregatorConv(
        receiver_tag=tfgnn.TARGET,
        sender_node_feature=_FEATURE_NAME,
        units=out_units)
    _ = conv(graph, edge_set_name="written")  # Build weights.
    weights = {v.name: v for v in conv.trainable_weights}
    self.assertLen(weights, 1)
    source_node_dims = 3
    weights["graph_sage_aggregator_conv/dense/kernel:0"].assign(
        [[1.]] * source_node_dims)
    actual = conv(graph, edge_set_name="written")
    expected_output = tf.constant([
        [6.],
        [6.],
        [0.],  # No neighbors.
        [6.]
    ])
    self.assertAllEqual(expected_output, actual)

  @parameterized.named_parameters(
      ("NoDropoutMeanAggKeras", 0.0, ReloadModel.KERAS),
      ("NoDropoutMeanAggSavedModel", 0.0, ReloadModel.SAVED_MODEL),
      ("DropoutMeanAggKeras", 0.9, ReloadModel.KERAS),
      ("DropoutMeanAggSavedModel", 0.9, ReloadModel.SAVED_MODEL))
  def testDropoutFullModel(self, dropout_rate, reload_model):
    tf.random.set_seed(0)
    graph = _get_test_graph()
    out_units = 30
    layer = graph_sage.GraphSAGEGraphUpdate(
        node_set_names={"topic"},
        l2_normalize=False,
        receiver_tag=tfgnn.TARGET,
        reduce_type="mean",
        use_pooling=False,
        dropout_rate=dropout_rate,
        units=out_units,
        feature_name=_FEATURE_NAME)
    _ = layer(graph)
    weights = {v.name: v for v in layer.trainable_weights}
    self.assertLen(weights, 3)
    node_dims = 30
    weights[
        "graph_sage/node_set_update/graph_sage_aggregator_conv/dense/kernel:0"
    ].assign(tf.eye(node_dims))
    weights[
        "graph_sage/node_set_update/graph_sage_next_state/dense_1/kernel:0"
    ].assign(tf.eye(node_dims))
    bias_shape = out_units
    weights["graph_sage/node_set_update/graph_sage_next_state/bias:0"].assign(
        [0.] * bias_shape)
    inputs = tf.keras.layers.Input(type_spec=graph.spec)
    outputs = layer(inputs)
    model = tf.keras.Model(inputs, outputs)
    if reload_model:
      export_dir = os.path.join(self.get_temp_dir(), "dropout-model")
      model.save(export_dir, include_optimizer=False)
      if reload_model == ReloadModel.KERAS:
        model = tf.keras.models.load_model(export_dir)
        # Check that from_config() worked, no fallback to a function trace, see
        # https://www.tensorflow.org/guide/keras/save_and_serialize#how_savedmodel_handles_custom_objects
        self.assertIsInstance(model.get_layer(index=1),
                              tfgnn.keras.layers.GraphUpdate)
      else:
        model = tf.saved_model.load(export_dir)

    # Actual value returns all 1s without dropout for both of the topic node
    # vectors. One of the nodes don't have any incoming edges, hence dropout is
    # verified for self node vectors and the other has one edge with it self
    # vector consisting of 0s, so that dropout is verified for edges only.
    # Applying dropout value 0.9, max entry after scaling the vector inputs is:
    # 1*1/(1-0.9) = 10.
    def min_max(vector):
      return [tf.reduce_min(vector), tf.reduce_max(vector)]

    def get_topic_vectors(**kwargs):
      out_gt = model(graph, **kwargs)
      out_nodes = out_gt.node_sets["topic"][_FEATURE_NAME]
      return out_nodes

    self.assertAllEqual(
        get_topic_vectors(training=False), [[1.] * node_dims, [1.] * node_dims])
    if dropout_rate != 0.0:
      topic_node_vectors = get_topic_vectors(training=True)
      self.assertAllClose(min_max(topic_node_vectors[0]), [0., 10.])
      self.assertAllClose(min_max(topic_node_vectors[1]), [0., 10.])

  def testAllNodeSets(self):
    graph = _get_test_graph()
    layer = graph_sage.GraphSAGEGraphUpdate(
        units=7, receiver_tag=tfgnn.TARGET, use_bias=False, use_pooling=False,
        feature_name=_FEATURE_NAME)
    _ = layer(graph)
    expected_weight_shapes = [
        # Node set "author":
        [3, 7],  # Conv on "written" from "paper".
        [4, 7],  # Conv on "affiliated_with" from "institution",
        [2, 7],  # Transformation of old state.
        # Node set "topic":
        [30, 7],  # Conv on "correlates" from "topic".
        [30, 7],  # Transformation of old state.
        # Node sets "paper" and "institution" are not receiving from any edges.
    ]
    self.assertCountEqual(expected_weight_shapes,
                          [v.shape.as_list() for v in layer.trainable_weights])

  @parameterized.named_parameters(
      ("E2ENormalizeNoConcatPooling", True, "sum", True, ReloadModel.SKIP),
      ("E2ENormalizeNoConcatAgg", True, "sum", False, ReloadModel.SKIP),
      ("E2ENormalizeConcatPooling", True, "concat", True, ReloadModel.SKIP),
      ("E2ENormalizeConcatAgg", True, "concat", False, ReloadModel.SKIP),
      ("E2ENoNormalizeConcatPooling", False, "concat", True, ReloadModel.SKIP),
      ("E2ENoNormalizeConcatAgg", False, "concat", False, ReloadModel.SKIP),
      ("E2ENoNormalizeNoConcatPooling", False, "sum", True, ReloadModel.SKIP),
      ("E2ENoNormalizeNoConcatAgg", False, "sum", False, ReloadModel.SKIP),
      ("E2ELoadKerasPooling", True, "concat", True, ReloadModel.KERAS),
      ("E2ELoadSavedModelPooling", True, "concat", True,
       ReloadModel.SAVED_MODEL))
  def testFullModel(self, normalize, combine_type, use_pooling, reload_model):
    graph = _get_test_graph()
    out_units = 1
    layer = graph_sage.GraphSAGEGraphUpdate(
        node_set_names={"author"},
        receiver_tag=tfgnn.TARGET,
        reduce_type="mean",
        use_pooling=use_pooling,
        units=out_units,
        hidden_units=out_units if use_pooling else None,
        l2_normalize=normalize,
        combine_type=combine_type,
        feature_name=_FEATURE_NAME)
    _ = layer(graph)
    weights = {v.name: v for v in layer.trainable_weights}
    if use_pooling:
      self.assertLen(weights, 8)
    else:
      self.assertLen(weights, 4)
    paper_node_dims = 3
    institution_node_dims = 4
    target_node_dims = 2
    if use_pooling:
      weights[
          "graph_sage/node_set_update/graph_sage_pooling_conv/dense/kernel:0"
      ].assign([[1.0]] * paper_node_dims)
      weights[
          "graph_sage/node_set_update/graph_sage_pooling_conv/dense/bias:0"
      ].assign([0.0])
      weights[
          "graph_sage/node_set_update/graph_sage_pooling_conv/dense_1/kernel:0"
      ].assign([[1.0]] * out_units)
      weights[
          "graph_sage/node_set_update/graph_sage_pooling_conv/dense_2/kernel:0"
      ].assign([[1.0]] * institution_node_dims)
      weights[
          "graph_sage/node_set_update/graph_sage_pooling_conv/dense_2/bias:0"
      ].assign([0.0])
      weights[
          "graph_sage/node_set_update/graph_sage_pooling_conv/dense_3/kernel:0"
      ].assign([[1.0]] * out_units)
      weights[
          "graph_sage/node_set_update/graph_sage_next_state/dense_4/kernel:0"
      ].assign([[1.0]] * target_node_dims)
    else:
      weights[
          "graph_sage/node_set_update/graph_sage_aggregator_conv/dense/kernel:0"
      ].assign([[1.0]] * paper_node_dims)
      weights[
          "graph_sage/node_set_update/graph_sage_aggregator_conv/dense_1/kernel:0"
      ].assign([[1.0]] * institution_node_dims)
      weights[
          "graph_sage/node_set_update/graph_sage_next_state/dense_2/kernel:0"
      ].assign([[1.0]] * target_node_dims)
    num_edge_type = 2
    bias_shape = out_units if combine_type == "sum" else out_units * (
        num_edge_type + 1)
    weights["graph_sage/node_set_update/graph_sage_next_state/bias:0"].assign(
        [0.] * bias_shape)
    inputs = tf.keras.layers.Input(type_spec=graph.spec)
    outputs = layer(inputs)
    model = tf.keras.Model(inputs, outputs)
    if reload_model:
      export_dir = os.path.join(self.get_temp_dir(), "gsage-model")
      model.save(export_dir, include_optimizer=False)
      if reload_model == ReloadModel.KERAS:
        model = tf.keras.models.load_model(export_dir)
        # Check that from_config() worked, no fallback to a function trace, see
        # https://www.tensorflow.org/guide/keras/save_and_serialize#how_savedmodel_handles_custom_objects
        self.assertIsInstance(model.get_layer(index=1),
                              tfgnn.keras.layers.GraphUpdate)
      else:
        model = tf.saved_model.load(export_dir)

    actual_graph = model(graph)
    actual = actual_graph.node_sets["author"][_FEATURE_NAME]
    # maps normalize-to-combine_type expected states.
    expected_outputs = {
        True: {
            "concat":
                tf.constant([[
                    1. / math.sqrt(1**2 + 6**2 * 2),
                    6. / math.sqrt(1**2 + 6**2 * 2),
                    6. / math.sqrt(1**2 + 6**2 * 2)
                ],
                             [
                                 2. / math.sqrt(2**2 + 6**2 * 2),
                                 6. / math.sqrt(2**2 + 6**2 * 2),
                                 6. / math.sqrt(2**2 + 6**2 * 2)
                             ],
                             [
                                 1. / math.sqrt(1**2 + 0**2 + 6**2),
                                 6. / math.sqrt(1**2 + 0**2 + 6**2),
                                 0. / math.sqrt(1**2 + 0**2 + 6**2)
                             ],
                             [
                                 2. / math.sqrt(2**2 + 6**2 * 2),
                                 6. / math.sqrt(2**2 + 6**2 * 2),
                                 6. / math.sqrt(2**2 + 6**2 * 2)
                             ]]),
            "sum":
                tf.constant([[1.], [1.], [1.], [1.]])
        },
        False: {
            "concat":
                tf.constant([[1., 6., 6.], [2., 6., 6.], [1., 6., 0.],
                             [2., 6., 6.]]),
            "sum":
                tf.constant([[13.], [14.], [7.], [14.]])
        }
    }
    self.assertAllClose(actual, expected_outputs[normalize][combine_type])

  def testReceivingRequired(self):
    graph = _get_test_graph()
    assert not any(edge_set.adjacency.target_name == "paper"
                   for edge_set in graph.edge_sets.values())
    layer = graph_sage.GraphSAGEGraphUpdate(
        node_set_names={"author", "paper"},
        receiver_tag=tfgnn.TARGET,
        units=1,
        hidden_units=1,
        feature_name=_FEATURE_NAME)
    with self.assertRaisesRegex(ValueError, r"not .* from any edge set.*paper"):
      _ = layer(graph)

  @parameterized.named_parameters(
      ("E2ELoadKerasMeanPool", "mean", True, ReloadModel.KERAS),
      ("E2ELoadKerasMeanAgg", "mean", False, ReloadModel.KERAS),
      ("E2ELoadKerasMaxPool", "max", True, ReloadModel.KERAS),
      ("E2ELoadKerasMaxAgg", "max", False, ReloadModel.KERAS),
      ("E2ELoadKerasMaxNoInfPool", "max_no_inf", True, ReloadModel.KERAS),
      ("E2ELoadKerasMaxNoInfAgg", "max_no_inf", False, ReloadModel.KERAS),
      ("E2ELoadSavedModelMaxPool", "max", True, ReloadModel.SAVED_MODEL),
      ("E2ELoadSavedModelMaxAgg", "max", False, ReloadModel.SAVED_MODEL),
      ("E2ELoadSavedModelMaxNoInfPool", "max_no_inf", True,
       ReloadModel.SAVED_MODEL), ("E2ELoadSavedModelMaxNoInfAgg", "max_no_inf",
                                  False, ReloadModel.SAVED_MODEL),
      ("E2ELoadSavedModelMeanPool", "mean", True, ReloadModel.SAVED_MODEL),
      ("E2ELoadSavedModelMeanAgg", "mean", False, ReloadModel.SAVED_MODEL))
  def testModelLoad(self, reduce_operation, use_pooling, reload_model):
    graph = _get_test_graph()
    out_units = 1
    layer = graph_sage.GraphSAGEGraphUpdate(
        node_set_names={"author"},
        receiver_tag=tfgnn.TARGET,
        reduce_type=reduce_operation,
        combine_type="concat",
        use_pooling=use_pooling,
        units=out_units,
        hidden_units=out_units if use_pooling else None,
        feature_name=_FEATURE_NAME)
    _ = layer(graph)
    weights = {v.name: v for v in layer.trainable_weights}
    if use_pooling:
      self.assertLen(weights, 8)
    else:
      self.assertLen(weights, 4)
    paper_node_dims = 3
    institution_node_dims = 4
    target_node_dims = 2
    if use_pooling:
      weights[
          "graph_sage/node_set_update/graph_sage_pooling_conv/dense/kernel:0"
      ].assign([[1.0]] * paper_node_dims)
      weights[
          "graph_sage/node_set_update/graph_sage_pooling_conv/dense/bias:0"
      ].assign([0.0])
      weights[
          "graph_sage/node_set_update/graph_sage_pooling_conv/dense_1/kernel:0"
      ].assign([[1.0]] * out_units)
      weights[
          "graph_sage/node_set_update/graph_sage_pooling_conv/dense_2/kernel:0"
      ].assign([[1.0]] * institution_node_dims)
      weights[
          "graph_sage/node_set_update/graph_sage_pooling_conv/dense_2/bias:0"
      ].assign([0.0])
      weights[
          "graph_sage/node_set_update/graph_sage_pooling_conv/dense_3/kernel:0"
      ].assign([[1.0]] * out_units)
      weights[
          "graph_sage/node_set_update/graph_sage_next_state/dense_4/kernel:0"
      ].assign([[1.0]] * target_node_dims)
    else:
      weights[
          "graph_sage/node_set_update/graph_sage_aggregator_conv/dense/kernel:0"
      ].assign([[1.0]] * paper_node_dims)
      weights[
          "graph_sage/node_set_update/graph_sage_aggregator_conv/dense_1/kernel:0"
      ].assign([[1.0]] * institution_node_dims)
      weights[
          "graph_sage/node_set_update/graph_sage_next_state/dense_2/kernel:0"
      ].assign([[1.0]] * target_node_dims)
    num_edge_type = 2
    bias_shape = out_units * (num_edge_type + 1)
    weights["graph_sage/node_set_update/graph_sage_next_state/bias:0"].assign(
        [0.] * bias_shape)
    inputs = tf.keras.layers.Input(type_spec=graph.spec)
    outputs = layer(inputs)
    model = tf.keras.Model(inputs, outputs)
    if reload_model:
      export_dir = os.path.join(self.get_temp_dir(), "gsage-model")
      model.save(export_dir, include_optimizer=False)
      if reload_model == ReloadModel.KERAS:
        model = tf.keras.models.load_model(export_dir)
      else:
        model = tf.saved_model.load(export_dir)

    actual_graph = model(graph)
    actual = actual_graph.node_sets["author"][_FEATURE_NAME]
    expected = tf.constant([[
        1. / math.sqrt(1**2 + 6**2 * 2), 6. / math.sqrt(1**2 + 6**2 * 2),
        6. / math.sqrt(1**2 + 6**2 * 2)
    ],
                            [
                                2. / math.sqrt(2**2 + 6**2 * 2),
                                6. / math.sqrt(2**2 + 6**2 * 2),
                                6. / math.sqrt(2**2 + 6**2 * 2)
                            ],
                            [
                                1. / math.sqrt(1**2 + 0**2 + 6**2),
                                6. / math.sqrt(1**2 + 0**2 + 6**2),
                                0. / math.sqrt(1**2 + 0**2 + 6**2)
                            ],
                            [
                                2. / math.sqrt(2**2 + 6**2 * 2),
                                6. / math.sqrt(2**2 + 6**2 * 2),
                                6. / math.sqrt(2**2 + 6**2 * 2)
                            ]])
    self.assertAllClose(actual, expected)

  @parameterized.named_parameters(
      ("E2ELoadKerasGCNConv", ReloadModel.KERAS),
      ("E2ELoadSavedModelGCNConv", ReloadModel.SAVED_MODEL))
  def testGCNConvolutionModelLoad(self, reload_model):
    graph = _get_test_graph()
    message_units = 1
    conv = graph_sage.GCNGraphSAGENodeSetUpdate(
        edge_set_names=["written", "affiliated_with"],
        receiver_tag=tfgnn.TARGET,
        self_node_feature=_FEATURE_NAME,
        sender_node_feature=_FEATURE_NAME,
        units=message_units,
        use_bias=True)
    layer = tfgnn.keras.layers.GraphUpdate(node_sets={"author": conv})
    _ = layer(graph)  # Build weights.
    weights = {v.name: v for v in layer.trainable_weights}
    self.assertLen(weights, 4)
    paper_feature_dim = 3
    institution_feature_dim = 4
    author_feature_dim = 2
    weights["graph_update/graph_sage_gcn_update/dense/kernel:0"].assign(
        [[1.0]] * paper_feature_dim)
    weights["graph_update/graph_sage_gcn_update/dense_1/kernel:0"].assign(
        [[1.0]] * institution_feature_dim)
    weights["graph_update/graph_sage_gcn_update/dense_2/kernel:0"].assign(
        [[1.0]] * author_feature_dim)
    weights["bias:0"].assign([0.] * message_units)
    inputs = tf.keras.layers.Input(type_spec=graph.spec)
    outputs = layer(inputs)
    model = tf.keras.Model(inputs, outputs)
    if reload_model:
      export_dir = os.path.join(self.get_temp_dir(), "gsage-model")
      model.save(export_dir, include_optimizer=False)
      if reload_model == ReloadModel.KERAS:
        model = tf.keras.models.load_model(export_dir)
      else:
        model = tf.saved_model.load(export_dir)
    actual_graph = model(graph)
    actual = actual_graph.node_sets["author"]
    expected_output = tf.constant([[4.3333335], [4.6666665], [3.5],
                                   [4.6666665]])
    self.assertAllEqual(expected_output, actual[_FEATURE_NAME])

  @parameterized.named_parameters(("WithSelfLoop", True), ("NoSelfLoop", False))
  def testGCNConvolutionSharedWeights(self, add_self_loop):
    graph = _get_test_graph()
    message_units = 1
    conv = graph_sage.GCNGraphSAGENodeSetUpdate(
        edge_set_names=["correlates"],
        receiver_tag=tfgnn.TARGET,
        self_node_feature=_FEATURE_NAME,
        sender_node_feature=_FEATURE_NAME,
        units=message_units,
        use_bias=True,
        share_weights=True,
        add_self_loop=add_self_loop)
    _ = conv(graph, node_set_name="topic")  # Build weights.
    weights = {v.name: v for v in conv.trainable_weights}
    self.assertLen(weights, 2)
    topic_feature_dim = 30
    weights["graph_sage_gcn_update/dense/kernel:0"].assign([[1.0]] *
                                                           topic_feature_dim)
    weights["bias:0"].assign([0.] * message_units)
    actual = conv(graph, node_set_name="topic")
    expected_output = {
        True: tf.constant([[30.], [15.]]),
        False: tf.constant([[0.], [30.]])
    }
    self.assertAllEqual(expected_output[add_self_loop], actual[_FEATURE_NAME])

  def testGCNConvolutionFail(self):
    graph = _get_test_graph()
    message_units = 1
    conv = graph_sage.GCNGraphSAGENodeSetUpdate(
        edge_set_names=["correlates", "affiliated_with"],
        receiver_tag=tfgnn.TARGET,
        self_node_feature=_FEATURE_NAME,
        sender_node_feature=_FEATURE_NAME,
        units=message_units,
        use_bias=True)
    self.assertRaisesRegex(
        ValueError,
        r"Incorrect .* that has a different node at receiver_tag:.* other than .*.",
        lambda: conv(graph, node_set_name="author"))
    conv = graph_sage.GCNGraphSAGENodeSetUpdate(
        edge_set_names=["correlates", "affiliated_with"],
        receiver_tag=tfgnn.TARGET,
        self_node_feature=_FEATURE_NAME,
        sender_node_feature=_FEATURE_NAME,
        units=message_units,
        use_bias=True,
        reduce_type="max_no_inf")
    self.assertRaisesRegex(ValueError,
                           r".* isn't supported, please instead use any of .*",
                           lambda: conv(graph, node_set_name="author"))


class GraphSAGETFLiteTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("WithPooling", True, 4, "sum"),
      ("SumAggregation", False, None, "sum"),
      ("ConcatAggregation", False, None, "concat"))
  def testBasic(self, use_pooling, hidden_units, combine_type):
    test_graph_1_dict = {
        # We care that the TFLite interpreter gives the same output as the
        # model, which was tested separately.
        "source": tf.constant([0, 1, 2, 0, 2, 1]),
        "target": tf.constant([1, 2, 0, 2, 1, 0]),
        "node_features": tf.constant([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
        ]),
        "edge_features": tf.constant([
            [3.0],
            [6.0],
            [9.0],
            [2.0],
            [6.0],
            [4.0],
        ]),
    }

    layer = graph_sage.GraphSAGEGraphUpdate(
        units=4,
        receiver_tag=tfgnn.TARGET,
        use_bias=True,
        use_pooling=use_pooling,
        hidden_units=hidden_units,
        combine_type=combine_type,
        feature_name=tfgnn.HIDDEN_STATE,
    )

    inputs = {
        "node_features": tf.keras.Input([4], None, "node_features", tf.float32),
        "source": tf.keras.Input([], None, "source", tf.int32),
        "target": tf.keras.Input([], None, "target", tf.int32),
        "edge_features": tf.keras.Input([1], None, "edge_features", tf.float32),
    }
    graph_in = _MakeGraphTensor()(inputs)
    graph_out = layer(graph_in)
    outputs = tf.keras.layers.Layer(name="final_node_states")(
        graph_out.node_sets["nodes"][tfgnn.HIDDEN_STATE]
    )
    model = tf.keras.Model(inputs, outputs)

    # The other unit tests should verify that this is correct
    expected = model(test_graph_1_dict).numpy()

    # TODO(b/276291104): Remove when TF 2.11+ is required by all of TFGNN
    if tf.__version__.startswith("2.10."):
      self.skipTest("GNN models are unsupported in TFLite until TF 2.11 but "
                    f"got TF {tf.__version__}")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    model_content = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=model_content)
    signature_runner = interpreter.get_signature_runner("serving_default")
    obtained = signature_runner(**test_graph_1_dict)["final_node_states"]
    self.assertAllClose(expected, obtained)


# TODO(b/274779989): Replace this layer with a more standard representation
# of GraphTensor as a dict of plain Tensors.
class _MakeGraphTensor(tf.keras.layers.Layer):
  """Makes a homogeneous GraphTensor of rank 0 with a single component."""

  def call(self, inputs):
    node_sizes = tf.shape(inputs["node_features"])[0]
    edge_sizes = tf.shape(inputs["edge_features"])[0]
    return tfgnn.GraphTensor.from_pieces(
        node_sets={
            "nodes": tfgnn.NodeSet.from_fields(
                sizes=tf.expand_dims(node_sizes, axis=0),
                features={tfgnn.HIDDEN_STATE: inputs["node_features"]},
            )
        },
        edge_sets={
            "edges": tfgnn.EdgeSet.from_fields(
                sizes=tf.expand_dims(edge_sizes, axis=0),
                adjacency=tfgnn.Adjacency.from_indices(
                    ("nodes", inputs["source"]), ("nodes", inputs["target"])
                ),
                features={tfgnn.HIDDEN_STATE: inputs["edge_features"]},
            )
        },
    )


if __name__ == "__main__":
  tf.test.main()
