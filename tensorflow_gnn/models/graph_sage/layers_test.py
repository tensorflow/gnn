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
        "graph_sage/node_set_update/graph_sage_aggregator_conv/dense/kernel:0"].assign(
            tf.eye(node_dims))
    weights[
        "graph_sage/node_set_update/graph_sage_next_state/dense_1/kernel:0"].assign(
            tf.eye(node_dims))
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
      self.assertLen(weights, 5)
    else:
      self.assertLen(weights, 3)
    source_node_dims = 3
    target_node_dims = 2
    if use_pooling:
      weights[
          "graph_sage/node_set_update/graph_sage_pooling_conv/dense/kernel:0"].assign(
              [[1.]] * source_node_dims)
      weights[
          "graph_sage/node_set_update/graph_sage_pooling_conv/dense/bias:0"].assign(
              [0.])
      weights[
          "graph_sage/node_set_update/graph_sage_pooling_conv/dense_1/kernel:0"].assign(
              [[1.]] * out_units)
      weights[
          "graph_sage/node_set_update/graph_sage_next_state/dense_2/kernel:0"].assign(
              [[1.]] * target_node_dims)
    else:
      weights[
          "graph_sage/node_set_update/graph_sage_aggregator_conv/dense/kernel:0"].assign(
              [[1.]] * source_node_dims)
      weights[
          "graph_sage/node_set_update/graph_sage_next_state/dense_1/kernel:0"].assign(
              [[1.]] * target_node_dims)
    num_edge_type = 1
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
      else:
        model = tf.saved_model.load(export_dir)

    actual_graph = model(graph)
    actual = actual_graph.node_sets["author"][_FEATURE_NAME]
    # maps normalize-to-combine_type expected states.
    expected_outputs = {
        True: {
            "concat":
                tf.constant(
                    [[1. / math.sqrt(1**2 + 6**2), 6. / math.sqrt(1**2 + 6**2)],
                     [2. / math.sqrt(2**2 + 6**2), 6. / math.sqrt(2**2 + 6**2)],
                     [1., 0.],
                     [2. / math.sqrt(2**2 + 6**2),
                      6. / math.sqrt(2**2 + 6**2)]]),
            "sum":
                tf.constant([[1.], [1.], [1.], [1.]])
        },
        False: {
            "concat": tf.constant([[1., 6.], [2., 6.], [1., 0.], [2., 6.]]),
            "sum": tf.constant([[7.], [8.], [1.], [8.]])
        }
    }
    self.assertAllClose(actual, expected_outputs[normalize][combine_type])

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
        node_set_names={"author", "paper"},
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
      self.assertLen(weights, 5)
    else:
      self.assertLen(weights, 3)
    source_node_dims = 3
    target_node_dims = 2
    if use_pooling:
      weights[
          "graph_sage/node_set_update/graph_sage_pooling_conv/dense/kernel:0"].assign(
              [[1.]] * source_node_dims)
      weights[
          "graph_sage/node_set_update/graph_sage_pooling_conv/dense/bias:0"].assign(
              [0.])
      weights[
          "graph_sage/node_set_update/graph_sage_pooling_conv/dense_1/kernel:0"].assign(
              [[1.]] * out_units)
      weights[
          "graph_sage/node_set_update/graph_sage_next_state/dense_2/kernel:0"].assign(
              [[1.]] * target_node_dims)
    else:
      weights[
          "graph_sage/node_set_update/graph_sage_aggregator_conv/dense/kernel:0"].assign(
              [[1.]] * source_node_dims)
      weights[
          "graph_sage/node_set_update/graph_sage_next_state/dense_1/kernel:0"].assign(
              [[1.]] * target_node_dims)
    num_edge_type = 1
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
    expected = tf.constant([[0.16439898, 0.9863939], [0.31622776, 0.94868326],
                            [0.99999994, 0.], [0.31622776, 0.94868326]])
    self.assertAllClose(actual, expected)


if __name__ == "__main__":
  tf.test.main()
