"""Tests for the layers from graph_update.py."""

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph.keras.layers import graph_ops
from tensorflow_gnn.graph.keras.layers import graph_update
from tensorflow_gnn.graph.keras.layers import graph_update_options as opt
from tensorflow_gnn.graph.keras.utils import fnn_factory


NEVER = opt.UpdateInputEnabled.NEVER
ALWAYS = opt.UpdateInputEnabled.ALWAYS
ON_UPDATE = opt.UpdateInputEnabled.ON_UPDATE


class CombinedUpdateTest(tf.test.TestCase, parameterized.TestCase):

  def testFromDefaults(self):
    values = dict(nodes=tf.constant([[1.], [2.], [10.]]))
    example_graph = _make_test_graph_01into2(values)

    options = opt.GraphUpdateOptions(graph_tensor_spec=example_graph.spec)
    options.node_set_default.update_fn_factory = fnn_factory.get_fnn_factory(
        output_dim=1, activation=None, name="node_set_update")
    options.edge_set_default.update_fn_factory = fnn_factory.get_fnn_factory(
        output_dim=1, activation=None, name="edge_set_update")
    model = tf.keras.Sequential([
        graph_update.EdgeSetUpdate("edges", options=options),
        graph_update.NodeSetUpdate("nodes", options=options),
        graph_ops.Readout(node_set_name="nodes")])
    _ = model(example_graph)  # Trigger building the model.
    self._set_weights_to_add_neighbors_twice(model)
    self.assertAllClose(model(example_graph), [[1.], [2.], [10. + 2. + 4.]])

  def _set_weights_to_add_neighbors_twice(self, model):
    weights = {v.name: v for v in model.trainable_weights}
    self.assertLen(weights, 4)
    # The edge state is twice the source state.
    weights["edge_set_update/output/kernel:0"].assign([[2.], [0.]])
    weights["edge_set_update/output/bias:0"].assign([0.])
    # The node state is updated by adding the incoming edges.
    weights["node_set_update/output/kernel:0"].assign([[1.], [1.]])
    weights["node_set_update/output/bias:0"].assign([0.])


def _make_test_graph_01into2(values):
  """Returns GraphTensor for [v0] --e0--> [v2] <-e1-- [v1] with values."""
  def maybe_features(key):
    features = {const.DEFAULT_STATE_NAME: values[key]} if key in values else {}
    return dict(features=features)
  graph = gt.GraphTensor.from_pieces(
      context=gt.Context.from_fields(**maybe_features("context")),
      node_sets={"nodes": gt.NodeSet.from_fields(
          sizes=tf.constant([3]), **maybe_features("nodes"))},
      edge_sets={"edges": gt.EdgeSet.from_fields(
          sizes=tf.constant([2]),
          adjacency=adj.Adjacency.from_indices(("nodes", tf.constant([0, 1])),
                                               ("nodes", tf.constant([2, 2]))),
          **maybe_features("edges"))})
  return graph


class GraphPieceUpdateBaseTest(tf.test.TestCase, parameterized.TestCase):
  """Tests argument handling of _GraphPieceUpdateBase via EdgeSetUpdate."""

  def testInputArg(self):
    extra_name = "extra_state"
    example_graph = gt.GraphTensor.from_pieces(
        node_sets={"nodes": gt.NodeSet.from_fields(
            sizes=tf.constant([3]),
            features={
                const.DEFAULT_STATE_NAME: tf.constant([[2.], [4.], [10.]]),
                extra_name: tf.constant([[6.], [8.], [12.]])})},
        edge_sets={"edges": gt.EdgeSet.from_fields(
            sizes=tf.constant([2]),
            adjacency=adj.Adjacency.from_indices(
                ("nodes", tf.constant([0, 1])),
                ("nodes", tf.constant([2, 2]))),
            features={})})
    options = opt.GraphUpdateOptions()
    options.edge_set_default.update_fn_factory = fnn_factory.get_fnn_factory(
        output_dim=1, activation=None, name="edge_set_update")

    # Baseline: defaults.
    update = graph_update.EdgeSetUpdate("edges", options=options)
    _ = update(example_graph)  # Trigger build.
    self._set_weights_for_half_source(update)
    result = update(example_graph).edge_sets["edges"].get_features_dict()
    self.assertLen(result, 1)
    self.assertAllClose(result[const.DEFAULT_STATE_NAME], [[1.], [2.]])

    # Setting input_fns via argument.
    options.edge_set_default.update_input_fn_factories = [self._do_not_call]
    options.edge_sets["edges"].update_input_fn_factories = [self._do_not_call]
    input_fns = [  # Note reverse order and non-standard feature name.
        graph_ops.Broadcast(const.TARGET, feature_name=extra_name),
        graph_ops.Broadcast(const.SOURCE, feature_name=extra_name)]
    update = graph_update.EdgeSetUpdate("edges", options=options,
                                        input_fns=input_fns)
    _ = update(example_graph)  # Trigger build.
    self._set_weights_for_half_source(update)  # With reversal: half target.
    result = update(example_graph).edge_sets["edges"].get_features_dict()
    self.assertLen(result, 1)
    self.assertAllClose(result[const.DEFAULT_STATE_NAME], [[6.], [6.]])

    # Setting input_fns via edge set options.
    input_fn_factories = [lambda: input_fns[0], lambda: input_fns[1]]
    options.edge_sets["edges"].update_input_fn_factories = input_fn_factories
    update = graph_update.EdgeSetUpdate("edges", options=options)
    _ = update(example_graph)
    self._set_weights_for_half_source(update)  # With reversal: half target.
    result = update(example_graph).edge_sets["edges"].get_features_dict()
    self.assertLen(result, 1)
    self.assertAllClose(result[const.DEFAULT_STATE_NAME], [[6.], [6.]])

    # Setting input_fns via edge set default options.
    options.edge_sets["edges"].update_input_fn_factories = None
    options.edge_set_default.update_input_fn_factories = input_fn_factories
    update = graph_update.EdgeSetUpdate("edges", options=options)
    _ = update(example_graph)
    self._set_weights_for_half_source(update)  # With reversal: half target.
    result = update(example_graph).edge_sets["edges"].get_features_dict()
    self.assertLen(result, 1)
    self.assertAllClose(result[const.DEFAULT_STATE_NAME], [[6.], [6.]])

  def testCombinerArg(self):
    values = dict(nodes=tf.constant([[2.], [4.], [10.]]))
    example_graph = _make_test_graph_01into2(values)
    options = opt.GraphUpdateOptions()
    options.edge_set_default.update_fn_factory = fnn_factory.get_fnn_factory(
        output_dim=1, activation=None, name="edge_set_update")

    # Baseline: defaults.
    update = graph_update.EdgeSetUpdate("edges", options=options)
    _ = update(example_graph)  # Trigger build.
    self._set_weights_for_half_source(update)
    result = update(example_graph).edge_sets["edges"].get_features_dict()
    self.assertLen(result, 1)
    self.assertAllClose(result[const.DEFAULT_STATE_NAME], [[1.], [2.]])

    # Setting combiner_fn via argument.
    options.edge_set_default.update_combiner_fn = "baddd"
    options.edge_sets["edges"].update_combiner_fn = "wronggg"
    combiner_fn = ReverseConcatenate()
    update = graph_update.EdgeSetUpdate("edges", options=options,
                                        combiner_fn=combiner_fn)
    _ = update(example_graph)  # Trigger build.
    self._set_weights_for_half_source(update)  # With reversal: half target.
    result = update(example_graph).edge_sets["edges"].get_features_dict()
    self.assertLen(result, 1)
    self.assertAllClose(result[const.DEFAULT_STATE_NAME], [[5.], [5.]])

    # Setting combiner_fn via edge set options.
    options.edge_sets["edges"].update_combiner_fn = combiner_fn
    update = graph_update.EdgeSetUpdate("edges", options=options)
    _ = update(example_graph)
    self._set_weights_for_half_source(update)  # With reversal: half target.
    result = update(example_graph).edge_sets["edges"].get_features_dict()
    self.assertLen(result, 1)
    self.assertAllClose(result[const.DEFAULT_STATE_NAME], [[5.], [5.]])

    # Setting combiner_fn via edge set default options.
    options.edge_sets["edges"].update_combiner_fn = None
    options.edge_set_default.update_combiner_fn = combiner_fn
    update = graph_update.EdgeSetUpdate("edges", options=options)
    _ = update(example_graph)
    self._set_weights_for_half_source(update)  # With reversal: half target.
    result = update(example_graph).edge_sets["edges"].get_features_dict()
    self.assertLen(result, 1)
    self.assertAllClose(result[const.DEFAULT_STATE_NAME], [[5.], [5.]])

  def testUpdateArg(self):
    values = dict(nodes=tf.constant([[2.], [4.], [10.]]))
    example_graph = _make_test_graph_01into2(values)
    options = opt.GraphUpdateOptions()

    # Setting update_fn via argument.
    options.edge_set_default.update_fn_factory = self._do_not_call
    options.edge_sets["edges"].update_fn_factory = self._do_not_call
    update_fn = tf.keras.layers.Dense(
        1, use_bias=False,
        kernel_initializer=tf.keras.initializers.Constant([[3.], [0.]]))
    update = graph_update.EdgeSetUpdate("edges", options=options,
                                        update_fn=update_fn)
    result = update(example_graph).edge_sets["edges"].get_features_dict()
    self.assertLen(result, 1)
    self.assertAllClose(result[const.DEFAULT_STATE_NAME], [[6.], [12.]])

    # Setting update_fn via edge set options.
    options.edge_sets["edges"].update_fn_factory = lambda: update_fn
    update = graph_update.EdgeSetUpdate("edges", options=options)
    result = update(example_graph).edge_sets["edges"].get_features_dict()
    self.assertLen(result, 1)
    self.assertAllClose(result[const.DEFAULT_STATE_NAME], [[6.], [12.]])

    # Setting update_fn via edge set default options.
    options.edge_sets["edges"].update_fn_factory = None
    options.edge_set_default.update_fn_factory = lambda: update_fn
    update = graph_update.EdgeSetUpdate("edges", options=options)
    result = update(example_graph).edge_sets["edges"].get_features_dict()
    self.assertLen(result, 1)
    self.assertAllClose(result[const.DEFAULT_STATE_NAME], [[6.], [12.]])

    # Not setting anything is an error.
    with self.assertRaisesRegex(ValueError, r"provide update_fn"):
      _ = graph_update.EdgeSetUpdate("edges", options=opt.GraphUpdateOptions())

  def testUpdateIsSublayer(self):
    """Tests that Keras finds weights, losses, training= arg of update_fn."""
    values = dict(edges=tf.constant([[2.], [3.]]))
    input_graph = _make_test_graph_01into2(values)
    update_fn = tf.keras.Sequential([
        tf.keras.layers.Dense(
            12, use_bias=False, name="repeat_dozen",
            kernel_initializer=tf.keras.initializers.Constant([[1.]] * 12),
            kernel_regularizer=tf.keras.regularizers.L2(0.01)),
        tf.keras.layers.Dropout(0.5)])
    update = graph_update.EdgeSetUpdate(
        "edges",
        input_fns=[graph_ops.Readout()],
        update_fn=update_fn)
    graph = update(input_graph)
    self.assertAllClose(graph.edge_sets["edges"][const.DEFAULT_STATE_NAME],
                        [[2.]*12, [3.]*12])
    self.assertAllClose(update.losses, [0.12])
    self.assertSameElements([v.name for v in update.trainable_weights],
                            ["repeat_dozen/kernel:0"])

    for _ in range(100):
      # Skip the unlucky case of getting all zeros in one of the features.
      graph = update(input_graph, training=True)
      max_after_dropout = tf.reduce_max(
          graph.edge_sets["edges"][const.DEFAULT_STATE_NAME], axis=1)
      if (max_after_dropout > tf.constant([0., 0.])).numpy().all():
        break
    self.assertAllClose(max_after_dropout, [4., 6.])

  def testOutputFeatureArg(self):
    values = dict(nodes=tf.constant([[2.], [4.], [10.]]))
    example_graph = _make_test_graph_01into2(values)
    options = opt.GraphUpdateOptions()
    options.edge_set_default.update_fn_factory = fnn_factory.get_fnn_factory(
        output_dim=1, activation=None, name="edge_set_update")

    # Baseline: defaults.
    update = graph_update.EdgeSetUpdate("edges", options=options)
    _ = update(example_graph)  # Trigger build.
    self._set_weights_for_half_source(update)
    result = update(example_graph).edge_sets["edges"].get_features_dict()
    self.assertLen(result, 1)
    self.assertAllClose(result[const.DEFAULT_STATE_NAME], [[1.], [2.]])

    # Setting output_feature via argument.
    options.edge_set_default.update_output_feature = "badd"
    options.edge_sets["edges"].update_output_feature = "worng"
    output_feature = "extra_state"
    update = graph_update.EdgeSetUpdate("edges", options=options,
                                        output_feature=output_feature)
    _ = update(example_graph)
    self._set_weights_for_half_source(update)
    result = update(example_graph).edge_sets["edges"].get_features_dict()
    self.assertLen(result, 1)
    self.assertAllClose(result[output_feature], [[1.], [2.]])

    # Setting output_feature via edge set options.
    options.edge_sets["edges"].update_output_feature = output_feature
    update = graph_update.EdgeSetUpdate("edges", options=options)
    _ = update(example_graph)
    self._set_weights_for_half_source(update)
    result = update(example_graph).edge_sets["edges"].get_features_dict()
    self.assertLen(result, 1)
    self.assertAllClose(result[output_feature], [[1.], [2.]])

    # Setting output_feaure via edge set default options.
    options.edge_sets["edges"].update_output_feature = None
    options.edge_set_default.update_output_feature = output_feature
    update = graph_update.EdgeSetUpdate("edges", options=options)
    _ = update(example_graph)
    self._set_weights_for_half_source(update)
    result = update(example_graph).edge_sets["edges"].get_features_dict()
    self.assertLen(result, 1)
    self.assertAllClose(result[output_feature], [[1.], [2.]])

  def testOutputFeatureIsList(self):
    values = dict(nodes=tf.constant([[1.], [2.], [10.]]))
    example_graph = _make_test_graph_01into2(values)

    update = graph_update.EdgeSetUpdate(
        "edges",
        update_fn=TwiceAndThrice(),
        output_feature=["twice", "thrice"])
    result = update(example_graph).edge_sets["edges"].get_features_dict()
    self.assertLen(result, 2)
    self.assertAllClose(result["twice"], [[2., 20.], [4., 20.]])
    self.assertAllClose(result["thrice"], [[3., 30.], [6., 30.]])

  def _set_weights_for_half_source(self, update):
    weights = update.trainable_weights
    self.assertLen(weights, 2)
    for w in weights:
      if w.shape.rank == 1:  # Bias.
        w.assign([0.])
      elif w.shape.rank == 2:  # Kernel.
        w.assign([[.5], [0.]])
      else:
        self.fail("Missing a case somehow")

  def _do_not_call(self, graph):
    del graph  # Unused.
    self.fail("The do_not_call factory was called")


class ReverseConcatenate(tf.keras.layers.Layer):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._concatenate = tf.keras.layers.Concatenate()

  def call(self, inputs):
    inputs.reverse()
    return self._concatenate(inputs)


class TwiceAndThrice(tf.keras.layers.Layer):

  def call(self, x):
    return [tf.multiply(2., x), tf.multiply(3., x)]


class EdgeSetUpdateTest(tf.test.TestCase, parameterized.TestCase):
  """Tests EdgeSetUpdate beyond _GraphPieceUpdateBase."""

  def testFromConfig(self):
    readout = graph_ops.Readout()
    combiner_fn = tf.keras.layers.Concatenate(name="test_concat")
    update_fn = tf.keras.layers.Dense(
        1, name="update_fn",
        kernel_initializer=tf.keras.initializers.Constant([[2.]]))
    kwargs = dict(edge_set_name="edges", input_fns=[readout],
                  combiner_fn=combiner_fn, update_fn=update_fn,
                  output_feature="out_feature", name="test_update")
    config = graph_update.EdgeSetUpdate(**kwargs).get_config()
    self.assertDictContainsSubset(kwargs, config)
    update = graph_update.EdgeSetUpdate.from_config(config)

    values = dict(edges=tf.constant([[5.], [6.]]))
    input_graph = _make_test_graph_01into2(values)
    graph = update(input_graph)
    self.assertAllClose(graph.edge_sets["edges"]["out_feature"],
                        [[10.], [12.]])

  @parameterized.parameters(False, True)
  def testFeaturesPreservedExceptOutput(self, new_output_feature):
    input_graph = _make_test_graph_red_blue()
    output_feature = "new" if new_output_feature else "even"
    update = graph_update.EdgeSetUpdate(
        "purple",
        input_fns=[
            graph_ops.Readout(feature_name="even"),
            graph_ops.Broadcast(const.SOURCE, feature_name="even"),
            graph_ops.Broadcast(const.CONTEXT, feature_name="even")],
        update_fn=tf.keras.layers.Dense(
            1, use_bias=False,
            kernel_initializer=tf.keras.initializers.Constant(
                [[1.], [1.], [1.]])),  # Sum the concatenated inputs.
        output_feature=output_feature)
    graph = update(input_graph)

    # Features are unchanged, except for the designated output_feature.
    self.assertLen(graph.context.features, 2)
    self.assertAllClose(graph.context["even"], [[2.]])
    self.assertAllClose(graph.context["odd"], [[3.]])
    self.assertLen(graph.node_sets, 2)
    self.assertLen(graph.node_sets["red"].features, 2)
    self.assertAllClose(graph.node_sets["red"]["even"], [[4.], [16.]])
    self.assertAllClose(graph.node_sets["red"]["odd"], [[5.], [17.]])
    self.assertLen(graph.node_sets["blue"].features, 1)
    self.assertAllClose(graph.node_sets["blue"]["other"], [[64.]])
    self.assertLen(graph.edge_sets, 2)
    self.assertLen(graph.edge_sets["red"].features, 2)
    self.assertAllClose(graph.edge_sets["red"]["even"], [[32.]])
    self.assertAllClose(graph.edge_sets["red"]["odd"], [[33.]])
    self.assertLen(graph.edge_sets["purple"].features, 2 + new_output_feature)
    self.assertAllClose(graph.edge_sets["purple"][output_feature], [[14.]])
    if new_output_feature:
      self.assertAllClose(graph.edge_sets["purple"]["even"], [[8.]])
    self.assertAllClose(graph.edge_sets["purple"]["odd"], [[9.]])

  def testPoolForbidden(self):
    values = dict(nodes=tf.constant([[1.], [2.], [10.]]),
                  edges=tf.constant([[5.], [6.]]))
    graph = _make_test_graph_01into2(values)
    options = opt.GraphUpdateOptions()
    options.edge_set_default.update_fn_factory = fnn_factory.get_fnn_factory(
        output_dim=32, activation="relu", name="edge_set_update")
    with self.assertRaisesRegex(ValueError, r"does not expect Pool"):
      update = graph_update.EdgeSetUpdate(
          "edges", options=options,
          input_fns=[graph_ops.Readout(), graph_ops.Pool(const.CONTEXT, "sum")])
      _ = update(graph)

  def testUserDefinedInputAndTrainingMode(self):
    """Tests input with training=... that is not UpdateInputLayerExtended."""
    values = dict(edges=tf.constant([[1., 2.], [5., 6.]]))
    input_graph = _make_test_graph_01into2(values)
    l2 = 0.01
    update = graph_update.EdgeSetUpdate(
        "edges",
        input_fns=[DoubleInputFromEdge("edges",
                                       const.DEFAULT_STATE_NAME, l2=l2)],
        update_fn=tf.keras.layers.Dense(
            2, use_bias=False, name="update",
            kernel_initializer=tf.keras.initializers.Constant([[3., 0.],
                                                               [0., 3.]]),
            kernel_regularizer=tf.keras.regularizers.L2(l2)))
    graph = update(input_graph)
    self.assertAllClose(graph.edge_sets["edges"][const.DEFAULT_STATE_NAME],
                        [[6., 12.], [30., 36.]])  # 2x in input, 3x in update.
    # Variables and regularizers are tracked.
    self.assertAllClose(sorted(update.losses), [0.04, 2 * 0.09])
    self.assertSameElements([v.name for v in update.trainable_weights],
                            ["multiplier:0", "edge_set_update/update/kernel:0"])
    # Keras takes care to forward training=True to the inner layer
    # althout the outer layer (subject under test) does not mention it.
    graph = update(input_graph, training=True)
    self.assertAllClose(graph.edge_sets["edges"][const.DEFAULT_STATE_NAME],
                        [[12., 6.], [36., 30.]])  # Reversed.

  @parameterized.named_parameters(
      ("Baseline", None, None, None, [[3.], [3.]]),
      ("BaselineExplicit", [const.SOURCE, const.TARGET], False, False,
       [[3.], [3.]]),
      ("SourceOnly", [const.SOURCE], False, False, [[1.], [1.]]),
      ("TargetOnly", [const.TARGET], False, False, [[2.], [2.]]),
      ("Recurrent", None, True, False, [[7.], [7.]]),
      ("RecurrentOnly", [], True, False, [[4.], [4.]]),
      ("Context", None, False, True, [[11.], [11.]]),
      ("ContextAndRecurrent", None, True, True, [[15.], [15.]]))
  def testDefaultInputs(self, use_node_tags, use_recurrent_state, use_context,
                        expected):
    values = dict(nodes=tf.constant([[1.], [1.], [2.]]),
                  edges=tf.constant([[4.], [4.]]),
                  context=tf.constant([[8.]]))
    input_graph = _make_test_graph_01into2(values)
    update_fn = tf.keras.layers.Dense(1, use_bias=False)
    options = opt.GraphUpdateOptions()
    options.edge_set_default.update_use_node_tags = use_node_tags
    options.edge_set_default.update_use_recurrent_state = use_recurrent_state
    options.edge_set_default.update_use_context = use_context
    update = graph_update.EdgeSetUpdate(
        "edges", update_fn=update_fn, options=options)
    _ = update(input_graph)  # Trigger building.
    kernel, = update_fn.trainable_weights  # Unpack.
    kernel.assign(tf.ones_like(kernel))  # Sum all inputs.
    graph = update(input_graph)
    self.assertAllClose(graph.edge_sets["edges"][const.DEFAULT_STATE_NAME],
                        expected)


def _make_test_graph_red_blue():
  return gt.GraphTensor.from_pieces(
      context=gt.Context.from_fields(
          features={"even": tf.constant([[2.]]),
                    "odd": tf.constant([[3.]])}),
      node_sets={
          "red": gt.NodeSet.from_fields(
              sizes=tf.constant([2]),
              features={"even": tf.constant([[4.], [16.]]),
                        "odd": tf.constant([[5.], [17.]])}),
          "blue": gt.NodeSet.from_fields(
              sizes=tf.constant([1]),
              features={"other": tf.constant([[64.]])})},
      edge_sets={
          "red": gt.EdgeSet.from_fields(
              sizes=tf.constant([1]),
              adjacency=adj.Adjacency.from_indices(
                  ("red", tf.constant([0])),
                  ("red", tf.constant([1]))),
              features={"even": tf.constant([[32.]]),
                        "odd": tf.constant([[33.]])}),
          "purple": gt.EdgeSet.from_fields(
              sizes=tf.constant([1]),
              adjacency=adj.Adjacency.from_indices(
                  ("red", tf.constant([0])),
                  ("blue", tf.constant([0]))),
              features={"even": tf.constant([[8.]]),
                        "odd": tf.constant([[9.]])})})


class DoubleInputFromEdge(tf.keras.layers.Layer):
  """Has a weight, does not implement UpdateInputLayerExtended, trains funny."""

  def __init__(self, edge_set_name, feature_name, l2=None, **kwargs):
    kwargs.setdefault("name", "double_input")
    super().__init__(**kwargs)
    self._multiplier = self.add_weight(
        name="multiplier", shape=[], trainable=True,
        initializer=tf.keras.initializers.Constant(2.),
        regularizer=tf.keras.regularizers.L2(l2) if l2 else None)
    self._edge_set_name = edge_set_name
    self._feature_name = feature_name

  def call(self, graph, training=None):
    feature = graph.edge_sets[self._edge_set_name][self._feature_name]
    if training:
      feature = tf.reverse(feature, axis=[1])
    return tf.multiply(self._multiplier, feature)


class NodeSetUpdateTest(tf.test.TestCase, parameterized.TestCase):
  """Tests NodeSetUpdate beyond _GraphPieceUpdateBase."""

  def testFromConfig(self):
    readout = graph_ops.Readout()
    combiner_fn = tf.keras.layers.Concatenate(name="test_concat")
    update_fn = tf.keras.layers.Dense(
        1, name="update_fn",
        kernel_initializer=tf.keras.initializers.Constant([[2.]]))
    kwargs = dict(node_set_name="nodes", input_fns=[readout],
                  combiner_fn=combiner_fn, update_fn=update_fn,
                  output_feature="out_feature", name="test_update")
    config = graph_update.NodeSetUpdate(**kwargs).get_config()
    self.assertDictContainsSubset(kwargs, config)
    update = graph_update.NodeSetUpdate.from_config(config)

    values = dict(nodes=tf.constant([[5.], [6.], [7.]]))
    input_graph = _make_test_graph_01into2(values)
    graph = update(input_graph)
    self.assertAllClose(graph.node_sets["nodes"]["out_feature"],
                        [[10.], [12.], [14.]])

  @parameterized.parameters(False, True)
  def testFeaturesPreservedExceptOutput(self, new_output_feature):
    input_graph = _make_test_graph_red_blue()
    output_feature = "new" if new_output_feature else "even"
    update = graph_update.NodeSetUpdate(
        "red",
        input_fns=[
            graph_ops.Readout(feature_name="even"),
            graph_ops.Pool(const.TARGET, "sum", edge_set_name="red",
                           feature_name="even"),
            graph_ops.Pool(const.SOURCE, "sum", edge_set_name="purple",
                           feature_name="even")],
        update_fn=tf.keras.layers.Dense(
            1, use_bias=False,
            kernel_initializer=tf.keras.initializers.Constant(
                [[1.], [1.], [1.]])),  # Sum the concatenated inputs.
        output_feature=output_feature)
    graph = update(input_graph)

    # Features are unchanged, except for the designated output_feature.
    self.assertLen(graph.context.features, 2)
    self.assertAllClose(graph.context["even"], [[2.]])
    self.assertAllClose(graph.context["odd"], [[3.]])
    self.assertLen(graph.node_sets, 2)
    self.assertLen(graph.node_sets["red"].features, 2 + new_output_feature)
    self.assertAllClose(graph.node_sets["red"][output_feature], [[12.], [48.]])
    if new_output_feature:
      self.assertAllClose(graph.node_sets["red"]["even"], [[4.], [16.]])
    self.assertAllClose(graph.node_sets["red"]["odd"], [[5.], [17.]])
    self.assertLen(graph.node_sets["blue"].features, 1)
    self.assertAllClose(graph.node_sets["blue"]["other"], [[64.]])
    self.assertLen(graph.edge_sets, 2)
    self.assertLen(graph.edge_sets["red"].features, 2)
    self.assertAllClose(graph.edge_sets["red"]["even"], [[32.]])
    self.assertAllClose(graph.edge_sets["red"]["odd"], [[33.]])
    self.assertLen(graph.edge_sets["purple"].features, 2)
    self.assertAllClose(graph.edge_sets["purple"]["even"], [[8.]])
    self.assertAllClose(graph.edge_sets["purple"]["odd"], [[9.]])

  def testBroadcastFromContext(self):
    values = dict(context=tf.constant([[42.]]))
    input_graph = _make_test_graph_01into2(values)
    update = graph_update.NodeSetUpdate(
        "nodes",
        input_fns=[graph_ops.Broadcast(const.CONTEXT)],
        update_fn=tf.keras.layers.Dense(
            1, use_bias=False, name="update",
            kernel_initializer=tf.keras.initializers.Constant([[1.]])))
    graph = update(input_graph)
    self.assertAllClose(graph.node_sets["nodes"][const.DEFAULT_STATE_NAME],
                        [[42.]]*3)

  def testPoolFromContextForbidden(self):
    values = dict(nodes=tf.constant([[1.], [2.], [10.]]),
                  context=tf.constant([[42.]]))
    graph = _make_test_graph_01into2(values)
    options = opt.GraphUpdateOptions()
    options.node_set_default.update_fn_factory = fnn_factory.get_fnn_factory(
        output_dim=32, activation="relu", name="edge_set_update")
    with self.assertRaisesRegex(ValueError, r"does not expect Pool.*CONTEXT"):
      update = graph_update.NodeSetUpdate(
          "nodes", options=options,
          input_fns=[graph_ops.Readout(), graph_ops.Pool(const.CONTEXT, "sum")])
      _ = update(graph)

  def testUserDefinedInputAndTrainingMode(self):
    """Tests input with training=... that is not UpdateInputLayerExtended."""
    values = dict(nodes=tf.constant([[1., 2.], [3., 4.], [5., 6.]]))
    input_graph = _make_test_graph_01into2(values)
    l2 = 0.01
    update = graph_update.NodeSetUpdate(
        "nodes",
        input_fns=[DoubleInputFromNode("nodes",
                                       const.DEFAULT_STATE_NAME, l2=l2)],
        update_fn=tf.keras.layers.Dense(
            2, use_bias=False, name="update",
            kernel_initializer=tf.keras.initializers.Constant([[3., 0.],
                                                               [0., 3.]]),
            kernel_regularizer=tf.keras.regularizers.L2(l2)))
    graph = update(input_graph)
    self.assertAllClose(graph.node_sets["nodes"][const.DEFAULT_STATE_NAME],
                        # 2x in input, 3x in update.
                        [[6., 12.], [18., 24.], [30., 36.]])
    # Variables and regularizers are tracked.
    self.assertAllClose(sorted(update.losses), [0.04, 2 * 0.09])
    self.assertSameElements([v.name for v in update.trainable_weights],
                            ["multiplier:0", "node_set_update/update/kernel:0"])
    # Keras takes care to forward training=True to the inner layer
    # althout the outer layer (subject under test) does not mention it.
    graph = update(input_graph, training=True)
    self.assertAllClose(graph.node_sets["nodes"][const.DEFAULT_STATE_NAME],
                        [[12., 6.], [24., 18.], [36., 30.]])  # Reversed.

  @parameterized.named_parameters(
      ("Default", None, None, None, None, [[2.], [4.], [8.+16.+32.]]),
      ("DefaultContext", None, True, None, None, [[3.], [5.], [8.+16.+32.+1]]),
      ("DefaultRecurrent", None, None, True, None, [[2.], [4.], [8.+16.+32.]]),
      ("DefaultNonRecurrent", None, None, False, None, [[0.], [0.], [16.+32.]]),
      ("Target", [const.TARGET], False, None, None, [[2.], [4.], [8.+16.+32.]]),
      ("TargetContext", [const.TARGET], True, None, None,
       [[3.], [5.], [8.+16.+32.+1]]),
      ("Source", [const.SOURCE], False, None, None, [[2.+16.], [4.+32.], [8.]]),
      ("SourceContext", [const.SOURCE], True, None, None,
       [[2.+16.+1.], [4.+32.+1.], [9.]]),
      ("SourceTarget", [const.SOURCE, const.TARGET], False, None, None,
       [[2.+16], [4.+32.], [8.+16.+32]]),
      ("SourceTargetAlways", [const.SOURCE, const.TARGET], False, None, ALWAYS,
       [[2.+16], [4.+32.], [8.+16.+32]]),
      ("SourceTargetNever", [const.SOURCE, const.TARGET], False, None, NEVER,
       [[2.], [4.], [8.]]),
      ("Empty", [], False, None, None, [[2.], [4.], [8.]]))
  def testHomogeneousDefaultInputs(
      self, pool_tags, use_context, use_recurrent_state, enable_edges,
      expected):
    """Tests node_pool_tags, use_context, and the default reduce_type "sum"."""
    values = dict(context=tf.constant([[1.]]),
                  nodes=tf.constant([[2.], [4.], [8.]]),
                  edges=tf.constant([[16.], [32.]]))
    input_graph = _make_test_graph_01into2(values)
    update_fn = tf.keras.layers.Dense(1, use_bias=False)
    options = opt.GraphUpdateOptions(graph_tensor_spec=input_graph.spec)
    options.node_set_default.update_use_recurrent_state = use_recurrent_state
    options.node_set_default.update_use_context = use_context
    options.edge_set_default.node_pool_enabled = enable_edges
    options.edge_set_default.node_pool_tags = pool_tags
    update = graph_update.NodeSetUpdate(
        "nodes", update_fn=update_fn, options=options)
    _ = update(input_graph)  # Trigger building.
    kernel, = update_fn.trainable_weights  # Unpack.
    kernel.assign(tf.ones_like(kernel))  # Sum all inputs.
    graph = update(input_graph)
    self.assertAllClose(graph.node_sets["nodes"][const.DEFAULT_STATE_NAME],
                        expected)

  @parameterized.parameters(
      ("sum", [[2.], [4.], [8. + 16. + 32.]]),
      ("mean", [[2.], [4.], [8. + (16. + 32.)/2]]))
  def testDefaultInputFactory(self, reduce_type, expected):
    """Tests node_pool_factory."""
    values = dict(nodes=tf.constant([[2.], [4.], [8.]]),
                  edges=tf.constant([[16.], [32.]]))
    input_graph = _make_test_graph_01into2(values)
    update_fn = tf.keras.layers.Dense(1, use_bias=False)
    options = opt.GraphUpdateOptions(graph_tensor_spec=input_graph.spec)
    options.edge_set_default.node_pool_factory = (
        # pylint: disable=g-long-lambda
        lambda tag, edge_set_name: graph_ops.Pool(tag, reduce_type,
                                                  edge_set_name=edge_set_name))
    update = graph_update.NodeSetUpdate(
        "nodes", update_fn=update_fn, options=options)
    _ = update(input_graph)  # Trigger building.
    kernel, = update_fn.trainable_weights  # Unpack.
    kernel.assign(tf.ones_like(kernel))  # Sum all inputs.
    graph = update(input_graph)
    self.assertAllClose(graph.node_sets["nodes"][const.DEFAULT_STATE_NAME],
                        expected)

  def testHeterogeneousDefaultInputs(self):
    """Tests selection of incident edge sets."""
    # Construct a graph with singleton node sets "a" and "b",
    # and between them singleton edge sets with distinct power-of-two values
    # for all combinations of source, target, and node_pool_tags of size 1.
    options = opt.GraphUpdateOptions()
    node_sets = {}
    node_value_a = 1.
    for name, value in [("a", node_value_a), ("b", 100.)]:
      node_sets[name] = gt.NodeSet.from_fields(
          sizes=tf.constant([1]),
          features={const.DEFAULT_STATE_NAME: tf.constant([[value]])})
    edge_sets = {}
    edge_set_values = {}
    value = 2.
    for org in ["a", "b"]:
      for dst in ["a", "b"]:
        for node_pool_tags, short_tags in [([const.SOURCE], "src"),
                                           ([const.TARGET], "trg")]:
          edge_set_name = f"{org}2{dst}@{short_tags}"
          options.edge_sets[edge_set_name].node_pool_tags = node_pool_tags
          edge_sets[edge_set_name] = gt.EdgeSet.from_fields(
              sizes=tf.constant([1]),
              adjacency=adj.Adjacency.from_indices((org, tf.constant([0])),
                                                   (dst, tf.constant([0]))),
              features={const.DEFAULT_STATE_NAME: tf.constant([[value]])})
          edge_set_values[edge_set_name] = value
          value *= 2.
    self.assertEqual(value, 2**9)
    input_graph = gt.GraphTensor.from_pieces(node_sets=node_sets,
                                             edge_sets=edge_sets)
    options.graph_tensor_spec = input_graph.spec

    # Pooling at "a" is expected to get the values from exactly these edge sets.
    expected_edge_sets = ["a2a@src", "a2a@trg", "a2b@src", "b2a@trg"]

    update_fn = tf.keras.layers.Dense(1, use_bias=False)
    update = graph_update.NodeSetUpdate(
        "a", update_fn=update_fn, options=options)
    _ = update(input_graph)  # Trigger building.
    kernel, = update_fn.trainable_weights  # Unpack.
    kernel.assign(tf.ones_like(kernel))  # Sum all inputs.
    graph = update(input_graph)
    self.assertAllClose(
        graph.node_sets["a"][const.DEFAULT_STATE_NAME],
        [[node_value_a + sum(edge_set_values[k] for k in expected_edge_sets)]])


class DoubleInputFromNode(tf.keras.layers.Layer):
  """Has a weight, does not implement UpdateInputLayerExtended, trains funny."""

  def __init__(self, node_set_name, feature_name, l2=None, **kwargs):
    kwargs.setdefault("name", "double_input")
    super().__init__(**kwargs)
    self._multiplier = self.add_weight(
        name="multiplier", shape=[], trainable=True,
        initializer=tf.keras.initializers.Constant(2.),
        regularizer=tf.keras.regularizers.L2(l2) if l2 else None)
    self._node_set_name = node_set_name
    self._feature_name = feature_name

  def call(self, graph, training=None):
    feature = graph.node_sets[self._node_set_name][self._feature_name]
    if training:
      feature = tf.reverse(feature, axis=[1])
    return tf.multiply(self._multiplier, feature)


class ContextUpdateTest(tf.test.TestCase, parameterized.TestCase):
  """Tests ContextUpdate beyond _GraphPieceUpdateBase."""

  def testFromConfig(self):
    readout = graph_ops.Readout()
    combiner_fn = tf.keras.layers.Concatenate(name="test_concat")
    update_fn = tf.keras.layers.Dense(
        1, name="update_fn",
        kernel_initializer=tf.keras.initializers.Constant([[2.]]))
    kwargs = dict(input_fns=[readout],
                  combiner_fn=combiner_fn, update_fn=update_fn,
                  output_feature="out_feature", name="test_update")
    config = graph_update.ContextUpdate(**kwargs).get_config()
    self.assertDictContainsSubset(kwargs, config)
    update = graph_update.ContextUpdate.from_config(config)

    values = dict(context=tf.constant([[5.]]))
    input_graph = _make_test_graph_01into2(values)
    graph = update(input_graph)
    self.assertAllClose(graph.context["out_feature"], [[10.]])

  @parameterized.parameters(False, True)
  def testFeaturesPreservedExceptOutput(self, new_output_feature):
    input_graph = _make_test_graph_red_blue()
    output_feature = "new" if new_output_feature else "even"
    update = graph_update.ContextUpdate(
        input_fns=[
            graph_ops.Readout(feature_name="even"),
            graph_ops.Pool(const.CONTEXT, "sum", node_set_name="red",
                           feature_name="even"),
            graph_ops.Pool(const.CONTEXT, "sum", edge_set_name="purple",
                           feature_name="even")],
        update_fn=tf.keras.layers.Dense(
            1, use_bias=False,
            kernel_initializer=tf.keras.initializers.Constant(
                [[1.], [1.], [1.]])),  # Sum the concatenated inputs.
        output_feature=output_feature)
    graph = update(input_graph)

    # Features are unchanged, except for the designated output_feature.
    self.assertLen(graph.context.features, 2 + new_output_feature)
    self.assertAllClose(graph.context[output_feature], [[30.]])
    if new_output_feature:
      self.assertAllClose(graph.context["even"], [[2.]])
    self.assertAllClose(graph.context["odd"], [[3.]])
    self.assertLen(graph.node_sets, 2)
    self.assertLen(graph.node_sets["red"].features, 2)
    self.assertAllClose(graph.node_sets["red"]["even"], [[4.], [16.]])
    self.assertAllClose(graph.node_sets["red"]["odd"], [[5.], [17.]])
    self.assertLen(graph.node_sets["blue"].features, 1)
    self.assertAllClose(graph.node_sets["blue"]["other"], [[64.]])
    self.assertLen(graph.edge_sets, 2)
    self.assertLen(graph.edge_sets["red"].features, 2)
    self.assertAllClose(graph.edge_sets["red"]["even"], [[32.]])
    self.assertAllClose(graph.edge_sets["red"]["odd"], [[33.]])
    self.assertLen(graph.edge_sets["purple"].features, 2)
    self.assertAllClose(graph.edge_sets["purple"]["even"], [[8.]])
    self.assertAllClose(graph.edge_sets["purple"]["odd"], [[9.]])

  def testBroadcastAndPoolSourceTargetForbidden(self):
    values = dict(nodes=tf.constant([[1.], [2.], [10.]]),
                  edges=tf.constant([[3.], [4.]]),
                  context=tf.constant([[42.]]))
    graph = _make_test_graph_01into2(values)
    options = opt.GraphUpdateOptions()
    options.context.update_fn_factory = fnn_factory.get_fnn_factory(
        output_dim=32, activation="relu", name="edge_set_update")
    with self.assertRaisesRegex(ValueError, r"does not expect Broadcast"):
      update = graph_update.ContextUpdate(
          options=options,
          input_fns=[graph_ops.Readout(), graph_ops.Broadcast(const.CONTEXT)])
      _ = update(graph)
    for tag in [const.SOURCE, const.TARGET]:
      with self.assertRaisesRegex(ValueError, r"expects Pool.*CONTEXT"):
        update = graph_update.ContextUpdate(
            options=options,
            input_fns=[graph_ops.Readout(), graph_ops.Pool(tag, "sum")])
        _ = update(graph)

  def testUserDefinedInputAndTrainingMode(self):
    """Tests input with training=... that is not UpdateInputLayerExtended."""
    values = dict(context=tf.constant([[1., 2.]]))
    input_graph = _make_test_graph_01into2(values)
    l2 = 0.01
    update = graph_update.ContextUpdate(
        input_fns=[DoubleInputFromContext(const.DEFAULT_STATE_NAME, l2=l2)],
        update_fn=tf.keras.layers.Dense(
            2, use_bias=False, name="update",
            kernel_initializer=tf.keras.initializers.Constant([[3., 0.],
                                                               [0., 3.]]),
            kernel_regularizer=tf.keras.regularizers.L2(l2)))
    graph = update(input_graph)
    self.assertAllClose(graph.context[const.DEFAULT_STATE_NAME],
                        [[6., 12.]])  # 2x in input, 3x in update.
    # Variables and regularizers are tracked.
    self.assertAllClose(sorted(update.losses), [0.04, 2 * 0.09])
    self.assertSameElements([v.name for v in update.trainable_weights],
                            ["multiplier:0", "context_update/update/kernel:0"])
    # Keras takes care to forward training=True to the inner layer
    # althout the outer layer (subject under test) does not mention it.
    graph = update(input_graph, training=True)
    self.assertAllClose(graph.context[const.DEFAULT_STATE_NAME],
                        [[12., 6.]])  # Reversed.

  @parameterized.named_parameters(
      ("Default", None, None, None, [[1. + (2.+4.+8.)]]),
      ("DefaultExplicit", True, ALWAYS, NEVER, [[1. + (2.+4.+8.)]]),
      ("NotRecurrent", False, None, None, [[(2.+4.+8.)]]),
      ("WithNeither", None, NEVER, NEVER, [[1.]]),
      ("WithNodes", None, ALWAYS, NEVER, [[1. + (2.+4.+8.)]]),
      ("WithEdges", None, NEVER, ALWAYS, [[1. + (16.+32.)]]),
      ("WithBoth", None, ALWAYS, ALWAYS, [[1. + (2.+4.+8.) + (16.+32.)]]),
  )
  def testDefaultInputs(
      self, use_recurrent_state, enable_nodes, enable_edges, expected):
    """Tests use of node and edge sets, and the default reduce_type "sum"."""
    values = dict(context=tf.constant([[1.]]),
                  nodes=tf.constant([[2.], [4.], [8.]]),
                  edges=tf.constant([[16.], [32.]]))
    input_graph = _make_test_graph_01into2(values)
    options = opt.GraphUpdateOptions(graph_tensor_spec=input_graph.spec)
    options.context.update_use_recurrent_state = use_recurrent_state
    options.node_set_default.context_pool_enabled = enable_nodes
    options.edge_set_default.context_pool_enabled = enable_edges
    update_fn = tf.keras.layers.Dense(1, use_bias=False)
    update = graph_update.ContextUpdate(update_fn=update_fn, options=options)
    _ = update(input_graph)  # Trigger building.
    kernel, = update_fn.trainable_weights  # Unpack.
    kernel.assign(tf.ones_like(kernel))  # Sum all inputs.
    graph = update(input_graph)
    self.assertAllClose(graph.context[const.DEFAULT_STATE_NAME], expected)

  @parameterized.parameters(
      ("sum", "sum", [[1. + (6.+24.+96.) + (128.+256.)]]),
      ("mean", "sum", [[1. + (6.+24.+96.)/3. + (128.+256.)]]),
      ("sum", "mean", [[1. + (6.+24.+96.) + (128.+256)/2.]]))
  def testDefaultInputFactory(self, node_reduce_type, edge_reduce_type,
                              expected):
    """Tests context_pool_factory of node sets and edge sets."""
    values = dict(context=tf.constant([[1.]]),
                  nodes=tf.constant([[6.], [24.], [96.]]),
                  edges=tf.constant([[128.], [256.]]))
    input_graph = _make_test_graph_01into2(values)
    update_fn = tf.keras.layers.Dense(1, use_bias=False)
    options = opt.GraphUpdateOptions(graph_tensor_spec=input_graph.spec)
    # Pooling nodes to context is enabled by default.
    # pylint: disable=g-long-lambda
    options.node_set_default.context_pool_factory = (
        lambda node_set_name: graph_ops.Pool(const.CONTEXT, node_reduce_type,
                                             node_set_name=node_set_name))
    options.edge_set_default.context_pool_enabled = ALWAYS
    options.edge_set_default.context_pool_factory = (
        lambda edge_set_name: graph_ops.Pool(const.CONTEXT, edge_reduce_type,
                                             edge_set_name=edge_set_name))
    update = graph_update.ContextUpdate(update_fn=update_fn, options=options)
    _ = update(input_graph)  # Trigger building.
    kernel, = update_fn.trainable_weights  # Unpack.
    kernel.assign(tf.ones_like(kernel))  # Sum all inputs.
    graph = update(input_graph)
    self.assertAllClose(graph.context[const.DEFAULT_STATE_NAME],
                        expected)

  def testReadoutFirstNode(self):
    """Tests input from ReadoutFirstNode."""
    values = dict(context=tf.constant([[1.]]),
                  nodes=tf.constant([[2.], [4.], [8.]]))
    input_graph = _make_test_graph_01into2(values)
    update = graph_update.ContextUpdate(
        input_fns=[graph_ops.ReadoutFirstNode(node_set_name="nodes")],
        update_fn=tf.keras.layers.Dense(
            1, use_bias=False, name="update",
            kernel_initializer=tf.keras.initializers.Constant([[10.]])))
    graph = update(input_graph)
    self.assertAllClose(graph.context[const.DEFAULT_STATE_NAME], [[20.]])


class DoubleInputFromContext(tf.keras.layers.Layer):
  """Has a weight, does not implement UpdateInputLayerExtended, trains funny."""

  def __init__(self, feature_name, l2=None, **kwargs):
    kwargs.setdefault("name", "double_input")
    super().__init__(**kwargs)
    self._multiplier = self.add_weight(
        name="multiplier", shape=[], trainable=True,
        initializer=tf.keras.initializers.Constant(2.),
        regularizer=tf.keras.regularizers.L2(l2) if l2 else None)
    self._feature_name = feature_name

  def call(self, graph, training=None):
    feature = graph.context[self._feature_name]
    if training:
      feature = tf.reverse(feature, axis=[1])
    return tf.multiply(self._multiplier, feature)


class GraphUpdateTest(tf.test.TestCase, parameterized.TestCase):
  """Tests GraphUpdate."""

  @parameterized.named_parameters(
      ("Basic", False, False),
      ("WithContext", False, True),
      ("FromConfig", True, True))
  def testFullGraphUpdate(self, from_config, update_context):
    """Tests update of everything if no node sets no edge sets are given."""
    input_graph = _make_test_graph_with_singleton_node_sets(
        [("u", [1.]), ("v", [2.]), ("w", [4.])],
        [("u", "u", [0.]), ("u", "v", [0.]), ("u", "w", [0.])])
    options = opt.GraphUpdateOptions(graph_tensor_spec=input_graph.spec,
                                     update_context=update_context)
    # Edge set updates just copy the source value.
    # pylint: disable=g-long-lambda
    options.edge_set_default.update_use_node_tags = [const.SOURCE]
    options.edge_set_default.update_fn_factory = (
        lambda: tf.keras.layers.Dense(
            1, kernel_initializer=tf.keras.initializers.Ones(), use_bias=False))
    # Node set updates just add the value coming in from the one edge set.
    options.node_set_default.update_fn_factory = (
        lambda: tf.keras.layers.Dense(
            1, kernel_initializer=tf.keras.initializers.Ones(), use_bias=False))
    options.context.update_use_recurrent_state = False
    if update_context:
      options.context.update_fn_factory = (
          lambda: tf.keras.layers.Dense(
              1, kernel_initializer=tf.keras.initializers.Ones(),
              use_bias=False))
    else:
      options.context.update_fn_factory = (
          lambda: self.fail("Unexpected call to context.update_fn_factory"))

    update = graph_update.GraphUpdate(options=options)
    if from_config:
      update = graph_update.GraphUpdate.from_config(update.get_config())
    graph = update(input_graph)

    def _node_states(node_set_name):
      return graph.node_sets[node_set_name][const.DEFAULT_STATE_NAME]
    # The changed node state shows that the NodeSetUpdates correctly read
    # edge states *after* EdgeSetUpdates, because they had state zero before.
    self.assertAllClose(_node_states("u"), [[1. + 1.]])
    self.assertAllClose(_node_states("v"), [[2. + 1.]])
    self.assertAllClose(_node_states("w"), [[4. + 1.]])
    if update_context:
      # A context state of 10 (not 6 = 1+2+4) shows the ContextUpdate correctly
      # read the node states *after* NodeSetUpdates.
      self.assertAllClose(graph.context[const.DEFAULT_STATE_NAME], [[10.]])
    else:
      self.assertEmpty(graph.context.features)

  @parameterized.named_parameters(
      ("All", ALWAYS, ALWAYS, [[16. + 4. + 1. + 18. + 4. + 8.]]),
      ("AllEdges", ALWAYS, NEVER, [[16. + 4.]]),
      ("UpdatedEdges", ON_UPDATE, NEVER, [[4.]]),
      ("AllNodes", NEVER, ALWAYS, [[1. + 18. + 4. + 8.]]),
      ("UpdatedNodes", NEVER, ON_UPDATE, [[18.]]),
      ("Default", None, None, [[18.]]),
  )
  def testFullyExplicitAndContextInputs(
      self, context_enable_edges, context_enable_nodes, expected_context):
    """Tests update of explicitly given node sets and edge sets."""
    input_graph = _make_test_graph_with_singleton_node_sets(
        [("u", [1.]), ("v", [2.]), ("w", [4.]), ("x", [8.])],
        [("u", "v", [16.]), ("w", "x", [32.])])
    options = opt.GraphUpdateOptions(graph_tensor_spec=input_graph.spec)
    # Edge set updates just copy the source value, discarding the old state.
    # pylint: disable=g-long-lambda
    options.edge_set_default.update_use_node_tags = [const.SOURCE]
    options.edge_set_default.update_fn_factory = (
        lambda: tf.keras.layers.Dense(
            1, kernel_initializer=tf.keras.initializers.Ones(), use_bias=False))
    # Node set updates just add the value coming in from the one edge set.
    options.edge_set_default.node_pool_enabled = ALWAYS
    options.node_set_default.update_fn_factory = (
        lambda: tf.keras.layers.Dense(
            1, kernel_initializer=tf.keras.initializers.Ones(), use_bias=False))
    # Context updates sum everything that's enabled.
    options.update_context = True
    options.edge_set_default.context_pool_enabled = context_enable_edges
    options.node_set_default.context_pool_enabled = context_enable_nodes
    options.context.update_use_recurrent_state = False
    options.context.update_fn_factory = (
        lambda: tf.keras.layers.Dense(
            1, kernel_initializer=tf.keras.initializers.Ones(), use_bias=False))

    update = graph_update.GraphUpdate(node_set_names=["v"],
                                      edge_set_names=["w->x"],  # Unrelated.
                                      options=options)
    graph = update(input_graph)

    def _edge_states(edge_set_name):
      return graph.edge_sets[edge_set_name][const.DEFAULT_STATE_NAME]
    self.assertAllClose(_edge_states("u->v"), [[16.]])  # Unchanged.
    self.assertAllClose(_edge_states("w->x"), [[4.]])  # From "w".

    def _node_states(node_set_name):
      return graph.node_sets[node_set_name][const.DEFAULT_STATE_NAME]
    self.assertAllClose(_node_states("u"), [[1.]])  # Unchanged.
    self.assertAllClose(_node_states("v"), [[2. + 16.]])  # From "u->v".
    self.assertAllClose(_node_states("w"), [[4.]])  # Unchanged.
    self.assertAllClose(_node_states("x"), [[8.]])  # Unchanged.

    self.assertAllClose(graph.context[const.DEFAULT_STATE_NAME],
                        expected_context)

  @parameterized.named_parameters(
      ("NoEdges", NEVER, [[4.]]),
      ("AllEdges", ALWAYS, [[4. + 1. + 32.]]),
      ("UpdatedEdges", ON_UPDATE, [[4. + 1.]]),
      ("Default", None, [[4 + 1.]]))
  def testExplicitEdgesAndNodeInputs(self, enable_edges, expected):
    """Tests update with explicit edge sets and inferred node sets."""
    input_graph = _make_test_graph_with_singleton_node_sets(
        [("u", [1.]), ("v", [2.]), ("w", [4.]), ("x", [8.])],
        [("u", "w", [16.]), ("v", "w", [32.]), ("v", "x", [64.])])
    options = opt.GraphUpdateOptions(graph_tensor_spec=input_graph.spec)
    # Edge set updates just copy the source value, discarding the old state.
    # pylint: disable=g-long-lambda
    options.edge_set_default.update_use_node_tags = [const.SOURCE]
    options.edge_set_default.update_fn_factory = (
        lambda: tf.keras.layers.Dense(
            1, kernel_initializer=tf.keras.initializers.Ones(), use_bias=False))
    # Node set updates just add the value coming in from the one edge set.
    options.edge_set_default.node_pool_enabled = enable_edges
    options.node_set_default.update_fn_factory = (
        lambda: tf.keras.layers.Dense(
            1, kernel_initializer=tf.keras.initializers.Ones(), use_bias=False))

    update = graph_update.GraphUpdate(edge_set_names=["u->w"], options=options)
    graph = update(input_graph)

    def _edge_states(edge_set_name):
      return graph.edge_sets[edge_set_name][const.DEFAULT_STATE_NAME]
    self.assertAllClose(_edge_states("u->w"), [[1.]])
    self.assertAllClose(_edge_states("v->w"), [[32.]])  # Unchanged.
    self.assertAllClose(_edge_states("v->x"), [[64.]])  # Unchanged.

    def _node_states(node_set_name):
      return graph.node_sets[node_set_name][const.DEFAULT_STATE_NAME]
    self.assertAllClose(_node_states("u"), [[1.]])  # Unchanged.
    self.assertAllClose(_node_states("v"), [[2.]])  # Unchanged.
    self.assertAllClose(_node_states("w"), expected)
    self.assertAllClose(_node_states("x"), [[8.]])  # Unchanged.

  def testExplicitNodes(self):
    """Tests update with explicit node sets and inferred edge sets."""
    input_graph = _make_test_graph_with_singleton_node_sets(
        [("u", [1.]), ("v", [2.]), ("w", [4.]), ("x", [8.])],
        [("u", "w", [16.]), ("v", "w", [32.]), ("v", "x", [64.])])
    options = opt.GraphUpdateOptions(graph_tensor_spec=input_graph.spec)
    # Edge set updates just copy the source value, discarding the old state.
    # pylint: disable=g-long-lambda
    options.edge_set_default.update_use_node_tags = [const.SOURCE]
    options.edge_set_default.update_fn_factory = (
        lambda: tf.keras.layers.Dense(
            1, kernel_initializer=tf.keras.initializers.Ones(), use_bias=False))
    # Node set updates just add the value coming in from the one edge set.
    options.node_set_default.update_fn_factory = (
        lambda: tf.keras.layers.Dense(
            1, kernel_initializer=tf.keras.initializers.Ones(),
            bias_initializer=tf.keras.initializers.Constant([100.])))

    update = graph_update.GraphUpdate(node_set_names=["v", "w"],
                                      options=options)
    graph = update(input_graph)

    def _edge_states(edge_set_name):
      return graph.edge_sets[edge_set_name][const.DEFAULT_STATE_NAME]
    self.assertAllClose(_edge_states("u->w"), [[1.]])
    self.assertAllClose(_edge_states("v->w"), [[2.]])
    self.assertAllClose(_edge_states("v->x"), [[64.]])  # Unchanged.

    def _node_states(node_set_name):
      return graph.node_sets[node_set_name][const.DEFAULT_STATE_NAME]
    self.assertAllClose(_node_states("u"), [[1.]])  # Unchanged.
    self.assertAllClose(_node_states("v"), [[102.]])
    self.assertAllClose(_node_states("w"), [[107.]])
    self.assertAllClose(_node_states("x"), [[8.]])  # Unchanged.


def _make_test_graph_with_singleton_node_sets(nodes, edges):
  """Returns graph with singleton node sets and edge sets of given values."""
  # pylint: disable=g-complex-comprehension
  return gt.GraphTensor.from_pieces(
      node_sets={
          name: gt.NodeSet.from_fields(
              sizes=tf.constant([1]),
              features={const.DEFAULT_STATE_NAME: tf.constant([value])})
          for name, value in nodes},
      edge_sets={
          f"{src}->{dst}": gt.EdgeSet.from_fields(
              sizes=tf.constant([1]),
              adjacency=adj.Adjacency.from_indices(
                  (src, tf.constant([0])),
                  (dst, tf.constant([0]))),
              features={const.DEFAULT_STATE_NAME: tf.constant([value])})
          for src, dst, value in edges})


if __name__ == "__main__":
  tf.test.main()
