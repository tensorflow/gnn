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
"""Tests for graph_ops Keras layers."""
from unittest import mock

from absl.testing import parameterized
from packaging import version
import tensorflow as tf

from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.keras.layers import graph_ops


class ReadoutTest(tf.test.TestCase, parameterized.TestCase):

  def testFeatureName(self):
    red_values = tf.constant([[11., 12.]])
    blue_values = tf.constant([[21., 22.]])
    default_values = tf.constant([[31., 32.]])
    graph = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(
            features={"red": red_values, "blue": blue_values,
                      const.HIDDEN_STATE: default_values}))

    readout = graph_ops.Readout(from_context=True, feature_name="red")
    self.assertEqual("red", readout.feature_name)
    self.assertAllEqual(red_values, readout(graph))

    with self.assertRaisesRegex(ValueError, r"initialized .* but called with"):
      readout(graph, feature_name="blue")

    self.assertAllEqual(default_values,
                        graph_ops.Readout(from_context=True)(graph))

  @parameterized.parameters("context", "nodes", "edges")
  def testFeatureLocation(self, origin):
    values = dict(
        context=tf.constant([[1.0, 1.5]]),
        nodes=tf.constant([[11.0 + k, 11.5 + k] for k in range(3)]),
        edges=tf.constant([[21.0 + k, 21.5 + k] for k in range(4)]))
    graph = self._make_test_graph_134(values)
    value = values[origin]
    location_kwarg = dict(context=dict(from_context=True),
                          nodes=dict(node_set_name="nodes"),
                          edges=dict(edge_set_name="edges"))[origin]

    readout = graph_ops.Readout(feature_name="value", **location_kwarg)
    self.assertEqual(location_kwarg, readout.location)
    self.assertAllEqual(value, readout(graph))
    self.assertAllEqual(value, readout(graph, feature_name="value"))
    self.assertAllEqual(value, readout(graph, **location_kwarg))
    self.assertAllEqual(value, readout(graph, feature_name="value",
                                       **location_kwarg))

    readout = graph_ops.Readout(**location_kwarg)
    self.assertEqual(location_kwarg, readout.location)
    self.assertAllEqual(value, readout(graph, feature_name="value"))

    readout = graph_ops.Readout(feature_name="value")
    self.assertEqual({}, readout.location)
    self.assertAllEqual(value, readout(graph, **location_kwarg))

    readout = graph_ops.Readout()
    self.assertEqual({}, readout.location)
    self.assertAllEqual(value,
                        readout(graph, feature_name="value", **location_kwarg))

  @parameterized.named_parameters(
      ("Nodes", dict(node_set_name="nodes")),
      ("Edges", dict(edge_set_name="edges")),
      ("Context", dict(from_context=True)))
  def testConflictingFeatureLocation(self, location_kwarg):
    graph = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(features={"value": tf.constant([[0.]])}))
    with self.assertRaisesRegex(ValueError, r"initialized .* but called with"):
      readout = graph_ops.Readout(node_set_name="wronk", feature_name="value")
      _ = readout(graph, feature_name="value", **location_kwarg)

  @parameterized.named_parameters(
      ("ContextAndNodes", dict(from_context=True, node_set_name="nodes")),
      ("ContextAndEdges", dict(from_context=True, edge_set_name="edges")),
      ("NodesAndEdges", dict(node_set_name="nodes", edge_set_name="edges")),
      ("AllThree", dict(from_context=True, node_set_name="nodes",
                        edge_set_name="edges")))
  def testTooManyFeatureLocations(self, location_kwargs):
    graph = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(features={"value": tf.constant([[0.]])}))
    with self.assertRaisesRegex(ValueError, "at most one of"):
      graph_ops.Readout(**location_kwargs)
    with self.assertRaisesRegex(ValueError, "at most one of"):
      graph_ops.Readout()(graph, **location_kwargs)

  def testNoFeatureLocation(self):
    graph = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(features={"value": tf.constant([[0.]])}))
    with self.assertRaisesRegex(ValueError, "requires one of"):
      graph_ops.Readout(feature_name="value")(graph)

  @parameterized.parameters("context", "nodes", "edges")
  def testFromConfig(self, location):
    values = dict(
        context=tf.constant([[1.0, 1.5]]),
        nodes=tf.constant([[11.0 + v, 11.5 + v] for v in range(3)]),
        edges=tf.constant([[21.0 + v, 21.5 + v] for v in range(4)]))
    graph = self._make_test_graph_134(values)
    value = values[location]
    location_kwarg = dict(context=dict(from_context=True),
                          nodes=dict(node_set_name="nodes"),
                          edges=dict(edge_set_name="edges"))[location]
    kwargs = dict(location_kwarg, feature_name="value", name="test_readout")
    config = graph_ops.Readout(**kwargs).get_config()
    self.assertDictContainsSubset(kwargs, config)

    readout = graph_ops.Readout.from_config(config)
    self.assertEqual("value", readout.feature_name)
    self.assertEqual(location_kwarg, readout.location)
    self.assertAllEqual(value, readout(graph))

  def _make_test_graph_134(self, values):
    graph = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(features={"value": values["context"]}),
        node_sets={"nodes": gt.NodeSet.from_fields(
            sizes=tf.constant([3]), features={"value": values["nodes"]})},
        edge_sets={"edges": gt.EdgeSet.from_fields(
            sizes=tf.constant([4]),
            features={"value": values["edges"]},
            adjacency=adj.Adjacency.from_indices(   # 0 <-> 1 <-> 2.
                ("nodes", tf.constant([0, 1, 1, 2])),
                ("nodes", tf.constant([1, 0, 2, 1]))))})
    return graph


class ReadoutFirstNodeTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("Dense", "dense", [[11.], [13.]]),
      ("Ragged", "ragged", [[110., 111.], [130.]]))
  def testFeatureName(self, feature_name, expected):
    # TODO(b/266817638): Remove when fixed
    if version.parse(tf.__version__) < version.parse(
        "2.11.0"
    ) and feature_name == "ragged":
      self.skipTest("Bad Test")

    graph = self._make_test_graph_22()

    readout = graph_ops.ReadoutFirstNode(node_set_name="nodes",
                                         feature_name=feature_name)
    self.assertEqual(feature_name, readout.feature_name)
    self.assertAllEqual(expected, readout(graph))

    with self.assertRaisesRegex(ValueError, r"initialized .* but called with"):
      readout(graph, feature_name="other")

  def testFeatureNameDefault(self):
    graph = self._make_test_graph_22()
    self.assertAllEqual(
        [[1.], [3.]],
        graph_ops.ReadoutFirstNode(node_set_name="nodes")(graph))

  def testFeatureLocation(self):
    graph = self._make_test_graph_22()
    value = [[1.], [3.]]
    readout = graph_ops.ReadoutFirstNode(node_set_name="nodes")
    self.assertEqual(dict(node_set_name="nodes"), readout.location)
    self.assertAllEqual(value, readout(graph))
    self.assertAllEqual(value, readout(graph, node_set_name="nodes"))

    readout = graph_ops.ReadoutFirstNode()
    self.assertEqual({}, readout.location)
    self.assertAllEqual(value, readout(graph, node_set_name="nodes"))

  def testBadFeatureLocation(self):
    graph = self._make_test_graph_22()
    with self.assertRaisesRegex(ValueError, r"initialized .* but called with"):
      readout = graph_ops.ReadoutFirstNode(node_set_name="wronk")
      _ = readout(graph, node_set_name="nodes")
    with self.assertRaisesRegex(ValueError, "requires node_set_name"):
      graph_ops.ReadoutFirstNode()(graph)

  def testFromConfig(self):
    graph = self._make_test_graph_22()
    value = [[11.], [13.]]
    kwargs = dict(node_set_name="nodes", feature_name="dense",
                  name="test_readout_first")
    config = graph_ops.ReadoutFirstNode(**kwargs).get_config()
    self.assertDictContainsSubset(kwargs, config)

    readout = graph_ops.ReadoutFirstNode.from_config(config)
    self.assertEqual("dense", readout.feature_name)
    self.assertEqual(dict(node_set_name="nodes"), readout.location)
    self.assertAllEqual(value, readout(graph))

  def _make_test_graph_22(self):
    graph = gt.GraphTensor.from_pieces(
        node_sets={"nodes": gt.NodeSet.from_fields(
            sizes=tf.constant([2, 2]),
            features={
                "dense": tf.constant([[11.], [12.], [13.], [14.]]),
                "ragged": tf.ragged.constant([
                    [110., 111.], [120.], [130.], [140., 141.]]),
                const.HIDDEN_STATE: tf.constant([[1.], [2.], [3.], [4.]]),
            })})
    return graph


class BroadcastTest(tf.test.TestCase, parameterized.TestCase):

  def testFeatureName(self):
    red_values = [[11., 12.]]
    blue_values = [[21., 22.]]
    default_values = [[31., 32.]]
    graph = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(
            features={"red": tf.constant(red_values),
                      "blue": tf.constant(blue_values),
                      const.HIDDEN_STATE: tf.constant(default_values)}),
        node_sets={"nodes": gt.NodeSet.from_fields(
            sizes=tf.constant([2]), features={})})

    broadcast = graph_ops.Broadcast(const.CONTEXT, node_set_name="nodes",
                                    feature_name="red")
    self.assertEqual("red", broadcast.feature_name)
    self.assertAllEqual(red_values * 2, broadcast(graph))

    with self.assertRaisesRegex(ValueError, r"initialized .* but called with"):
      _ = broadcast(graph, feature_name="blue")

    self.assertAllEqual(
        default_values * 2,
        graph_ops.Broadcast(const.CONTEXT, node_set_name="nodes")(graph))

  @parameterized.named_parameters(
      ("ContextToNodes", const.CONTEXT, "nodes", [[10.], [10.], [10.]]),
      ("ContextToEdges", const.CONTEXT, "edges", [[10.], [10.]]),
      ("SourceToEdges", const.SOURCE, "edges", [[21.], [21.]]),
      ("TargetToEdges", const.TARGET, "edges", [[20.], [22.]]))
  def testTagAndLocation(self, tag, location, expected):
    values = dict(context=tf.constant([[10.]]),
                  nodes=tf.constant([[20. + k] for k in range(3)]),
                  edges=tf.constant([[30. + k] for k in range(2)]))
    graph = _make_test_graph_132(values)
    location_kwarg = (dict(node_set_name="nodes") if location == "nodes" else
                      dict(edge_set_name="edges"))

    # Initialized with all three args, called with zero to three (redundantly).
    broadcast = graph_ops.Broadcast(tag, feature_name="value", **location_kwarg)
    self.assertEqual(tag, broadcast.tag)
    self.assertEqual(location_kwarg, broadcast.location)
    self.assertAllEqual(expected, broadcast(graph))
    self.assertAllEqual(expected, broadcast(graph, tag=tag))
    self.assertAllEqual(expected, broadcast(graph, feature_name="value"))
    self.assertAllEqual(expected, broadcast(graph, **location_kwarg))
    self.assertAllEqual(expected, broadcast(graph, tag=tag,
                                            feature_name="value"))
    self.assertAllEqual(expected, broadcast(graph, tag=tag,
                                            **location_kwarg))
    self.assertAllEqual(expected, broadcast(graph, feature_name="value",
                                            **location_kwarg))
    self.assertAllEqual(expected, broadcast(graph, tag=tag,
                                            feature_name="value",
                                            **location_kwarg))

    # Initialized with one arg, called with the other two.
    broadcast = graph_ops.Broadcast(tag)
    self.assertEqual(tag, broadcast.tag)
    self.assertEqual({}, broadcast.location)
    self.assertAllEqual(expected, broadcast(graph, feature_name="value",
                                            **location_kwarg))

    broadcast = graph_ops.Broadcast(**location_kwarg)
    self.assertIsNone(broadcast.tag)
    self.assertEqual(location_kwarg, broadcast.location)
    self.assertAllEqual(expected, broadcast(graph, tag=tag,
                                            feature_name="value"))

    broadcast = graph_ops.Broadcast(feature_name="value")
    self.assertIsNone(broadcast.tag)
    self.assertEqual({}, broadcast.location)
    self.assertAllEqual(expected, broadcast(graph, tag=tag, **location_kwarg))

    # Initialized with zero args, called with all.
    broadcast = graph_ops.Broadcast()
    self.assertIsNone(broadcast.tag)
    self.assertEqual({}, broadcast.location)
    self.assertAllEqual(expected, broadcast(graph, tag=tag,
                                            feature_name="value",
                                            **location_kwarg))

  def testConflictingTag(self):
    graph = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(features={"value": tf.constant([[0.]])}))
    with self.assertRaisesRegex(ValueError, r"initialized .* but called with"):
      broadcast = graph_ops.Broadcast(const.SOURCE, edge_set_name="wronk",
                                      feature_name="value")
      _ = broadcast(graph, tag=const.CONTEXT)

  @parameterized.named_parameters(
      ("Nodes", dict(node_set_name="nodes")),
      ("Edges", dict(edge_set_name="edges")))
  def testConflictingLocation(self, location_kwarg):
    graph = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(features={"value": tf.constant([[0.]])}))
    with self.assertRaisesRegex(ValueError, r"initialized .* but called with"):
      broadcast = graph_ops.Broadcast(const.CONTEXT, node_set_name="wronk",
                                      feature_name="value")
      _ = broadcast(graph, feature_name="value", **location_kwarg)

  def testTooFewOrManyLocations(self):
    graph = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(features={"value": tf.constant([[0.]])}))
    with self.assertRaisesRegex(ValueError, "at most one of"):
      graph_ops.Broadcast(const.CONTEXT,
                          node_set_name="nodes", edge_set_name="edges")
    with self.assertRaisesRegex(ValueError, "requires exactly one of"):
      graph_ops.Broadcast(const.CONTEXT)(graph)

  @parameterized.parameters(const.SOURCE, const.TARGET)
  def testNodeSetNameVsNonContext(self, origin):
    with self.assertRaisesRegex(ValueError, "requires edge_set_name"):
      graph_ops.Broadcast(origin, node_set_name="nodes")

  @parameterized.named_parameters(
      ("ContextToNodes", const.CONTEXT, "nodes", [[10.], [10.], [10.]]),
      ("ContextToEdges", const.CONTEXT, "edges", [[10.], [10.]]),
      ("SourceToEdges", const.SOURCE, "edges", [[21.], [21.]]),
      ("TargetToEdges", const.TARGET, "edges", [[20.], [22.]]))
  def testFromConfig(self, tag, location, expected):
    values = dict(context=tf.constant([[10.]]),
                  nodes=tf.constant([[20. + k] for k in range(3)]),
                  edges=tf.constant([[30. + k] for k in range(2)]))
    graph = _make_test_graph_132(values)
    location_kwarg = (dict(node_set_name="nodes") if location == "nodes" else
                      dict(edge_set_name="edges"))
    kwargs = dict(location_kwarg, tag=tag, feature_name="value",
                  name="test_broadcast")
    config = graph_ops.Broadcast(**kwargs).get_config()
    self.assertDictContainsSubset(kwargs, config)

    broadcast = graph_ops.Broadcast.from_config(config)
    self.assertEqual(tag, broadcast.tag)
    self.assertEqual("value", broadcast.feature_name)
    self.assertEqual(location_kwarg, broadcast.location)
    self.assertAllEqual(expected, broadcast(graph))


def _make_test_graph_132(values):
  """Returns GraphTensor for [v0] <-e0-- [v1] --e1--> [v2] with values."""
  def maybe_features(key):
    return dict(features={"value": values[key]} if key in values else {})
  graph = gt.GraphTensor.from_pieces(
      context=gt.Context.from_fields(**maybe_features("context")),
      node_sets={"nodes": gt.NodeSet.from_fields(
          sizes=tf.constant([3]), **maybe_features("nodes"))},
      edge_sets={"edges": gt.EdgeSet.from_fields(
          sizes=tf.constant([2]),
          adjacency=adj.Adjacency.from_indices(("nodes", tf.constant([1, 1])),
                                               ("nodes", tf.constant([0, 2]))),
          **maybe_features("edges"))})
  return graph


class AddSelfLoopsTest(tf.test.TestCase, parameterized.TestCase):
  """Ensures that AddSelfLoops invokes well-tested function add_self_loops."""

  def testAddSelfLoopLayerCallAddSelfLoopsFnReturningItsValue(self):
    mock.MagicMock()
    with mock.patch.object(
        graph_ops.ops, "add_self_loops", autospec=True) as mock_one:
      mock_one.return_value = "testReturn"

      layer = graph_ops.AddSelfLoops("some_edge_set")
      self.assertEqual("testReturn", layer("some_graph_tensor"))
      mock_one.assert_called_once_with("some_graph_tensor", "some_edge_set")


class PoolTest(tf.test.TestCase, parameterized.TestCase):

  def testFeatureName(self):
    red_values = [[11., 12.], [13., 14]]
    blue_values = [[21., 22.], [23., 24]]
    default_values = [[31., 32.], [33., 34.]]
    graph = gt.GraphTensor.from_pieces(
        node_sets={"nodes": gt.NodeSet.from_fields(
            sizes=tf.constant([2]),
            features={"red": tf.constant(red_values),
                      "blue": tf.constant(blue_values),
                      const.HIDDEN_STATE: tf.constant(default_values)})})

    pool = graph_ops.Pool(const.CONTEXT, "sum", node_set_name="nodes",
                          feature_name="red")
    self.assertEqual("red", pool.feature_name)
    self.assertAllEqual([[24., 26.]], pool(graph))

    with self.assertRaisesRegex(ValueError, r"initialized .* but called with"):
      _ = pool(graph, feature_name="blue")

    self.assertAllEqual(
        [[64., 66.]],
        graph_ops.Pool(const.CONTEXT, "sum", node_set_name="nodes")(graph))

  @parameterized.parameters(("sum", [[12.+14., 11+13.]]),
                            ("mean", [[13., 12.]]),
                            ("max", [[14., 13.]]),
                            ("min", [[12., 11.]]))
  def testReduceType(self, reduce_type, expected):
    values = [[12., 11.],
              [14., 13.]]
    graph = gt.GraphTensor.from_pieces(
        node_sets={"nodes": gt.NodeSet.from_fields(
            sizes=tf.constant([2]),
            features={const.HIDDEN_STATE: tf.constant(values)})})

    pool = graph_ops.Pool(const.CONTEXT, node_set_name="nodes")
    self.assertIsNone(pool.reduce_type)
    self.assertAllEqual(expected, pool(graph, reduce_type=reduce_type))

    with self.assertRaisesRegex(ValueError, r"requires reduce_type"):
      _ = pool(graph)

    pool = graph_ops.Pool(const.CONTEXT, reduce_type, node_set_name="nodes")
    self.assertEqual(reduce_type, pool.reduce_type)
    self.assertAllEqual(expected, pool(graph))
    self.assertAllEqual(expected, pool(graph, reduce_type=reduce_type))

    other = "max" if reduce_type != "max" else "min"
    with self.assertRaisesRegex(ValueError, r"initialized .* but called with"):
      _ = pool(graph, reduce_type=other)

  @parameterized.named_parameters(
      ("NodesToContext", const.CONTEXT, "nodes", "sum", [[20. + 21. + 22.]]),
      ("EdgesToContext", const.CONTEXT, "edges", "sum", [[30. + 31.]]),
      ("EdgesToSource", const.SOURCE, "edges", "sum", [[0.], [30.+31.], [0.]]),
      ("EdgesToTarget", const.TARGET, "edges", "sum", [[30.], [0.], [31.]]))
  def testTagAndLocation(self, tag, location, reduce_type, expected):
    values = dict(context=tf.constant([[10.]]),
                  nodes=tf.constant([[20. + k] for k in range(3)]),
                  edges=tf.constant([[30. + k] for k in range(2)]))
    graph = _make_test_graph_132(values)
    location_kwarg = (dict(node_set_name="nodes") if location == "nodes" else
                      dict(edge_set_name="edges"))

    # Initialized with all four args, called with zero, one or all args.
    pool = graph_ops.Pool(tag, reduce_type, feature_name="value",
                          **location_kwarg)
    self.assertEqual(tag, pool.tag)
    self.assertEqual(location_kwarg, pool.location)
    self.assertAllEqual(expected, pool(graph))
    self.assertAllEqual(expected, pool(graph, tag=tag))
    self.assertAllEqual(expected, pool(graph, reduce_type=reduce_type))
    self.assertAllEqual(expected, pool(graph, feature_name="value"))
    self.assertAllEqual(expected, pool(graph, **location_kwarg))
    self.assertAllEqual(expected, pool(graph, tag=tag, reduce_type=reduce_type,
                                       feature_name="value", **location_kwarg))

    # Initialized with one arg, called with the other three.
    pool = graph_ops.Pool(tag)
    self.assertEqual(tag, pool.tag)
    self.assertEqual({}, pool.location)
    self.assertAllEqual(expected, pool(graph, reduce_type=reduce_type,
                                       feature_name="value", **location_kwarg))

    pool = graph_ops.Pool(reduce_type=reduce_type)
    self.assertIsNone(pool.tag)
    self.assertEqual({}, pool.location)
    self.assertAllEqual(expected, pool(graph, tag=tag,
                                       feature_name="value", **location_kwarg))

    pool = graph_ops.Pool(**location_kwarg)
    self.assertIsNone(pool.tag)
    self.assertEqual(location_kwarg, pool.location)
    self.assertAllEqual(expected, pool(graph, tag=tag, reduce_type=reduce_type,
                                       feature_name="value"))

    pool = graph_ops.Pool(feature_name="value")
    self.assertIsNone(pool.tag)
    self.assertEqual({}, pool.location)
    self.assertAllEqual(expected, pool(graph, tag=tag, reduce_type=reduce_type,
                                       **location_kwarg))

    # Initialized with zero args, called with all.
    pool = graph_ops.Pool()
    self.assertIsNone(pool.tag)
    self.assertEqual({}, pool.location)
    self.assertAllEqual(expected, pool(graph, tag=tag, reduce_type=reduce_type,
                                       feature_name="value", **location_kwarg))

  def testConflictingTag(self):
    graph = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(features={"value": tf.constant([[0.]])}))
    with self.assertRaisesRegex(ValueError, r"initialized .* but called with"):
      pool = graph_ops.Pool(const.SOURCE, "sum", edge_set_name="wronk",
                            feature_name="value")
      _ = pool(graph, tag=const.CONTEXT)

  @parameterized.named_parameters(
      ("Nodes", dict(node_set_name="nodes")),
      ("Edges", dict(edge_set_name="edges")))
  def testConflictingLocation(self, location_kwarg):
    graph = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(features={"value": tf.constant([[0.]])}))
    with self.assertRaisesRegex(ValueError, r"initialized .* but called with"):
      pool = graph_ops.Pool(const.CONTEXT, "sum", node_set_name="wronk",
                            feature_name="value")
      _ = pool(graph, feature_name="value", **location_kwarg)

  def testTooFewOrManyLocations(self):
    graph = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(features={"value": tf.constant([[0.]])}))
    with self.assertRaisesRegex(ValueError, "at most one of"):
      graph_ops.Pool(const.CONTEXT, "sum", node_set_name="nodes",
                     edge_set_name="edges")
    with self.assertRaisesRegex(ValueError, "requires exactly one of"):
      graph_ops.Pool(const.CONTEXT, "sum")(graph)

  @parameterized.parameters(const.SOURCE, const.TARGET)
  def testNodeSetNameVsNonContext(self, origin):
    with self.assertRaisesRegex(ValueError, "requires edge_set_name"):
      graph_ops.Pool(origin, "sum", node_set_name="nodes")

  @parameterized.named_parameters(
      ("NodesToContext", const.CONTEXT, "nodes", "mean", [[(20.+21.+22.)/3.]]),
      ("EdgesToContext", const.CONTEXT, "edges", "max", [[31.]]),
      ("EdgesToSource", const.SOURCE, "edges", "sum", [[0.], [30.+31.], [0.]]),
      ("EdgesToTarget", const.TARGET, "edges", "sum", [[30.], [0.], [31.]]))
  def testFromConfig(self, tag, location, reduce_type, expected):
    values = dict(context=tf.constant([[10.]]),
                  nodes=tf.constant([[20. + k] for k in range(3)]),
                  edges=tf.constant([[30. + k] for k in range(2)]))
    graph = _make_test_graph_132(values)
    location_kwarg = (dict(node_set_name="nodes") if location == "nodes" else
                      dict(edge_set_name="edges"))
    kwargs = dict(location_kwarg, reduce_type=reduce_type, tag=tag,
                  feature_name="value", name="test_pool")
    config = graph_ops.Pool(**kwargs).get_config()
    self.assertDictContainsSubset(kwargs, config)

    pool = graph_ops.Pool.from_config(config)
    self.assertEqual(tag, pool.tag)
    self.assertEqual(reduce_type, pool.reduce_type)
    self.assertEqual("value", pool.feature_name)
    self.assertEqual(location_kwarg, pool.location)
    self.assertAllEqual(expected, pool(graph))


if __name__ == "__main__":
  tf.test.main()
