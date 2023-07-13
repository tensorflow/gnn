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
"""Tests for MapFeatures."""

import collections
import enum
import functools
import os

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.keras import keras_tensors  # For registration. pylint: disable=unused-import
from tensorflow_gnn.keras.layers import map_features


def double_fn(inputs, **_):
  """Returns twice the value of each input feature."""
  return {k: tf.add(v, v) for k, v in inputs.features.items()}


class ReloadModel(int, enum.Enum):
  """Controls how to reload a model for further testing after saving."""
  SKIP = 0
  SAVED_MODEL = 1
  KERAS = 2


class MapFeaturesTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("All", True, True, True),
      ("Context", True, False, False),
      ("Nodes", False, True, False),
      ("Edges", False, False, True))
  def testBasicCallbackApplication(self, map_context, map_nodes, map_edges):
    input_graph = _make_scalar_test_graph()

    layer = map_features.MapFeatures(
        context_fn=double_fn if map_context else None,
        node_sets_fn=double_fn if map_nodes else None,
        edge_sets_fn=double_fn if map_edges else None)
    graph = layer(input_graph)
    context = graph.context
    nodes = graph.node_sets["nodes"]
    edges = graph.edge_sets["edges"]

    rc = tf.ragged.constant

    self.assertCountEqual(["c_dense", "c_ragged"], context.features.keys())
    self.assertAllEqual(
        [[2., 4.]] if map_context else [[1., 2.]],
        context["c_dense"])
    self.assertAllEqual(
        rc([[[2, 4], [8]]] if map_context else [[[1, 2], [4]]]),
        context["c_ragged"])

    self.assertCountEqual(["n_dense", "n_ragged"], nodes.features.keys())
    self.assertAllEqual(
        [[22., 24.], [42., 44.]] if map_nodes else [[11., 12.], [21., 22.]],
        nodes["n_dense"])
    self.assertAllEqual(
        rc([[2, 4], [8]] if map_nodes else [[1, 2], [4]]),
        nodes["n_ragged"])

    self.assertCountEqual(["e_dense", "e_ragged"], edges.features.keys())
    self.assertAllEqual(
        [[62.], [64.]] if map_edges else [[31.], [32.]],
        edges["e_dense"])
    self.assertAllEqual(
        rc([[2], [4, 8]] if map_edges else [[1], [2, 4]]),
        edges["e_ragged"])

  @parameterized.named_parameters(
      ("Default", {}, {"ordinary_nodes": 1, "ordinary_edges": 1}),
      ("AllowReadoutNodes",
       {"allowed_aux_node_sets_pattern": r"_readout\W.*"},
       {"ordinary_nodes": 1, "ordinary_edges": 1,
        "_readout:test": 1, "_readout:train": 1}),
      ("AllowExtraEdges",
       {"allowed_aux_edge_sets_pattern": r"_extra.*"},
       {"ordinary_nodes": 1, "ordinary_edges": 1, "_extra_edges": 1}))
  def testAuxPieces(self, kwargs_to_test, expected_call_counts):
    input_graph = gt.GraphTensor.from_pieces(
        node_sets={
            "ordinary_nodes": gt.NodeSet.from_fields(sizes=tf.constant([1])),
            "_extra_nodes": gt.NodeSet.from_fields(sizes=tf.constant([1])),
            "_readout:train": gt.NodeSet.from_fields(sizes=tf.constant([1])),
            "_readout:test": gt.NodeSet.from_fields(sizes=tf.constant([1])),
        },
        edge_sets={
            "ordinary_edges": gt.EdgeSet.from_fields(
                sizes=tf.constant([]),
                adjacency=adj.Adjacency.from_indices(
                    ("ordinary_nodes", tf.constant([0])),
                    ("ordinary_nodes", tf.constant([0])))),
            "_extra_edges": gt.EdgeSet.from_fields(
                sizes=tf.constant([]),
                adjacency=adj.Adjacency.from_indices(
                    ("_extra_nodes", tf.constant([0])),
                    ("_extra_nodes", tf.constant([0])))),
            "_readout:train/seed": gt.EdgeSet.from_fields(
                sizes=tf.constant([]),
                adjacency=adj.Adjacency.from_indices(
                    ("ordinary_nodes", tf.constant([0])),
                    ("_readout:train", tf.constant([0])))),
        })

    call_counts = collections.defaultdict(lambda: 0)
    def node_sets_fn(node_set, node_set_name):
      call_counts[node_set_name] += 1
      return node_set.features
    def edge_sets_fn(edge_set, edge_set_name):
      call_counts[edge_set_name] += 1
      return edge_set.features
    layer = map_features.MapFeatures(node_sets_fn=node_sets_fn,
                                     edge_sets_fn=edge_sets_fn,
                                     **kwargs_to_test)
    _ = layer(input_graph)
    self.assertDictEqual(expected_call_counts, call_counts)

  def testFilterFeatures(self):
    input_graph = _make_scalar_test_graph()

    def keep_all(the_input_graph_piece):  # For context.
      return the_input_graph_piece.features  # Test this ImmutableMapping works.
    def keep_dense_fn(inputs, *, node_set_name):  # For nodes.
      self.assertEqual(node_set_name, "nodes")
      return {k: v for k, v in inputs.features.items() if k.endswith("dense")}
    def keep_none(unused_funny_arg_name, *, edge_set_name):  # For edges.
      self.assertEqual(edge_set_name, "edges")
      return {}
    layer = map_features.MapFeatures(context_fn=keep_all,
                                     node_sets_fn=keep_dense_fn,
                                     edge_sets_fn=keep_none)
    graph = layer(input_graph)
    context = graph.context
    nodes = graph.node_sets["nodes"]
    edges = graph.edge_sets["edges"]

    # Context is unchanged.
    self.assertCountEqual(["c_dense", "c_ragged"], context.features.keys())
    self.assertAllEqual([[1., 2.]], context["c_dense"])

    # Nodes keep "n_dense" unchanged, lose "n_ragged".
    self.assertCountEqual(["n_dense"], nodes.features.keys())
    self.assertAllEqual([[11., 12.], [21., 22.]], nodes["n_dense"])

    # Edges lose all features.
    self.assertEmpty(edges.features)

  @parameterized.named_parameters(
      ("", ReloadModel.SKIP),
      ("Restored", ReloadModel.SAVED_MODEL),
      ("RestoredKeras", ReloadModel.KERAS))
  def testNewFeature(self, reload_model):
    input_graph = _make_scalar_test_graph()

    # Replaces all features by hidden state [1., 1., 1.] for each node/edge.
    state_dim = 3
    def fn(inputs, *, node_set_name=None, edge_set_name=None):
      del node_set_name, edge_set_name  # Unused.
      total_size = inputs.total_size
      self.assertEqual(total_size.shape.rank, 0)  # A scalar Tensor.
      target_shape = tf.stack([total_size, tf.constant(state_dim)])
      return tf.ones(target_shape)

    # Build a Model around the Layer.
    inputs = tf.keras.layers.Input(type_spec=input_graph.spec)
    layer = map_features.MapFeatures(node_sets_fn=fn, edge_sets_fn=fn,
                                     context_fn=fn)
    outputs = layer(inputs)
    model = tf.keras.Model(inputs, outputs)
    _ = model(input_graph)  # Trigger building.

    # Do an optional round-trip through SavedModel before testing the layer's
    # behavior.
    if reload_model:
      export_dir = os.path.join(self.get_temp_dir(), "new-features-model")
      model.save(export_dir, include_optimizer=False)
      if reload_model == ReloadModel.KERAS:
        model = tf.keras.models.load_model(export_dir)
        # Check that from_config() worked, no fallback to a function trace, see
        # https://www.tensorflow.org/guide/keras/save_and_serialize#how_savedmodel_handles_custom_objects
        self.assertIsInstance(model.get_layer(index=1),
                              map_features.MapFeatures)
      else:
        model = tf.saved_model.load(export_dir)

    graph = model(input_graph)
    self.assertAllEqual(
        [[1., 1., 1.]],
        graph.context[const.HIDDEN_STATE])
    self.assertAllEqual(
        [[1., 1., 1.],
         [1., 1., 1.]],
        graph.node_sets["nodes"][const.HIDDEN_STATE])
    self.assertAllEqual(
        [[1., 1., 1.],
         [1., 1., 1.]],
        graph.edge_sets["edges"][const.HIDDEN_STATE])

  @parameterized.named_parameters(
      ("OnEdgeSet", "edge_sets_fn"),
      ("OnNodeSet", "node_sets_fn"))
  def testNewFeatureBadOutput(self, kwarg_to_test):
    input_graph = _make_scalar_test_graph()

    def bad_fn(inputs, **_):
      state_dim = 3
      # Getting the static total size like this (and not with inputs.total_size)
      # is an easy mistake to make, because it works in raw TensorFlow
      # but violates the requirement of the Keras functional API that
      # outputs must be KerasTensors computed from inputs.
      # We provide a special error message for this and similar cases.
      total_size = inputs.spec.total_size
      target_shape = tf.constant([total_size, state_dim])
      return tf.ones(target_shape)

    with self.assertRaisesRegex(ValueError, r"return KerasTensor.*total_size"):
      layer = map_features.MapFeatures(**{kwarg_to_test: bad_fn})
      _ = layer(input_graph)  # Trigger building.

  @parameterized.named_parameters(
      ("Basic", ReloadModel.SKIP),
      ("Restored", ReloadModel.SAVED_MODEL),
      ("RestoredKeras", ReloadModel.KERAS))
  def testEmbeddingTable(self, reload_model):
    input_graph = _make_scalar_test_graph()

    # Replaces the "[cne]_ragged" feature with embeddings of its values.
    # Integer value v maps to embedding [v, v+delta] for v < input_dim
    # where delta is set differently for context, nodes and edges.
    def fn(inputs, *, node_set_name=None, edge_set_name=None, input_dim, delta):
      name = node_set_name or edge_set_name or "context"
      features = inputs.get_features_dict()
      hashed_ids = features.pop("{}_ragged".format(name[0]))
      range_start = [.0, delta]
      embedding = tf.keras.layers.Embedding(
          input_dim, len(range_start), MyRangeInitializer(range_start),
          name=f"embed_{name}")
      features["embedded"] = embedding(hashed_ids)
      return features

    inputs = tf.keras.layers.Input(type_spec=input_graph.spec)
    layer = map_features.MapFeatures(
        context_fn=functools.partial(fn, input_dim=10, delta=0.25),
        node_sets_fn=functools.partial(fn, input_dim=100, delta=0.5),
        edge_sets_fn=functools.partial(fn, input_dim=1000, delta=0.75))
    outputs = layer(inputs)
    model = tf.keras.Model(inputs, outputs)
    # Trigger building.
    _ = model(input_graph)

    # Test that the model tracks weights that the callback created on the fly.
    # (That's the main advantage of using the Keras functional API here.)
    weight_shapes = {w.name: w.shape for w in model.weights}
    self.assertEqual(
        {"embed_context/embeddings:0": tf.TensorShape([10, 2]),
         "embed_nodes/embeddings:0": tf.TensorShape([100, 2]),
         "embed_edges/embeddings:0": tf.TensorShape([1000, 2])},
        weight_shapes)

    # Do an optional round-trip through SavedModel before testing the layer's
    # behavior. In particular, this restores the checkpointed embedding tables.
    if reload_model:
      export_dir = os.path.join(self.get_temp_dir(), "embedding-model")
      model.save(export_dir, include_optimizer=False)
      if reload_model == ReloadModel.KERAS:
        model = tf.keras.models.load_model(export_dir)
        # Check that from_config() worked, no fallback to a function trace, see
        # https://www.tensorflow.org/guide/keras/save_and_serialize#how_savedmodel_handles_custom_objects
        self.assertIsInstance(model.get_layer(index=1),
                              map_features.MapFeatures)
      else:
        model = tf.saved_model.load(export_dir)

    graph = model(input_graph)
    context = graph.context
    nodes = graph.node_sets["nodes"]
    edges = graph.edge_sets["edges"]

    self.assertCountEqual(["c_dense", "embedded"], context.features.keys())
    self.assertCountEqual(["n_dense", "embedded"], nodes.features.keys())
    self.assertCountEqual(["e_dense", "embedded"], edges.features.keys())

    rc = lambda x: tf.ragged.constant(x, inner_shape=(2,))
    self.assertAllEqual(rc([[[[1., 1.25], [2., 2.25]], [[4., 4.25]]]]),
                        context["embedded"])
    self.assertAllEqual(rc([[[1., 1.5], [2., 2.5]], [[4., 4.5]]]),
                        nodes["embedded"])
    self.assertAllEqual(rc([[[1., 1.75]], [[2., 2.75], [4., 4.75]]]),
                        edges["embedded"])

  @parameterized.named_parameters(
      ("OnContext", "context_fn"),
      ("OnNodes", "node_sets_fn"),
      ("OnEdges", "edge_sets_fn"))
  def testCheckSameFeatures(self, kwarg_to_test):
    graph_with_dense = _make_scalar_test_graph(dense=True, ragged=False)
    graph_with_ragged = _make_scalar_test_graph(dense=False, ragged=True)
    graph_with_both = _make_scalar_test_graph(dense=True, ragged=True)
    graph_without = _make_scalar_test_graph(dense=False, ragged=False)

    # Changes to the feature set of a mapped graph piece raise an error.
    def keep_all(graph_piece, **_):
      return graph_piece.features
    layer = map_features.MapFeatures(**{kwarg_to_test: keep_all})
    _ = layer(graph_with_ragged)  # First call defines the feature set.
    _ = layer(graph_with_ragged)
    regex = r"feature set .*has changed"
    with self.assertRaisesRegex(ValueError, regex):
      _ = layer(graph_with_dense)
    with self.assertRaisesRegex(ValueError, regex):
      _ = layer(graph_with_both)
    with self.assertRaisesRegex(ValueError, regex):
      _ = layer(graph_without)

    # Changes to the feature set of an ignored graph piece are ignored,
    # both for the kward_to_test (which returns None) and the two unset
    # kwargs (which are None).
    def ignore(unused_graph_piece, **_):
      return None
    layer = map_features.MapFeatures(**{kwarg_to_test: ignore})
    _ = layer(graph_with_ragged)  # First call defines the feature set.
    _ = layer(graph_with_ragged)
    _ = layer(graph_with_dense)
    _ = layer(graph_with_both)
    _ = layer(graph_without)

  @parameterized.named_parameters(
      ("EdgeSet", "edge_sets_fn", dict(edges=False), dict(edges=True)),
      ("NodeSet", "node_sets_fn",
       dict(nodes=False, edges=False), dict(nodes=True, edges=False)))
  def testCheckUnexpected(self, kwarg_to_test, make_without, make_with):
    graph_without = _make_scalar_test_graph(**make_without)
    graph_with_readout = _make_scalar_test_graph(**make_without, readout=True)
    graph_with_unexpected = _make_scalar_test_graph(**make_with)
    def keep_all(graph_piece, **_):
      return graph_piece.features
    layer = map_features.MapFeatures(**{kwarg_to_test: keep_all})
    _ = layer(graph_without)  # First call defines the known graph pieces.
    _ = layer(graph_without)  # OK to call again.
    _ = layer(graph_with_readout)  # OK to call with new aux pieces.
    with self.assertRaisesRegex(KeyError, r"Unexpected"):
      _ = layer(graph_with_unexpected)

  def testBatchedGraphTensorRaggedReduce(self):
    rc = tf.ragged.constant
    # Test input is a GraphTensor of shape [3].
    # Each element has components with the given number of nodes in each.
    # The nodes of each element have a "ragged_id" feature with
    # alternatingly 1 or 2 values, numbered consecutively.
    # Its shape is [*graph_shape, (num_nodes), (num_ids_per_node)].
    input_graph = _make_batched_test_graph([[3], [2, 1], [1]],
                                           add_ragged_ids=True)
    self.assertEqual([3], input_graph.shape)
    self.assertAllEqual(
        rc([[[0], [1, 2], [3]], [[100], [101, 102], [103]], [[200]]],
           ragged_rank=2),
        input_graph.node_sets["nodes"]["ragged_ids"])

    # Reduces the "ragged_ids" of each node to a scalar: their sum.
    # The resulting shape is [*graph_shape, (num_nodes)].
    def sum_ids_on_node(node_set, node_set_name):
      self.assertEqual("nodes", node_set_name)
      features = node_set.get_features_dict()
      ragged_ids = features.pop("ragged_ids")
      features["summed_ids"] = tf.reduce_sum(ragged_ids, axis=-1)
      return features

    graph = map_features.MapFeatures(node_sets_fn=sum_ids_on_node)(input_graph)
    nodes = graph.node_sets["nodes"]

    self.assertCountEqual(["summed_ids"], nodes.features.keys())
    self.assertAllEqual(
        rc([[0, 1+2, 3], [100, 101+102, 103], [200]], ragged_rank=1),
        nodes["summed_ids"])

  def testBatchedGraphTensorNewFeatures(self):
    rc = tf.ragged.constant
    # Test input is a GraphTensor of shape [3].
    # Each element has components with the given number of nodes in each.
    input_graph = _make_batched_test_graph([[4], [2], [1]],
                                           add_ragged_ids=True)

    def node_sets_fn(node_set, **_):
      features = {}
      set_sizes = tf.reduce_sum(node_set.sizes, axis=-1)  # Smash components.
      values = tf.ones([tf.reduce_sum(set_sizes), 3])
      features["three_ones"] = tf.RaggedTensor.from_row_lengths(values,
                                                                set_sizes)
      return features

    graph = map_features.MapFeatures(node_sets_fn=node_sets_fn)(input_graph)

    self.assertAllEqual(
        rc([[[1., 1., 1.]] * 4,
            [[1., 1., 1.]] * 2,
            [[1., 1., 1.]] * 1], ragged_rank=1),
        graph.node_sets["nodes"]["three_ones"])


@tf.keras.utils.register_keras_serializable(package="GNNtesting")
class MyRangeInitializer(tf.keras.initializers.Initializer):
  """Initializes with 2D value [start, start+1, start+2, ...]."""

  def __init__(self, start):
    self.start = start

  def __call__(self, shape, dtype=None, **kwargs):
    num_values, value_dim = tf.TensorShape(shape).as_list()
    start = tf.constant(self.start)
    start.shape.assert_is_compatible_with([value_dim])
    offsets = tf.range(num_values, dtype=dtype)
    result = tf.add(  # Broadcasting will create shape [num_values, value_dim].
        tf.expand_dims(offsets, axis=1),  # [num_values, 1]
        tf.expand_dims(start, axis=0))    # [1, value_dim]
    return result

  def get_config(self):
    return dict(start=self.start)


def _make_scalar_test_graph(*, context=True, nodes=True, edges=True,
                            dense=True, ragged=True, readout=False):
  c = tf.constant
  rc = tf.ragged.constant

  context_obj = None
  if context:
    features = {
        **({"c_dense": c([[1., 2.]])} if dense else {}),
        **({"c_ragged": rc([[[1, 2], [4]]])} if ragged else {})}
    context_obj = gt.Context.from_fields(features=features)

  node_sets = {}
  if nodes:
    features = {
        **({"n_dense": c([[11., 12.], [21., 22.]])} if dense else {}),
        **({"n_ragged": rc([[1, 2], [4]])} if ragged else {})}
    node_sets["nodes"] = gt.NodeSet.from_fields(
        sizes=c([2]), features=features)

  edge_sets = {}
  if edges:
    assert nodes, "Can't have edges without nodes."
    adjacency = adj.Adjacency.from_indices(
        ("nodes", c([0, 1])),
        ("nodes", c([1, 0])))
    features = {
        **({"e_dense": c([[31.], [32.]])} if dense else {}),
        **({"e_ragged": rc([[1], [2, 4]])} if ragged else {})}
    edge_sets["edges"] = gt.EdgeSet.from_fields(
        sizes=c([2]), adjacency=adjacency, features=features)

  if readout:  # Independent of ordinary nodes and edges.
    node_sets["_readout"] = gt.NodeSet.from_fields(sizes=c([1]))
    edge_sets["_readout/seed"] = gt.EdgeSet.from_fields(
        sizes=c([1]),
        adjacency=adj.Adjacency.from_indices(("nodes", c([0])),
                                             ("nodes", c([0]))))

  return gt.GraphTensor.from_pieces(
      context=context_obj, node_sets=node_sets, edge_sets=edge_sets)


def _make_test_graph_for_batching(node_sizes, start_id):
  """Returns graph with "nodes" with 1 or 2 "ragged_ids"."""
  nodes_features = {}
  if start_id is not None:
    ragged_ids = []
    next_id = start_id
    for node_id in range(sum(node_sizes)):
      num_ids_on_node = node_id % 2 + 1
      ragged_ids.append(list(range(next_id, next_id + num_ids_on_node)))
      next_id += num_ids_on_node
    nodes_features["ragged_ids"] = tf.ragged.constant(ragged_ids)

  graph = gt.GraphTensor.from_pieces(
      node_sets={"nodes": gt.NodeSet.from_fields(
          sizes=tf.constant(node_sizes),
          features=nodes_features)})
  return graph


def _make_batched_test_graph(all_node_sizes, *, add_ragged_ids=False):
  """Returns batch of graphs with "nodes" with "ragged_ids"."""
  nodes_features_spec = {}
  if add_ragged_ids:
    nodes_features_spec["ragged_ids"] = tf.RaggedTensorSpec(
        tf.TensorShape([None, None]),  # [num_nodes, (num_ids_per_node)].
        dtype=tf.int32, ragged_rank=1)
  graph_spec = gt.GraphTensorSpec.from_piece_specs(
      node_sets_spec={
          "nodes": gt.NodeSetSpec.from_field_specs(
              features_spec=nodes_features_spec,
              sizes_spec=tf.TensorSpec(shape=(None,),  # [num_components].
                                       dtype=tf.int32))},
      edge_sets_spec={})

  def generate_graphs():
    for i, node_sizes in enumerate(all_node_sizes):
      yield _make_test_graph_for_batching(
          node_sizes,
          start_id=100*i if add_ragged_ids else None)

  ds = tf.data.Dataset.from_generator(generate_graphs,
                                      output_signature=graph_spec)
  ds = ds.batch(len(all_node_sizes))
  return ds.get_single_element()


class MakeEmptyFeatureTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("Basic", ReloadModel.SKIP),
      ("Restored", ReloadModel.SAVED_MODEL),
      ("RestoredKeras", ReloadModel.KERAS))
  def testNodesScalar(self, reload_model):
    input_graph = _make_scalar_test_graph(dense=False, ragged=False)

    def node_sets_fn(node_set, node_set_name):
      self.assertEqual(node_set_name, "nodes")
      return {"empty": map_features.MakeEmptyFeature()(node_set)}
    inputs = tf.keras.layers.Input(type_spec=input_graph.spec)
    outputs = map_features.MapFeatures(node_sets_fn=node_sets_fn)(inputs)
    model = tf.keras.Model(inputs, outputs)
    # Trigger building.
    _ = model(input_graph)

    if reload_model:
      export_dir = os.path.join(self.get_temp_dir(),
                                "nodes-scalar-empty-features")
      model.save(export_dir, include_optimizer=False)
      if reload_model == ReloadModel.KERAS:
        model = tf.keras.models.load_model(export_dir)
        # Check that from_config() worked, no fallback to a function trace, see
        # https://www.tensorflow.org/guide/keras/save_and_serialize#how_savedmodel_handles_custom_objects
        self.assertIsInstance(model.get_layer(index=1),
                              map_features.MapFeatures)
      else:
        model = tf.saved_model.load(export_dir)

    empty_feature = model(input_graph).node_sets["nodes"]["empty"]
    self.assertIsInstance(empty_feature, tf.Tensor)  # Not ragged.
    self.assertAllEqual(empty_feature.shape, [2, 0])

  @parameterized.named_parameters(
      ("Basic", ReloadModel.SKIP),
      ("Restored", ReloadModel.SAVED_MODEL),
      ("RestoredKeras", ReloadModel.KERAS))
  def testNodesBatched(self, reload_model):
    # The test input has six graphs, with variable sizes per component.
    all_node_sizes = [[4, 3], [3], [2, 1, 2], [4, 1, 2, 3], [11], [4, 5, 3]]
    input_graph = _make_batched_test_graph(all_node_sizes)
    all_num_nodes = [sum(x) for x in all_node_sizes]

    def node_sets_fn(node_set, node_set_name):
      self.assertEqual(node_set_name, "nodes")
      return {"empty": map_features.MakeEmptyFeature()(node_set)}
    inputs = tf.keras.layers.Input(type_spec=input_graph.spec)
    outputs = map_features.MapFeatures(node_sets_fn=node_sets_fn)(inputs)
    model = tf.keras.Model(inputs, outputs)
    # Trigger building.
    _ = model(input_graph)

    if reload_model:
      export_dir = os.path.join(self.get_temp_dir(), "batched-empty-features")
      model.save(export_dir, include_optimizer=False)
      if reload_model == ReloadModel.KERAS:
        model = tf.keras.models.load_model(export_dir)
        # Check that from_config() worked, no fallback to a function trace, see
        # https://www.tensorflow.org/guide/keras/save_and_serialize#how_savedmodel_handles_custom_objects
        self.assertIsInstance(model.get_layer(index=1),
                              map_features.MapFeatures)
      else:
        model = tf.saved_model.load(export_dir)

    empty_feature = model(input_graph).node_sets["nodes"]["empty"]
    self.assertIsInstance(empty_feature, tf.RaggedTensor)
    self.assertAllEqual(empty_feature.flat_values, [[]] * sum(all_num_nodes))
    for idx, num_nodes in enumerate(all_num_nodes):
      self.assertAllEqual([[]] * num_nodes, empty_feature[idx],
                          msg=f"at index {idx}")

  def testEdges(self):
    input_graph = _make_scalar_test_graph(dense=False, ragged=False)

    def edge_sets_fn(edge_set, edge_set_name):
      self.assertEqual(edge_set_name, "edges")
      return {"empty": map_features.MakeEmptyFeature()(edge_set)}
    inputs = tf.keras.layers.Input(type_spec=input_graph.spec)
    outputs = map_features.MapFeatures(edge_sets_fn=edge_sets_fn)(inputs)
    model = tf.keras.Model(inputs, outputs)
    # Trigger building.
    _ = model(input_graph)

    empty_feature = model(input_graph).edge_sets["edges"]["empty"]
    self.assertIsInstance(empty_feature, tf.Tensor)
    self.assertAllEqual(empty_feature.shape, [2, 0])

  def testContext(self):
    input_graph = _make_scalar_test_graph(dense=False, ragged=False)

    def context_fn(context):
      return {"empty": map_features.MakeEmptyFeature()(context)}
    inputs = tf.keras.layers.Input(type_spec=input_graph.spec)
    outputs = map_features.MapFeatures(context_fn=context_fn)(inputs)
    model = tf.keras.Model(inputs, outputs)
    # Trigger building.
    _ = model(input_graph)

    empty_feature = model(input_graph).context["empty"]
    self.assertIsInstance(empty_feature, tf.Tensor)
    self.assertAllEqual(empty_feature.shape, [1, 0])


if __name__ == "__main__":
  tf.test.main()
