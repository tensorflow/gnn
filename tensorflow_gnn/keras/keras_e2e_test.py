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
"""End-to-end tests for Keras Models."""

import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn  # Test user-visibe names.

as_tensor = tf.convert_to_tensor


class ExportedKerasNamesTest(tf.test.TestCase):
  """Tests symbols exist in tfgnn.keras.*."""

  def assertIsSubclass(self, first, second, msg=None):
    if msg is None:
      msg = f'{repr(first)} is not a subclass of {repr(second)}'
    self.assertTrue(issubclass(first, second), msg=msg)

  def assertCallable(self, expr, msg=None):
    if msg is None:
      msg = f'{repr(expr)} is not callable'
    self.assertTrue(callable(expr), msg=msg)

  def testLayers(self):
    Layer = tf.keras.layers.Layer  # pylint: disable=invalid-name
    self.assertIsSubclass(tfgnn.keras.layers.MapFeatures, Layer)
    self.assertIsSubclass(tfgnn.keras.layers.MakeEmptyFeature, Layer)
    self.assertIsSubclass(tfgnn.keras.layers.PadToTotalSizes, Layer)
    self.assertIsSubclass(tfgnn.keras.layers.Broadcast, Layer)
    self.assertIsSubclass(tfgnn.keras.layers.Pool, Layer)
    self.assertIsSubclass(tfgnn.keras.layers.Readout, Layer)
    self.assertIsSubclass(tfgnn.keras.layers.ReadoutFirstNode, Layer)
    self.assertIsSubclass(tfgnn.keras.layers.StructuredReadout, Layer)
    self.assertIsSubclass(tfgnn.keras.layers.StructuredReadoutIntoFeature,
                          Layer)
    self.assertIsSubclass(tfgnn.keras.layers.AddReadoutFromFirstNode, Layer)
    self.assertIsSubclass(tfgnn.keras.layers.AnyToAnyConvolutionBase, Layer)
    self.assertIsSubclass(tfgnn.keras.layers.SimpleConv, Layer)
    self.assertIsSubclass(tfgnn.keras.layers.ItemDropout, Layer)
    self.assertIsSubclass(tfgnn.keras.layers.NextStateFromConcat, Layer)
    self.assertIsSubclass(tfgnn.keras.layers.ResidualNextState, Layer)
    self.assertIsSubclass(tfgnn.keras.layers.EdgeSetUpdate, Layer)
    self.assertIsSubclass(tfgnn.keras.layers.NodeSetUpdate, Layer)
    self.assertIsSubclass(tfgnn.keras.layers.ContextUpdate, Layer)
    self.assertIsSubclass(tfgnn.keras.layers.GraphUpdate, Layer)

  def testBuilders(self):
    self.assertIsSubclass(tfgnn.keras.ConvGNNBuilder, object)

  def testInitializers(self):
    self.assertCallable(tfgnn.keras.clone_initializer)


# An example of a custom Keras layer used by tests below.
class AddWeightedSwappedInEdges(tf.keras.layers.Layer):
  """Adds weighted sum of coordinate-swapped neighbor states to each node."""

  def __init__(self, supports_get_config=True, supports_from_config=True,
               **kwargs):
    kwargs.setdefault('name', 'add_weighted_swapped_in_edges')
    super().__init__(**kwargs)
    self.supports_get_config = supports_get_config
    self.supports_from_config = supports_from_config
    self.fnn = tf.keras.layers.Dense(
        units=2,
        name='swap_node_state_coordinates',
        use_bias=False,
        kernel_initializer=tf.keras.initializers.Constant([[0., 1.], [1., 0.]]))

  def get_config(self):
    if self.supports_get_config:
      return dict(
          supports_from_config=self.supports_from_config,
          **super().get_config())
    else:
      raise NotImplementedError('unsupported')

  @classmethod
  def from_config(cls, config):
    if config['supports_from_config']:
      return cls(**config)
    else:
      raise ValueError('Let\'s pretend there was a problem.')

  def call(self, graph):
    weight = graph.edge_sets['edge']['edge_weight']
    node_state = graph.node_sets['node']['hidden_state']
    source_value = tf.gather(graph.node_sets['node']['hidden_state'],
                             graph.edge_sets['edge'].adjacency[tfgnn.SOURCE])
    message = tf.multiply(weight, source_value)
    pooled_message = tf.math.unsorted_segment_sum(
        message, graph.edge_sets['edge'].adjacency[tfgnn.TARGET],
        graph.node_sets['node'].total_size)
    node_updates = self.fnn(pooled_message)
    node_state += node_updates
    return graph.replace_features(
        node_sets={'node': {
            'hidden_state': node_state
        }})


# A similar example of model building with tfgnn.keras.layers.*.
def add_weighted_swapped_in_edges(graph, use_deferred_init):

  def _source_times_weight(inputs):
    edge_inputs, node_inputs, _ = inputs
    return tf.multiply(edge_inputs['edge_weight'], node_inputs[tfgnn.SOURCE])

  def get_kwargs(graph_tensor_spec=None):
    del graph_tensor_spec  # Unused.
    return dict(
        edge_sets={
            'edge':
                tfgnn.keras.layers.EdgeSetUpdate(
                    tf.keras.layers.Lambda(_source_times_weight),
                    edge_input_feature=['edge_weight'],
                    node_input_tags=[tfgnn.SOURCE])
        },
        node_sets={
            'node':
                tfgnn.keras.layers.NodeSetUpdate(
                    {'edge': tfgnn.keras.layers.Pool(tfgnn.TARGET, 'sum')},
                    tfgnn.keras.layers.NextStateFromConcat(
                        tf.keras.layers.Dense(
                            units=2,
                            name='add_swapped_message',
                            use_bias=False,
                            kernel_initializer=tf.keras.initializers.Constant(
                                [[1., 0., 0., 1.], [0., 1., 1., 0.]]))))
        })

  if use_deferred_init:
    update = tfgnn.keras.layers.GraphUpdate(deferred_init_callback=get_kwargs)
  else:
    update = tfgnn.keras.layers.GraphUpdate(**get_kwargs())
  return update(graph)


class GraphTensorKerasModelTest(tf.test.TestCase, parameterized.TestCase):

  def _create_graph_tensor(self, static_shapes, factor):
    """Returns a graph with one component, as depicted below.

            /--  0.5 -->>
     [10, 0]             [12, 0]
            <<-- -0.5 --/

    Args:
      static_shapes: If true, shape dimensions reflect the concrete values. If
        false, shape dimensions are set to None.
      factor: size multiplier.
    """
    factor = tf.cast(factor, tf.int32)

    def tile(tensor, factor):
      assert tensor.shape.rank in (1, 2)
      return tf.tile(tensor,
                     [factor] if tensor.shape.rank == 1 else [factor, 1])

    return tfgnn.GraphTensor.from_pieces(
        edge_sets={
            'edge':
                tfgnn.EdgeSet.from_fields(
                    features={
                        'edge_weight':
                            tile(
                                as_tensor([[0.5], [-0.5]], tf.float32), factor)
                    },
                    sizes=as_tensor([2]) * factor,
                    adjacency=tfgnn.HyperAdjacency.from_indices(
                        indices={
                            tfgnn.SOURCE: ('node',
                                           tile(as_tensor([0, 1]), factor)),
                            tfgnn.TARGET: ('node',
                                           tile(as_tensor([1, 0]), factor)),
                        }))
        },
        node_sets={
            'node':
                tfgnn.NodeSet.from_fields(
                    features={
                        'hidden_state':
                            tile(
                                as_tensor([[10, 0.], [12., 0.]], tf.float32),
                                factor)
                    },
                    sizes=as_tensor([2]) * factor)
        })

  def _get_input_spec(self, static_shapes):
    """Returns a GraphTensorSpec for a homogeneous scalar graph.

    The number of components is indeterminate ((suitable for model computations
    after merging a batch of inputs into components of a singe graph).
    Each node has a state of shape [2] and each edge has a weight of shape [1].

    Args:
      static_shapes: If true, shape dimensions reflect the concrete values. If
        false, shape dimensions are set to None.
    """
    if static_shapes:
      spec = self._create_graph_tensor(static_shapes, 1).spec
      # Check that dataset spec has static component dimensions.
      self.assertAllEqual(spec.edge_sets_spec['edge']['edge_weight'],
                          tf.TensorSpec(tf.TensorShape([2, 1]), tf.float32))
      return spec

    ds = tf.data.Dataset.range(
        1,
        3).map(lambda factor: self._create_graph_tensor(static_shapes, factor))
    spec = ds.element_spec
    # Check that dataset spec has relaxed component dimensions.
    self.assertAllEqual(spec.edge_sets_spec['edge']['edge_weight'],
                        tf.TensorSpec(tf.TensorShape([None, 1]), tf.float32))
    return spec

  @parameterized.named_parameters(('StaticShapes', True),
                                  ('DynamicShapes', False),
                                  ('DeferredInit', False, True))
  def testStdLayerModel(self, static_shapes, use_deferred_init=False):

    # A Keras Model build from tfgnn.keras.layers.*.
    inputs = tf.keras.layers.Input(
        type_spec=self._get_input_spec(static_shapes))
    graph = add_weighted_swapped_in_edges(
        inputs, use_deferred_init=use_deferred_init)
    outputs = tfgnn.keras.layers.Readout(node_set_name='node')(graph)
    model = tf.keras.Model(inputs, outputs)

    expected_1 = as_tensor([[10., -6.], [12., 5.]], tf.float32)
    graph_1 = self._create_graph_tensor(static_shapes, factor=1)
    if use_deferred_init:
      # Must call to initialize before saving.
      self.assertAllClose(model(graph_1), expected_1)

    # Save and restore the model.
    export_dir = os.path.join(self.get_temp_dir(), 'stdlayer-tf')
    tf.saved_model.save(model, export_dir)
    restored_model = tf.saved_model.load(export_dir)

    self.assertAllClose(model(graph_1), expected_1)
    self.assertAllClose(restored_model(graph_1), expected_1)

  @parameterized.parameters([True, False])
  def testCustomGraphToGraphModel(self, static_shapes):

    # A Keras Model that inputs and outputs a GraphTensor.
    inputs = tf.keras.layers.Input(
        type_spec=self._get_input_spec(static_shapes))
    outputs = AddWeightedSwappedInEdges(supports_get_config=False)(inputs)
    model = tf.keras.Model(inputs, outputs)
    # Save and restore the model.
    export_dir = os.path.join(self.get_temp_dir(), 'graph2graph-tf')
    tf.saved_model.save(model, export_dir)
    restored_model = tf.saved_model.load(export_dir)

    def readout(graph):
      return graph.node_sets['node']['hidden_state']

    expected_1 = as_tensor([[10., -6.], [12., 5.]], tf.float32)
    graph_1 = self._create_graph_tensor(static_shapes, factor=1)
    self.assertAllClose(readout(model(graph_1)), expected_1)
    self.assertAllClose(readout(restored_model(graph_1)), expected_1)

  def testCustomModelWithReadoutOp(self, static_shapes=True):

    # A Keras Model that maps a GraphTensor to a Tensor,
    # using subscripting provided by GraphKerasTensor.
    inputs = net = tf.keras.layers.Input(
        type_spec=self._get_input_spec(static_shapes))
    net = AddWeightedSwappedInEdges(supports_get_config=False)(net)
    net = net.node_sets['node']['hidden_state']
    model = tf.keras.Model(inputs, net)
    # Save and restore the model.
    export_dir = os.path.join(self.get_temp_dir(), 'graph2tensor-op-tf')
    tf.saved_model.save(model, export_dir)
    restored_model = tf.saved_model.load(export_dir)

    expected_1 = as_tensor([[10., -6.], [12., 5.]], tf.float32)
    graph_1 = self._create_graph_tensor(static_shapes, factor=1)
    self.assertAllClose(model(graph_1), expected_1)
    self.assertAllClose(restored_model(graph_1), expected_1)

  @parameterized.named_parameters(
      ('Basic', True, True),
      ('DynamicShapes', False, True),
      ('FallbackToTfFunction', True, False))
  def testCustomModelKerasRestore(self, static_shapes, from_config):

    # A Keras Model that maps a GraphTensor to a Tensor.
    inputs = net = tf.keras.layers.Input(
        type_spec=self._get_input_spec(static_shapes))
    net = AddWeightedSwappedInEdges(supports_get_config=True,
                                    supports_from_config=from_config)(net)
    net = tfgnn.keras.layers.Readout(
        node_set_name='node', feature_name='hidden_state')(
            net)
    model = tf.keras.Model(inputs, net)
    # Save and restore the model as a Keras model.
    export_dir = os.path.join(self.get_temp_dir(), 'graph2tensor-keras')
    model.save(export_dir)
    restored_model = tf.keras.models.load_model(
        export_dir,
        custom_objects=dict(
            AddWeightedSwappedInEdges=AddWeightedSwappedInEdges))
    self.assertIsInstance(restored_model, tf.keras.Model)
    if from_config:  # The common case.
      self.assertIsInstance(
          restored_model.get_layer(index=1), AddWeightedSwappedInEdges)
    else:
      # Model loading wraps a tf.function as a one-off Layer type.
      # This used to fail (b/217370590) when the layer's input is a GraphTensor.
      self.assertNotIsInstance(
          restored_model.get_layer(index=1), AddWeightedSwappedInEdges)
    self.assertIsInstance(
        restored_model.get_layer(index=2), tfgnn.keras.layers.Readout)

    expected_1 = as_tensor([[10., -6.], [12., 5.]], tf.float32)
    graph_1 = self._create_graph_tensor(static_shapes, factor=1)
    self.assertAllClose(model(graph_1), expected_1)
    self.assertAllClose(restored_model(graph_1), expected_1)

  def testPredict(self):

    def features_fn(index):
      label = tf.cast(index, tf.float32)
      graph = tfgnn.GraphTensor.from_pieces(
          context=tfgnn.Context.from_fields(
              features={'h': tf.expand_dims(label, -1)}))
      return graph, label

    ds = tf.data.Dataset.range(4)
    ds = ds.map(features_fn)
    ds = ds.batch(2)
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Lambda(lambda gt: tf.squeeze(gt.context['h'], -1))])

    model.compile(loss='mae')
    model.fit(ds)
    predictions = model.predict(ds.map(lambda graph, _: graph))
    self.assertAllClose(predictions, np.array([0., 1., 2., 3.]))

    graph_batch = next(iter(ds))[0]
    predictions_on_batch = model.predict_on_batch(graph_batch)
    self.assertAllClose(predictions_on_batch, np.array([0., 1.]))


if __name__ == '__main__':
  tf.test.main()
