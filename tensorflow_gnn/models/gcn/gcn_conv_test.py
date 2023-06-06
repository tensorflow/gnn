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
"""Tests for gcn_conv."""
import enum
import math
import os

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models.gcn import gcn_conv


class ReloadModel(int, enum.Enum):
  """Controls how to reload a model for further testing after saving."""

  SKIP = 0
  SAVED_MODEL = 1
  KERAS = 2


class GcnConvTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for GCNConv layer."""

  @parameterized.named_parameters(
      dict(
          testcase_name='noSelfLoops',
          graph=tfgnn.GraphTensor.from_pieces(
              node_sets={
                  tfgnn.NODES: tfgnn.NodeSet.from_fields(
                      sizes=[2],
                      features={
                          tfgnn.HIDDEN_STATE: tf.constant(
                              [[1.0, 0, 0], [0, 1, 0]]
                          )
                      },
                  )
              },
              edge_sets={
                  tfgnn.EDGES: tfgnn.EdgeSet.from_fields(
                      sizes=[2],
                      adjacency=tfgnn.Adjacency.from_indices(
                          source=(
                              tfgnn.NODES,
                              tf.constant([0, 1], dtype=tf.int64),
                          ),
                          target=(
                              tfgnn.NODES,
                              tf.constant([1, 0], dtype=tf.int64),
                          ),
                      ),
                  )
              },
          ),
          add_self_loops=False,
          expected_result=tf.constant([[0, 1, 0],
                                       [1.0, 0, 0]]),
      ),
      dict(
          testcase_name='selfLoops',
          graph=tfgnn.GraphTensor.from_pieces(
              node_sets={
                  tfgnn.NODES: tfgnn.NodeSet.from_fields(
                      sizes=[2],
                      features={
                          tfgnn.HIDDEN_STATE: tf.constant(
                              [[1.0, 0, 0],
                               [0, 1, 0]]
                          )
                      },
                  )
              },
              edge_sets={
                  tfgnn.EDGES: tfgnn.EdgeSet.from_fields(
                      sizes=[2],
                      adjacency=tfgnn.Adjacency.from_indices(
                          source=(
                              tfgnn.NODES,
                              tf.constant([0, 1], dtype=tf.int64),
                          ),
                          target=(
                              tfgnn.NODES,
                              tf.constant([1, 0], dtype=tf.int64),
                          ),
                      ),
                  )
              },
          ),
          add_self_loops=True,
          expected_result=tf.constant([[0.5, 0.5, 0],
                                       [0.5, 0.5, 0]]),
      ),
      dict(
          testcase_name='discreteComponents',
          graph=tfgnn.GraphTensor.from_pieces(
              node_sets={
                  tfgnn.NODES: tfgnn.NodeSet.from_fields(
                      sizes=[2, 2],
                      features={
                          tfgnn.HIDDEN_STATE: tf.constant(
                              [[1.0, 0, 0], [0, 1, 0]] * 2
                          )
                      },
                  )
              },
              edge_sets={
                  tfgnn.EDGES: tfgnn.EdgeSet.from_fields(
                      sizes=[2, 2],
                      adjacency=tfgnn.Adjacency.from_indices(
                          source=(
                              tfgnn.NODES,
                              tf.constant([0, 1, 2, 3], dtype=tf.int64),
                          ),
                          target=(
                              tfgnn.NODES,
                              tf.constant([1, 0, 3, 2], dtype=tf.int64),
                          ),
                      ),
                  )
              },
          ),
          add_self_loops=False,
          expected_result=tf.constant([[0, 1.0, 0], [1, 0, 0]] * 2),
      ),
  )
  def test_gcnconv(self, graph, add_self_loops, expected_result):
    """Tests that gcn_conv returns the correct values."""
    conv = gcn_conv.GCNConv(
        units=3,
        activation=None,
        use_bias=False,
        add_self_loops=add_self_loops,
        kernel_initializer=tf.keras.initializers.Constant(tf.eye(3)),
    )
    self.assertAllClose(
        expected_result,
        conv(graph, edge_set_name=tfgnn.EDGES),
        rtol=1e-06,
        atol=1e-06,
    )

  def test_gcnconv_activation(self):
    """Tests that the activation function is correctly passed through."""
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            tfgnn.NODES: tfgnn.NodeSet.from_fields(
                sizes=[2],
                features={
                    tfgnn.HIDDEN_STATE: tf.constant([[-1.0, 0, 0], [0, 1, 0]])
                },
            )
        },
        edge_sets={
            tfgnn.EDGES: tfgnn.EdgeSet.from_fields(
                sizes=[2],
                adjacency=tfgnn.Adjacency.from_indices(
                    source=(tfgnn.NODES, tf.constant([0, 1], dtype=tf.int64)),
                    target=(tfgnn.NODES, tf.constant([1, 0], dtype=tf.int64)),
                ),
            )
        },
    )
    conv_relu = gcn_conv.GCNConv(
        units=3,
        use_bias=False,
        add_self_loops=True,
        kernel_initializer=tf.keras.initializers.Constant(tf.eye(3)),
    )
    expected_result = tf.constant([[0.0, 0.5, 0], [0, 0.5, 0]])
    self.assertAllClose(
        expected_result,
        conv_relu(graph, edge_set_name=tfgnn.EDGES),
        rtol=1e-06,
        atol=1e-06,
    )

  def test_gcnconv_heterogeneous(self):
    """Tests that heterogeneous edges throw an error."""
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            tfgnn.NODES: tfgnn.NodeSet.from_fields(
                sizes=[2],
                features={
                    tfgnn.HIDDEN_STATE: tf.constant([[1.0, 0, 0], [0, 1, 0]])
                },
            )
        },
        edge_sets={
            tfgnn.EDGES: tfgnn.EdgeSet.from_fields(
                sizes=[2],
                adjacency=tfgnn.Adjacency.from_indices(
                    source=(tfgnn.NODES, tf.constant([0, 1], dtype=tf.int64)),
                    target=('antinodes', tf.constant([1, 0], dtype=tf.int64)),
                ),
            )
        },
    )
    conv = gcn_conv.GCNConv(units=3)
    self.assertRaisesRegex(
        ValueError,
        'source and target node sets must be the same for edge set edges ',
        lambda: conv(graph, edge_set_name=tfgnn.EDGES),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='noSelfLoops_noBias',
          use_bias=False,
          add_self_loops=False,
      ),
  )
  def test_gcnconv_with_edge_weights_ones(self, use_bias, add_self_loops):
    """Tests that gcn_conv returns the correct values with edge weights."""
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            tfgnn.NODES: tfgnn.NodeSet.from_fields(
                sizes=[2],
                features={
                    tfgnn.HIDDEN_STATE: tf.constant([[1.0, 0.0], [0.0, 1.0]])
                },
            )
        },
        edge_sets={
            tfgnn.EDGES: tfgnn.EdgeSet.from_fields(
                sizes=[2],
                features={'weights': tf.constant([1.0, 1.0], dtype=tf.float32)},
                adjacency=tfgnn.Adjacency.from_indices(
                    source=(tfgnn.NODES, tf.constant([0, 1], dtype=tf.int64)),
                    target=(tfgnn.NODES, tf.constant([1, 0], dtype=tf.int64)),
                ),
            )
        },
    )
    conv_with_edge_weights = gcn_conv.GCNConv(
        units=2,
        use_bias=use_bias,
        add_self_loops=add_self_loops,
        kernel_initializer=tf.keras.initializers.Constant(tf.eye(2)),
        edge_weight_feature_name='weights',
    )
    conv_without_edge_weights = gcn_conv.GCNConv(
        units=2,
        use_bias=use_bias,
        add_self_loops=add_self_loops,
        kernel_initializer=tf.keras.initializers.Constant(tf.eye(2)),
    )
    self.assertAllClose(
        conv_with_edge_weights(graph, edge_set_name=tfgnn.EDGES),
        conv_without_edge_weights(graph, edge_set_name=tfgnn.EDGES),
        rtol=1e-06,
        atol=1e-06,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='noSelfLoops_noBias',
          use_bias=False,
          add_self_loops=False,
          expected_result=tf.constant(
              [[0.0, 4.0 / (2.0 * 2.0)], [9.0 / (3.0 * 3.0), 0.0]]
          ),
      ),
      dict(
          testcase_name='selfLoops_noBias',
          use_bias=False,
          add_self_loops=True,
          expected_result=tf.constant([
              [
                  1.0 / (math.sqrt(5.0) * math.sqrt(10.0)),
                  4.0 / (math.sqrt(5.0) * math.sqrt(5.0)),
              ],
              [
                  9.0 / (math.sqrt(10.0) * math.sqrt(10.0)),
                  1.0 / (math.sqrt(10.0) * math.sqrt(5.0)),
              ],
          ]),
      ),
  )
  def test_gcnconv_with_edge_weights(
      self, use_bias, add_self_loops, expected_result
  ):
    """Tests that gcn_conv returns the correct values with edge weights."""
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            tfgnn.NODES: tfgnn.NodeSet.from_fields(
                sizes=[2],
                features={
                    tfgnn.HIDDEN_STATE: tf.constant([[1.0, 0.0], [0.0, 1.0]])
                },
            )
        },
        edge_sets={
            tfgnn.EDGES: tfgnn.EdgeSet.from_fields(
                sizes=[2],
                features={'weights': tf.constant([9.0, 4.0], dtype=tf.float32)},
                adjacency=tfgnn.Adjacency.from_indices(
                    source=(tfgnn.NODES, tf.constant([0, 1], dtype=tf.int64)),
                    target=(tfgnn.NODES, tf.constant([1, 0], dtype=tf.int64)),
                ),
            )
        },
    )
    conv = gcn_conv.GCNConv(
        units=2,
        use_bias=use_bias,
        add_self_loops=add_self_loops,
        kernel_initializer=tf.keras.initializers.Constant(tf.eye(2)),
        edge_weight_feature_name='weights',
    )

    self.assertAllClose(
        expected_result,
        conv(graph, edge_set_name=tfgnn.EDGES),
        rtol=1e-06,
        atol=1e-06,
    )

  def test_gcnconv_with_edge_weights_missing(self):
    """Tests that missing given edge weights feature name in the graph tensor throws an error."""
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            tfgnn.NODES: tfgnn.NodeSet.from_fields(
                sizes=[2],
                features={
                    tfgnn.HIDDEN_STATE: tf.constant([[1.0, 0.0], [0.0, 1.0]])
                },
            )
        },
        edge_sets={
            tfgnn.EDGES: tfgnn.EdgeSet.from_fields(
                sizes=[2],
                adjacency=tfgnn.Adjacency.from_indices(
                    source=(tfgnn.NODES, tf.constant([0, 1], dtype=tf.int64)),
                    target=(tfgnn.NODES, tf.constant([1, 0], dtype=tf.int64)),
                ),
            )
        },
    )

    conv = gcn_conv.GCNConv(units=2, edge_weight_feature_name='weights')
    self.assertRaisesRegex(
        ValueError,
        'weights is not given for edge set edges ',
        lambda: conv(graph, edge_set_name=tfgnn.EDGES),
    )

  def test_gcnconv_with_bad_shaped_edge_weights(self):
    """Tests that given edge weights feature with bad shape throws an error."""
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            tfgnn.NODES: tfgnn.NodeSet.from_fields(
                sizes=[2],
                features={
                    tfgnn.HIDDEN_STATE: tf.constant([[1.0, 0.0], [0.0, 1.0]])
                },
            )
        },
        edge_sets={
            tfgnn.EDGES: tfgnn.EdgeSet.from_fields(
                sizes=[2],
                features={
                    'weights': tf.constant([[9.0], [4.0]], dtype=tf.float32)
                },
                adjacency=tfgnn.Adjacency.from_indices(
                    source=(tfgnn.NODES, tf.constant([0, 1], dtype=tf.int64)),
                    target=(tfgnn.NODES, tf.constant([1, 0], dtype=tf.int64)),
                ),
            )
        },
    )

    conv = gcn_conv.GCNConv(units=2, edge_weight_feature_name='weights')
    self.assertRaisesRegex(
        ValueError,
        'Expecting vector for edge weights. Received rank 2.',
        lambda: conv(graph, edge_set_name=tfgnn.EDGES),
    )

  def test_gcnconv_with_unknown_normalization(self):
    """Tests that unknown degree normalization throws an error."""
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            tfgnn.NODES: tfgnn.NodeSet.from_fields(
                sizes=[2],
                features={
                    tfgnn.HIDDEN_STATE: tf.constant([[1.0, 0.0], [0.0, 1.0]])
                },
            )
        },
        edge_sets={
            tfgnn.EDGES: tfgnn.EdgeSet.from_fields(
                sizes=[2],
                adjacency=tfgnn.Adjacency.from_indices(
                    source=(tfgnn.NODES, tf.constant([0, 1], dtype=tf.int64)),
                    target=(tfgnn.NODES, tf.constant([1, 0], dtype=tf.int64)),
                ),
            )
        },
    )

    conv = gcn_conv.GCNConv(units=2, degree_normalization='unknown')
    self.assertRaisesRegex(
        ValueError,
        (
            'Expecting degree_normalization to be `none`, `in`, `out`,'
            ' `in_out`, or `in_in`'
        ),
        lambda: conv(graph, edge_set_name=tfgnn.EDGES),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='noneDegreeNormalization',
          degree_normalization='none',
          expected_result=tf.constant(
              [[1.0, 0.0, 3.0, 0.0], [2.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]
          ),
      ),
      dict(
          testcase_name='outDegreeNormalization',
          degree_normalization='out',
          expected_result=tf.constant([
              [1.0 / 3.0, 0.0, 3.0 / 4.0, 0.0],
              [2.0 / 3.0, 1.0 / 2.0, 0.0, 0.0],
              [0.0, 1.0 / 2.0, 1.0 / 4.0, 0.0],
          ]),
      ),
      dict(
          testcase_name='inDegreeNormalization',
          degree_normalization='in',
          expected_result=tf.constant([
              [1.0 / 4.0, 0.0, 3.0 / 4.0, 0.0],
              [2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0],
              [0.0, 1.0 / 2.0, 1.0 / 2.0, 0.0],
          ]),
      ),
      dict(
          testcase_name='in_outDegreeNormalization',
          degree_normalization='in_out',
          expected_result=tf.constant([
              [
                  1.0 / (math.sqrt(4.0) * math.sqrt(3.0)),
                  0.0,
                  3.0 / (math.sqrt(4.0) * math.sqrt(4.0)),
                  0.0,
              ],
              [
                  2.0 / (math.sqrt(3.0) * math.sqrt(3.0)),
                  1.0 / (math.sqrt(3.0) * math.sqrt(2.0)),
                  0.0,
                  0.0,
              ],
              [
                  0.0,
                  1.0 / (math.sqrt(2.0) * math.sqrt(2.0)),
                  1.0 / (math.sqrt(2.0) * math.sqrt(4.0)),
                  0.0,
              ],
          ]),
      ),
      dict(
          testcase_name='in_inDegreeNormalization',
          degree_normalization='in_in',
          expected_result=tf.constant([
              [
                  1.0 / (math.sqrt(4.0) * math.sqrt(4.0)),
                  0.0,
                  3.0 / (math.sqrt(4.0) * math.sqrt(2.0)),
                  0.0,
              ],
              [
                  2.0 / (math.sqrt(3.0) * math.sqrt(4.0)),
                  1.0 / (math.sqrt(3.0) * math.sqrt(3.0)),
                  0.0,
                  0.0,
              ],
              [
                  0.0,
                  1.0 / (math.sqrt(2.0) * math.sqrt(3.0)),
                  1.0 / (math.sqrt(2.0) * math.sqrt(2.0)),
                  0.0,
              ],
          ]),
      ),
  )
  def test_gcnconv_degree_normalization(
      self, degree_normalization, expected_result
  ):
    """Tests that gcn_conv returns the correct values with an asymmetric adjacency."""
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            tfgnn.NODES: tfgnn.NodeSet.from_fields(
                sizes=[4],
                features={
                    tfgnn.HIDDEN_STATE: tf.constant([
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ])
                },
            )
        },
        edge_sets={
            tfgnn.EDGES: tfgnn.EdgeSet.from_fields(
                sizes=[3],
                features={
                    'weights': tf.constant([2.0, 1.0, 3.0], dtype=tf.float32)
                },
                adjacency=tfgnn.Adjacency.from_indices(
                    source=(
                        tfgnn.NODES,
                        tf.constant([0, 1, 2], dtype=tf.int64),
                    ),
                    target=(
                        tfgnn.NODES,
                        tf.constant([1, 2, 0], dtype=tf.int64),
                    ),
                ),
            )
        },
    )

    conv = gcn_conv.GCNConv(
        units=4,
        use_bias=False,
        add_self_loops=True,
        kernel_initializer=tf.keras.initializers.Constant(tf.eye(4)),
        edge_weight_feature_name='weights',
        degree_normalization=degree_normalization,
    )

    self.assertAllClose(
        expected_result,
        conv(graph, edge_set_name=tfgnn.EDGES),
        rtol=1e-06,
        atol=1e-06,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='inDegreeNormalization',
          degree_normalization='in',
          # Same as the expected result in the previous test.
          expected_result=tf.constant([
              [1.0 / 4.0, 0.0, 3.0 / 4.0],
              [2.0 / 3.0, 1.0 / 3.0, 0.0],
              [0.0, 1.0 / 2.0, 1.0 / 2.0],
          ]),
      ),
      dict(
          testcase_name='outDegreeNormalization',
          degree_normalization='out',
          # Same as the expected result in the previous test.
          expected_result=tf.constant([
              [1.0 / 3.0, 0.0, 3.0 / 4.0],
              [2.0 / 3.0, 1.0 / 2.0, 0.0],
              [0.0, 1.0 / 2.0, 1.0 / 4.0],
          ]),
      ),
  )
  def test_gcnconv_degree_normalization_reverse_edges(
      self, degree_normalization, expected_result
  ):
    """Tests that gcn_conv returns the correct values with an asymmetric adjacency."""
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            tfgnn.NODES: tfgnn.NodeSet.from_fields(
                sizes=[3],
                features={
                    tfgnn.HIDDEN_STATE: tf.constant(
                        [[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0],]
                    )
                },
            )
        },
        edge_sets={
            tfgnn.EDGES: tfgnn.EdgeSet.from_fields(
                sizes=[3],
                features={
                    'weights': tf.constant([2.0, 1.0, 3.0], dtype=tf.float32)
                },
                adjacency=tfgnn.Adjacency.from_indices(
                    # Swapping source and target compared to the previous test.
                    target=(
                        tfgnn.NODES,
                        tf.constant([0, 1, 2], dtype=tf.int64),
                    ),
                    source=(
                        tfgnn.NODES,
                        tf.constant([1, 2, 0], dtype=tf.int64),
                    ),
                ),
            )
        },
    )

    conv = gcn_conv.GCNConv(
        units=3,
        # Changing the receiver to be the source node instead of the target node
        receiver_tag=tfgnn.SOURCE,
        use_bias=False,
        add_self_loops=True,
        kernel_initializer=tf.keras.initializers.Constant(tf.eye(3)),
        edge_weight_feature_name='weights',
        degree_normalization=degree_normalization,
    )

    self.assertAllClose(
        expected_result,
        conv(graph, edge_set_name=tfgnn.EDGES),
        rtol=1e-06,
        atol=1e-06,
    )

  def test_gcnconv_symmetric_adj(self):
    """Tests that gcn_conv returns the same values for a symmetric adjacency with in_in and in_out normalizations."""
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            tfgnn.NODES: tfgnn.NodeSet.from_fields(
                sizes=[3],
                features={
                    tfgnn.HIDDEN_STATE: tf.constant(
                        [[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]]
                    )
                },
            )
        },
        edge_sets={
            tfgnn.EDGES: tfgnn.EdgeSet.from_fields(
                sizes=[4],
                adjacency=tfgnn.Adjacency.from_indices(
                    source=(
                        tfgnn.NODES,
                        tf.constant([0, 0, 1, 2], dtype=tf.int64),
                    ),
                    target=(
                        tfgnn.NODES,
                        tf.constant([1, 2, 0, 0], dtype=tf.int64),
                    ),
                ),
            )
        },
    )

    conv_in_out = gcn_conv.GCNConv(
        units=3,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.Constant(tf.eye(3)),
        degree_normalization='in_out',
    )

    conv_in_in = gcn_conv.GCNConv(
        units=3,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.Constant(tf.eye(3)),
        degree_normalization='in_in',
    )
    self.assertAllEqual(
        conv_in_in(graph, edge_set_name=tfgnn.EDGES),
        conv_in_out(graph, edge_set_name=tfgnn.EDGES),
    )

  @parameterized.named_parameters(
      ('', ReloadModel.SKIP),
      ('Restored', ReloadModel.SAVED_MODEL),
      ('RestoredKeras', ReloadModel.KERAS),
  )
  def test_full_model(self, reload_model):
    """Tests GCNGraphUpdate in a full Model (incl. saving) with edge input."""
    gt_input = tfgnn.GraphTensor.from_pieces(
        node_sets={
            tfgnn.NODES: tfgnn.NodeSet.from_fields(
                sizes=[2],
                features={
                    tfgnn.HIDDEN_STATE: tf.constant([[1.0, 0, 0], [0, 1, 0]])
                },
            )
        },
        edge_sets={
            tfgnn.EDGES: tfgnn.EdgeSet.from_fields(
                sizes=[2],
                adjacency=tfgnn.Adjacency.from_indices(
                    source=(tfgnn.NODES, tf.constant([0, 1], dtype=tf.int64)),
                    target=(tfgnn.NODES, tf.constant([1, 0], dtype=tf.int64)),
                ),
            )
        },
    )
    l2reg = 0.1  # Coefficient for L2 regularization.
    layer = gcn_conv.GCNHomGraphUpdate(
        units=3,
        add_self_loops=True,
        kernel_regularizer=tf.keras.regularizers.l2(l2reg),
    )
    _ = layer(gt_input)  # Build weights.
    weights = {v.name: v for v in layer.trainable_weights}
    self.assertLen(weights, 2)
    weights['gcn/node_set_update/gcn_conv/dense/bias:0'].assign([0.0, 0.0, 0.0])
    weights['gcn/node_set_update/gcn_conv/dense/kernel:0'].assign([
        [1.0, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    # Build a Model around the Layer, possibly saved and restored.
    inputs = tf.keras.layers.Input(type_spec=gt_input.spec)
    outputs = layer(inputs)
    model = tf.keras.Model(inputs, outputs)
    if reload_model:
      export_dir = os.path.join(self.get_temp_dir(), 'gcn')
      model.save(export_dir, include_optimizer=False)
      if reload_model == ReloadModel.KERAS:
        model = tf.keras.models.load_model(export_dir)
        # Check that from_config() worked, no fallback to a function trace, see
        # https://www.tensorflow.org/guide/keras/save_and_serialize#how_savedmodel_handles_custom_objects
        self.assertIsInstance(
            model.get_layer(index=1), tfgnn.keras.layers.GraphUpdate
        )
      else:
        model = tf.saved_model.load(export_dir)

    if not reload_model or reload_model == ReloadModel.KERAS:
      # Model.losses only works on Keras models. tf.saved_model.load(), however,
      # does not return a Keras model. See:
      # https://www.tensorflow.org/api_docs/python/tf/saved_model/load
      kernel_variables = [
          v for v in model.trainable_variables if '/kernel:0' in v.name
      ]
      self.assertLen(kernel_variables, 1)  # 1 kernel variable per `weights[]`.
      self.assertLen(model.losses, 1)  # One loss term per kernel variable.

      expected_model_loss = tf.reduce_sum(kernel_variables[0] ** 2) * l2reg
      self.assertAllClose(model.losses[0], expected_model_loss)

    got_gt = model(gt_input)
    got = got_gt.node_sets['nodes'][tfgnn.HIDDEN_STATE]

    # The fourth column with values x.y from nodes is analogous to the
    # testBasic test above, with the contribution x from the favored
    # input before the decimal dot and the other contribution y after.
    # The fifth column with values (2x).(3y) is from edges, with the
    # multipliers 2 and 3 used above in setting up the edge features.
    want = tf.constant([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
    self.assertAllEqual(got.shape, (2, 3))
    self.assertAllClose(got, want, atol=0.0001)

  def test_full_model_heterogeneous(self):
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            'paper': tfgnn.NodeSet.from_fields(
                features={'f': tf.constant([[1.0, 2.0, 3.0],
                                            [2.0, 1.0, 3.0]])},
                sizes=tf.constant([1, 1]),
            ),
            'author': tfgnn.NodeSet.from_fields(
                features={'f': tf.constant([[1.0, 0.0], [0.0, 2.0]] * 2)},
                sizes=tf.constant([2, 2]),
            ),
        },
        edge_sets={
            'written': tfgnn.EdgeSet.from_fields(
                features={},
                sizes=tf.constant([2, 1]),
                adjacency=tfgnn.Adjacency.from_indices(
                    ('paper', tf.constant([0, 0, 1])),
                    ('author', tf.constant([1, 0, 3])),
                ),
            ),
        },
    )
    layer = gcn_conv.GCNHomGraphUpdate(units=3)
    self.assertRaises(ValueError, lambda: layer(graph))

  def test_no_nans(self):
    gt_input = tfgnn.GraphTensor.from_pieces(
        node_sets={
            tfgnn.NODES: tfgnn.NodeSet.from_fields(
                sizes=[2],
                features={
                    tfgnn.HIDDEN_STATE: tf.constant([[1.0, 0, 0],
                                                     [0, 1, 0]])
                },
            )
        },
        edge_sets={
            tfgnn.EDGES: tfgnn.EdgeSet.from_fields(
                sizes=[2],
                adjacency=tfgnn.Adjacency.from_indices(
                    source=(tfgnn.NODES, tf.constant([0, 1], dtype=tf.int64)),
                    target=(tfgnn.NODES, tf.constant([0, 0], dtype=tf.int64)),
                ),
            )
        },
    )
    l2reg = 0.1  # Coefficient for L2 regularization.
    layer = gcn_conv.GCNHomGraphUpdate(
        units=3,
        add_self_loops=False,
        kernel_regularizer=tf.keras.regularizers.l2(l2reg),
    )
    _ = layer(gt_input)  # Build weights.
    weights = {v.name: v for v in layer.trainable_weights}
    self.assertLen(weights, 2)
    weights['gcn/node_set_update/gcn_conv/dense/bias:0'].assign([0.0, 0.0, 0.0])
    weights['gcn/node_set_update/gcn_conv/dense/kernel:0'].assign([
        [1.0, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    node_feats = layer(gt_input).node_sets['nodes'][tfgnn.HIDDEN_STATE]
    second_row = node_feats[1]
    # Although no leading connections, there should be 0's rather than NaNs.
    self.assertAllClose(second_row, tf.zeros_like(second_row))


class GCNTFLiteTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('Simplest', False, None),
      ('OnlySelfLoops', True, None),
      ('AllOps', True, 'edge_weights'),
  )
  def testBasic(self, add_self_loops, edge_weight_feature_name):
    test_graph_1_dict = {
        # We care that the TFLite interpreter gives the same output as the
        # initialized weights that we keep here).
        'source': tf.constant([0, 1, 2, 3]),
        'target': tf.constant([1, 0, 3, 2]),
        'node_features': tf.constant([[1.0, 0, 0], [0, 1, 0]] * 2),
        'edge_weights': tf.constant([1.2, 0.8, 0.9, 1.1])
    }

    layer = gcn_conv.GCNHomGraphUpdate(
        units=4,
        add_self_loops=add_self_loops,
        edge_weight_feature_name=edge_weight_feature_name,
    )

    inputs = {
        'node_features': tf.keras.Input([3], None, 'node_features', tf.float32),
        'source': tf.keras.Input([], None, 'source', tf.int32),
        'target': tf.keras.Input([], None, 'target', tf.int32),
        'edge_weights': tf.keras.Input([], None, 'edge_weights', tf.float32),
    }
    graph_in = _MakeGraphTensor()(inputs)
    graph_out = layer(graph_in)
    outputs = tf.keras.layers.Layer(name='final_node_states')(
        graph_out.node_sets['nodes'][tfgnn.HIDDEN_STATE]
    )
    model = tf.keras.Model(inputs, outputs)

    # The other unit tests should verify that this is correct
    expected = model(test_graph_1_dict).numpy()

    # TODO(b/276291104): Remove when TF 2.11+ is required by all of TFGNN
    if tf.__version__.startswith('2.10.'):
      self.skipTest('GNN models are unsupported in TFLite until TF 2.11 but '
                    f'got TF {tf.__version__}')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    model_content = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=model_content)
    signature_runner = interpreter.get_signature_runner('serving_default')
    obtained = signature_runner(**test_graph_1_dict)['final_node_states']
    self.assertAllClose(expected, obtained)


# TODO(b/274779989): Replace this layer with a more standard representation
# of GraphTensor as a dict of plain Tensors.
class _MakeGraphTensor(tf.keras.layers.Layer):
  """Makes a homogeneous GraphTensor of rank 0 with a single component."""

  def call(self, inputs):
    node_sizes = tf.shape(inputs['node_features'])[0]
    edge_sizes = tf.shape(inputs['edge_weights'])[0]
    return tfgnn.GraphTensor.from_pieces(
        node_sets={
            'nodes': tfgnn.NodeSet.from_fields(
                sizes=tf.expand_dims(node_sizes, axis=0),
                features={tfgnn.HIDDEN_STATE: inputs['node_features']},
            )
        },
        edge_sets={
            'edges': tfgnn.EdgeSet.from_fields(
                sizes=tf.expand_dims(edge_sizes, axis=0),
                adjacency=tfgnn.Adjacency.from_indices(
                    ('nodes', inputs['source']), ('nodes', inputs['target'])
                ),
                features={'edge_weights': inputs['edge_weights']},
            )
        },
    )


if __name__ == '__main__':
  tf.test.main()
