"""Tests for gcn_conv."""
import enum
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
                  tfgnn.NODES:
                      tfgnn.NodeSet.from_fields(
                          sizes=[2],
                          features={
                              tfgnn.HIDDEN_STATE:
                                  tf.constant([[1., 0, 0], [0, 1, 0]])
                          },
                      )
              },
              edge_sets={
                  tfgnn.EDGES:
                      tfgnn.EdgeSet.from_fields(
                          sizes=[2],
                          adjacency=tfgnn.Adjacency.from_indices(
                              source=(tfgnn.NODES,
                                      tf.constant([0, 1], dtype=tf.int64)),
                              target=(tfgnn.NODES,
                                      tf.constant([1, 0], dtype=tf.int64)),
                          ))
              }),
          add_self_loops=False,
          expected_result=tf.constant([[0, 1, 0], [1., 0, 0]])),
      dict(
          testcase_name='selfLoops',
          graph=tfgnn.GraphTensor.from_pieces(
              node_sets={
                  tfgnn.NODES:
                      tfgnn.NodeSet.from_fields(
                          sizes=[2],
                          features={
                              tfgnn.HIDDEN_STATE:
                                  tf.constant([[1., 0, 0], [0, 1, 0]])
                          },
                      )
              },
              edge_sets={
                  tfgnn.EDGES:
                      tfgnn.EdgeSet.from_fields(
                          sizes=[2],
                          adjacency=tfgnn.Adjacency.from_indices(
                              source=(tfgnn.NODES,
                                      tf.constant([0, 1], dtype=tf.int64)),
                              target=(tfgnn.NODES,
                                      tf.constant([1, 0], dtype=tf.int64)),
                          ))
              }),
          add_self_loops=True,
          expected_result=tf.constant([[0.5, 0.5, 0], [0.5, 0.5, 0]]),
      ),
      dict(
          testcase_name='discreteComponents',
          graph=tfgnn.GraphTensor.from_pieces(
              node_sets={
                  tfgnn.NODES:
                      tfgnn.NodeSet.from_fields(
                          sizes=[2, 2],
                          features={
                              tfgnn.HIDDEN_STATE:
                                  tf.constant([[1., 0, 0], [0, 1, 0]] * 2)
                          },
                      )
              },
              edge_sets={
                  tfgnn.EDGES:
                      tfgnn.EdgeSet.from_fields(
                          sizes=[2, 2],
                          adjacency=tfgnn.Adjacency.from_indices(
                              source=(tfgnn.NODES,
                                      tf.constant([0, 1, 2, 3],
                                                  dtype=tf.int64)),
                              target=(tfgnn.NODES,
                                      tf.constant([1, 0, 3, 2],
                                                  dtype=tf.int64)),
                          ))
              }),
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
        kernel_initializer=tf.keras.initializers.Constant(tf.eye(3)))
    self.assertAllClose(expected_result,
                        conv(graph, edge_set_name=tfgnn.EDGES),
                        rtol=1e-06, atol=1e-06)

  def test_gcnconv_activation(self):
    """Tests that the activation function is correctly passed through."""
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            tfgnn.NODES:
                tfgnn.NodeSet.from_fields(
                    sizes=[2],
                    features={
                        tfgnn.HIDDEN_STATE:
                            tf.constant([[-1., 0, 0], [0, 1, 0]])
                    },
                )
        },
        edge_sets={
            tfgnn.EDGES:
                tfgnn.EdgeSet.from_fields(
                    sizes=[2],
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=(tfgnn.NODES, tf.constant([0, 1],
                                                         dtype=tf.int64)),
                        target=(tfgnn.NODES, tf.constant([1, 0],
                                                         dtype=tf.int64)),
                    ))
        })
    conv_relu = gcn_conv.GCNConv(
        units=3,
        use_bias=False,
        add_self_loops=True,
        kernel_initializer=tf.keras.initializers.Constant(tf.eye(3)))
    expected_result = tf.constant([[0., 0.5, 0], [0, 0.5, 0]])
    self.assertAllClose(expected_result,
                        conv_relu(graph, edge_set_name=tfgnn.EDGES),
                        rtol=1e-06, atol=1e-06)

  def test_gcnconv_heterogeneous(self):
    """Tests that heterogeneous edges throw an error."""
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            tfgnn.NODES:
                tfgnn.NodeSet.from_fields(
                    sizes=[2],
                    features={
                        tfgnn.HIDDEN_STATE:
                            tf.constant([[1., 0, 0], [0, 1, 0]])
                    },
                )
        },
        edge_sets={
            tfgnn.EDGES:
                tfgnn.EdgeSet.from_fields(
                    sizes=[2],
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=(tfgnn.NODES, tf.constant([0, 1],
                                                         dtype=tf.int64)),
                        target=('antinodes', tf.constant([1, 0],
                                                         dtype=tf.int64)),
                    ))
        })
    conv = gcn_conv.GCNConv(units=3)
    self.assertRaisesRegex(ValueError,
                           ('source and target node sets must be the same '
                            'for edge set edges '),
                           lambda: conv(graph, edge_set_name=tfgnn.EDGES))

  @parameterized.named_parameters(
      ('', ReloadModel.SKIP),
      ('Restored', ReloadModel.SAVED_MODEL),
      ('RestoredKeras', ReloadModel.KERAS))
  def test_full_model(self, reload_model):
    """Tests GCNGraphUpdate in a full Model (incl. saving) with edge input."""
    gt_input = tfgnn.GraphTensor.from_pieces(
        node_sets={
            tfgnn.NODES:
                tfgnn.NodeSet.from_fields(
                    sizes=[2],
                    features={
                        tfgnn.HIDDEN_STATE:
                            tf.constant([[1., 0, 0], [0, 1, 0]])
                    },
                )
        },
        edge_sets={
            tfgnn.EDGES:
                tfgnn.EdgeSet.from_fields(
                    sizes=[2],
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=(tfgnn.NODES, tf.constant([0, 1],
                                                         dtype=tf.int64)),
                        target=(tfgnn.NODES, tf.constant([1, 0],
                                                         dtype=tf.int64)),
                    ))
        })
    layer = gcn_conv.GCNHomGraphUpdate(units=3, add_self_loops=True)
    _ = layer(gt_input)  # Build weights.
    weights = {v.name: v for v in layer.trainable_weights}
    self.assertLen(weights, 2)
    weights['gcn/node_set_update/gcn_conv/dense/bias:0'].assign(
        [0., 0., 0.])
    weights['gcn/node_set_update/gcn_conv/dense/kernel:0'].assign(
        [[1., 0, 0,],
         [0, 1, 0,],
         [0, 0, 1,]])

    # Build a Model around the Layer, possibly saved and restored.
    inputs = tf.keras.layers.Input(type_spec=gt_input.spec)
    outputs = layer(inputs)
    model = tf.keras.Model(inputs, outputs)
    if reload_model:
      export_dir = os.path.join(self.get_temp_dir(), 'gcn')
      model.save(export_dir, include_optimizer=False)
      if reload_model == ReloadModel.KERAS:
        model = tf.keras.models.load_model(export_dir)
      else:
        model = tf.saved_model.load(export_dir)

    got_gt = model(gt_input)
    got = got_gt.node_sets['nodes'][tfgnn.HIDDEN_STATE]

    # The fourth column with values x.y from nodes is analogous to the
    # testBasic test above, with the contribution x from the favored
    # input before the decimal dot and the other contribution y after.
    # The fifth column with values (2x).(3y) is from edges, with the
    # multipliers 2 and 3 used above in setting up the edge features.
    want = tf.constant([[0.5, 0.5, 0.],
                        [0.5, 0.5, 0.]])
    self.assertAllEqual(got.shape, (2, 3))
    self.assertAllClose(got, want, atol=.0001)

  def test_full_model_heterogeneous(self):
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            'paper':
                tfgnn.NodeSet.from_fields(
                    features={'f': tf.constant([[1., 2., 3.], [2., 1., 3.]])},
                    sizes=tf.constant([1, 1])),
            'author':
                tfgnn.NodeSet.from_fields(
                    features={'f': tf.constant([[1., 0.], [0., 2.]] * 2)},
                    sizes=tf.constant([2, 2])),
        },
        edge_sets={
            'written':
                tfgnn.EdgeSet.from_fields(
                    features={},
                    sizes=tf.constant([2, 1]),
                    adjacency=tfgnn.Adjacency.from_indices(
                        ('paper', tf.constant([0, 0, 1])),
                        ('author', tf.constant([1, 0, 3])),
                    )),
        },
    )
    layer = gcn_conv.GCNHomGraphUpdate(units=3)
    self.assertRaises(ValueError, lambda: layer(graph))

if __name__ == '__main__':
  tf.test.main()
