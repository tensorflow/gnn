"""Tests for gcn_conv."""
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models.gcn import gcn_conv


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
                              tfgnn.DEFAULT_STATE_NAME:
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
                              tfgnn.DEFAULT_STATE_NAME:
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
                              tfgnn.DEFAULT_STATE_NAME:
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
                        tfgnn.DEFAULT_STATE_NAME:
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
                        tfgnn.DEFAULT_STATE_NAME:
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


if __name__ == '__main__':
  tf.test.main()
