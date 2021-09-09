from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn


ct = tf.constant
rt = tf.ragged.constant


class NormalizationOpsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('simple',
       tfgnn.GraphTensor.from_pieces(
           node_sets={
               'signals':
                   tfgnn.NodeSet.from_fields(
                       features={
                           tfgnn.DEFAULT_STATE_NAME: ct([1., 5., 3., 2., 4.]),
                       },
                       sizes=[5]),
           },
           edge_sets={
               'edges':
                   tfgnn.EdgeSet.from_fields(
                       sizes=[25],
                       adjacency=tfgnn.Adjacency.from_indices(
                           ('signals',
                            ct([
                                0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3,
                                3, 3, 3, 3, 4, 4, 4, 4, 4
                            ])), ('signals',
                                  ct([
                                      0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3,
                                      4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4
                                  ])))),
           }), tf.repeat(ct([0.0116, 0.6364, 0.0861, 0.0316, 0.234]), 5)),
      ('two_components',
       tfgnn.GraphTensor.from_pieces(
           node_sets={
               'signals':
                   tfgnn.NodeSet.from_fields(
                       features={
                           tfgnn.DEFAULT_STATE_NAME:
                               rt([[1., 5., 3., 2., 4.], [1., 2., 3.]]),
                       },
                       sizes=[[5], [3]]),
           },
           edge_sets={
               'edges':
                   tfgnn.EdgeSet.from_fields(
                       sizes=[[25], [9]],
                       adjacency=tfgnn.Adjacency.from_indices(
                           ('signals',
                            rt([[
                                0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3,
                                3, 3, 3, 3, 4, 4, 4, 4, 4
                            ], [0, 0, 0, 1, 1, 1, 2, 2, 2]])),
                           ('signals',
                            rt([[
                                0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0,
                                1, 2, 3, 4, 0, 1, 2, 3, 4
                            ], [0, 1, 2, 0, 1, 2, 0, 1, 2]])))),
           }).merge_batch_to_components(),
       tf.concat([
           tf.repeat(ct([0.0116, 0.6364, 0.0861, 0.0316, 0.234]), 5),
           tf.repeat(ct([.0900, .2447, .6654]), 3)
       ],
                 axis=0)),
      ('one_component_empty',
       tfgnn.GraphTensor.from_pieces(
           node_sets={
               'signals':
                   tfgnn.NodeSet.from_fields(
                       features={
                           tfgnn.DEFAULT_STATE_NAME: ct([]),
                       }, sizes=[0]),
           },
           edge_sets={
               'edges':
                   tfgnn.EdgeSet.from_fields(
                       sizes=[0],
                       adjacency=tfgnn.Adjacency.from_indices(
                           ('signals', ct([], dtype=tf.int32)),
                           ('signals', ct([], dtype=tf.int32)))),
           }), ct([])),
      ('one_nonempty_one_empty',
       tfgnn.GraphTensor.from_pieces(
           node_sets={
               'signals':
                   tfgnn.NodeSet.from_fields(
                       features={
                           tfgnn.DEFAULT_STATE_NAME:
                               rt([[1., 5., 3., 2., 4.], []]),
                       },
                       sizes=[[5], [0]]),
           },
           edge_sets={
               'edges':
                   tfgnn.EdgeSet.from_fields(
                       sizes=[[25], [0]],
                       adjacency=tfgnn.Adjacency.from_indices(
                           ('signals',
                            rt([[
                                0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3,
                                3, 3, 3, 3, 4, 4, 4, 4, 4
                            ], []])), ('signals',
                                       rt([[
                                           0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1,
                                           2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4
                                       ], []])))),
           }).merge_batch_to_components(),
       tf.concat(
           [tf.repeat(ct([0.0116, 0.6364, 0.0861, 0.0316, 0.234]), 5),
            ct([])],
           axis=0)),
  )
  def testSoftmax(self, gt_input, want):
    """Unit tests for the softmax function."""
    broadcasted = tfgnn.broadcast_node_to_edges(
        gt_input, 'edges', tfgnn.SOURCE, feature_name=tfgnn.DEFAULT_STATE_NAME)
    got = tfgnn.softmax_edges_per_node(
        gt_input, 'edges', tfgnn.TARGET, feature_value=broadcasted)
    self.assertAllClose(got, want, atol=.001)


if __name__ == '__main__':
  tf.test.main()
