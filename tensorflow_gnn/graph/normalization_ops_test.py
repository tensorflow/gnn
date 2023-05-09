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
from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import broadcast_ops
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import normalization_ops


ct = tf.constant
rt = tf.ragged.constant


class NormalizationOpsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('simple',
       gt.GraphTensor.from_pieces(
           node_sets={
               'signals':
                   gt.NodeSet.from_fields(
                       features={
                           const.HIDDEN_STATE: ct([1., 5., 3., 2., 4.]),
                       },
                       sizes=[5]),
           },
           edge_sets={
               'edges':
                   gt.EdgeSet.from_fields(
                       sizes=[25],
                       adjacency=adj.Adjacency.from_indices(
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
       gt.GraphTensor.from_pieces(
           node_sets={
               'signals':
                   gt.NodeSet.from_fields(
                       features={
                           const.HIDDEN_STATE:
                               rt([[1., 5., 3., 2., 4.], [1., 2., 3.]]),
                       },
                       sizes=[[5], [3]]),
           },
           edge_sets={
               'edges':
                   gt.EdgeSet.from_fields(
                       sizes=[[25], [9]],
                       adjacency=adj.Adjacency.from_indices(
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
       gt.GraphTensor.from_pieces(
           node_sets={
               'signals':
                   gt.NodeSet.from_fields(
                       features={
                           const.HIDDEN_STATE: ct([]),
                       }, sizes=[0]),
           },
           edge_sets={
               'edges':
                   gt.EdgeSet.from_fields(
                       sizes=[0],
                       adjacency=adj.Adjacency.from_indices(
                           ('signals', ct([], dtype=tf.int32)),
                           ('signals', ct([], dtype=tf.int32)))),
           }), ct([])),
      ('one_nonempty_one_empty',
       gt.GraphTensor.from_pieces(
           node_sets={
               'signals':
                   gt.NodeSet.from_fields(
                       features={
                           const.HIDDEN_STATE:
                               rt([[1., 5., 3., 2., 4.], []]),
                       },
                       sizes=[[5], [0]]),
           },
           edge_sets={
               'edges':
                   gt.EdgeSet.from_fields(
                       sizes=[[25], [0]],
                       adjacency=adj.Adjacency.from_indices(
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
    broadcasted = broadcast_ops.broadcast_node_to_edges(
        gt_input, 'edges', const.SOURCE, feature_name=const.HIDDEN_STATE)
    got = normalization_ops.softmax_edges_per_node(
        gt_input, 'edges', const.TARGET, feature_value=broadcasted)
    self.assertAllClose(got, want, atol=.001)

  def testSoftmaxMultipleEdgeSets(self):
    """Tests softmax() across multiple edge sets."""
    graph_tensor = gt.GraphTensor.from_pieces(
        node_sets={
            'air': gt.NodeSet.from_fields(sizes=[3],),
            'ground': gt.NodeSet.from_fields(sizes=[2],)
        },
        edge_sets={
            'aa': gt.EdgeSet.from_fields(
                sizes=[4],
                adjacency=adj.Adjacency.from_indices(
                    ('air', [0, 1, 2, 1]),
                    ('air', [1, 2, 1, 0]))),
            'ga': gt.EdgeSet.from_fields(
                sizes=[3],
                adjacency=adj.Adjacency.from_indices(
                    ('ground', [0, 1, 0]),
                    ('air', [2, 0, 2]))),
        })
    edge_set_names = ['aa', 'ga']
    # Feature values log(x_i) are chosen such that sum(x_i) == 100
    # at each target node.
    feature_values = [tf.math.log(tf.constant([60., 50., 40., 20.])),
                      tf.math.log(tf.constant([25., 80., 25.]))]
    expected_aa = tf.constant([0.6, 0.5, 0.4, 0.2])
    expected_ga = tf.constant([0.25, 0.8, 0.25])
    actual = normalization_ops.softmax(
        graph_tensor, const.TARGET,
        edge_set_name=edge_set_names, feature_value=feature_values)
    self.assertLen(actual, len(edge_set_names))
    actual_aa, actual_ga = actual
    self.assertAllClose(actual_aa, expected_aa)
    self.assertAllClose(actual_ga, expected_ga)

  @parameterized.product(
      # The descriptive names are meant to make test output easier to read.
      relation=['EdgeToNode', 'EdgeToContext', 'NodeToContext'],
      value=['Balanced', 'Huge', 'Tiny'])
  def testSoftmaxGradient(self, relation, value):
    """Tests softmax() and its derivative, also for large offsets."""

    # A graph with two-to-one relationships off all kinds: edge to target node,
    # edge to context, source node to context.
    graph_tensor = gt.GraphTensor.from_pieces(
        node_sets={
            'source': gt.NodeSet.from_fields(sizes=[2]),
            'target': gt.NodeSet.from_fields(sizes=[2]),  # 2nd node unused.
        },
        edge_sets={
            'edges': gt.EdgeSet.from_fields(
                sizes=[2],
                adjacency=adj.Adjacency.from_indices(
                    ('source', ct([0, 1], dtype=tf.int32)),
                    ('target', ct([0, 0], dtype=tf.int32)))),
        })

    # Get the args that define the relation in the graph for which
    # softmax normalization is done.
    per_tag, name_kwarg = {
        'EdgeToNode': (const.TARGET, dict(edge_set_name='edges')),
        'EdgeToContext': (const.CONTEXT, dict(edge_set_name='edges')),
        'NodeToContext': (const.CONTEXT, dict(node_set_name='source'))
    }[relation]

    # Get the target weights, before normalizing to a total of 1.
    # Gradient computation is tested w.r.t. the first weight.
    epsilon = 1e-5
    unnormalized_target = {
        'Balanced': [2., 3.],
        'Huge': [5.-epsilon, epsilon],
        'Tiny': [epsilon, 5.-epsilon],
    }[value]

    # Mathematically, softmax() is invariant under adding a common offset to all
    # inputs, but the implementation has to take care that the values of exp(_)
    # don't overflow to infs or underflow to erase the distinction between
    # inputs. Some graceful degradation is expected once the offset is so big
    # that adding it to the inputs starts to cancel out theit differences even
    # before softmax() is applied.
    for offset, atol in [(0., 1e-6),
                         (10., 1e-6), (100., 1e-6),
                         (-10., 1e-6), (-100., 1e-6),
                         (1000., 1e-5), (10000., 1e-4), (100000., 1e-3)]:
      msg = f'for offset {offset}'
      with tf.GradientTape() as tape:
        exp_x = tf.constant(unnormalized_target)  # Ground truth without offset.
        x = tf.math.log(exp_x) + offset
        tape.watch(x)
        y_actual = normalization_ops.softmax(
            graph_tensor, per_tag, **name_kwarg, feature_value=x)
        y0_actual = y_actual[0]
      # Recall y[i] = exp(x[i]) / (exp(x[0]) + exp(x[1])).
      y_expected = exp_x / tf.reduce_sum(exp_x)
      self.assertAllClose(y_expected, y_actual, atol=atol, rtol=0., msg=msg)

      dy0_dx_actual = tape.gradient(y0_actual, x)
      # The https://en.wikipedia.org/wiki/Quotient_rule yields:
      dy0_dx_expected = [
          exp_x[0] * exp_x[1], -exp_x[0] * exp_x[1]
      ] / (exp_x[0] + exp_x[1])**2
      self.assertAllClose(dy0_dx_expected, dy0_dx_actual,
                          atol=atol, rtol=0., msg=msg)


if __name__ == '__main__':
  tf.test.main()
