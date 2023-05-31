# Copyright 2023 The TensorFlow GNN Authors. All Rights Reserved.
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
"""Tests for pool_ops.py."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import pool_ops


class PoolWrappersTest(tf.test.TestCase, parameterized.TestCase):
  """Tests forwarding from pool_*_to_*() to pool()."""

  def testPoolNodesToContext(self):
    input_graph = _get_test_graph_abc_efx()
    self.assertAllClose(
        tf.constant([[3./2.], [4.]]),
        pool_ops.pool_nodes_to_context(
            input_graph, "a", "mean", feature_name="feat"))
    self.assertAllClose(
        tf.constant([30./2., 40.]),
        pool_ops.pool_nodes_to_context(
            input_graph, "a", "mean",
            feature_value=tf.constant([10., 20., 40.])))

  def testPoolEdgesToContext(self):
    input_graph = _get_test_graph_abc_efx()
    self.assertAllClose(
        tf.constant([[30./2.], [40.]]),
        pool_ops.pool_edges_to_context(
            input_graph, "e", "mean", feature_name="feat"))
    self.assertAllClose(
        tf.constant([3./2., 4.]),
        pool_ops.pool_edges_to_context(
            input_graph, "e", "mean",
            feature_value=tf.constant([1., 2., 4.])))

  def testPoolEdgesToNode(self):
    input_graph = _get_test_graph_abc_efx()
    self.assertAllClose(
        tf.constant([[10.], [20.], [40.]]),
        pool_ops.pool_edges_to_node(
            input_graph, "e", const.SOURCE, "mean", feature_name="feat"))
    self.assertAllClose(
        tf.constant([1., 2., 4.]),
        pool_ops.pool_edges_to_node(
            input_graph, "e", const.SOURCE, "mean",
            feature_value=tf.constant([1., 2., 4.])))
    self.assertAllClose(
        tf.constant([[30./2.], [40.]]),
        pool_ops.pool_edges_to_node(
            input_graph, "e", const.TARGET, "mean", feature_name="feat"))
    self.assertAllClose(
        tf.constant([3./2., 4.]),
        pool_ops.pool_edges_to_node(
            input_graph, "e", const.TARGET, "mean",
            feature_value=tf.constant([1., 2., 4.])))


class PoolTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for pool(), excluding specifics of indivdual reduce_types."""

  def testPoolNodesToContext(self):
    input_graph = _get_test_graph_abc_efx()
    self.assertAllClose(
        tf.constant([[3.], [4.]]),
        pool_ops.pool_v2(input_graph, const.CONTEXT, reduce_type="sum",
                         node_set_name="a", feature_name="feat"))

  @parameterized.named_parameters(
      ("ToSource", const.SOURCE, tf.constant([[10.], [20.], [40.]])),
      ("ToTarget", const.TARGET, tf.constant([[30.], [40.]])),
      ("ToContext", const.CONTEXT, tf.constant([[30.], [40.]])))
  def testPoolEdges(self, to_tag, expected):
    input_graph = _get_test_graph_abc_efx()
    self.assertAllClose(
        expected,
        pool_ops.pool_v2(input_graph, to_tag, reduce_type="sum",
                         edge_set_name="e", feature_name="feat"))

  def testPoolHyperedges(self):
    input_graph = gt.GraphTensor.from_pieces(
        node_sets={
            "v": gt.NodeSet.from_fields(sizes=tf.constant([2])),
            "w": gt.NodeSet.from_fields(sizes=tf.constant([1])),
        },
        edge_sets={
            "hyper": gt.EdgeSet.from_fields(
                sizes=tf.constant([1]),
                features={"feat": tf.constant([[1.]])},
                adjacency=adj.HyperAdjacency.from_indices(
                    {6: ("v", tf.constant([0])),
                     8: ("w", tf.constant([0])),
                     9: ("v", tf.constant([1]))}))})
    self.assertAllClose(
        tf.constant([[1.], [0.]]),
        pool_ops.pool_v2(input_graph, 6, reduce_type="sum",
                         edge_set_name="hyper", feature_name="feat"))
    self.assertAllClose(
        tf.constant([[1.]]),
        pool_ops.pool_v2(input_graph, 8, reduce_type="sum",
                         edge_set_name="hyper", feature_name="feat"))
    self.assertAllClose(
        tf.constant([[0.], [1.]]),
        pool_ops.pool_v2(input_graph, 9, reduce_type="sum",
                         edge_set_name="hyper", feature_name="feat"))
    self.assertAllClose(
        tf.constant([[1.]]),
        pool_ops.pool_v2(input_graph, const.CONTEXT, reduce_type="sum",
                         edge_set_name="hyper", feature_name="feat"))

  @parameterized.named_parameters(
      ("SumMean", "sum|mean", tf.constant(
          [[[60., 63., 60./3, 63./3], [66., 69., 66./3, 69./3]],
           [[90., 92., 90./2, 92./2], [94., 96., 94./2, 96./2]]])),
      ("SumMinMax", "sum|min|max", tf.constant(
          [[[60., 63., 10., 11., 30., 31.], [66., 69., 12., 13., 32., 33.]],
           [[90., 92., 40., 41., 50., 51.], [94., 96., 42., 43., 52., 53.]]])),
      ("MaxSumMin", "max|sum|min", tf.constant(  # Reordered.
          [[[30., 31., 60., 63., 10., 11.], [32., 33., 66., 69., 12., 13.]],
           [[50., 51., 90., 92., 40., 41.], [52., 53., 94., 96., 42., 43.]]])),
  )
  def testReduceTypes(self, reduce_type, expected):
    """Test concatenation of reduce types along the innermost axis."""
    input_graph = _get_test_graph_abc_efx()
    feature_value = [
        tf.constant([[[10., 11.], [12., 13.]],
                     [[20., 21.], [22., 23.]],
                     [[40., 41.], [42., 43.]]]),
        tf.constant([[[30., 31.], [32., 33.]],
                     [[50., 51.], [52., 53.]]]),
    ]
    self.assertAllClose(
        expected,
        pool_ops.pool_v2(input_graph, const.TARGET, reduce_type=reduce_type,
                         edge_set_name=["e", "f"], feature_value=feature_value))

  def testSingleFeatureValueInputs(self):
    """Tests the ways of inputting a single tensor with edge feature values."""
    input_graph = _get_test_graph_abc_efx()
    self.assertAllClose(
        tf.constant([[15.], [40.]]),
        pool_ops.pool_v2(input_graph, const.TARGET, reduce_type="mean",
                         edge_set_name="e",  # Not a list.
                         feature_name="feat"))
    self.assertAllClose(
        tf.constant([[15.], [40.]]),
        pool_ops.pool_v2(input_graph, const.TARGET, reduce_type="mean",
                         edge_set_name=["e"],  # List.
                         feature_name="feat"))
    self.assertAllClose(
        tf.constant([[115.], [140.]]),
        pool_ops.pool_v2(input_graph, const.TARGET, reduce_type="mean",
                         edge_set_name="e",  # Not a list.
                         feature_value=tf.constant([[110.], [120.], [140.]])))
    self.assertAllClose(
        tf.constant([[115.], [140.]]),
        pool_ops.pool_v2(input_graph, const.TARGET, reduce_type="mean",
                         edge_set_name=["e"],  # List.
                         feature_value=[tf.constant([[110.], [120.], [140.]])]))

  def testMultiFeatureValueInputs(self):
    """Tests the ways of inputting multiple tensors with edge feature values."""
    input_graph = _get_test_graph_abc_efx()
    self.assertAllClose(
        tf.constant([[20.], [45.]]),
        pool_ops.pool_v2(input_graph, const.TARGET, reduce_type="mean",
                         edge_set_name=["e", "f"],
                         feature_name="feat"))
    self.assertAllClose(
        tf.constant([[120.], [145.]]),
        pool_ops.pool_v2(input_graph, const.TARGET, reduce_type="mean",
                         edge_set_name=["e", "f"],
                         feature_value=[tf.constant([[110.], [120.], [140.]]),
                                        tf.constant([[130.], [150.]])]))

  def testNodeFeatureValueInputs(self):
    """Tests the ways of inputting tensors with node feature values."""
    input_graph = _get_test_graph_abc_efx()
    self.assertAllClose(
        tf.constant([[2.0], [4.5]]),
        pool_ops.pool_v2(input_graph, const.CONTEXT, reduce_type="mean",
                         node_set_name=["a", "b"],
                         feature_name="feat"))
    self.assertAllClose(
        tf.constant([[12.0], [14.5]]),
        pool_ops.pool_v2(input_graph, const.CONTEXT, reduce_type="mean",
                         node_set_name=["a", "b"],
                         feature_value=[tf.constant([[11.], [12.], [14.]]),
                                        tf.constant([[13.], [15.]])]))


def _get_test_graph_abc_efx():
  return gt.GraphTensor.from_pieces(
      node_sets={
          "a": gt.NodeSet.from_fields(
              sizes=tf.constant([2, 1]),
              features={"feat": tf.constant([[1.], [2.], [4.]])}),
          "b": gt.NodeSet.from_fields(
              sizes=tf.constant([1, 1]),
              features={"feat": tf.constant([[3.], [5.]])}),
          "c": gt.NodeSet.from_fields(
              sizes=tf.constant([1, 1]),
              features={"feat": tf.constant([[-91.], [-92.]])}),
      },
      edge_sets={
          "e": gt.EdgeSet.from_fields(
              sizes=tf.constant([2, 1]),
              features={"feat": tf.constant([[10.], [20.], [40.]])},
              adjacency=adj.Adjacency.from_indices(
                  ("a", tf.constant([0, 1, 2])),
                  ("c", tf.constant([0, 0, 1])))),
          "f": gt.EdgeSet.from_fields(
              sizes=tf.constant([1, 1]),
              features={"feat": tf.constant([[30.], [50.]])},
              adjacency=adj.Adjacency.from_indices(
                  ("b", tf.constant([0, 1])),
                  ("c", tf.constant([0, 1])))),
          "x": gt.EdgeSet.from_fields(
              sizes=tf.constant([1, 0]),
              features={"feat": tf.constant([[-99.]])},
              adjacency=adj.Adjacency.from_indices(
                  ("a", tf.constant([0])),
                  ("b", tf.constant([0])))),
      })


# NOTE: Testing with TFLite requires a Keras wrapper and therefore is delegated
# to tensorflow_gnn/keras/layers/graph_ops_test.py.
class PoolReduceTypesTest(tf.test.TestCase, parameterized.TestCase):
  """Tests GraphPieceReducer and MultiReducer for *all* reduce_types."""

  @parameterized.named_parameters(
      ("Sum", "sum", tf.constant([0., 10., 20.+30.])),
      ("Prod", "prod", tf.constant([1., 10., 20.*30.])),
      ("Mean", "mean", tf.constant([0., 10./1., (20.+30.)/2.])),
      ("Max", "max", tf.constant([tf.float32.min, 10., 30.])),
      ("MaxNoInf", "max_no_inf", tf.constant([0., 10., 30.])),
      ("Min", "min", tf.constant([tf.float32.max, 10., 20.])),
      ("MinNoInf", "min_no_inf", tf.constant([0., 10., 20.])),
  )
  def testSingleRank1(self, reduce_type, expected):
    input_graph = _get_test_graph_0123()
    actual = pool_ops.pool_v2(
        input_graph, const.TARGET,
        edge_set_name="e",
        reduce_type=reduce_type,
        feature_value=tf.constant([20., 10., 30.]))
    self.assertAllClose(expected, actual)

  @parameterized.named_parameters(
      ("Sum", "sum", tf.constant([0., 10., 20.+30.+40.])),
      ("Prod", "prod", tf.constant([1., 10., 20.*30.*40])),
      ("Mean", "mean", tf.constant([0., 10./1., (20.+30.+40.)/3.])),
      ("Max", "max", tf.constant([tf.float32.min, 10., 40.])),
      ("MaxNoInf", "max_no_inf", tf.constant([0., 10., 40.])),
      ("Min", "min", tf.constant([tf.float32.max, 10., 20.])),
      ("MinNoInf", "min_no_inf", tf.constant([0., 10., 20.])),
  )
  def testMultiRank1(self, reduce_type, expected):
    input_graph = _get_test_graph_0123()
    actual = pool_ops.pool_v2(
        input_graph, const.TARGET,
        edge_set_name=["e", "f", "g"],
        reduce_type=reduce_type,
        feature_value=[tf.constant([20., 10., 30.]),
                       tf.constant([40.]),
                       tf.zeros([0], tf.float32)])
    self.assertAllClose(expected, actual)

  @parameterized.named_parameters(
      ("Sum", "sum", tf.constant(
          [[0., 0.], [10., 11.], [20.+30., 21.+31.]])),
      ("Prod", "prod", tf.constant(
          [[1., 1.], [10., 11.], [20.*30., 21.*31.]])),
      ("Mean", "mean", tf.constant(
          [[0., 0.], [10./1., 11./1.], [(20.+30.)/2., (21.+31.)/2.]])),
      # Extra test because "mean|sum" is computed from "sum" and "count".
      ("MeanAndSum", "mean|sum", tf.constant(
          [[0., 0., 0., 0], [10./1., 11./1., 10., 11.],
           [(20.+30.)/2., (21.+31.)/2., 20.+30., 21.+31.]])),
      ("Max", "max", tf.constant(
          [[tf.float32.min]*2, [10., 11.], [30., 31.]])),
      ("MaxNoInf", "max_no_inf", tf.constant(
          [[0., 0.], [10., 11.], [30., 31.]])),
      ("Min", "min", tf.constant(
          [[tf.float32.max]*2, [10., 11.], [20., 21.]])),
      ("MinNoInf", "min_no_inf", tf.constant(
          [[0., 0.], [10., 11.], [20., 21.]])),
  )
  def testSingleRank2(self, reduce_type, expected):
    input_graph = _get_test_graph_0123()
    actual = pool_ops.pool_v2(
        input_graph, const.TARGET,
        edge_set_name="e",
        reduce_type=reduce_type,
        feature_value=tf.constant([[20., 21.],
                                   [10., 11.],
                                   [30., 31.]]))
    self.assertAllClose(expected, actual)

  @parameterized.named_parameters(
      ("Sum", "sum", tf.constant(
          [[0., 0.], [10., 11.], [20.+30.+40., 21.+31.-41.]])),
      ("Prod", "prod", tf.constant(
          [[1., 1.], [10., 11.], [20.*30.*40., 21.*31.*-41.]])),
      ("Mean", "mean", tf.constant(
          [[0., 0.], [10./1., 11./1.], [(20.+30.+40.)/3., (21.+31.-41.)/3.]])),
      ("Max", "max", tf.constant(
          [[tf.float32.min]*2, [10., 11.], [40., 31.]])),
      ("MaxNoInf", "max_no_inf", tf.constant(
          [[0., 0.], [10., 11.], [40., 31.]])),
      ("Min", "min", tf.constant(
          [[tf.float32.max]*2, [10., 11.], [20., -41.]])),
      ("MinNoInf", "min_no_inf", tf.constant(
          [[0., 0.], [10., 11.], [20., -41.]])),
  )
  def testMultiRank2(self, reduce_type, expected):
    input_graph = _get_test_graph_0123()
    actual = pool_ops.pool_v2(
        input_graph, const.TARGET,
        edge_set_name=["e", "f", "g"],
        reduce_type=reduce_type,
        feature_value=[tf.constant([[20., 21.],
                                    [10., 11.],
                                    [30., 31.]]),
                       tf.constant([[40., -41.]]),
                       tf.zeros([0, 2], tf.float32)])
    self.assertAllClose(expected, actual)

  @parameterized.named_parameters(
      ("Sum", "sum", tf.constant(
          [[[0., 0.], [0., 0.]],
           [[10., 11.], [12., 13.]],
           [[20.+30.+40., 21.+31.+41.], [22.+32.-42., 23.+33.-43.]]])),
      ("Prod", "prod", tf.constant(
          [[[1., 1.], [1., 1.]],
           [[10., 11.], [12., 13.]],
           [[20.*30.*40., 21.*31.*41.], [22.*32.*-42., 23.*33.*-43.]]])),
      ("Mean", "mean", tf.constant(
          [[[0., 0.], [0., 0.]],
           [[10., 11.], [12., 13.]],
           [[(20.+30.+40.)/3., (21.+31.+41.)/3.],
            [(22.+32.-42.)/3., (23.+33.-43.)/3.]]])),
      ("Max", "max", tf.constant(
          [[[tf.float32.min]*2]*2,
           [[10., 11.], [12., 13.]],
           [[40., 41.], [32., 33.]]])),
      ("MaxNoInf", "max_no_inf", tf.constant(
          [[[0., 0.], [0., 0.]],
           [[10., 11.], [12., 13.]],
           [[40., 41.], [32., 33.]]])),
      ("Min", "min", tf.constant(
          [[[tf.float32.max]*2]*2,
           [[10., 11.], [12., 13.]],
           [[20., 21.], [-42., -43.]]])),
      ("MinNoInf", "min_no_inf", tf.constant(
          [[[0., 0.], [0., 0.]],
           [[10., 11.], [12., 13.]],
           [[20., 21.], [-42., -43.]]])),
  )
  def testMultiRank3(self, reduce_type, expected):
    input_graph = _get_test_graph_0123()
    actual = pool_ops.pool_v2(
        input_graph, const.TARGET,
        edge_set_name=["e", "f", "g"],
        reduce_type=reduce_type,
        feature_value=[tf.constant([[[20., 21.], [22., 23.]],
                                    [[10., 11.], [12., 13.]],
                                    [[30., 31.], [32., 33.]]]),
                       tf.constant([[[40., 41.], [-42., -43.]]]),
                       tf.zeros([0, 2, 2], tf.float32)])
    self.assertAllClose(expected, actual)

  @parameterized.named_parameters(
      ("Sum", "sum", tf.ragged.constant(
          [[],
           [[10.], [12., 13.]],
           [[20.+30., 21.+31.], [22.+32.]]])),
      ("Prod", "prod", tf.ragged.constant(
          [[],
           [[10.], [12., 13]],
           [[20.*30., 21.*31.], [22.*32.]]])),
      ("Mean", "mean", tf.ragged.constant(
          [[],
           [[10.,], [12., 13]],
           [[(20.+30.)/2., (21.+31.)/2.], [(22.+32.)/2.]]])),
      # Extra test because "mean|sum" is computed from "sum" and "count".
      ("MeanAndSum", "mean|sum", tf.ragged.constant(
          [[],
           [[10., 10.], [12., 13., 12., 13.]],
           [[(20.+30.)/2., (21.+31.)/2., 20.+30., 21.+31.],
            [(22.+32.)/2., 22.+32.]]])),
      ("Max", "max", tf.ragged.constant(
          [[],
           [[10.], [12., 13.]],
           [[30., 31.], [32.]]])),
      ("MaxNoInf", "max_no_inf", tf.ragged.constant(
          [[],
           [[10.], [12., 13.]],
           [[30., 31.], [32.]]])),
      ("Min", "min", tf.ragged.constant(
          [[],
           [[10.], [12., 13.]],
           [[20., 21.], [22.]]])),
      ("MinNoInf", "min_no_inf", tf.ragged.constant(
          [[],
           [[10.], [12., 13.]],
           [[20., 21.], [22.]]])),
  )
  def testRaggedWithScalarFlatValues(self, reduce_type, expected):
    input_graph = _get_test_graph_0123()
    actual = pool_ops.pool_v2(
        input_graph, const.TARGET,
        edge_set_name="e",
        reduce_type=reduce_type,
        feature_value=tf.ragged.constant([[[20., 21.], [22.]],
                                          [[10.], [12., 13]],
                                          [[30., 31.], [32.]]]))
    self.assertAllClose(expected, actual)

  @parameterized.named_parameters(
      ("Sum", "sum", tf.ragged.constant(
          [[],
           [[[10., 0.]], [[12., 2.], [13., 3.]]],
           [[[20.+30., 0.+0.], [21.+31., 1.+1.]], [[22.+32., 2.+2.]]]],
          inner_shape=(2,))),
      ("Prod", "prod", tf.ragged.constant(
          [[],
           [[[10., 0.]], [[12., 2.], [13., 3.]]],
           [[[20.*30., 0.*0.], [21.*31., 1.*1.]], [[22.*32., 2.*2.]]]],
          inner_shape=(2,))),
      ("Mean", "mean", tf.ragged.constant(
          [[],
           [[[10., 0.]], [[12., 2.], [13., 3.]]],
           [[[(20.+30.)/2., 0.], [(21.+31.)/2., 1.]], [[(22.+32.)/2., 2.]]]],
          inner_shape=(2,))),
      # # Extra test because "mean|sum" is computed from "sum" and "count".
      ("MeanAndSum", "mean|sum", tf.ragged.constant(
          [[],
           [[[10., 0., 10., 0.]], [[12., 2., 12., 2.], [13., 3., 13., 3.]]],
           [[[(20.+30.)/2., 0., 20.+30., 0.+0.],
             [(21.+31.)/2., 1., 21.+31., 1.+1.]],
            [[(22.+32.)/2., 2., 22.+32., 2.+2.]]]],
          inner_shape=(4,))),
      ("Max", "max", tf.ragged.constant(
          [[],
           [[[10., 0.]], [[12., 2.], [13., 3.]]],
           [[[30., 0.], [31., 1.]], [[32., 2.]]]],
          inner_shape=(2,))),
      ("MaxNoInf", "max_no_inf", tf.ragged.constant(
          [[],
           [[[10., 0.]], [[12., 2.], [13., 3.]]],
           [[[30., 0.], [31., 1.]], [[32., 2.]]]],
          inner_shape=(2,))),
      ("Min", "min", tf.ragged.constant(
          [[],
           [[[10., 0.]], [[12., 2.], [13., 3.]]],
           [[[20., 0.], [21., 1.]], [[22., 2.]]]],
          inner_shape=(2,))),
      ("MinNoInf", "min_no_inf", tf.ragged.constant(
          [[],
           [[[10., 0.]], [[12., 2.], [13., 3.]]],
           [[[20., 0.], [21., 1.]], [[22., 2.]]]],
          inner_shape=(2,))),
  )
  # Positive inner rank used to trigger b/216278499.
  def testRaggedWithFlatValuesOfRank1(self, reduce_type, expected):
    input_graph = _get_test_graph_0123()
    actual = pool_ops.pool_v2(
        input_graph, const.TARGET,
        edge_set_name="e",
        reduce_type=reduce_type,
        feature_value=tf.ragged.constant(
            [[[[20., 0.], [21., 1.]], [[22., 2.]]],
             [[[10., 0.]], [[12., 2.], [13., 3.]]],
             [[[30., 0.], [31., 1.]], [[32., 2.]]]],
            inner_shape=(2,)))
    self.assertAllClose(expected, actual)


def _get_test_graph_0123():
  return gt.GraphTensor.from_pieces(
      node_sets={
          "v": gt.NodeSet.from_fields(sizes=tf.constant([3])),
      },
      edge_sets={
          # Using edge_set_name="e" tests 0, 1 and 2 inputs.
          "e": gt.EdgeSet.from_fields(
              sizes=tf.constant([3]),
              adjacency=adj.Adjacency.from_indices(
                  ("v", tf.constant([0, 0, 0])),
                  ("v", tf.constant([2, 1, 2])))),  # Unsorted.
          # Using edge_set_name=["e", "f"] tests 0+0, 1+0 and 2+1 inputs,
          # thus exercisig the handling of zero inputs at both levels.
          "f": gt.EdgeSet.from_fields(
              sizes=tf.constant([1]),
              adjacency=adj.Adjacency.from_indices(
                  ("v", tf.constant([0])),
                  ("v", tf.constant([2])))),
          # Using edge_set_name=["e", "f", "g"] additionally tests
          # the case of a completely empty edge set.
          "g": gt.EdgeSet.from_fields(
              sizes=tf.constant([0]),
              adjacency=adj.Adjacency.from_indices(
                  ("v", tf.constant([], tf.int32)),
                  ("v", tf.constant([], tf.int32)))),
      })


if __name__ == "__main__":
  tf.test.main()
