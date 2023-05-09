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
"""Tests for tag_utils."""

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import tag_utils


class ReverseTagTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("Source", const.SOURCE, const.TARGET),
      ("Target", const.TARGET, const.SOURCE))
  def test(self, tag, expected):
    actual = tag_utils.reverse_tag(tag)
    self.assertEqual(expected, actual)

  def testError(self):
    with self.assertRaisesRegex(ValueError, r"Expected tag .*got: 3"):
      _ = tag_utils.reverse_tag(3)


class GetEdgeOrNodeSetNameArgsForTagTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("OneEdgeSet", dict(edge_set_name="a->b"), ["a->b"], None, False),
      ("TwoEdgeSets", dict(edge_set_name=["a->b", "b->b"]),
       ["a->b", "b->b"], None, True),
      ("OneNodeSet", dict(node_set_name="a"), None, ["a"], False),
      ("TwoNodeSets", dict(node_set_name=["a", "b"]), None, ["a", "b"], True))
  def testContext(self, call_kwargs, expected_edge_set_names,
                  expected_node_set_names, expected_got_sequence_args):
    spec = _make_test_graph_spec()
    edge_set_names, node_set_names, got_sequence_args = (
        tag_utils.get_edge_or_node_set_name_args_for_tag(
            spec, const.CONTEXT, **call_kwargs))
    self.assertEqual(expected_edge_set_names, edge_set_names)
    self.assertEqual(expected_node_set_names, node_set_names)
    self.assertEqual(expected_got_sequence_args, got_sequence_args)

  @parameterized.named_parameters(
      ("ZeroEdgeSets", dict(edge_set_name=[]), r"requires.*non-empty"),
      ("ZeroNodeSets", dict(node_set_name=[]), r"requires.*non-empty"),
      ("NeitherArg", dict(), r"requires.*exactly one"),
      ("BothArgs", dict(edge_set_name="a->b", node_set_name="b"),
       r"requires.*exactly one"))
  def testContextRaises(self, call_kwargs, regex):
    spec = _make_test_graph_spec()
    with self.assertRaisesRegex(ValueError, regex):
      tag_utils.get_edge_or_node_set_name_args_for_tag(
          spec, const.CONTEXT, **call_kwargs)

  @parameterized.named_parameters(
      ("OneEdgeSet", dict(edge_set_name="a->b"), ["a->b"], False),
      ("TwoEdgeSets", dict(edge_set_name=["a->b", "b->b"]),
       ["a->b", "b->b"], True))
  def testTarget(self, call_kwargs, expected_edge_set_names,
                 expected_got_sequence_args):
    spec = _make_test_graph_spec()
    edge_set_names, node_set_names, got_sequence_args = (
        tag_utils.get_edge_or_node_set_name_args_for_tag(
            spec, const.TARGET, **call_kwargs))
    self.assertIsNone(node_set_names)
    self.assertEqual(expected_edge_set_names, edge_set_names)
    self.assertEqual(expected_got_sequence_args, got_sequence_args)

  @parameterized.named_parameters(
      ("ZeroEdgeSets", dict(edge_set_name=[]), r"requires.*non-empty"),
      ("UnequalEndpoints", dict(edge_set_name=["a->a", "a->b"]),
       r"requires.*same endpoint"),
      ("NodeSet", dict(node_set_name="a"), r"requires.*edge_set_name"),
      ("NeitherArg", dict(), r"requires.*edge_set_name"),
      ("BothArgs", dict(edge_set_name="a->b", node_set_name="b"), r""))
  def testTargetRaises(self, call_kwargs, regex):
    spec = _make_test_graph_spec()
    with self.assertRaisesRegex(ValueError, regex):
      tag_utils.get_edge_or_node_set_name_args_for_tag(
          spec, const.TARGET, **call_kwargs)


def _make_test_graph_spec():
  return gt.GraphTensorSpec.from_piece_specs(
      node_sets_spec={
          n: gt.NodeSetSpec.from_field_specs(
              sizes_spec=tf.TensorSpec([None], const.default_indices_dtype))
          for n in ["a", "b"]},
      edge_sets_spec={
          f"{s}->{t}": gt.EdgeSetSpec.from_field_specs(
              sizes_spec=tf.TensorSpec([None], const.default_indices_dtype),
              adjacency_spec=adj.AdjacencySpec.from_incident_node_sets(s, t))
          for s, t in [("a", "a"), ("a", "b"), ("b", "b")]})


if __name__ == "__main__":
  absltest.main()
