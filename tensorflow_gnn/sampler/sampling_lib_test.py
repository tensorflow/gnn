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
"""Tests for sampling_lib."""

import math
from typing import Iterable, List, Mapping, Tuple

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.sampler import sampling_lib as lib
from tensorflow_gnn.sampler import sampling_spec_pb2
from tensorflow_gnn.sampler import subgraph_pb2

from google.protobuf import text_format

PCollection = beam.PCollection
SamplingOp = sampling_spec_pb2.SamplingOp
SamplingSpec = sampling_spec_pb2.SamplingSpec
SamplingStrategy = sampling_spec_pb2.SamplingStrategy

Edge = subgraph_pb2.Node.Edge
Node = subgraph_pb2.Node
NodeId = lib.NodeId
SampleId = lib.SampleId


def _get_op(strategy: SamplingStrategy, sample_size: int) -> SamplingOp:
  result = SamplingOp()
  result.sample_size = sample_size
  result.strategy = strategy
  return result


def _get_test_edge(weight: float) -> Edge:
  edge = Edge()
  edge.features.feature["weight"].float_list.value.append(weight)
  return edge


class TestHelperFunctions(parameterized.TestCase):

  @parameterized.parameters(0.0, 1.0, 2.0)
  def test_sampling_weight_fn_top_k(self, edge_weight: float):
    weight_fn = lib.create_sampling_weight_fn(
        _get_op(SamplingStrategy.TOP_K, 10), lib.get_weight_feature)
    for _ in range(1_000):
      edge = _get_test_edge(edge_weight)
      self.assertEqual(weight_fn(edge), edge_weight)

  @parameterized.parameters(0.0, 1.0, 2.0)
  def test_sampling_weight_fn_random_uniform(self, edge_weight: float):
    weight_fn = lib.create_sampling_weight_fn(
        _get_op(SamplingStrategy.RANDOM_UNIFORM, 10), lib.get_weight_feature)

    weights = [weight_fn(_get_test_edge(edge_weight)) for _ in range(1_000)]
    hist = set(int(weight * 10.0) for weight in weights)
    # For the uniform distribution the probability that any histogram has not
    # values is 0.9^1_000.
    self.assertCountEqual(hist, range(10))

  def test_sampling_weight_fn_random_weighted(self):
    weight_fn = lib.create_sampling_weight_fn(
        _get_op(SamplingStrategy.RANDOM_WEIGHTED, 10), lib.get_weight_feature)

    self.assertLess(weight_fn(_get_test_edge(0.0)), -1.0e10)

    n_samples = 10_000
    w1 = [(weight_fn(_get_test_edge(1.0)), 1) for i in range(10 * n_samples)]
    w3 = [(weight_fn(_get_test_edge(3.0)), 3) for i in range(10 * n_samples)]

    w1_count = 0
    for _, group in sorted(w1 + w3, reverse=True)[:n_samples]:
      self.assertIn(group, [1, 3])
      w1_count += int(group == 1)

    ratio = 1.0 * w1_count / n_samples
    std = math.sqrt(0.25 * 0.75 / n_samples)
    self.assertLess(ratio, 0.25 + 6.0 * std)
    self.assertGreater(ratio, 0.25 - 6.0 * std)

  def test_is_deterministic(self):
    self.assertTrue(lib.is_deterministic(_get_op(SamplingStrategy.TOP_K, 1)))
    self.assertFalse(
        lib.is_deterministic(_get_op(SamplingStrategy.RANDOM_UNIFORM, 1)))
    self.assertFalse(
        lib.is_deterministic(_get_op(SamplingStrategy.RANDOM_WEIGHTED, 1)))


def _create_test_node(node_id: int, neighbor_ids: List[int]) -> Node:
  node = Node()
  node.id = str(node_id).encode()
  for neighbor_id in neighbor_ids:
    edge: Edge = node.outgoing_edges.add()
    edge.MergeFrom(_get_test_edge(float(neighbor_id)))
    edge.neighbor_id = str(neighbor_id).encode()
  return node


def _create_test_nodes(
    inputs: List[Tuple[SampleId, int,
                       List[int]]]) -> List[Tuple[SampleId, Node]]:
  result = []
  for sample_id, node_index, neighbors in inputs:
    result.append((sample_id, _create_test_node(node_index, neighbors)))
  return result


class EdgeSamplingTestBase(tf.test.TestCase, parameterized.TestCase):

  def sampled_edges_matcher(self, expected: Iterable[Tuple[SampleId, Node]]):

    def key_fn(item):
      sample_id, node = item
      return sample_id, node.id

    sorted_expected = sorted(expected, key=key_fn)

    def _equal(actual):
      sorted_actual = sorted(actual, key=key_fn)

      msg = f"Failed assert: {sorted_expected} == {sorted_actual}"

      self.assertEqual(len(sorted_expected), len(sorted_actual), msg=msg)

      for l, r in zip(sorted_expected, sorted_actual):
        l_sample_id, l_node = l
        r_sample_id, r_node = r
        self.assertProtoEquals(l_node, r_node, msg=msg)
        self.assertEqual(l_sample_id, r_sample_id, msg=msg)

    return _equal


class TestReservoirEdgeSamplingFn(EdgeSamplingTestBase):

  def _add_num_paths(
      self, nodes: List[Tuple[SampleId, Node]],
      num_paths: List[int]) -> List[Tuple[SampleId, Tuple[int, Node]]]:
    self.assertEqual(len(nodes), len(nodes))
    return [(sample_id, (count, node))
            for (sample_id, node), count in zip(nodes, num_paths)]

  @parameterized.parameters([
      SamplingStrategy.TOP_K, SamplingStrategy.RANDOM_UNIFORM,
      SamplingStrategy.RANDOM_WEIGHTED
  ])
  def test_all(self, strategy: SamplingStrategy):
    sampling_op = _get_op(strategy, 3)
    sampling_fn = lib.ResevoirEdgeSamplingFn(
        lib.create_sampling_weight_fn(sampling_op, lib.get_weight_feature),
        sampling_op.sample_size,
        resample_for_each_path=not lib.is_deterministic(sampling_op))
    with beam.Pipeline() as root:
      nodes = _create_test_nodes([(b"sample.1", 1, [3, 2, 1]),
                                  (b"sample.1", 2, [2, 1]),
                                  (b"sample.2", 3, [2])])
      sampled_edges, new_frontier = (
          root | beam.Create(self._add_num_paths(nodes, [10, 20, 30]))
          | beam.ParDo(sampling_fn).with_outputs(
              lib._FROTIER_OUTPUT_TAG, main="sampled_edges"))
      util.assert_that(
          sampled_edges, self.sampled_edges_matcher(nodes), label="samples")
      util.assert_that(
          new_frontier,
          util.equal_to([
              ((b"sample.1", b"1"), 10),
              ((b"sample.1", b"2"), 10),
              ((b"sample.1", b"3"), 10),
              ((b"sample.1", b"1"), 20),
              ((b"sample.1", b"2"), 20),
              ((b"sample.2", b"2"), 30),
          ]),
          label="frontier")

  def test_top_2(self):
    sampling_op = _get_op(SamplingStrategy.TOP_K, 2)
    sampling_fn = lib.ResevoirEdgeSamplingFn(
        lib.create_sampling_weight_fn(sampling_op, lib.get_weight_feature),
        sampling_op.sample_size,
        resample_for_each_path=False)
    with beam.Pipeline() as root:
      nodes = _create_test_nodes([(b"sample.1", 1, [3, 2, 1]),
                                  (b"sample.1", 2, [2, 1]),
                                  (b"sample.2", 3, [2])])
      sampled_edges, new_frontier = (
          root | beam.Create(self._add_num_paths(nodes, [10, 20, 30]))
          | beam.ParDo(sampling_fn).with_outputs(
              lib._FROTIER_OUTPUT_TAG, main="sampled_edges"))
      expected_nodes = _create_test_nodes([(b"sample.1", 1, [3, 2]),
                                           (b"sample.1", 2, [2, 1]),
                                           (b"sample.2", 3, [2])])
      util.assert_that(
          sampled_edges,
          self.sampled_edges_matcher(expected_nodes),
          label="samples")
      util.assert_that(
          new_frontier,
          util.equal_to([
              ((b"sample.1", b"2"), 10),
              ((b"sample.1", b"3"), 10),
              ((b"sample.1", b"1"), 20),
              ((b"sample.1", b"2"), 20),
              ((b"sample.2", b"2"), 30),
          ]),
          label="frontier")

  def test_top_1(self):
    sampling_op = _get_op(SamplingStrategy.TOP_K, 1)
    sampling_fn = lib.ResevoirEdgeSamplingFn(
        lib.create_sampling_weight_fn(sampling_op, lib.get_weight_feature),
        sampling_op.sample_size,
        resample_for_each_path=False)
    with beam.Pipeline() as root:
      nodes = _create_test_nodes([(b"sample.1", 1, [1, 2, 3, 1]),
                                  (b"sample.1", 2, [2, 1]),
                                  (b"sample.2", 3, [2])])
      sampled_edges, new_frontier = (
          root | beam.Create(self._add_num_paths(nodes, [1, 1, 1]))
          | beam.ParDo(sampling_fn).with_outputs(
              lib._FROTIER_OUTPUT_TAG, main="sampled_edges"))
      expected_nodes = _create_test_nodes([(b"sample.1", 1, [3]),
                                           (b"sample.1", 2, [2]),
                                           (b"sample.2", 3, [2])])
      util.assert_that(
          sampled_edges,
          self.sampled_edges_matcher(expected_nodes),
          label="samples")
      util.assert_that(
          new_frontier,
          util.equal_to([
              ((b"sample.1", b"3"), 1),
              ((b"sample.1", b"2"), 1),
              ((b"sample.2", b"2"), 1),
          ]),
          label="frontier")

  def test_non_deterministic(self):
    mock_weights = {b"1": [2, 1, 2], b"2": [1, 2, 1], b"3": [0, 0, 0]}

    def mock_weight_fn(edge: Edge) -> float:
      return mock_weights[edge.neighbor_id].pop(0)

    sampling_op = _get_op(SamplingStrategy.RANDOM_UNIFORM, 1)
    sampling_fn = lib.ResevoirEdgeSamplingFn(
        mock_weight_fn, sampling_op.sample_size, resample_for_each_path=True)
    with beam.Pipeline() as root:
      nodes = _create_test_nodes([(b"sample.1", 1, [1, 2, 3])])
      sampled_edges, new_frontier = (
          root | beam.Create(self._add_num_paths(nodes, [3]))
          | beam.ParDo(sampling_fn).with_outputs(
              lib._FROTIER_OUTPUT_TAG, main="sampled_edges"))
      expected_nodes = _create_test_nodes([(b"sample.1", 1, [1, 2])])
      util.assert_that(
          sampled_edges,
          self.sampled_edges_matcher(expected_nodes),
          label="samples")
      util.assert_that(
          new_frontier,
          util.equal_to([
              ((b"sample.1", b"1"), 2),
              ((b"sample.1", b"2"), 1),
          ]),
          label="frontier")


class TestEdgeSampling(EdgeSamplingTestBase):

  @parameterized.named_parameters(
      dict(
          testcase_name="empty",
          sampling_spec="""
            seed_op {
              op_name: 'seed'
              node_set_name: 'node'
            }
          """,
          seeds={"node": []},
          adj_lists={},
          expected_result={}),
      dict(
          testcase_name="no_sampling",
          sampling_spec="""
            seed_op {
              op_name: 'seed'
              node_set_name: 'node'
            }
          """,
          seeds={"node": [(b"s.1", b"1")]},
          adj_lists={"edge": [_create_test_node(1, [1, 2])]},
          expected_result={}),
      dict(
          testcase_name="one_hop",
          sampling_spec="""
            seed_op {
              op_name: 'seed'
              node_set_name: 'node'
            }
            sampling_ops {
              op_name: 'hop-1'
              input_op_names: [ 'seed' ]
              strategy: TOP_K
              sample_size: 2
              edge_set_name: 'edge'
            }
          """,
          seeds={"node": [
              (b"s.1.0", b"1"),
              (b"s.1.1", b"1"),
              (b"s.2", b"2"),
          ]},
          adj_lists={
              "edge": [
                  _create_test_node(1, [1, 2, 3]),
                  _create_test_node(2, [1])
              ]
          },
          expected_result={
              "edge": [
                  (b"s.1.0", _create_test_node(1, [3, 2])),
                  (b"s.1.1", _create_test_node(1, [3, 2])),
                  (b"s.2", _create_test_node(2, [1])),
              ]
          }),
      dict(
          testcase_name="two_hops",
          sampling_spec="""
            seed_op {
              op_name: 'seed'
              node_set_name: 'node'
            }
            sampling_ops {
              op_name: 'hop-1'
              input_op_names: [ 'seed' ]
              strategy: TOP_K
              sample_size: 2
              edge_set_name: 'edge'
            }
            sampling_ops {
              op_name: 'hop-2'
              input_op_names: [ 'hop-1' ]
              strategy: TOP_K
              sample_size: 2
              edge_set_name: 'edge'
            }
          """,
          seeds={"node": [(b"s.1", b"1"),]},
          adj_lists={
              "edge": [
                  _create_test_node(1, [1, 2, 3]),
                  _create_test_node(2, [1, 4, 3, 2]),
                  _create_test_node(3, [1]),
                  _create_test_node(4, [1, 4])
              ]
          },
          expected_result={
              "edge": [
                  (b"s.1", _create_test_node(1, [3, 2])),
                  (b"s.1", _create_test_node(2, [4, 3])),
                  (b"s.1", _create_test_node(3, [1])),
              ]
          }),
      dict(
          testcase_name="merging",
          sampling_spec="""
            seed_op {
              op_name: 'seed'
              node_set_name: 'A'
            }
            sampling_ops {
              op_name: 'A->B:1'
              input_op_names: [ 'seed' ]
              strategy: TOP_K
              sample_size: 2
              edge_set_name: 'A->B:1'
            }
            sampling_ops {
              op_name: 'A->B:2'
              input_op_names: [ 'seed' ]
              strategy: TOP_K
              sample_size: 2
              edge_set_name: 'A->B:2'
            }
            sampling_ops {
              op_name: 'B->A'
              input_op_names: [ 'A->B:1',  'A->B:2']
              strategy: TOP_K
              sample_size: 2
              edge_set_name: 'B->A'
            }
          """,
          seeds={"A": [(b"s.1", b"1"), (b"s.2", b"2")]},
          adj_lists={
              "A->B:1": [
                  _create_test_node(1, [1, 2, 3]),
                  _create_test_node(2, [1, 4, 5, 2]),
              ],
              "A->B:2": [_create_test_node(1, [2]),],
              "B->A": [_create_test_node(3, [1]),]
          },
          expected_result={
              "A->B:1": [(b"s.1", _create_test_node(1, [3, 2])),
                         (b"s.2", _create_test_node(2, [5, 4]))],
              "A->B:2": [(b"s.1", _create_test_node(1, [2])),],
              "B->A": [(b"s.1", _create_test_node(3, [1])),]
          }),
  )
  def test_logic(self, sampling_spec: str,
                 seeds: Mapping[tfgnn.NodeSetName, Iterable[Tuple[SampleId,
                                                                  NodeId]]],
                 adj_lists: Mapping[tfgnn.EdgeSetName, Iterable[Node]],
                 expected_result: Mapping[tfgnn.EdgeSetName,
                                          Iterable[Tuple[SampleId, Node]]]):
    sampling_spec = text_format.Parse(sampling_spec, SamplingSpec())
    with beam.Pipeline() as root:
      seeds = {
          set_name: root | f"Seeds/{set_name}" >> beam.Create(values)
          for set_name, values in seeds.items()
      }

      adj_lists = {
          set_name: root | f"AdjLists/{set_name}" >> beam.Create(values)
          for set_name, values in adj_lists.items()
      }

      actual_result = lib.sample_edges(sampling_spec, seeds, adj_lists)
      self.assertCountEqual(actual_result.keys(), expected_result.keys())
      for edge_set_name in expected_result:
        util.assert_that(
            actual_result[edge_set_name],
            self.sampled_edges_matcher(expected_result[edge_set_name]),
            label=edge_set_name)


def _test_example(value: bytes) -> tf.train.Example:
  result = tf.train.Example()
  result.features.feature["s"].bytes_list.value.append(value)
  return result


class TestAdjacencyLists(tf.test.TestCase, parameterized.TestCase):

  def _nodes_matcher(self, expected: Iterable[Node]):
    key_fn = lambda node: node.id
    sorted_expected = sorted(expected, key=key_fn)

    def _equal(actual):
      sorted_actual = sorted(actual, key=key_fn)
      msg = f"Failed assert: {sorted_expected} == {sorted_actual}"
      self.assertEqual(len(sorted_expected), len(sorted_actual), msg=msg)

      for l_node, r_node in zip(sorted_expected, sorted_actual):
        self.assertProtoEquals(l_node, r_node, msg=msg)

    return _equal

  @parameterized.named_parameters(
      dict(
          testcase_name="empty", edges={}, sort_edges=False,
          expected_result={}),
      dict(
          testcase_name="single_edge",
          edges={
              "edge": [(b"1", b"2", tf.train.Example())],
          },
          sort_edges=False,
          expected_result={
              "edge": [
                  """
                  id: '1'
                  outgoing_edges { neighbor_id: '2' edge_index: 0 }
                  """,
              ],
          }),
      dict(
          testcase_name="homogeneous",
          edges={
              "edge": [(b"1", b"2", _test_example(b"12")),
                       (b"1", b"3", _test_example(b"13")),
                       (b"2", b"1", _test_example(b"21"))]
          },
          sort_edges=True,
          expected_result={
              "edge": [
                  """
                  id: '1'
                  outgoing_edges {
                    neighbor_id: '2'
                    edge_index: 0
                    features {
                      feature {
                        key: 's'
                        value { bytes_list { value : ['12'] } }
                      }
                    }
                  }
                  outgoing_edges {
                    neighbor_id: '3'
                    edge_index: 1
                    features {
                      feature {
                        key: 's'
                        value { bytes_list { value : ['13'] } }
                      }
                    }
                  }
                  """,
                  """
                  id: '2'
                  outgoing_edges {
                    neighbor_id: '1'
                    edge_index: 0
                    features {
                      feature {
                        key: 's'
                        value { bytes_list { value : ['21'] } }
                      }
                    }
                  }
                  """,
              ]
          }),
      dict(
          testcase_name="heterogeneous",
          edges={
              "a->b": [(b"1", b"2", tf.train.Example()),
                       (b"1", b"3", tf.train.Example()),
                       (b"1", b"1", tf.train.Example())],
              "a->c": [
                  (b"2", b"3", tf.train.Example()),
                  (b"3", b"1", tf.train.Example()),
                  (b"2", b"1", tf.train.Example()),
                  (b"3", b"2", tf.train.Example()),
              ]
          },
          sort_edges=True,
          expected_result={
              "a->b": [
                  """
                  id: '1'
                  outgoing_edges { neighbor_id: '1' edge_index: 0 }
                  outgoing_edges { neighbor_id: '2' edge_index: 1 }
                  outgoing_edges { neighbor_id: '3' edge_index: 2 }
                  """,
              ],
              "a->c": [
                  """
                  id: '2'
                  outgoing_edges { neighbor_id: '1' edge_index: 0 }
                  outgoing_edges { neighbor_id: '3' edge_index: 1 }
                  """,
                  """
                  id: '3'
                  outgoing_edges { neighbor_id: '1' edge_index: 0 }
                  outgoing_edges { neighbor_id: '2' edge_index: 1 }
                  """,
              ]
          }),
  )
  def test_logic(self, edges: Mapping[tfgnn.EdgeSetName,
                                      Iterable[Tuple[NodeId, NodeId, str]]],
                 sort_edges: bool, expected_result: Mapping[tfgnn.EdgeSetName,
                                                            Iterable[str]]):
    expected_result = tf.nest.map_structure(
        lambda v: text_format.Parse(v, Node()), expected_result)

    with beam.Pipeline() as root:
      edges = {
          edge_set_name: root | f"Create/{edge_set_name}" >> beam.Create(values)
          for edge_set_name, values in edges.items()
      }
      actual_result = lib.create_adjacency_lists({"edges": edges},
                                                 sort_edges=sort_edges)
      self.assertCountEqual(actual_result.keys(), expected_result.keys())
      for edge_set_name in expected_result:
        util.assert_that(
            actual_result[edge_set_name],
            self._nodes_matcher(expected_result[edge_set_name]),
            label=edge_set_name)


class TestFindConnectingNodes(EdgeSamplingTestBase):

  @parameterized.named_parameters(
      dict(
          testcase_name="homogeneous",
          schema="""
            node_sets {
              key: "node"
            }
            edge_sets {
              key: "edge"
              value { source: "node" target: "node" }
            }
          """,
          incident_nodes={
              "node": [(b"s.1", [b"1", b"2", b"4"]), (b"s.2", [b"1", b"3"]),
                       (b"s.3", [b"x", b"y"])]
          },
          adj_lists={
              "edge": [
                  _create_test_node(1, [1, 2, 3]),
                  _create_test_node(2, [1, 4, 3, 2]),
                  _create_test_node(3, [1]),
                  _create_test_node(4, [1, 4])
              ]
          },
          expected_result={
              "edge": [
                  (b"s.1", _create_test_node(1, [1, 2])),
                  (b"s.1", _create_test_node(2, [1, 4, 2])),
                  (b"s.1", _create_test_node(4, [1, 4])),
                  (b"s.2", _create_test_node(1, [1, 3])),
                  (b"s.2", _create_test_node(3, [1])),
              ]
          }),
      dict(
          testcase_name="heterogeneous",
          schema="""
            node_sets { key: "a" }
            node_sets { key: "b" }
            node_sets { key: "c" }
            edge_sets {
              key: "a->b"
              value { source: "a" target: "b" }
            }
            edge_sets {
              key: "b->c"
              value { source: "b" target: "c" }
            }
            edge_sets {
              key: "c->a"
              value { source: "c" target: "a" }
            }
          """,
          incident_nodes={
              "a": [(b"s.1", [b"1"]),],
              "b": [(b"s.1", [b"2"]),],
              "c": [(b"s.1", [b"3"]),],
          },
          adj_lists={
              "a->b": [
                  _create_test_node(1, [2, 3, 4]),
                  _create_test_node(2, [1]),
              ],
              "b->c": [
                  _create_test_node(1, [2, 3, 4]),
                  _create_test_node(2, [3, 1]),
              ],
              "c->a": [_create_test_node(3, [3, 1]),],
          },
          expected_result={
              "a->b": [(b"s.1", _create_test_node(1, [2])),],
              "b->c": [(b"s.1", _create_test_node(2, [3])),],
              "c->a": [(b"s.1", _create_test_node(3, [1])),],
          }),
  )
  def test_logic(self, schema: str,
                 incident_nodes: Mapping[tfgnn.NodeSetName,
                                         Iterable[Tuple[SampleId,
                                                        List[NodeId]]]],
                 adj_lists: Mapping[tfgnn.EdgeSetName, Iterable[Node]],
                 expected_result: Mapping[tfgnn.EdgeSetName,
                                          Iterable[Tuple[SampleId, Node]]]):
    schema = text_format.Parse(schema, tfgnn.GraphSchema())

    with beam.Pipeline() as root:
      incident_nodes = {
          set_name: root | f"Nodes/{set_name}" >> beam.Create(values)
          for set_name, values in incident_nodes.items()
      }
      adj_lists = {
          set_name: root | f"AdjLists/{set_name}" >> beam.Create(values)
          for set_name, values in adj_lists.items()
      }
      actual_result = lib.find_connecting_edges(schema, incident_nodes,
                                                adj_lists)
      self.assertCountEqual(actual_result.keys(), expected_result.keys())
      for edge_set_name in expected_result:
        util.assert_that(
            actual_result[edge_set_name],
            self.sampled_edges_matcher(expected_result[edge_set_name]),
            label=edge_set_name)


class TestCreateUniqueNodeIds(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="seeds_only",
          schema="""
            node_sets {
              key: "node"
            }
            edge_sets {
              key: "edge"
              value { source: "node" target: "node" }
            }
          """,
          seeds={"node": [
              (b"s.1", b"1"),
              (b"s.2", b"1"),
          ]},
          edges={"edge": []},
          expected_result={"node": [
              (b"s.1", [b"1"]),
              (b"s.2", [b"1"]),
          ]}),
      dict(
          testcase_name="homogeneous",
          schema="""
            node_sets {
              key: "node"
            }
            edge_sets {
              key: "edge"
              value { source: "node" target: "node" }
            }
          """,
          seeds={"node": [
              (b"s.1", b"1"),
              (b"s.2", b"2"),
          ]},
          edges={
              "edge": [
                  (b"s.1", _create_test_node(1, [2])),
                  (b"s.1", _create_test_node(2, [1])),
                  (b"s.2", _create_test_node(2, [1, 3])),
                  (b"s.2", _create_test_node(3, [1])),
              ]
          },
          expected_result={
              "node": [
                  (b"s.1", [b"1", b"2"]),
                  (b"s.2", [b"1", b"2", b"3"]),
              ]
          }),
      dict(
          testcase_name="heterogeneous",
          schema="""
            node_sets { key: "A" }
            node_sets { key: "B" }
            edge_sets { key: "A->B" value { source: "A" target: "B" } }
            edge_sets { key: "B->A" value { source: "B" target: "A" } }
          """,
          seeds={"A": [
              (b"s.1", b"1"),
              (b"s.2", b"2"),
          ]},
          edges={
              "A->B": [
                  (b"s.1", _create_test_node(1, [1])),
                  (b"s.2", _create_test_node(2, [1])),
              ],
              "B->A": [
                  (b"s.1", _create_test_node(1, [1])),
                  (b"s.2", _create_test_node(1, [3])),
              ]
          },
          expected_result={
              "A": [
                  (b"s.1", [b"1"]),
                  (b"s.2", [b"2", b"3"]),
              ],
              "B": [
                  (b"s.1", [b"1"]),
                  (b"s.2", [b"1"]),
              ]
          }),
  )
  def test_logic(self, schema: str, seeds: Mapping[tfgnn.NodeSetName,
                                                   Iterable[Tuple[SampleId,
                                                                  NodeId]]],
                 edges: Mapping[tfgnn.EdgeSetName, Iterable[Tuple[SampleId,
                                                                  Node]]],
                 expected_result: Mapping[tfgnn.NodeSetName,
                                          List[Tuple[SampleId, List[bytes]]]]):

    def ids_matcher(expected):

      def _sorted(values):
        result = [(k, sorted(v)) for k, v in values]
        return sorted(result, key=lambda item: item[0])

      sorted_expected = _sorted(expected)

      def _equal(actual):
        sorted_actual = _sorted(actual)
        if sorted_expected != sorted_actual:
          raise util.BeamAssertException("Failed assert: %r == %r" %
                                         (sorted_expected, sorted_actual))

      return _equal

    schema = text_format.Parse(schema, tfgnn.GraphSchema())

    with beam.Pipeline() as root:
      seeds = {
          set_name: root | f"Seeds/{set_name}" >> beam.Create(values)
          for set_name, values in seeds.items()
      }
      edges = {
          set_name: root | f"Edges/{set_name}" >> beam.Create(values)
          for set_name, values in edges.items()
      }
      actual_result = lib.create_unique_node_ids(schema, seeds, edges)
      self.assertCountEqual(actual_result.keys(), expected_result.keys())
      for node_set_name in expected_result:
        util.assert_that(
            actual_result[node_set_name],
            ids_matcher(expected_result[node_set_name]),
            label=node_set_name)


class TestLookupNodeFeatures(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="homogeneous",
          node_ids={
              "node": [
                  (b"s.1", [b"1", b"2", b"3"]),
                  (b"s.2", [b"1"]),
                  (b"s.3", [b"x"]),
              ]
          },
          node_features={"node": {
              b"1": b"foo",
              b"3": b"bar",
          }},
          expected_result={
              "node": [
                  (b"s.1", (b"1", b"foo")),
                  (b"s.1", (b"3", b"bar")),
                  (b"s.2", (b"1", b"foo")),
              ]
          }),
      dict(
          testcase_name="heterogeneous",
          node_ids={
              "a": [
                  (b"s.1", [b"1", b"2", b"3"]),
                  (b"s.2", [b"3"]),
              ],
              "b": [(b"s.1", [b"1", b"2"]),]
          },
          node_features={
              "a": {
                  b"1": b"f.a1",
                  b"3": b"f.a3",
              },
              "b": {
                  b"2": b"f.b2",
              }
          },
          expected_result={
              "a": [
                  (b"s.1", (b"1", b"f.a1")),
                  (b"s.1", (b"3", b"f.a3")),
                  (b"s.2", (b"3", b"f.a3")),
              ],
              "b": [(b"s.1", (b"2", b"f.b2")),]
          }),
  )
  def test_logic(self, node_ids: Mapping[tfgnn.NodeSetName,
                                         List[Tuple[SampleId, List[bytes]]]],
                 node_features: Mapping[tfgnn.NodeSetName, Mapping[NodeId,
                                                                   bytes]],
                 expected_result: Mapping[tfgnn.NodeSetName,
                                          List[Tuple[SampleId, bytes]]]):

    def as_example(value: bytes) -> tf.train.Example:
      result = tf.train.Example()
      result.features.feature["s"].bytes_list.value.append(value)
      return result

    def extract_feature(
        sample_id: SampleId, values: Tuple[NodeId, tf.train.Features]
    ) -> Tuple[SampleId, Tuple[NodeId, bytes]]:
      node_id, features = values
      return sample_id, (node_id, features.feature["s"].bytes_list.value[0])

    with beam.Pipeline() as root:
      node_ids = {
          set_name: root | f"NodeIds/{set_name}" >> beam.Create(ids)
          for set_name, ids in node_ids.items()
      }

      node_examples = tf.nest.map_structure(as_example, node_features)
      node_examples = {
          set_name: root | f"Features/{set_name}" >> beam.Create(value.items())
          for set_name, value in node_examples.items()
      }
      actual_result = lib.lookup_node_features(node_ids, node_examples)
      self.assertCountEqual(actual_result.keys(), expected_result.keys())
      for node_set_name in expected_result:
        util.assert_that(
            actual_result[node_set_name]
            | f"ParseResult/{node_set_name}" >> beam.MapTuple(extract_feature),
            util.equal_to(expected_result[node_set_name]),
            label=node_set_name)


if __name__ == "__main__":
  tf.test.main()
