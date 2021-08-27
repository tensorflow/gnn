"""Tests for open source graph sampler."""

from os import path
import random
import tempfile

from typing import Dict, List, Union, Tuple, Any, Optional

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import networkx as nx
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.proto import examples_pb2
from tensorflow_gnn.sampler import graph_sampler as sampler
from tensorflow_gnn.sampler import sampling_spec_pb2
from tensorflow_gnn.sampler import subgraph
from tensorflow_gnn.sampler import subgraph_pb2
from tensorflow_gnn.utils import test_utils

from google.protobuf import text_format

Example = tf.train.Example
Node = subgraph_pb2.Node
PCollection = beam.PCollection


def _make_example(features_dict: Dict[str, Union[int, float, bytes, str]]):
  """Create a tf.train.Example from a dict."""
  ex = tf.train.Example()
  for key, values in features_dict.items():
    if isinstance(values, list):
      if values:
        value = values[0]
        if not all(isinstance(v, type(value)) for v in values):
          raise ValueError("Inconsistent types: {}".format(values))
    else:
      values = [values]
    if values and isinstance(values[0], str):
      values = [s.encode("utf8") for s in values]
    for value in values:
      feature = ex.features.feature[key]
      if isinstance(value, int):
        flist = feature.int64_list.value
      elif isinstance(value, float):
        flist = feature.float_list.value
      elif isinstance(value, bytes):
        flist = feature.bytes_list.value
      else:
        raise ValueError(
            f"Expected int, float, or bytes for feature, got: {value}")
      flist.append(value)
  return ex


_FLORENTINE_FAMILIES_GRAPH_SCHEMA = """
  node_sets {
    key: "family"
    value {
      features {
        key: "name"
        value {
          dtype: DT_STRING
        }
      }
    }
  }
  edge_sets {
    key: "marriage"
    value {
      source: "family"
      target: "family"
      features {
        key: "love"
        value {
          dtype: DT_FLOAT
        }
      }
    }
  }
"""


def _make_random_graph_tables(
    parent: PCollection,
    seed: int) -> Tuple[tfgnn.GraphSchema, PCollection, PCollection]:
  """Generates an in-memory version of a unigraph dataset."""
  random.seed(seed)
  schema = text_format.Parse(_FLORENTINE_FAMILIES_GRAPH_SCHEMA,
                             tfgnn.GraphSchema())
  graph = nx.florentine_families_graph()
  nodes = []
  for name in graph.nodes:
    node = Node()
    node.id = name.encode("utf8")
    for target in graph[name]:
      edge = node.outgoing_edges.add()
      edge.neighbor_id = target.encode("utf8")
      edge.features.feature["weight"].float_list.value.append(random.random())
    nodes.append(node)
  nodes_coll = (
      parent
      | "Nodes" >> beam.Create([(node.id, node) for node in nodes]))
  features_coll = (
      parent | "Features" >> beam.Create(
          [(node.id, _make_example({"letter": node.id[0]})) for node in nodes]))
  return schema, nodes_coll, features_coll


class TestSampleGraph(tf.test.TestCase):

  def test_create_adjacency(self):
    with beam.Pipeline() as p:
      nodes = p | "Nodes" >> beam.Create([(b"nsn", bytes((key,)), Example())
                                          for key in b"abcde"])
      edges = p | "Edges" >> beam.Create(
          [(b"esn", bytes((edge[0],)), bytes((edge[1],)), Example())
           for edge in [b"ab", b"ac", b"bd", b"be", b"ee", b"da", b"dc"]])
      node_map = sampler.create_adjacency_list(nodes, edges)

      exp_items = [(b"a", ("""id: "a" node_set_name: "nsn"
                   outgoing_edges { neighbor_id: "b" edge_set_name: "esn"}
                   outgoing_edges { neighbor_id: "c" edge_set_name: "esn"}""")),
                   (b"b", ("""id: "b" node_set_name: "nsn"
                   outgoing_edges { neighbor_id: "d" edge_set_name: "esn"}
                   outgoing_edges { neighbor_id: "e" edge_set_name: "esn"}""")),
                   (b"e", ("""id: "e" node_set_name: "nsn"
                   outgoing_edges { neighbor_id: "e" edge_set_name: "esn"}""")),
                   (b"d", ("""id: "d" node_set_name: "nsn"
                   outgoing_edges { neighbor_id: "a" edge_set_name: "esn"}
                   outgoing_edges { neighbor_id: "c" edge_set_name: "esn"}""")),
                   (b"c", 'id: "c" node_set_name: "nsn"')]
      expected_node_map = {
          key: text_format.Parse(value, Node()) for key, value in exp_items
      }

      def _assert_fn(data):
        self.assertEqual(len(data), len(expected_node_map))
        for key, proto in data:
          self.assertProtoEquals(expected_node_map[key], proto)

      util.assert_that(node_map, _assert_fn)

  def test_sample_graph(self):
    with beam.Pipeline() as p:
      unused_schema, nodes, unused_features = _make_random_graph_tables(p, 42)
      seeds = nodes | beam.Keys()
      spec = text_format.Parse(
          """
        insert_sample_ids: True
        seed_op: <
          op_name: 'seed'
          node_set_name: 'fruits'
        >
        sampling_ops: <
          op_name: 'hop-1'
          input_op_names: [ 'seed' ]
          strategy: TOP_K
          sample_size: 8
          edge_set_name: 'tastelike'
        >
        sampling_ops: <
          op_name: 'hop-2'
          input_op_names: [ 'hop-1' ]
          strategy: TOP_K
          sample_size: 4
          edge_set_name: 'tastelike'
        >
      """, sampling_spec_pb2.SamplingSpec())
      subgraphs = sampler.sample_graph(nodes, seeds, spec)

      def _assert_fn(data):
        self.assertEqual(len(data), 15)
        for key, proto in data:
          self.assertIsInstance(key, bytes)
          self.assertIsInstance(proto, subgraph_pb2.Subgraph)

      util.assert_that(subgraphs, _assert_fn)


def read_tfrecords_of_examples(filename: str) -> List[Example]:
  protos = []
  for example_str in tf.data.TFRecordDataset(filename):
    example = Example()
    example.ParseFromString(example_str.numpy())
    protos.append(example)
  return protos


def read_proto_examples(filename: str) -> List[Example]:
  """Read a text-formatted file of example proto."""
  with open(filename) as fp:
    example_list = text_format.Parse(fp.read(), examples_pb2.ExampleList())
  return list(example_list.examples)


def approximate_examples_equal(example1: Example,
                               example2: Example) -> Optional[str]:
  """Performs an approximate comparison of two example protos."""
  # This ignores the ordering of values and compares them as sets.
  keys1 = example1.features.feature.keys()
  keys2 = example2.features.feature.keys()
  for name in sorted(set(keys1) | set(keys2)):
    if name not in example1.features.feature:
      return f"Feature {name} not in first example"
    if name not in example2.features.feature:
      return f"Feature {name} not in second example"
    values1 = subgraph.get_feature_values(example1.features.feature[name])
    values2 = subgraph.get_feature_values(example2.features.feature[name])
    if values1 is None:
      return f"Feature {name} not in first example."
    if values2 is None:
      return f"Feature {name} not in second example."
    set1, set2 = set(values1), set(values2)
    if set1 != set2:
      return f"Values for feature {name} not equal: {set1} != {set2}"
  return None


def get_key(example: Example):
  """Gets the node key from the given example."""
  return example.features.feature["context/seed_id"].bytes_list.value[0]


class TestSamplePipeline(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("homogeneous", "testdata/homogeneous/citrus.pbtxt", """
        insert_sample_ids: True
        seed_op: <
          op_name: 'seed'
          node_set_name: 'fruits'
        >
        sampling_ops: <
          op_name: 'hop-1'
          input_op_names: [ 'seed' ]
          strategy: TOP_K
          sample_size: 100
          edge_set_name: 'tastelike'
        >
        sampling_ops: <
          op_name: 'hop-2'
          input_op_names: [ 'hop-1' ]
          strategy: TOP_K
          sample_size: 100
          edge_set_name: 'tastelike'
        >
      """, "testdata/homogeneous/sampler_golden.ascii"),
      ("heterogeneous", "testdata/heterogeneous/graph.pbtxt", """
        insert_sample_ids: True
        seed_op: <
          op_name: 'seed'
          node_set_name: 'customer'
        >
        sampling_ops: <
          op_name: 'hop-1'
          input_op_names: [ 'seed' ]
          strategy: TOP_K
          sample_size: 1
          edge_set_name: 'owns_card'
        >
        sampling_ops: <
          op_name: 'hop-2'
          input_op_names: [ 'hop-1' ]
          strategy: TOP_K
          sample_size: 1
          edge_set_name: 'owns_card'
        >
      """, "testdata/heterogeneous/sampler_golden.ascii"))
  def test_against_golden_data(self, schema_file, spec_text, golden_file):
    spec = text_format.Parse(spec_text, sampling_spec_pb2.SamplingSpec())
    schema_file = test_utils.get_resource(schema_file)
    with tempfile.TemporaryDirectory() as tmpdir:
      graph_tensor_filename = path.join(tmpdir, "graph_tensors.tfrecords")
      sampler.run_sample_graph_pipeline(schema_file, spec,
                                        graph_tensor_filename)

      self._assert_parseable(
          path.join(tmpdir, "schema.pbtxt"), graph_tensor_filename)

      actual_protos = {
          get_key(example): example
          for example in read_tfrecords_of_examples(graph_tensor_filename)
      }

    # Read expected protos from golden file.
    golden_filename = test_utils.get_resource(golden_file)
    expected_protos = {
        get_key(example): example
        for example in read_proto_examples(golden_filename)
    }

    # Map by the first node, which is the seed, in order to compare.
    for seed_name, expected_proto in expected_protos.items():
      actual_proto = actual_protos[seed_name]
      self.assertIsNone(
          approximate_examples_equal(expected_proto, actual_proto),
          "Examples differ: {} != {}".format(expected_proto, actual_proto))

  def _assert_parseable(self, schema_file: str, records_file: str) -> None:
    """Asserts that the tensors in the given `filename` are parseable.

    Args:
      schema_file: The path to the file containing the Graph schema.
      records_file: The path to the file with tfrecords to parse.

    Raises:
      An error if the tensors are not parseable by the TF-GNN IO library.
    """
    schema = tfgnn.read_schema(schema_file)
    # Assert the schema has no metadata attached to it.
    self.assertFalse(
        schema.context.metadata.HasField("cardinality"),
        msg="Expected context metadata cardinality to be empty.")
    self.assertFalse(
        schema.context.metadata.HasField("filename"),
        msg="Expected context metadata filename to be empty.")
    for node_set_name, node_set in schema.node_sets.items():
      self.assertFalse(
          node_set.metadata.HasField("cardinality"),
          msg=f"Expected node {node_set_name} metadata cardinality to be empty."
      )
      self.assertFalse(
          node_set.metadata.HasField("filename"),
          msg=f"Expected node {node_set_name} metadata filename to be empty.")
    for edge_set_name, edge_set in schema.edge_sets.items():
      self.assertFalse(
          edge_set.metadata.HasField("cardinality"),
          msg=f"Expected edge {edge_set_name} metadata cardinality to be empty."
      )
      self.assertFalse(
          edge_set.metadata.HasField("filename"),
          msg=f"Expected edge {edge_set_name} metadata filename to be empty.")
    graph_spec = tfgnn.create_graph_spec_from_schema_pb(schema)
    for example_str in tf.data.TFRecordDataset(records_file):
      example = tfgnn.parse_example(graph_spec, tf.expand_dims(example_str, 0))
      self.assertIsNotNone(example)

  @parameterized.named_parameters(("no_seed_op_name", """
        seed_op: <
          node_set_name: 'fruits'
        >
      """, "seed operation name"), ("invalid_input", """
        seed_op: <
          op_name: 'seed'
          node_set_name: 'fruits'
        >
        sampling_ops: <
          op_name: 'hop-1'
          input_op_names: [ 'foo' ]
          strategy: TOP_K
          sample_size: 1
          edge_set_name: 'tastelike'
        >
      """, "'foo' from operation 'hop-1'"), ("invalid_sample_size", """
        seed_op: <
          op_name: 'seed'
          node_set_name: 'fruits'
        >
        sampling_ops: <
          op_name: 'hop-1'
          input_op_names: [ 'seed' ]
          strategy: TOP_K
          sample_size: 0
          edge_set_name: 'tastelike'
        >
      """, "sample size 0"), ("input_wrong_order", """
        seed_op: <
          op_name: 'seed'
          node_set_name: 'fruits'
        >
        sampling_ops: <
          op_name: 'hop-1'
          input_op_names: [ 'hop-2' ]
          strategy: TOP_K
          sample_size: 1
          edge_set_name: 'tastelike'
        >
        sampling_ops: <
          op_name: 'hop-2'
          input_op_names: [ 'seed' ]
          strategy: TOP_K
          sample_size: 1
          edge_set_name: 'tastelike'
        >
      """, "'hop-2' from operation 'hop-1'"), ("invalid_seed_node_set_name", """
        seed_op: <
          op_name: 'seed'
          node_set_name: 'foo'
        >
      """, "node set name 'foo'"), ("invalid_sampling_edge_set_name", """
        seed_op: <
          op_name: 'seed'
          node_set_name: 'fruits'
        >
        sampling_ops: <
          op_name: 'hop-1'
          input_op_names: [ 'seed' ]
          strategy: TOP_K
          sample_size: 1
          edge_set_name: 'foo'
        >
      """, "edge set name 'foo'"))
  def test_invalid_spec(self, sampling_spec, error_regex):
    """Tests that the given spec fails with the given error regex."""
    spec = text_format.Parse(sampling_spec, sampling_spec_pb2.SamplingSpec())
    with self.assertRaisesRegex(ValueError, error_regex):
      sampler.run_sample_graph_pipeline(
          test_utils.get_resource("testdata/homogeneous/citrus.pbtxt"), spec,
          "graph_tensors.tfrecords")

  @parameterized.named_parameters(
      ("no_sampling", "testdata/homogeneous/citrus.pbtxt", """
        insert_sample_ids: True
        seed_op: <
          op_name: 'seed'
          node_set_name: 'fruits'
        >
      """, "testdata/homogeneous/one_seed.csv", [{
          "nodes/fruits.name": [b"Amanatsu"],
      }]), ("sequential_hops", "testdata/homogeneous/citrus.pbtxt", """
        insert_sample_ids: True
        seed_op: <
          op_name: 'seed'
          node_set_name: 'fruits'
        >
        sampling_ops: <
          op_name: 'hop-1'
          input_op_names: [ 'seed' ]
          strategy: TOP_K
          sample_size: 1
          edge_set_name: 'tastelike'
        >
        sampling_ops: <
          op_name: 'hop-2'
          input_op_names: [ 'hop-1' ]
          strategy: TOP_K
          sample_size: 1
          edge_set_name: 'tastelike'
        >
      """, "testdata/homogeneous/one_seed.csv", [{
          "nodes/fruits.name": [b"Amanatsu", b"Lumia", b"Daidai"],
      }]), ("two_branches_graph", "testdata/homogeneous/citrus.pbtxt", """
        insert_sample_ids: True
        seed_op: <
          op_name: 'seed'
          node_set_name: 'fruits'
        >
        sampling_ops: <
          op_name: 'hop-1'
          input_op_names: [ 'seed' ]
          strategy: TOP_K
          sample_size: 1
          edge_set_name: 'tastelike'
        >
        sampling_ops: <
          op_name: 'hop-2'
          input_op_names: [ 'seed' ]
          strategy: TOP_K
          sample_size: 1
          edge_set_name: 'tastelike'
        >
      """, "testdata/homogeneous/one_seed.csv", [{
          "nodes/fruits.name": [b"Amanatsu", b"Lumia"],
      }]), ("multiple_seeds", "testdata/homogeneous/citrus.pbtxt", """
        insert_sample_ids: True
        seed_op: <
          op_name: 'seed'
          node_set_name: 'fruits'
        >
        sampling_ops: <
          op_name: 'hop-1'
          input_op_names: [ 'seed' ]
          strategy: TOP_K
          sample_size: 1
          edge_set_name: 'tastelike'
        >
        sampling_ops: <
          op_name: 'hop-2'
          input_op_names: [ 'hop-1' ]
          strategy: TOP_K
          sample_size: 1
          edge_set_name: 'tastelike'
        >
      """, "testdata/homogeneous/two_seeds.csv", [{
          "nodes/fruits.name": [b"Amanatsu", b"Lumia", b"Daidai"],
      }, {
          "nodes/fruits.name": [b"Daidai"],
      }]), ("multiple_inputs", "testdata/homogeneous/citrus.pbtxt", """
        insert_sample_ids: True
        seed_op: <
          op_name: 'seed'
          node_set_name: 'fruits'
        >
        sampling_ops: <
          op_name: 'hop-1'
          input_op_names: [ 'seed' ]
          strategy: TOP_K
          sample_size: 1
          edge_set_name: 'tastelike'
        >
        sampling_ops: <
          op_name: 'hop-2'
          input_op_names: [ 'seed', 'hop-1' ]
          strategy: TOP_K
          sample_size: 1
          edge_set_name: 'tastelike'
        >
      """, "testdata/homogeneous/one_seed.csv", [{
          "nodes/fruits.name": [b"Amanatsu", b"Lumia", b"Daidai"],
      }]), ("heterogeneous_simple", "testdata/heterogeneous/graph.pbtxt", """
        insert_sample_ids: True
        seed_op: <
          op_name: 'seed'
          node_set_name: 'customer'
        >
        sampling_ops: <
          op_name: 'hop-1'
          input_op_names: [ 'seed' ]
          strategy: TOP_K
          sample_size: 1
          edge_set_name: 'owns_card'
        >
        sampling_ops: <
          op_name: 'hop-2'
          input_op_names: [ 'hop-1' ]
          strategy: TOP_K
          sample_size: 1
          edge_set_name: 'owns_card'
        >
      """, "testdata/heterogeneous/one_customer.csv", [{
          "nodes/customer.name": [b"Ji Grindstaff"],
          "nodes/creditcard.number": [16827485386298040],
          "nodes/transaction.merchant": [],
      }]), ("heterogeneous_multiple", "testdata/heterogeneous/graph.pbtxt", """
        insert_sample_ids: True
        seed_op: <
          op_name: 'seed'
          node_set_name: 'customer'
        >
        sampling_ops: <
          op_name: 'hop-1'
          input_op_names: [ 'seed' ]
          strategy: TOP_K
          sample_size: 1
          edge_set_name: 'owns_card'
        >
        sampling_ops: <
          op_name: 'hop-2'
          input_op_names: [ 'hop-1' ]
          strategy: TOP_K
          sample_size: 1
          edge_set_name: 'owns_card'
        >
      """, "testdata/heterogeneous/two_customers.csv", [{
          "nodes/customer.name": [b"Ji Grindstaff"],
          "nodes/creditcard.number": [16827485386298040],
          "nodes/transaction.merchant": [],
      }, {
          "nodes/customer.name": [b"Augustina Uren"],
          "nodes/creditcard.number": [11470379189154620],
          "nodes/transaction.merchant": [],
      }]))
  def test_nodes_sampled(self, schema_file, sampling_spec, seeds_file,
                         expected_keys):
    """Tests that the given nodes are indeed the ones that were sampled."""
    spec = text_format.Parse(sampling_spec, sampling_spec_pb2.SamplingSpec())
    schema_file = test_utils.get_resource(schema_file)
    with tempfile.TemporaryDirectory() as tmpdir:
      graph_tensor_filename = path.join(tmpdir, "graph_tensors.tfrecords")
      sampler.run_sample_graph_pipeline(schema_file, spec,
                                        graph_tensor_filename,
                                        test_utils.get_resource(seeds_file))

      self._assert_parseable(
          path.join(tmpdir, "schema.pbtxt"), graph_tensor_filename)
      expected_records = read_tfrecords_of_examples(graph_tensor_filename)

    # Construct the actual and expected objects.
    actual_key_maps: List[Dict[str, List[Any]]] = []
    for i, ex in enumerate(expected_records):
      actual_key_map: Dict[str, List[Any]] = {}
      for feature_name, feature in ex.features.feature.items():
        if feature_name in expected_keys[i]:
          actual_key_map[feature_name] = _extract_value(feature)

      # Add any keys that we want to ensure are empty.
      for feature_name, feature_value in expected_keys[i].items():
        if not feature_value and feature_name not in actual_key_map:
          actual_key_map[feature_name] = []
      actual_key_maps.append(actual_key_map)

    self.assertListEqual(actual_key_maps, expected_keys)

  def test_invalid_seed_id(self):
    """Tests that a warning is logged when a given seed does not exist."""
    spec = text_format.Parse(
        """
        insert_sample_ids: True
        seed_op: <
          op_name: 'seed'
          node_set_name: 'customer'
        >
        sampling_ops: <
          op_name: 'hop-1'
          input_op_names: [ 'seed' ]
          strategy: TOP_K
          sample_size: 1
          edge_set_name: 'owns_card'
        >""", sampling_spec_pb2.SamplingSpec())
    with tempfile.TemporaryDirectory() as tmpdir:
      graph_tensor_filename = path.join(tmpdir, "graph_tensors.tfrecords")
      schema_file = "testdata/heterogeneous/graph.pbtxt"
      seeds_file = "testdata/heterogeneous/invalid_customer.csv"

      with self.assertLogs(level="WARN") as logs:
        sampler.run_sample_graph_pipeline(
            test_utils.get_resource(schema_file), spec, graph_tensor_filename,
            test_utils.get_resource(seeds_file))

      want = "Seed node with ID b'1'"
      log_found = any(want in log for log in logs.output)
      self.assertTrue(
          log_found, msg=f"Message '{want}' not found in logs: {logs.output}")


def _extract_value(feature: tf.train.Feature) -> List[Any]:
  if feature.bytes_list.value:
    return feature.bytes_list.value
  if feature.float_list.value:
    return feature.float_list.value
  if feature.int64_list.value:
    return feature.int64_list.value
  return []


if __name__ == "__main__":
  tf.test.main()
