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
"""Tests for GraphSchema utils (go/tf-gnn-api)."""

from absl.testing import parameterized
import google.protobuf.text_format as pbtext
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import schema_utils as su
import tensorflow_gnn.proto.graph_schema_pb2 as schema_pb2
from tensorflow_gnn.utils import test_utils

_SCHEMA_SPEC_MATCHING_PAIRS = [
    dict(
        testcase_name='context_schema',
        schema_pbtxt="""
          context {
            features {
              key: "label"
              value: {
                dtype: DT_STRING
              }
            }
            features {
              key: "embedding"
              value: {
                dtype: DT_FLOAT
                shape: { dim { size: 128 } }
              }
            }
          }
          """,
        graph_spec=gt.GraphTensorSpec.from_piece_specs(
            context_spec=gt.ContextSpec.from_field_specs(
                features_spec={
                    'label':
                        tf.TensorSpec(shape=(1,), dtype=tf.string),
                    'embedding':
                        tf.TensorSpec(shape=(1, 128), dtype=tf.float32),
                },
                indices_dtype=tf.int64))),
    dict(
        testcase_name='nodes_schema',
        schema_pbtxt="""
              node_sets {
                key: 'node'
                value {
                  features {
                    key: "id"
                    value: {
                      dtype: DT_INT32
                    }
                  }
                  features {
                    key: "words"
                    value: {
                      dtype: DT_STRING
                      shape: { dim { size: -1 } }
                    }
                  }
                }
              }
              """,
        graph_spec=gt.GraphTensorSpec.from_piece_specs(
            node_sets_spec={
                'node':
                    gt.NodeSetSpec.from_field_specs(
                        sizes_spec=tf.TensorSpec(
                            shape=(1,), dtype=tf.int64),
                        features_spec={
                            'id':
                                tf.TensorSpec(
                                    shape=(None,), dtype=tf.int32),
                            'words':
                                tf.RaggedTensorSpec(
                                    shape=(None, None),
                                    dtype=tf.string,
                                    ragged_rank=1,
                                    row_splits_dtype=tf.int64),
                        })
            })),
    dict(
        testcase_name='edges_schema',
        schema_pbtxt="""
              node_sets { key: 'node'}
              edge_sets {
                key: 'edge'
                value {
                  source: 'node'
                  target: 'node'
                  features {
                    key: "weight"
                    value: {
                      dtype: DT_FLOAT
                    }
                  }
                }
              }
              """,
        graph_spec=gt.GraphTensorSpec.from_piece_specs(
            node_sets_spec={
                'node':
                    gt.NodeSetSpec.from_field_specs(
                        sizes_spec=tf.TensorSpec(
                            shape=(1,), dtype=tf.int32))
            },
            edge_sets_spec={
                'edge':
                    gt.EdgeSetSpec.from_field_specs(
                        features_spec={
                            'weight':
                                tf.TensorSpec(
                                    shape=(None,), dtype=tf.float32)
                        },
                        sizes_spec=tf.TensorSpec(
                            shape=(1,), dtype=tf.int32),
                        adjacency_spec=(
                            adj.AdjacencySpec.from_incident_node_sets(
                                source_node_set='node',
                                target_node_set='node',
                                index_spec=tf.TensorSpec(
                                    shape=(None,), dtype=tf.int32))))
            }))
]


class SchemaUtilsTest(tf.test.TestCase):

  def test_iter_sets(self):
    schema = test_utils.get_proto_resource('testdata/homogeneous/citrus.pbtxt',
                                           schema_pb2.GraphSchema())

    self.assertSetEqual(
        set([('nodes', 'fruits'), ('edges', 'tastelike')]),
        set((stype, sname) for stype, sname, _ in su.iter_sets(schema)))

    # pylint: disable=pointless-statement
    schema.context.features['mutate_this']
    self.assertSetEqual(
        set([('context', ''), ('nodes', 'fruits'), ('edges', 'tastelike')]),
        set((stype, sname) for stype, sname, _ in su.iter_sets(schema)))

  def test_iter_features(self):
    schema = test_utils.get_proto_resource('testdata/homogeneous/citrus.pbtxt',
                                           schema_pb2.GraphSchema())
    self.assertSetEqual(
        set([('nodes', 'fruits'), ('edges', 'tastelike')]),
        set((stype, sname) for stype, sname, _ in su.iter_sets(schema)))


class SchemaToGraphTensorSpecTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for Graph Tensor specification."""

  def testInvariants(self):
    schema_pbtxt = """
              node_sets { key: 'node'}
              edge_sets {
                key: 'edge'
                value {
                  source: 'node'
                  target: 'node'
                  features {
                    key: "weight"
                    value: {
                      dtype: DT_FLOAT
                    }
                  }
                }
              }
              """
    schema_pb = pbtext.Merge(schema_pbtxt, schema_pb2.GraphSchema())
    result_spec = su.create_graph_spec_from_schema_pb(
        schema_pb, indices_dtype=tf.int32)
    self.assertAllEqual(result_spec.shape, tf.TensorShape([]))
    self.assertAllEqual(result_spec.indices_dtype, tf.int32)
    self.assertAllEqual(result_spec.total_num_components, 1)

    self.assertAllEqual(list(result_spec.node_sets_spec.keys()), ['node'])
    self.assertIsNone(result_spec.node_sets_spec['node'].total_size)
    self.assertEmpty(result_spec.node_sets_spec['node'].features_spec)

    self.assertAllEqual(list(result_spec.edge_sets_spec.keys()), ['edge'])
    edge_set_spec = result_spec.edge_sets_spec['edge']
    self.assertIsNone(edge_set_spec.total_size)
    self.assertEqual(list(edge_set_spec.features_spec.keys()), ['weight'])

  @parameterized.named_parameters(_SCHEMA_SPEC_MATCHING_PAIRS)
  def testParametrized(self, schema_pbtxt: str, graph_spec: gt.GraphTensorSpec):
    schema_pb = pbtext.Merge(schema_pbtxt, schema_pb2.GraphSchema())
    result_spec = su.create_graph_spec_from_schema_pb(
        schema_pb, indices_dtype=graph_spec.indices_dtype)
    self.assertAllEqual(graph_spec, result_spec)


class GraphTensorSpecToSchemaTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for Graph Tensor specification."""

  @parameterized.named_parameters(_SCHEMA_SPEC_MATCHING_PAIRS)
  def testParametrized(self, schema_pbtxt: str, graph_spec: gt.GraphTensorSpec):
    result_schema = su.create_schema_pb_from_graph_spec(graph_spec)
    expected_schema_pb = pbtext.Merge(schema_pbtxt, schema_pb2.GraphSchema())
    self.assertEqual(expected_schema_pb, result_schema)


class CompatibleWithSchemaTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for Graph Tensor specification."""

  @parameterized.named_parameters(_SCHEMA_SPEC_MATCHING_PAIRS)
  def testParametrized(self, schema_pbtxt: str, graph_spec: gt.GraphTensorSpec):
    # pylint: disable=protected-access
    schema_pb = pbtext.Merge(schema_pbtxt, schema_pb2.GraphSchema())
    su.check_compatible_with_schema_pb(graph_spec, schema_pb)
    self.assertRaisesRegex(
        ValueError,
        r'check_compatible_with_schema_pb\(\) requires a scalar GraphTensor',
        su.check_compatible_with_schema_pb, graph_spec._batch(None), schema_pb)

  def testFailsForMultipleComponents(self):
    # pylint: disable=protected-access
    schema_pbtxt = """
          context {
            features { key: "label" value: { dtype: DT_STRING } }
          }
          """
    graph_spec = gt.GraphTensorSpec.from_piece_specs(
        context_spec=gt.ContextSpec.from_field_specs(
            features_spec={
                'label': tf.TensorSpec(shape=(2,), dtype=tf.string),
            },
            indices_dtype=tf.int64))
    schema_pb = pbtext.Merge(schema_pbtxt, schema_pb2.GraphSchema())
    self.assertRaisesRegex(
        ValueError, (r'check_compatible_with_schema_pb\(\) requires scalar'
                     ' GraphTensor with a single graph component'),
        su.check_compatible_with_schema_pb, graph_spec, schema_pb)

  def testFailsForIncompatibleFeatures(self):
    # pylint: disable=protected-access
    schema_pbtxt = """
          context {
            features { key: "label" value: { dtype: DT_STRING } }
          }
          """
    graph_spec = gt.GraphTensorSpec.from_piece_specs(
        context_spec=gt.ContextSpec.from_field_specs(
            features_spec={
                'not_label': tf.TensorSpec(shape=(1,), dtype=tf.string),
            },
            indices_dtype=tf.int64))
    schema_pb = pbtext.Merge(schema_pbtxt, schema_pb2.GraphSchema())
    self.assertRaisesRegex(
        ValueError, (r'Graph is not compatible with the graph schema'),
        su.check_compatible_with_schema_pb, graph_spec, schema_pb)


if __name__ == '__main__':
  tf.test.main()
