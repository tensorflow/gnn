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
"""Unit tests for graph tensor container."""

import copy
from typing import List

from absl import logging
import mock
import tensorflow as tf

from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import schema_validation as sv
import tensorflow_gnn.proto.graph_schema_pb2 as schema_pb2

from google.protobuf import text_format


as_tensor = tf.convert_to_tensor


class GraphValidationTest(tf.test.TestCase):

  def test_validate_schema_feature_dtypes(self):
    schema = text_format.Parse("""
      node_sets {
        key: "queries"
        value {
          features {
            key: "num_words"
            value {
              description: "Number of stop words, regular words, frequent words"
            }
          }
        }
      }
    """, schema_pb2.GraphSchema())

    # Check that dtype is always set.
    with self.assertRaises(sv.ValidationError):
      sv._validate_schema_feature_dtypes(schema)

    for dtype in tf.string, tf.int64, tf.float32:
      num_words = schema.node_sets['queries'].features['num_words']
      num_words.dtype = dtype.as_datatype_enum
      sv._validate_schema_feature_dtypes(schema)

    for dtype in tf.int32, tf.float64:
      num_words = schema.node_sets['queries'].features['num_words']
      num_words.dtype = dtype.as_datatype_enum
      with self.assertRaises(sv.ValidationError):
        sv._validate_schema_feature_dtypes(schema)

  def test_validate_schema_shapes(self):
    schema = text_format.Parse("""
      node_sets {
        key: "queries"
        value {
          features {
            key: "num_words"
            value {
              description: "Number of stop words, regular words, frequent words"
              shape { dim { size: 1 } }
            }
          }
        }
      }
    """, schema_pb2.GraphSchema())

    # Test tensor shape protos with unknown ranks (not allowed).
    shape = schema.node_sets['queries'].features['num_words'].shape
    shape.dim[0].size = 2
    sv._validate_schema_shapes(schema)
    shape.unknown_rank = True
    with self.assertRaises(sv.ValidationError):
      sv._validate_schema_shapes(schema)

  def test_warn_schema_scalar_shapes(self):
    schema = text_format.Parse("""
      node_sets {
        key: "queries"
        value {
          features {
            key: "num_words"
            value {
              description: "Number of stop words, regular words, frequent words"
              shape { dim { size: 1 } }
            }
          }
        }
      }
    """, schema_pb2.GraphSchema())
    warnings = sv._warn_schema_scalar_shapes(schema)
    self.assertIsInstance(warnings, list)
    self.assertLen(warnings, 1)
    self.assertIsInstance(warnings[0], sv.ValidationError)

  def test_validate_schema_descriptions(self):
    schema = text_format.Parse("""
      node_sets {
        key: "queries"
        value {
          features {
            key: "num_words"
            value {
              description: "Number of stop words, regular words, frequent words"
              shape {
                dim {
                  size: 3
                  name: "Type of word"  # Legitimate usage.
                }
              }
            }
          }
        }
      }
    """, schema_pb2.GraphSchema())
    sv._validate_schema_descriptions(schema)

    schema.node_sets['queries'].features['num_words'].ClearField('description')
    with self.assertRaises(sv.ValidationError):
      sv._validate_schema_descriptions(schema)

  @mock.patch.object(logging, 'error')
  def test_validate_schema_reserved_feature_names(self, mock_error):
    # Invalidate feature names on node sets.
    for name in '#size', '#id':
      schema = schema_pb2.GraphSchema()
      _ = schema.node_sets['queries'].features['#size']
      with self.assertRaises(sv.ValidationError):
        sv._validate_schema_reserved_feature_names(schema)

    # Invalidate feature names on edge sets.
    for name in '#size', '#source', '#target':
      schema = schema_pb2.GraphSchema()
      _ = schema.edge_sets['documents'].features[name]
      with self.assertRaises(sv.ValidationError,
                             msg='Feature: {}'.format(name)):
        sv._validate_schema_reserved_feature_names(schema)

    # Check that an error is issued for other features.
    #
    # TODO(blais,aferludin): We cannot raise an exception yet because the graph
    # sampler uses a number of hardcoded features with '#' prefix. Remove those
    # features from the sampler.
    name = '#weight'
    schema = schema_pb2.GraphSchema()
    _ = schema.edge_sets['documents'].features[name]
    sv._validate_schema_reserved_feature_names(schema)
    mock_error.assert_called()

  def test_validate_schema_context_references(self):
    schema = text_format.Parse("""
      context {
        features {
          key: "embedding"
          value: { dtype: DT_FLOAT  shape: { dim { size: 10 } } }
        }
      }
      node_sets {
        key: "queries"
      }
      node_sets {
        key: "documents"
      }
      edge_sets {
        key: "clicks"
        value {
          source: "queries"
          target: "documents"
        }
      }
    """, schema_pb2.GraphSchema())
    schema.node_sets['queries'].context.append('embedding')
    sv._validate_schema_context_references(schema)
    schema.node_sets['queries'].context.append('devnull')
    with self.assertRaises(sv.ValidationError):
      sv._validate_schema_context_references(schema)
    schema.node_sets['queries'].context[:] = []

    schema.edge_sets['clicks'].context.append('embedding')
    sv._validate_schema_context_references(schema)
    schema.edge_sets['clicks'].context.append('devnull')
    with self.assertRaises(sv.ValidationError):
      sv._validate_schema_context_references(schema)

  def test_validate_schema_node_set_references(self):
    schema = text_format.Parse("""
      node_sets {
        key: "queries"
      }
      node_sets {
        key: "documents"
      }
      edge_sets {
        key: "clicks"
        value {
          source: "queries"
          target: "documents"
        }
      }
    """, schema_pb2.GraphSchema())
    sv._validate_schema_node_set_references(schema)

    bad_schema = copy.copy(schema)
    bad_schema.edge_sets['clicks'].source = 'devnull'
    with self.assertRaises(sv.ValidationError):
      sv._validate_schema_node_set_references(bad_schema)

    bad_schema = copy.copy(schema)
    bad_schema.edge_sets['clicks'].target = 'devnull'
    with self.assertRaises(sv.ValidationError):
      sv._validate_schema_node_set_references(bad_schema)

  def test_validate_schema_edge_set_empty_source_or_target(self):
    empty_source_schema = text_format.Parse(
        """
      edge_sets {
        key: "clicks"
        value {
            target: "documents"
        }
      }
    """, schema_pb2.GraphSchema())

    with self.assertRaises(sv.ValidationError):
      sv._validate_schema_node_set_references(empty_source_schema)

    empty_target_schema = text_format.Parse(
        """
      edge_sets {
        key: "clicks"
        value {
            source: "queries"
        }
      }
    """, schema_pb2.GraphSchema())
    with self.assertRaises(sv.ValidationError):
      sv._validate_schema_node_set_references(empty_target_schema)

  def test_validate_schema_readout(self):
    schema = text_format.Parse("""
      node_sets {
        key: "queries"
      }
      node_sets {
        key: "documents"
      }
      node_sets {
        key: "_readout"
      }
      edge_sets {
        key: "_readout/foo"
        value {
          source: "queries"
          target: "_readout"
        }
      }
    """, schema_pb2.GraphSchema())
    sv._validate_schema_readout(schema, readout_node_sets=['_readout'])
    sv._validate_schema_readout(schema)

    schema_without = copy.copy(schema)
    del schema_without.edge_sets['_readout/foo']
    del schema_without.node_sets['_readout']
    sv._validate_schema_readout(schema_without)

    with self.assertRaisesRegex(
        sv.ValidationError,
        r'lacks auxiliary node set.*_readout:train'):
      sv._validate_schema_readout(
          schema, readout_node_sets=['_readout', '_readout:train'])

    bad_schema = copy.copy(schema)
    bad_schema.edge_sets['_readout/foo'].target = 'documents'
    with self.assertRaisesRegex(
        sv.ValidationError,
        r'validate_graph_tensor_spec_for_readout'):
      sv._validate_schema_readout(bad_schema)


class SchemaTests(tf.test.TestCase):

  def test_check_required_features__missing_feature(self):
    required = text_format.Parse("""
      node_sets {
        key: "queries"
        value {
          features {
            key: "musthave"
          }
        }
      }
    """, schema_pb2.GraphSchema())
    given = text_format.Parse("""
      node_sets {
        key: "queries"
        value {
          features {
            key: "musthave"
          }
        }
      }
    """, schema_pb2.GraphSchema())
    sv.check_required_features(required, given)

    given.node_sets['queries'].features['extras'].CopyFrom(
        given.node_sets['queries'].features['musthave'])
    sv.check_required_features(required, given)

    del given.node_sets['queries'].features['musthave']
    with self.assertRaises(sv.ValidationError):
      sv.check_required_features(required, given)

  def test_check_required_features__invalid_dtype(self):
    required = text_format.Parse("""
      node_sets {
        key: "queries"
        value {
          features {
            key: "musthave"
            value { dtype: DT_STRING }
          }
        }
      }
    """, schema_pb2.GraphSchema())
    given = copy.copy(required)
    sv.check_required_features(required, given)

    given.node_sets['queries'].features['musthave'].dtype = (
        tf.dtypes.float32.as_datatype_enum)
    with self.assertRaises(sv.ValidationError):
      sv.check_required_features(required, given)

    required.node_sets['queries'].features['musthave'].ClearField('dtype')
    sv.check_required_features(required, given)

  def test_check_required_features__invalid_shape(self):
    required = text_format.Parse("""
      node_sets {
        key: "queries"
        value {
          features {
            key: "musthave"
            value { shape { dim { size: 10 } dim { size: 20 } } }
          }
        }
      }
    """, schema_pb2.GraphSchema())
    given = copy.copy(required)

    # Check matching.
    sv.check_required_features(required, given)

    # Check invalid size (both present).
    req_musthave = required.node_sets['queries'].features['musthave']
    req_musthave.shape.dim[0].size += 1
    with self.assertRaises(sv.ValidationError):
      sv.check_required_features(required, given)

    # Check ignoring dim.
    req_musthave.shape.dim[0].ClearField('size')
    sv.check_required_features(required, given)

    # Check ignoring dim, failing rank.
    del req_musthave.shape.dim[1]
    with self.assertRaises(sv.ValidationError):
      sv.check_required_features(required, given)

    # Check enabled for scalar feature.
    req_musthave.shape.ClearField('dim')
    with self.assertRaises(sv.ValidationError):
      sv.check_required_features(required, given)


# NOTE(blais): These tests are a holdover of the previous iteration where we did
# everything using dicts. Eventually they could find their way into the
# GraphTensor constructor itself.
class GraphConstraintsTest(tf.test.TestCase):

  def test_assert_constraints_feature_shape_prefix_nodes(self):
    # Check valid.
    testgraph = gt.GraphTensor.from_pieces(
        node_sets={
            'n': gt.NodeSet.from_fields(
                features={'f': as_tensor([3, 4, 5])},
                sizes=as_tensor([3]))})
    sv._assert_constraints_feature_shape_prefix(testgraph)

    # Corrupt and check invalid. (We could make mutation more robust, but it
    # comes in handy in this test.)
    size_name = testgraph.node_sets['n']._DATAKEY_SIZES
    testgraph.node_sets['n']._data[size_name] = as_tensor([[3]])
    with self.assertRaises(tf.errors.InvalidArgumentError):
      sv._assert_constraints_feature_shape_prefix(testgraph)

    size_name = testgraph.node_sets['n']._DATAKEY_SIZES
    testgraph.node_sets['n']._data[size_name] = as_tensor([4])
    with self.assertRaises(tf.errors.InvalidArgumentError):
      sv._assert_constraints_feature_shape_prefix(testgraph)

    # Check invalid prefix shape.
    testgraph = gt.GraphTensor.from_pieces(
        node_sets={
            'n': gt.NodeSet.from_fields(
                features={'f': as_tensor([3, 4, 5, 6])},
                sizes=as_tensor([3]))})
    with self.assertRaises(tf.errors.InvalidArgumentError):
      sv._assert_constraints_feature_shape_prefix(testgraph)

  def test_assert_constraints_feature_shape_prefix_edges(self):
    # Check valid.
    testgraph = gt.GraphTensor.from_pieces(
        node_sets={
            'n': gt.NodeSet.from_fields(
                features={'f': as_tensor(['a', 'b', 'c', 'd'])},
                sizes=as_tensor([4]))},
        edge_sets={
            'e': gt.EdgeSet.from_fields(
                features={'w': as_tensor([3, 4, 5])},
                sizes=as_tensor([3]),
                adjacency=adj.Adjacency.from_indices(
                    ('n', as_tensor([0, 1, 0])),
                    ('n', as_tensor([1, 0, 1])))
            )})
    sv._assert_constraints_feature_shape_prefix(testgraph)

    size_name = testgraph.edge_sets['e']._DATAKEY_SIZES
    testgraph.edge_sets['e']._data[size_name] = as_tensor([[3]])
    with self.assertRaises(tf.errors.InvalidArgumentError):
      sv._assert_constraints_feature_shape_prefix(testgraph)

    size_name = testgraph.edge_sets['e']._DATAKEY_SIZES
    testgraph.edge_sets['e']._data[size_name] = as_tensor([4])
    with self.assertRaises(tf.errors.InvalidArgumentError):
      sv._assert_constraints_feature_shape_prefix(testgraph)

  def _create_test_graph_with_indices(self,
                                      source: List[int],
                                      target: List[int]):
    assert len(source) == len(target)
    return gt.GraphTensor.from_pieces(
        node_sets={
            'n': gt.NodeSet.from_fields(
                features={'f': as_tensor(['a', 'b', 'c', 'd'])},
                sizes=as_tensor([4]))},
        edge_sets={
            'e': gt.EdgeSet.from_fields(
                features={'w': as_tensor([3, 4, 5])},
                sizes=as_tensor([2]),
                adjacency=adj.Adjacency.from_indices(
                    ('n', as_tensor(source)),
                    ('n', as_tensor(target)))
                )})

  def test_assert_constraints_edge_indices_range_valid(self):
    testgraph = self._create_test_graph_with_indices([0, 3], [0, 3])
    sv._assert_constraints_edge_indices_range(testgraph)

    # Underflow.
    testgraph = self._create_test_graph_with_indices([0, -1], [0, 3])
    with self.assertRaises(tf.errors.InvalidArgumentError):
      sv._assert_constraints_edge_indices_range(testgraph)

    # Overflow.
    testgraph = self._create_test_graph_with_indices([0, 4], [0, 3])
    with self.assertRaises(tf.errors.InvalidArgumentError):
      sv._assert_constraints_edge_indices_range(testgraph)

  def test_assert_constraints_edge_shapes(self):
    testgraph = self._create_test_graph_with_indices([0, 3], [0, 3])
    sv._assert_constraints_edge_shapes(testgraph)

    testgraph = self._create_test_graph_with_indices([0, 1, 2], [0, 1, 2])
    with self.assertRaises(tf.errors.InvalidArgumentError):
      sv._assert_constraints_edge_shapes(testgraph)

    testgraph = self._create_test_graph_with_indices([[0, 3]], [[0, 3]])
    with self.assertRaises(tf.errors.InvalidArgumentError):
      sv._assert_constraints_edge_shapes(testgraph)


if __name__ == '__main__':
  tf.test.main()
