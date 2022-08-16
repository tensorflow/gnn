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
"""Tests for gt.GraphTensor extension type (go/tf-gnn-api)."""

import functools
from typing import Any, Callable, Iterable, List, Mapping, Optional

from absl.testing import parameterized
import google.protobuf.text_format as pbtext
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as gc
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import graph_tensor_io as io
from tensorflow_gnn.graph import schema_utils as su
import tensorflow_gnn.proto.graph_schema_pb2 as schema_pb2

ResultValue = Mapping[str, Any]
ResultFn = Callable[[gt.GraphTensor], ResultValue]

as_tensor = tf.convert_to_tensor
as_ragged = tf.ragged.constant


class TfExampleParsingTestBase(tf.test.TestCase, parameterized.TestCase):

  def pbtxt_to_dataset(self, examples_pbtxt: List[str]) -> tf.data.Dataset:
    serialized = []
    for example_pbtxt in examples_pbtxt:
      serialized.append(
          pbtext.Merge(example_pbtxt, tf.train.Example()).SerializeToString())

    return tf.data.Dataset.from_tensor_slices(serialized)

  def assertFieldsEqual(self, expected: gt.Fields, actual: gt.Fields):
    self.assertAllEqual(expected.keys(), expected.keys())
    for k in expected.keys():
      self.assertAllEqual(expected[k], actual[k], msg=f'field={k}')

  def assertFieldsSeqEqual(self, expected: Iterable[gt.Fields],
                           actual: Iterable[gt.Fields]):
    expected = list(expected)
    actual = list(actual)
    self.assertEqual(len(expected), len(actual))
    for e, a in zip(expected, actual):
      self.assertFieldsEqual(e, a)


class TfExampleParsingFromSpecTest(TfExampleParsingTestBase):
  """Tests for TF Example to Graph Tensor parsing from the GraphTensorSpec."""

  @parameterized.parameters([
      dict(
          description='context dense features parsing',
          spec=gt.GraphTensorSpec.from_piece_specs(
              context_spec=gt.ContextSpec.from_field_specs(features_spec={
                  'v': tf.TensorSpec(shape=(2,), dtype=tf.int16),
                  'm': tf.TensorSpec(shape=(2, 3), dtype=tf.int32),
                  't': tf.TensorSpec(shape=(1, 1, 2), dtype=tf.int64),
              })),
          examples=[
              r"""
              features {
                feature {key: "context/v" value {int64_list {value: [1, 2]} } }
                feature {key: "context/m" value {int64_list {value: [1, 2, 3, 4, 5, 6]} } }
                feature {key: "context/t" value {int64_list {value: [1, 2] } } }
              }""", r"""
              features {
                feature {key: "context/v" value {int64_list {value: [9, 8]} } }
                feature {key: "context/m" value {int64_list {value: [9, 8, 7, 6, 5, 4]} } }
                feature {key: "context/t" value {int64_list {value: [9, 8]} } }
              }"""
          ],
          expected_values=[{
              'context/v': as_tensor([1, 2]),
              'context/m': as_tensor([[1, 2, 3], [4, 5, 6]]),
              'context/t': as_tensor([[[1, 2]]])
          }, {
              'context/v': as_tensor([9, 8]),
              'context/m': as_tensor([[9, 8, 7], [6, 5, 4]]),
              'context/t': as_tensor([[[9, 8]]])
          }]),
      dict(
          description='context ragged features parsing',
          spec=gt.GraphTensorSpec.from_piece_specs(
              context_spec=gt.ContextSpec.from_field_specs(features_spec={
                  'xyz':
                      tf.RaggedTensorSpec(
                          shape=(None, None, 3),
                          ragged_rank=1,
                          row_splits_dtype=tf.int32,
                          dtype=tf.float32)
              })),
          examples=[
              r"""
              features {
                feature {
                  key: "context/xyz"
                  value {
                    float_list {value: [-1., 0., 1., 1., 0., -1., 2., 2., 2.]}
                  }
                }
                feature {
                  key: "context/xyz.d1"
                  value {
                    int64_list {value: [0, 2, 0, 1]}
                  }
                }
              }"""
          ],
          expected_values=[{
              'context/xyz':
                  as_ragged([[], [[-1., 0., 1.], [1., 0., -1.]], [],
                             [[2., 2., 2.]]],
                            ragged_rank=1)
          }]),
      dict(
          description='node/edge features parsing',
          spec=gt.GraphTensorSpec.from_piece_specs(
              node_sets_spec={
                  'node':
                      gt.NodeSetSpec.from_field_specs(
                          features_spec={
                              'text':
                                  tf.RaggedTensorSpec(
                                      shape=(None, None, None),
                                      ragged_rank=2,
                                      row_splits_dtype=tf.int32,
                                      dtype=tf.string),
                          },
                          sizes_spec=tf.TensorSpec(
                              shape=(None,), dtype=tf.int32)),
              },
              edge_sets_spec={
                  'edge':
                      gt.EdgeSetSpec.from_field_specs(
                          features_spec={
                              'weight':
                                  tf.TensorSpec(
                                      shape=(None,), dtype=tf.float32),
                          },
                          sizes_spec=tf.TensorSpec(
                              shape=(None,), dtype=tf.int32),
                          adjacency_spec=(
                              adj.AdjacencySpec.from_incident_node_sets(
                                  source_node_set='node',
                                  target_node_set='node',
                                  index_spec=tf.TensorSpec(
                                      shape=(None,), dtype=tf.int32)))),
              }),
          examples=[
              r"""
          features {
            feature {key: "nodes/node.#size" value {int64_list {value: [1, 2]} } }
            feature {key: "nodes/node.text" value {bytes_list {value: ['a', 'b', 'c', 'd', 'e']} } }
            feature {key: "nodes/node.text.d1" value {int64_list {value: [2, 0, 1]} } }
            feature {key: "nodes/node.text.d2" value {int64_list {value: [2, 1, 2]} } }

            feature {key: "edges/edge.#size" value {int64_list {value: [2, 3]} } }
            feature {key: "edges/edge.#source" value {int64_list {value: [0, 1, 2, 2, 2]} } }
            feature {key: "edges/edge.#target" value {int64_list {value: [2, 1, 0, 0, 0]} } }
            feature {key: "edges/edge.weight" value {float_list {value: [1., 2., 3., 4., 5.]} } }
          }"""
          ],
          expected_values=[{
              'node/#size': [1, 2],
              'node/text':
                  as_ragged([[[b'a', b'b'], [b'c']], [], [[b'd', b'e']]],
                            ragged_rank=2,
                            dtype=tf.string,
                            row_splits_dtype=tf.int32),
              'edge/#size':
                  as_tensor([2, 3]),
              'edge/#source':
                  as_tensor([0, 1, 2, 2, 2]),
              'edge/#target':
                  as_tensor([2, 1, 0, 0, 0]),
              'edge/weight':
                  as_tensor([1., 2., 3., 4., 5.])
          }])
  ])
  def testSingleExampleParsing(
      self,
      description: str,
      spec: gt.GraphTensorSpec,
      examples: List[str],
      expected_values: List[gc.Fields],
  ):
    ds = self.pbtxt_to_dataset(examples)
    ds = ds.map(functools.partial(io.parse_single_example, spec))
    self.assertAllEqual(ds.element_spec, spec)
    ds = ds.map(_flatten_homogeneous_graph)
    self.assertFieldsSeqEqual(expected_values, ds)

  case1 = dict(
      description='context dense features parsing',
      drop_remainder=True,
      spec=gt.GraphTensorSpec.from_piece_specs(
          context_spec=gt.ContextSpec.from_field_specs(features_spec={
              'v': tf.TensorSpec(shape=(2,), dtype=tf.int16),
              'm': tf.TensorSpec(shape=(2, 3), dtype=tf.int32),
              't': tf.TensorSpec(shape=(1, 1, 2), dtype=tf.int64),
          })),
      examples=[
          r"""
          features {
            feature {key: "context/v" value {int64_list {value: [1, 2]} } }
            feature {key: "context/m" value {int64_list {value: [1, 2, 3, 4, 5, 6]} } }
            feature {key: "context/t" value {int64_list {value: [1, 2] } } }
          }""", r"""
          features {
            feature {key: "context/v" value {int64_list {value: [9, 8]} } }
            feature {key: "context/m" value {int64_list {value: [9, 8, 7, 6, 5, 4]} } }
            feature {key: "context/t" value {int64_list {value: [9, 8]} } }
          }"""
      ],
      expected={
          'context/v': as_tensor([[1, 2], [9, 8]]),
          'context/m': as_tensor([[[1, 2, 3], [4, 5, 6]],
                                  [[9, 8, 7], [6, 5, 4]]]),
          'context/t': as_tensor([[[[1, 2]]], [[[9, 8]]]])
      },
      prefix=None,
      validate=True)

  case2 = dict(
      description='ragged features parsing',
      drop_remainder=False,
      spec=gt.GraphTensorSpec.from_piece_specs(
          node_sets_spec={
              'node':
                  gt.NodeSetSpec.from_field_specs(
                      features_spec={
                          'words':
                              tf.RaggedTensorSpec(
                                  shape=(None, None),
                                  ragged_rank=1,
                                  row_splits_dtype=tf.int32,
                                  dtype=tf.string),
                      },
                      sizes_spec=tf.TensorSpec(shape=(1,), dtype=tf.int32)),
          },
          edge_sets_spec={
              'edge':
                  gt.EdgeSetSpec.from_field_specs(
                      features_spec={
                          'weight':
                              tf.TensorSpec(
                                  shape=(None,), dtype=tf.float32),
                      },
                      sizes_spec=tf.TensorSpec(shape=(1,), dtype=tf.int32),
                      adjacency_spec=(
                          adj.AdjacencySpec.from_incident_node_sets(
                              source_node_set='node',
                              target_node_set='node',
                              index_spec=tf.TensorSpec(
                                  shape=(None,), dtype=tf.int32)))),
          }),
      examples=[
          r"""
            features {
              feature {key: "nodes/node.#size" value {int64_list {value: [3]} } }
              feature {key: "nodes/node.words" value {bytes_list {value: ['a', 'b', 'c']} } }
              feature {key: "nodes/node.words.d1" value {int64_list {value: [2, 0, 1]} } }

              feature {key: "edges/edge.#size" value {int64_list {value: [5]} } }
              feature {key: "edges/edge.#source" value {int64_list {value: [0, 1, 2, 2, 2]} } }
              feature {key: "edges/edge.#target" value {int64_list {value: [2, 1, 0, 0, 0]} } }
              feature {key: "edges/edge.weight" value {float_list {value: [1., 2., 3., 4., 5.]} } }
            }""", r"""
            features {
              feature {key: "nodes/node.#size" value {int64_list {value: [1]} } }
              feature {key: "nodes/node.words" value {bytes_list {value: ['e', 'f']} } }
              feature {key: "nodes/node.words.d1" value {int64_list {value: [2]} } }
            }"""
      ],
      expected={
          'node/#size': as_tensor([[3], [1]]),
          'node/words': as_ragged([[[b'a', b'b'], [], [b'c']], [[b'e', b'f']]],
                                  ragged_rank=2,
                                  dtype=tf.string,
                                  row_splits_dtype=tf.int32),
          'edge/#size': as_tensor([[5], [0]]),
          'edge/#source': as_ragged([[0, 1, 2, 2, 2], []]),
          'edge/#target': as_ragged([[2, 1, 0, 0, 0], []]),
          'edge/weight': as_ragged([[1., 2., 3., 4., 5.], []])
      },
      prefix=None,
      validate=True)

  # pylint: disable=g-long-lambda
  case3 = dict(
      description='variable number of graph components',
      drop_remainder=False,
      spec=gt.GraphTensorSpec.from_piece_specs(
          context_spec=gt.ContextSpec.from_field_specs(
              features_spec={
                  'label': tf.TensorSpec(shape=(None,), dtype=tf.string)}),
          node_sets_spec={
              'node':
                  gt.NodeSetSpec.from_field_specs(
                      features_spec={
                          'words': tf.RaggedTensorSpec(
                              shape=(None, None),
                              ragged_rank=1,
                              row_splits_dtype=tf.int32,
                              dtype=tf.string),
                      },
                      sizes_spec=tf.TensorSpec(
                          shape=(None,), dtype=tf.int32)),
          },
          edge_sets_spec={
              'edge':
                  gt.EdgeSetSpec.from_field_specs(
                      features_spec={'weight': tf.TensorSpec(
                          shape=(None,), dtype=tf.float32)},
                      sizes_spec=tf.TensorSpec(
                          shape=(None,), dtype=tf.int32),
                      adjacency_spec=adj.AdjacencySpec.from_incident_node_sets(
                          source_node_set='node',
                          target_node_set='node',
                          index_spec=tf.TensorSpec(
                              shape=(None,), dtype=tf.int32))),
          }),
      examples=[
          r"""
            features {
              feature {key: "context/label" value {bytes_list {value: ['G', 'B']} } }

              feature {key: "nodes/node.#size" value {int64_list {value: [1, 2]} } }
              feature {key: "nodes/node.words" value {bytes_list {value: ['a', 'b', 'c']} } }
              feature {key: "nodes/node.words.d1" value {int64_list {value: [2, 0, 1]} } }

              feature {key: "edges/edge.#size" value {int64_list {value: [2, 3]} } }
              feature {key: "edges/edge.#source" value {int64_list {value: [0, 1, 2, 2, 2]} } }
              feature {key: "edges/edge.#target" value {int64_list {value: [2, 1, 0, 0, 0]} } }
              feature {key: "edges/edge.weight" value {float_list {value: [1., 2., 3., 4., 5.]} } }
            }""", r"""
            features {
              feature {key: "context/label" value {bytes_list {value: ['B']} } }

              feature {key: "nodes/node.#size" value {int64_list {value: [1]} } }
              feature {key: "nodes/node.words" value {bytes_list {value: ['e', 'f']} } }
              feature {key: "nodes/node.words.d1" value {int64_list {value: [2]} } }
              feature {key: "edges/edge.#size" value {int64_list {value: [0]} } }
            }"""
      ],
      expected={
          'context/label': as_ragged([[b'G', b'B'], [b'B']]),
          'node/#size': as_ragged([[1, 2], [1]]),
          'node/words': as_ragged([[[b'a', b'b'], [], [b'c']], [[b'e', b'f']]],
                                  ragged_rank=2,
                                  dtype=tf.string,
                                  row_splits_dtype=tf.int32),
          'edge/#size': as_ragged([[2, 3], [0]]),
          'edge/#source': as_ragged([[0, 1, 2, 2, 2], []]),
          'edge/#target': as_ragged([[2, 1, 0, 0, 0], []]),
          'edge/weight': as_ragged([[1., 2., 3., 4., 5.], []])
      },
      prefix=None,
      validate=True)

  case4 = case3.copy()
  case4['prefix'] = 'gnn_'
  case4['validate'] = False

  @parameterized.parameters([case1, case2, case3])
  def testExamplesParsing(
      self,
      description: str,
      spec: gt.GraphTensorSpec,
      drop_remainder: bool,
      examples: List[str],
      expected: gc.Fields,
      prefix: Optional[str],
      validate: bool,
  ):
    batch_size = len(examples)
    ds = self.pbtxt_to_dataset(examples)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.map(functools.partial(io.parse_example, spec,
                                  prefix=prefix, validate=validate))
    self.assertAllEqual(ds.element_spec,
                        spec._batch(batch_size if drop_remainder else None))
    ds = ds.map(_flatten_homogeneous_graph)
    self.assertFieldsSeqEqual([expected], ds)


class TfExampleParsingFromSchemaTest(TfExampleParsingTestBase):
  """Tests for TF Example to Graph Tensor parsing from the GraphSchema."""

  def _test_impl(self, schema_pb: schema_pb2.GraphSchema, examples: List[str],
                 expected_value: ResultValue, result_map_fn: ResultFn, *,
                 batch_then_parse: bool, drop_remainder: bool):
    assert isinstance(schema_pb, schema_pb2.GraphSchema)
    graph_spec = su.create_graph_spec_from_schema_pb(schema_pb)
    batch_size = len(examples)

    ds = self.pbtxt_to_dataset(examples)
    if batch_then_parse:
      ds = ds.batch(batch_size, drop_remainder=drop_remainder)
      ds = ds.map(functools.partial(io.parse_example, graph_spec))
    else:
      ds = ds.map(functools.partial(io.parse_single_example, graph_spec))
      ds = ds.batch(batch_size, drop_remainder=drop_remainder)

    ds = ds.map(result_map_fn)
    result_value = next(iter(ds))
    self.assertFieldsEqual(result_value, expected_value)

  def _test_all_cases(self, schema_pb: schema_pb2.GraphSchema,
                      examples: List[str], expected_value: ResultValue,
                      result_map_fn):
    test_case = functools.partial(self._test_impl, schema_pb, examples,
                                  expected_value, result_map_fn)
    test_case(batch_then_parse=True, drop_remainder=True)
    test_case(batch_then_parse=True, drop_remainder=False)
    test_case(batch_then_parse=False, drop_remainder=True)
    test_case(batch_then_parse=False, drop_remainder=False)

  @parameterized.parameters([
      dict(
          description='context dense features parsing',
          schema_pbtxt=r"""
          context {
            features {
              key: "s"
              value: {
                dtype: DT_INT16
              }
            }
            features {
              key: "v"
              value: {
                dtype: DT_INT32
                shape: { dim { size: 2 } }
              }
            }
            features {
              key: "m"
              value: {
                dtype: DT_INT32
                shape: { dim { size: 2 }, dim { size: 3 } }
              }
            }
            features {
              key: "t"
              value: {
                dtype: DT_INT64
                shape: { dim { size: 1 }, dim { size: 1 }, dim { size: 2 } }
              }
            }
            features {
              key: "r"
              value: {
                dtype: DT_UINT32
                shape: { dim { size: -1 } dim { size: -1 } }
              }
            }
          }""",
          examples=[
              r"""
          features {
            feature {key: "context/s" value {int64_list {value: [1]} } }
            feature {key: "context/v" value {int64_list {value: [1, 2]} } }
            feature {key: "context/m" value {int64_list {value: [1, 2, 3, 4, 5, 6]} } }
            feature {key: "context/t" value {int64_list {value: [1, 2] } } }
            feature {key: "context/r" value {int64_list {value: [1, 2, 3] } } }
            feature {key: "context/r.d1" value {int64_list {value: [2] } } }
            feature {key: "context/r.d2" value {int64_list {value: [1, 2] } } }
          }""", r"""
          features {
            feature {key: "context/s" value {int64_list {value: [9]} } }
            feature {key: "context/v" value {int64_list {value: [9, 8]} } }
            feature {key: "context/m" value {int64_list {value: [9, 8, 7, 6, 5, 4]} } }
            feature {key: "context/t" value {int64_list {value: [9, 8]} } }
            feature {key: "context/r.d1" value {int64_list {value: [0] } } }
          }"""
          ],
          expected_value={
              's':
                  as_tensor([[1], [9]]),
              'v':
                  as_tensor([[[1, 2]], [[9, 8]]]),
              'm':
                  as_tensor([[[[1, 2, 3], [4, 5, 6]]], [[[9, 8, 7], [6, 5,
                                                                     4]]]]),
              't':
                  as_tensor([[[[[1, 2]]]], [[[[9, 8]]]]]),
              'r':
                  as_ragged([[[[1], [2, 3]]], [[]]],
                            ragged_rank=3,
                            row_splits_dtype=tf.int32)
          })
  ])
  def testContextParsing(self, description: str, schema_pbtxt: str,
                         examples: List[str], expected_value: ResultValue):
    schema_pb = pbtext.Merge(schema_pbtxt, schema_pb2.GraphSchema())

    @tf.function
    def result_map_fn(g: gt.GraphTensor):
      return g.context.get_features_dict()

    self._test_all_cases(schema_pb, examples, expected_value, result_map_fn)

  @parameterized.parameters([
      dict(
          description='context dense features parsing',
          schema_pbtxt=r"""
          node_sets {
            key: "node"
            value {
              features {
                key: "id"
                value { dtype: DT_STRING }
              }
            }
            value {
              features {
                key: "fv"
                value {
                  dtype: DT_FLOAT
                  shape: { dim { size: 3 } }
                }
              }
            }
            value {
              features {
                key: "sr"
                value {
                  dtype: DT_STRING
                  shape: { dim { size: -1 } }
                }
              }
            }
          }

          """,
          examples=[
              r"""
          features {
            feature {key: "nodes/node.#size" value {int64_list {value: [1] } } }
            feature {key: "nodes/node.id" value {bytes_list {value: ['node.1.1'] } } }
            feature {key: "nodes/node.fv" value {float_list {value: [1., 2., 3.] } } }
            feature {key: "nodes/node.sr.d1" value {int64_list {value: [0]} } }
          }""", r"""
          features {
          }""", r"""
          features {
            feature {key: "nodes/node.#size" value {int64_list {value: [2] } } }
            feature {key: "nodes/node.id" value {bytes_list {value: ['node.3.1', 'node.3.2'] } } }
            feature {key: "nodes/node.fv" value {float_list {value: [4., 5., 6., 7., 8., 9.] } } }
            feature {key: "nodes/node.sr" value {bytes_list {value: ['a', 'b', 'c']} } }
            feature {key: "nodes/node.sr.d1" value {int64_list {value: [1, 2]} } }

          }"""
          ],
          expected_value={
              'id':
                  as_ragged([['node.1.1'], [], ['node.3.1', 'node.3.2']]),
              'fv':
                  as_ragged([[[1., 2., 3.]], [], [[4., 5., 6.], [7., 8., 9.]]],
                            ragged_rank=1,
                            row_splits_dtype=tf.int32),
              'sr':
                  as_ragged([[[]], [], [['a'], ['b', 'c']]],
                            ragged_rank=2,
                            row_splits_dtype=tf.int32),
              '#size':
                  as_tensor([[1], [0], [2]]),
          })
  ])
  def testNodeSetParsing(self, description: str, schema_pbtxt: str,
                         examples: List[str], expected_value: ResultValue):
    schema_pb = pbtext.Merge(schema_pbtxt, schema_pb2.GraphSchema())

    @tf.function
    def result_map_fn(g: gt.GraphTensor):
      node = g.node_sets['node']
      features = node.get_features_dict()
      features.update({'#size': node.sizes})
      return features

    self._test_all_cases(schema_pb, examples, expected_value, result_map_fn)

  @parameterized.parameters([
      dict(
          description='context dense features parsing',
          schema_pbtxt=r"""
          edge_sets {
            key: "edge"
            value {
              features {
                key: "id"
                value { dtype: DT_STRING }
              }
              source: 'node.a'
              target: 'node.b'
            }
          }
          """,
          examples=[
              r"""
          features {
          }""", r"""
          features {
            feature {key: "edges/edge.#size" value {int64_list {value: [1] } } }
            feature {key: "edges/edge.#source" value {int64_list {value: [0] } } }
            feature {key: "edges/edge.#target" value {int64_list {value: [0] } } }
            feature {key: "edges/edge.id" value {bytes_list {value: ['e.2.1'] } } }
          }""", r"""
          features {
            feature {key: "edges/edge.#size" value {int64_list {value: [2] } } }
            feature {key: "edges/edge.#source" value {int64_list {value: [0, 1] } } }
            feature {key: "edges/edge.#target" value {int64_list {value: [1, 0] } } }
            feature {key: "edges/edge.id" value {bytes_list {value: ['e.3.1', 'e.3.2'] } } }
          }"""
          ],
          expected_value={
              'id':
                  as_ragged([[], ['e.2.1'], ['e.3.1', 'e.3.2']]),
              f'#adj:{gc.SOURCE}:node.a':
                  as_ragged([[], [0], [0, 1]], row_splits_dtype=tf.int32),
              f'#adj:{gc.TARGET}:node.b':
                  as_ragged([[], [0], [1, 0]], row_splits_dtype=tf.int32),
              '#size':
                  as_tensor([[0], [1], [2]]),
          })
  ])
  def testEdgeSetParsing(self, description: str, schema_pbtxt: str,
                         examples: List[str], expected_value: ResultValue):
    schema_pb = pbtext.Merge(schema_pbtxt, schema_pb2.GraphSchema())

    @tf.function
    def result_map_fn(g: gt.GraphTensor):
      edge = g.edge_sets['edge']
      features = edge.get_features_dict()
      features.update({'#size': edge.sizes})
      features.update({
          f'#adj:{tag}:{name}': index
          for tag, (name, index) in edge.adjacency.get_indices_dict().items()
      })
      return features

    self._test_all_cases(schema_pb, examples, expected_value, result_map_fn)


def _flatten_homogeneous_graph(graph: gt.GraphTensor) -> gc.Fields:
  result = {}
  for name, value in graph.context.features.items():
    result[f'context/{name}'] = value

  if graph.node_sets:
    node_set = graph.node_sets['node']
    for name, value in node_set.features.items():
      result[f'node/{name}'] = value
    result['node/#size'] = node_set.sizes

  if graph.edge_sets:
    edge_set = graph.edge_sets['edge']
    for name, value in edge_set.features.items():
      result[f'edge/{name}'] = value
    result['edge/#size'] = edge_set.sizes
    result['edge/#source'] = edge_set.adjacency[gc.SOURCE]
    result['edge/#target'] = edge_set.adjacency[gc.TARGET]

  return result


if __name__ == '__main__':
  tf.test.main()
