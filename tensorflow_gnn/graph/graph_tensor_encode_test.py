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
"""Tests for encoder routines to tf.train.Exammple."""

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.graph import graph_constants as gc
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import graph_tensor_encode as ge
from tensorflow_gnn.graph import graph_tensor_io as io
from tensorflow_gnn.graph import graph_tensor_random as gr
from tensorflow_gnn.graph import schema_utils as su
import tensorflow_gnn.proto.graph_schema_pb2 as schema_pb2
from tensorflow_gnn.utils import test_utils

# Enables tests for graph pieces that are members of test classes.
gc.enable_graph_tensor_validation_at_runtime()


# TODO(blais): Move this to graph_tensor_test_utils once ported.
def _find_first_available_tensor(gtensor: gt.GraphTensor) -> gc.Field:
  for feature in gtensor.context.features.values():
    return feature
  for node_set in gtensor.node_sets.values():
    for feature in node_set.features.values():
      return feature
  for edge_set in gtensor.edge_sets.values():
    for feature in edge_set.features.values():
      return feature


class TestWriteExample(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    gc.enable_graph_tensor_validation_at_runtime()

  # TODO(blais,aferludin): Replace this with graph_tensor_test_utils
  def _compare_graph_tensors(self, rfeatures: gc.Field, pfeatures: gc.Field):
    self.assertEqual(rfeatures.shape.as_list(), pfeatures.shape.as_list())
    if rfeatures.dtype in (tf.float64,):
      # float64 => float32 conversions.
      cmp = self.assertAllClose
    else:
      cmp = self.assertAllEqual

    if isinstance(rfeatures, tf.RaggedTensor):
      cmp(rfeatures.flat_values, pfeatures.flat_values)
      rlist = rfeatures.nested_row_lengths()
      plist = pfeatures.nested_row_lengths()
      self.assertEqual(len(rlist), len(plist))
      for rlengths, plengths in zip(rlist, plist):
        self.assertAllEqual(rlengths, plengths)
    else:
      cmp(rfeatures, pfeatures)

  @parameterized.parameters((None, True),
                            (None, False),
                            ('someprefix_', True))
  def test_write_random_graph_tensors(self, prefix, validate):
    # Produce a stream of random graph tensors with a complex schema and verify
    # that they parse back.
    schema = test_utils.get_proto_resource(
        'testdata/feature_repr.pbtxt', schema_pb2.GraphSchema())
    spec = su.create_graph_spec_from_schema_pb(schema)

    # TODO(blais): Turn this into a utility.
    def random_graph_tensor_generator(spec) -> tf.data.Dataset:
      def generator():
        while True:
          yield gr.random_graph_tensor(spec)
      return tf.data.Dataset.from_generator(generator, output_signature=spec)

    for rgraph in random_graph_tensor_generator(spec).take(16):
      example = ge.write_example(rgraph, prefix=prefix)
      serialized = tf.constant(example.SerializeToString())
      pgraph = io.parse_single_example(spec, serialized,
                                       prefix=prefix, validate=validate)

      # TODO(blais): When graph_tensor_test_utils is ported, compare the entire
      # contents.
      rfeatures = _find_first_available_tensor(rgraph)
      pfeatures = _find_first_available_tensor(pgraph)
      self._compare_graph_tensors(rfeatures, pfeatures)

  def _roundtrip_test(self, shape, create_tensor):
    # Produce random tensors of various shapes, serialize them, and then run
    # them back through our parser and finally check that values are
    # identical.
    for dtype in (
        tf.bool,
        tf.int8,
        tf.uint8,
        tf.int16,
        tf.uint16,
        tf.int32,
        tf.uint32,
        tf.int32,
        tf.uint32,
        tf.int64,
        tf.uint64,
        tf.bfloat16,
        tf.float16,
        tf.float32,
        tf.float64,
        tf.string,
    ):
      shape = tf.TensorShape(shape)
      if shape[1:].is_fully_defined():
        tensor_spec = tf.TensorSpec(shape, dtype)
      else:
        ragged_rank = shape.rank - 1
        for dim in reversed(shape.as_list()):
          if dim is None:
            break
          ragged_rank -= 1

        tensor_spec = tf.RaggedTensorSpec(shape, dtype, ragged_rank=ragged_rank)
      spec = create_tensor(tensor_spec)
      spec = spec.relax(num_nodes=True, num_edges=True)
      rgraph = gr.random_graph_tensor(spec)
      example = ge.write_example(rgraph)
      serialized = tf.constant(example.SerializeToString())
      pgraph = io.parse_single_example(spec, serialized, validate=True)

      # Find the available tensor.
      rfeatures = _find_first_available_tensor(rgraph)
      pfeatures = _find_first_available_tensor(pgraph)
      with self.subTest(dtype=dtype):
        self._compare_graph_tensors(rfeatures, pfeatures)

  @parameterized.parameters(
      (shape,)
      for shape in [
          [1],
          [1, 2],
          [1, None],
          [1, 4, 3],
          [1, 4, None],
          [1, None, 4],
          [1, None, None],
      ]
  )
  def test_write_various_shapes_as_context(self, shape):
    def create_tensor(tensor_spec):
      return gt.GraphTensorSpec.from_piece_specs(
          context_spec=gt.ContextSpec.from_field_specs(
              features_spec={'wings': tensor_spec}
          )
      )

    self._roundtrip_test(shape, create_tensor)

  @parameterized.parameters(
      (shape,)
      for shape in [
          [4],
          [4, 3],
          [None, 4],
          [None, 4, 3],
          [None, None, 4],
          [None, None, 4, 3],
          [4, None],
          [4, 3, None],
          [4, None, None],
          [4, 3, None, None],
          [5, None, 4, None, 3],
          [None, 4, None, 3, None],
      ]
  )
  def test_write_various_shapes_as_node_set(self, shape):
    def create_tensor(tensor_spec):
      return gt.GraphTensorSpec.from_piece_specs(
          node_sets_spec={'butterfly': gt.NodeSetSpec.from_field_specs(
              sizes_spec=tf.TensorSpec([1], tf.int64),
              features_spec={'wings': tensor_spec})})
    self._roundtrip_test(shape, create_tensor)

  def testUInt64MaxRoundtrip(self):
    feat = tf.constant(tf.uint64.max, tf.uint64, shape=[1])
    rgraph = gt.GraphTensor.from_pieces(
        context=gt.Context.from_fields(features={'f': feat})
    )
    example = ge.write_example(rgraph)
    serialized = tf.constant(example.SerializeToString())
    pgraph = io.parse_single_example(rgraph.spec, serialized, validate=True)

    self.assertAllEqual(pgraph.context.features['f'], feat)


if __name__ == '__main__':
  tf.test.main()
