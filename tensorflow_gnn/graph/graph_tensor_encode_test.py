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


TEST_SHAPES = [[4],
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
               [None, 4, None, 3, None]]


class TestWriteExample(tf.test.TestCase, parameterized.TestCase):

  # TODO(blais,aferludin): Replace this with graph_tensor_test_utils
  def _compare_graph_tensors(self, rfeatures: gc.Field, pfeatures: gc.Field):
    self.assertEqual(rfeatures.shape.as_list(), pfeatures.shape.as_list())
    if isinstance(rfeatures, tf.RaggedTensor):
      self.assertAllEqual(rfeatures.flat_values, pfeatures.flat_values)
      rlist = rfeatures.nested_row_lengths()
      plist = pfeatures.nested_row_lengths()
      self.assertEqual(len(rlist), len(plist))
      for rlengths, plengths in zip(rlist, plist):
        self.assertAllEqual(rlengths, plengths)
    else:
      self.assertAllEqual(rfeatures, pfeatures)

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
    # them back through our parser and finally check that the shapes are
    # identical.
    dtype = tf.float32
    tensor_spec = (tf.TensorSpec(shape, dtype)
                   if tf.TensorShape(shape).is_fully_defined()
                   else tf.RaggedTensorSpec(shape, dtype))
    spec = create_tensor(tensor_spec)
    rgraph = gr.random_graph_tensor(spec, row_splits_dtype=tf.int64)
    example = ge.write_example(rgraph)
    serialized = tf.constant(example.SerializeToString())
    pgraph = io.parse_single_example(spec, serialized, validate=True)

    # Find the available tensor.
    # TODO(blais): Replaced these with self.assertGraphTensorEq(rgraph, pgraph).
    rfeatures = _find_first_available_tensor(rgraph)
    pfeatures = _find_first_available_tensor(pgraph)
    self._compare_graph_tensors(rfeatures, pfeatures)

  @parameterized.parameters((shape,) for shape in TEST_SHAPES)
  def test_write_various_shapes_as_context(self, shape):
    def create_tensor(tensor_spec):
      return gt.GraphTensorSpec.from_piece_specs(
          context_spec=gt.ContextSpec.from_field_specs(
              features_spec={'wings': tensor_spec}))
    self._roundtrip_test(shape, create_tensor)

  @parameterized.parameters((shape,) for shape in TEST_SHAPES)
  def test_write_various_shapes_as_node_set(self, shape):
    def create_tensor(tensor_spec):
      return gt.GraphTensorSpec.from_piece_specs(
          node_sets_spec={'butterfly': gt.NodeSetSpec.from_field_specs(
              sizes_spec=tf.TensorSpec([1], tf.int64),
              features_spec={'wings': tensor_spec})})
    self._roundtrip_test(shape, create_tensor)


if __name__ == '__main__':
  tf.test.main()
