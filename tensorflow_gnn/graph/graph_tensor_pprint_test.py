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
"""Tests for pretty-printing."""

import pprint

import tensorflow as tf
from tensorflow_gnn.graph import graph_tensor_pprint as gpp
from tensorflow_gnn.graph import graph_tensor_random as gr
from tensorflow_gnn.graph import schema_utils as su
import tensorflow_gnn.proto.graph_schema_pb2 as schema_pb2
from tensorflow_gnn.utils import test_utils


class TestConvertForPprint(tf.test.TestCase):

  def test_graph_tensor_to_values(self):
    schema = test_utils.get_proto_resource(
        'testdata/feature_repr.pbtxt', schema_pb2.GraphSchema())
    spec = su.create_graph_spec_from_schema_pb(schema)
    graph = gr.random_graph_tensor(spec, row_splits_dtype=tf.int64)
    values = gpp.graph_tensor_to_values(graph)
    text = pprint.pformat(values)
    # This just ensures there is no error in the genreation.
    self.assertIsInstance(text, str)


if __name__ == '__main__':
  tf.test.main()
