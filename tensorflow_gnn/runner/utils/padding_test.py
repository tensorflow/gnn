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
"""Tests for padding."""
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.runner import orchestration
from tensorflow_gnn.runner.utils import padding

SCHEMA = """
  node_sets {
    key: "node"
    value {
      features {
        key: "features"
        value {
          dtype: DT_FLOAT
          shape { dim { size: 4 } }
        }
      }
    }
  }
  edge_sets {
    key: "edge"
    value {
      source: "node"
      target: "node"
    }
  }"""


class PaddingTest(tf.test.TestCase):

  def _assert_fields_equal(self, a: tfgnn.Fields, b: tfgnn.Fields):
    self.assertCountEqual(a.keys(), b.keys())
    for k, v in a.items():
      self.assertAllEqual(v, b[k])

  @parameterized.named_parameters([
      dict(
          testcase_name="FitOrSkipPadding",
          klass=padding.FitOrSkipPadding,
      ),
      dict(
          testcase_name="TightPadding",
          klass=padding.TightPadding,
      ),
  ])
  def test_protocol_matches_type(self, klass: object):
    self.assertIsInstance(klass, orchestration.GraphTensorPadding)

  def test_parse_dataset(self):
    schema = tfgnn.parse_schema(SCHEMA)
    gtspec = tfgnn.create_graph_spec_from_schema_pb(schema)
    expected = tfgnn.random_graph_tensor(gtspec)
    example = tfgnn.write_example(expected)
    dataset = tf.data.Dataset.from_tensors([example.SerializeToString()])

    actual = next(iter(padding._parse_dataset(gtspec, dataset)))

    self.assertCountEqual(actual.node_set.keys(), expected.node_set.keys())
    self.assertCountEqual(actual.edge_set.keys(), expected.edge_set.keys())

    self._assert_fields_equal(
        actual.context.features,
        expected.context.features)

    for k, v in actual.node_set.items():
      self._assert_fields_equal(v, expected.node_set[k].features)

    for k, v in actual.edge_set.items():
      self._assert_fields_equal(v, expected.edge_set[k].features)


if __name__ == "__main__":
  tf.test.main()
