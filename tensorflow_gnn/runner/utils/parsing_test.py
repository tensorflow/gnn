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
"""Tests for parsing."""
import functools

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.runner.utils import parsing as parsing_utils

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
  }
"""

Fields = tfgnn.Fields
GraphTensor = tfgnn.GraphTensor

ds_from_tensor = tf.data.Dataset.from_tensors


def gtspec() -> tfgnn.GraphTensorSpec:
  return tfgnn.create_graph_spec_from_schema_pb(tfgnn.parse_schema(SCHEMA))


@functools.lru_cache(None)
def random_graph_tensor() -> tfgnn.GraphTensor:
  return tfgnn.random_graph_tensor(gtspec())


@functools.lru_cache(None)
def random_serialized_graph_tensor() -> str:
  return tfgnn.write_example(random_graph_tensor()).SerializeToString()


class ParsingTest(tf.test.TestCase, parameterized.TestCase):

  def _assert_fields_equal(self, a: Fields, b: Fields):
    self.assertCountEqual(a.keys(), b.keys())
    for k, v in a.items():
      self.assertAllEqual(v, b[k])

  def _assert_graph_tensors_equal(self, a: GraphTensor, b: GraphTensor):
    self.assertCountEqual(a.node_sets.keys(), b.node_sets.keys())
    self.assertCountEqual(a.edge_sets.keys(), b.edge_sets.keys())

    self._assert_fields_equal(a.context.features, b.context.features)

    for k, v in a.node_sets.items():
      self._assert_fields_equal(v.features, b.node_sets[k].features)

    for k, v in a.edge_sets.items():
      self._assert_fields_equal(v.features, b.edge_sets[k].features)

  @parameterized.named_parameters([
      dict(
          testcase_name="SerializedGraphTensorElement",
          ds=ds_from_tensor(random_serialized_graph_tensor()),
          spec=random_graph_tensor().spec,
          expected=random_graph_tensor(),
      ),
      dict(
          testcase_name="GraphTensorElement",
          ds=ds_from_tensor(random_graph_tensor()),
          spec=random_graph_tensor().spec,
          expected=random_graph_tensor(),
      ),
      dict(
          testcase_name="SerializedGraphTensorElements",
          ds=ds_from_tensor(random_serialized_graph_tensor()).repeat().batch(4),
          spec=random_graph_tensor().spec,
          expected=next(
              iter(
                  ds_from_tensor(random_graph_tensor()).repeat().batch(4)
                  )
              ),
      ),
      dict(
          testcase_name="GraphTensorElements",
          ds=ds_from_tensor(random_graph_tensor()).repeat().batch(4),
          spec=random_graph_tensor().spec,
          expected=next(
              iter(
                  ds_from_tensor(random_graph_tensor()).repeat().batch(4)
                  )
              ),
      ),
  ])
  def test_maybe_parse_graph_tensor_dataset(
      self,
      ds: tf.data.Dataset,
      spec: tfgnn.GraphTensorSpec,
      expected: tfgnn.GraphTensor):
    ds = parsing_utils.maybe_parse_graph_tensor_dataset(ds, spec)
    self._assert_graph_tensors_equal(
        expected,
        next(iter(ds)))

  @parameterized.named_parameters([
      dict(
          testcase_name="FloatElement",
          ds=ds_from_tensor(tf.constant(8191.)),
          spec=random_graph_tensor().spec,
          expected_failure=r"Expected `GraphTensorSpec` \(got .*\)",
      ),
      dict(
          testcase_name="IncompatibleGraphTensorElement",
          ds=ds_from_tensor(
              tfgnn.homogeneous(
                  source=tf.constant((0, 3)),
                  target=tf.constant((1, 2)),
                  node_set_sizes=tf.constant((4,))
                  ),
              ),
          spec=random_graph_tensor().spec,
          expected_failure=r"Graph is not compatible with the graph schema.*",
      ),
  ])
  def test_maybe_parse_graph_tensor_dataset_fails(
      self,
      ds: tf.data.Dataset,
      spec: tfgnn.GraphTensorSpec,
      expected_failure: str):
    with self.assertRaisesRegex(ValueError, expected_failure):
      _ = parsing_utils.maybe_parse_graph_tensor_dataset(ds, spec)

if __name__ == "__main__":
  tf.test.main()
