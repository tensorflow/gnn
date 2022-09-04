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
"""Tests for sampling_spec_builder."""

from absl.testing import absltest
from absl.testing import parameterized

import tensorflow_gnn as tfgnn
from tensorflow_gnn.sampler import sampling_spec_builder
from tensorflow_gnn.sampler import sampling_spec_pb2

from google.protobuf import text_format


def get_schema(edge_sets=('AA', 'AB', 'AC', 'BC', 'CD')) -> tfgnn.GraphSchema:
  schema = tfgnn.GraphSchema()
  # Schema has DAG like:
  # A -> B
  #  \    \
  #   +--> C -> D
  for edge_set_name in edge_sets:
    schema.edge_sets[edge_set_name].source = edge_set_name[0]
    schema.edge_sets[edge_set_name].target = edge_set_name[1]
    unused_node_set = schema.node_sets[edge_set_name[0]]  # To initalize.
    unused_node_set = schema.node_sets[edge_set_name[1]]

  return schema


class SamplingSpecBuilderTest(parameterized.TestCase):

  def test_line_to_sampling_spec(self):
    schema = get_schema()
    builder = sampling_spec_builder.SamplingSpecBuilder(
        schema, sampling_spec_pb2.SamplingStrategy.RANDOM_UNIFORM)
    proto = (builder.seed('A').sample(5, 'AB').sample(5, 'BC').sample(5, 'CD')
             .to_sampling_spec())

    expected_proto = text_format.Parse(
        """
        seed_op {
          op_name: "SEED->A"
          node_set_name: "A"
        }
        sampling_ops {
          op_name: "A->B"
          input_op_names: "SEED->A"
          edge_set_name: "AB"
          strategy: RANDOM_UNIFORM
          sample_size: 5
        }
        sampling_ops {
          op_name: "B->C"
          input_op_names: "A->B"
          edge_set_name: "BC"
          strategy: RANDOM_UNIFORM
          sample_size: 5
        }
        sampling_ops {
          op_name: "C->D"
          input_op_names: "B->C"
          edge_set_name: "CD"
          strategy: RANDOM_UNIFORM
          sample_size: 5
        }
        """, sampling_spec_pb2.SamplingSpec())
    self.assertEqual(expected_proto, proto)

  def test_dag_to_sampling_spec(self):
    schema = get_schema()
    builder = sampling_spec_builder.SamplingSpecBuilder(schema).seed('A')
    path1 = builder.sample(5, 'AB').sample(4, 'BC', op_name='A-B-C')
    path2 = builder.sample(7, 'AC', op_name='A-C')
    proto = (path1.join([path2]).sample(10, 'CD').to_sampling_spec())

    expected_proto = text_format.Parse(
        """
        seed_op {
          op_name: "SEED->A"
          node_set_name: "A"
        }
        sampling_ops {
          op_name: "A-C"
          input_op_names: "SEED->A"
          edge_set_name: "AC"
          strategy: TOP_K
          sample_size: 7
        }
        sampling_ops {
          op_name: "A->B"
          input_op_names: "SEED->A"
          edge_set_name: "AB"
          strategy: TOP_K
          sample_size: 5
        }
        sampling_ops {
          op_name: "A-B-C"
          input_op_names: "A->B"
          edge_set_name: "BC"
          strategy: TOP_K
          sample_size: 4
        }
        sampling_ops {
          op_name: "(A-B-C|A-C)->D"
          input_op_names: "A-B-C"
          input_op_names: "A-C"
          edge_set_name: "CD"
          strategy: TOP_K
          sample_size: 10
        }
        """, sampling_spec_pb2.SamplingSpec())
    self.assertEqual(expected_proto, proto)

  def test_sample_with_list_of_sizes(self):
    schema = get_schema()
    proto = (sampling_spec_builder.SamplingSpecBuilder(schema).seed('A')
             .sample([5, 3, 2], 'AA').to_sampling_spec())
    expected_proto = text_format.Parse(
        """
        seed_op {
          op_name: "SEED->A"
          node_set_name: "A"
        }
        sampling_ops {
          op_name: "A->A"
          input_op_names: "SEED->A"
          edge_set_name: "AA"
          strategy: TOP_K
          sample_size: 5
        }
        sampling_ops {
          op_name: "A->A.2"
          input_op_names: "A->A"
          edge_set_name: "AA"
          strategy: TOP_K
          sample_size: 3
        }
        sampling_ops {
          op_name: "A->A.3"
          input_op_names: "A->A.2"
          edge_set_name: "AA"
          strategy: TOP_K
          sample_size: 2
        }
        """, sampling_spec_pb2.SamplingSpec())
    self.assertEqual(expected_proto, proto)

  def test_no_required_edgeset_or_nodeset_names_for_homogeneous_graph(self):
    schema = get_schema(edge_sets=['AA'])  # Homogeneous graph.
    proto = (sampling_spec_builder.SamplingSpecBuilder(schema)
             .seed().sample([10, 5]).sample([2, 1]).to_sampling_spec())
    #                             # ^ could be combined with previous sample.

    expected_proto = text_format.Parse(
        """
        seed_op {
          op_name: "SEED->A"
          node_set_name: "A"
        }
        sampling_ops {
          op_name: "A->A"
          input_op_names: "SEED->A"
          edge_set_name: "AA"
          sample_size: 10
          strategy: TOP_K
        }
        sampling_ops {
          op_name: "A->A.2"
          input_op_names: "A->A"
          edge_set_name: "AA"
          sample_size: 5
          strategy: TOP_K
        }
        sampling_ops {
          op_name: "A->A.3"
          input_op_names: "A->A.2"
          edge_set_name: "AA"
          sample_size: 2
          strategy: TOP_K
        }
        sampling_ops {
          op_name: "A->A.4"
          input_op_names: "A->A.3"
          edge_set_name: "AA"
          sample_size: 1
          strategy: TOP_K
        }
        """, sampling_spec_pb2.SamplingSpec())
    self.assertEqual(expected_proto, proto)


if __name__ == '__main__':
  absltest.main()
