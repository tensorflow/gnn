# Copyright 2023 The TensorFlow GNN Authors. All Rights Reserved.
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
"""Unit tests for the sampler driver program.
"""
from __future__ import annotations

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn
from tensorflow_gnn.experimental.sampler.beam import sampler
import tensorflow_gnn.sampler as sampler_lib

from google.protobuf import text_format


class TestGetSamplingModel(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      ("one_hop", """
        seed_op <
          op_name: "seed"
          node_set_name: "paper"
        >
        sampling_ops <
          op_name: "seed->paper"
          input_op_names: "seed"
          edge_set_name: "cites"
          sample_size: 32
          strategy: RANDOM_UNIFORM
        >
      """, {"edges/cites_0": "edges/cites"}),
      ("two_hops", """
        seed_op <
          op_name: "seed"
          node_set_name: "paper"
        >
        sampling_ops <
          op_name: "seed->paper"
          input_op_names: "seed"
          edge_set_name: "cites"
          sample_size: 32
          strategy: RANDOM_UNIFORM
        >
        sampling_ops <
          op_name: "paper->paper"
          input_op_names: "seed->paper"
          edge_set_name: "cites"
          sample_size: 16
          strategy: RANDOM_UNIFORM
        >
       """, {"edges/cites_0": "edges/cites", "edges/cites_1": "edges/cites"})
  ])
  def test_correct_layer_names_dict(self,
                                    sampling_spec_pbtxt: str,
                                    expected_layer_name_dict: dict[str, str]):
    graph_schema = text_format.Parse("""
      node_sets {
        key: "author"
        value {
          features {
            key: "#id"
            value {
              dtype: DT_STRING
            }
          }
          metadata {}
        }
      }
      node_sets {
        key: "field_of_study"
        value {
          features {
            key: "#id"
            value {
              dtype: DT_STRING
            }
          }
          metadata {}
        }
      }
      node_sets {
        key: "institution"
        value {
          features {
            key: "#id"
            value {
              dtype: DT_STRING
            }
          }
          metadata {}
        }
      }
      node_sets {
        key: "paper"
        value {
          features {
            key: "#id"
            value {
              dtype: DT_STRING
            }
          }
          metadata {}
        }
      }
      edge_sets {
        key: "affiliated_with"
        value {
          source: "author"
          target: "institution"
          metadata {}
        }
      }
      edge_sets {
        key: "cites"
        value {
          source: "paper"
          target: "paper"
          metadata {
            filename: "edges-cites.tfrecords@120"
            cardinality: 5416271
          }
        }
      }
      edge_sets {
        key: "has_topic"
        value {
          source: "paper"
          target: "field_of_study"
          metadata {}
        }
      }
      edge_sets {
        key: "writes"
        value {
          source: "author"
          target: "paper"
          metadata {}
        }
      }
    """, tensorflow_gnn.GraphSchema())

    sampling_spec = text_format.Parse(
        sampling_spec_pbtxt, sampler_lib.SamplingSpec())
    _, layer_name_dict = sampler.get_sampling_model(graph_schema,
                                                    sampling_spec)
    self.assertDictEqual(
        expected_layer_name_dict,
        layer_name_dict)

if __name__ == "__main__":
  tf.test.main()
