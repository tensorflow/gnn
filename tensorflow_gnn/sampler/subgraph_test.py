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
import copy
import random

from typing import Mapping, List

from absl.testing import parameterized

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.sampler import subgraph
from tensorflow_gnn.sampler import subgraph_pb2

from google.protobuf import text_format


class TestSubgraph(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.subgraph = text_format.Parse(
        """
      sample_id: "SAMPLE_ID_1"
      seed_node_id: "center"
      nodes {
        id: "center"
        features {feature {key: "heft" value {int64_list {value: 50}}}}
        outgoing_edges {
          neighbor_id: "center"
          features {feature {key: "weight" value {float_list {value: 1.0}}}}
          edge_set_name: "relation"
        }
        outgoing_edges {
          neighbor_id: "left"
          features {feature {key: "weight" value {float_list {value: 0.8}}}}
          edge_set_name: "relation"
        }
        outgoing_edges {
          neighbor_id: "right"
          features {feature {key: "weight" value {float_list {value: 0.4}}}}
          edge_set_name: "relation"
        }
        node_set_name: "position"
      }
      nodes {
        id: "left"
        features {feature {key: "heft" value {int64_list {value: 100}}}}
        outgoing_edges {
          neighbor_id: "center"
          features {feature {key: "weight" value {float_list {value: 0.8}}}}
          edge_set_name: "relation"
        }
        node_set_name: "position"
      }
      nodes {
        id: "right"
        features {feature {key: "heft2" value {int64_list {value: 150}}}}
        outgoing_edges {
          neighbor_id: "right"
          features {feature {key: "weight2" value {float_list {value: 1.0}}}}
          edge_set_name: "relation2"
        }
        outgoing_edges {
          neighbor_id: "left"
          features {feature {key: "weight" value {float_list {value: 0.4}}}}
          edge_set_name: "relation"
        }
        node_set_name: "position2"
      }
      features {feature {key: "stype" value {bytes_list {value: "geometry"}}}}
    """, subgraph_pb2.Subgraph())

    self.schema = text_format.Parse(
        """
      context {
        features {
          key: "stype" value { dtype: DT_STRING }
        }
      }
      node_sets {
        key: "position"
        value {
          features {
            key: "heft" value {dtype: DT_INT64}
          }
        }
      }
      node_sets {
        key: "position2"
        value {
          features {
            key: "heft2" value {dtype: DT_INT64}
          }
        }
      }
      edge_sets {
        key: "relation"
        value {
          features {
            key: "weight" value {dtype: DT_FLOAT}
          }
        }
      }
      edge_sets {
        key: "relation2"
        value {
          features {
            key: "weight2" value {dtype: DT_FLOAT}
          }
        }
      }
    """, tfgnn.GraphSchema())

    self.expected = text_format.Parse(
        """
      features {
        feature {
          key: "context/stype"
          value {
            bytes_list {value: "geometry"}
          }
        }
        feature {
          key: "edges/relation.#size"
          value {
            int64_list {value: 5}
          }
        }
        feature {
          key: "edges/relation.#source"
          value {
            int64_list {value: 0 value: 0 value: 0 value: 1 value: 0}
          }
        }
        feature {
          key: "edges/relation.#target"
          value {
            int64_list {value: 0 value: 1 value: 0 value: 0 value: 1}
          }
        }
        feature {
          key: "edges/relation.weight"
          value {
            float_list {value: 1.0 value: 0.8 value: 0.4
                        value: 0.8 value: 0.4}
          }
        }
        feature {
          key: "edges/relation2.#size"
          value {
            int64_list {value: 1}
          }
        }
        feature {
          key: "edges/relation2.#source"
          value {
            int64_list {value: 0}
          }
        }
        feature {
          key: "edges/relation2.#target"
          value {
            int64_list {value: 0}
          }
        }
        feature {
          key: "edges/relation2.weight2"
          value {
            float_list {value: 1.0 }
          }
        }
        feature {
          key: "nodes/position.#size"
          value {
            int64_list {value: 2}
          }
        }
        feature {
          key: "nodes/position.heft"
          value {
            int64_list {value: 50 value: 100}
          }
        }
        feature {
          key: "nodes/position2.#size"
          value {
            int64_list {value: 1}
          }
        }
        feature {
          key: "nodes/position2.heft2"
          value {
            int64_list {value: 150}
          }
        }
      }
    """, tf.train.Example())

  def test_valid(self):
    example = subgraph.encode_subgraph_to_example(self.schema, self.subgraph)
    self.assertProtoEquals(self.expected, example)

  def test_extra_features(self):
    # Insert extra features, ensure they're ignored.
    subgraph_copy = copy.copy(self.subgraph)
    for node in subgraph_copy.nodes:
      node.features.feature["extra"].float_list.value.append(42)
    example = subgraph.encode_subgraph_to_example(self.schema, subgraph_copy)
    self.assertProtoEquals(self.expected, example)

  def test_missing_node_features(self):
    # Remove node feature, ensure error raised.
    subgraph_copy = copy.copy(self.subgraph)
    node = random.choice(subgraph_copy.nodes)
    del node.features.feature["heft"]
    with self.assertRaises(ValueError):
      subgraph.encode_subgraph_to_example(self.schema, subgraph_copy)

  def test_missing_edge_features(self):
    # Remove edge feature, ensure error raised.
    subgraph_copy = copy.copy(self.subgraph)
    node = random.choice(subgraph_copy.nodes)
    edge = random.choice(node.outgoing_edges)
    del edge.features.feature["weight"]
    with self.assertRaises(ValueError):
      subgraph.encode_subgraph_to_example(self.schema, subgraph_copy)

  def test_invalid_length(self):
    # Insert feature with an irregular number of values, ensure error raised.
    subgraph_copy = copy.copy(self.subgraph)
    node = random.choice(subgraph_copy.nodes)
    edge = random.choice(node.outgoing_edges)
    edge.features.feature["weight"].float_list.value.append(42.)
    with self.assertRaises(ValueError):
      subgraph.encode_subgraph_to_example(self.schema, subgraph_copy)


class TestSubgraphPiecesToExample(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="empty",
          schema="",
          seeds={},
          context="",
          node_sets={},
          edge_sets={},
          expected_result=""),
      dict(
          testcase_name="context_only",
          schema="""
            context {
              features { key: "stype" value { dtype: DT_STRING } }
              features {
                key: "itype"
                value {
                  dtype: DT_INT64
                  shape { dim { size: 2 } }
                }
              }
            }
          """,
          seeds={},
          context="""
            feature { key: "stype" value { bytes_list: { value: ["X"] } } }
            feature { key: "itype" value { int64_list: { value: [1, 2] } } }
          """,
          node_sets={},
          edge_sets={},
          expected_result="""
            features {
              feature {
                key: "context/stype" value { bytes_list: { value: ["X"] } }
              }
              feature {
                key: "context/itype" value { int64_list: { value: [1, 2] } }
              }
            }
          """),
      dict(
          testcase_name="nodes_only",
          schema="""
            node_sets {
              key: "A"
              value {
                features { key: "stype" value { dtype: DT_STRING } }
              }
            }
            node_sets {
              key: "B"
              value {
                features { key: "ftype" value {dtype: DT_FLOAT} }
              }
            }
          """,
          seeds={"A": [b"a2"]},
          context="",
          node_sets={
              "A": {
                  b"a1": """
                    feature {
                      key: "stype"
                      value { bytes_list: { value: ["1"] } }
                    }
                  """,
                  b"a2": """
                    feature {
                      key: "stype"
                      value { bytes_list: { value: ["2"] } }
                    }
                  """,
                  b"a3": """
                    feature {
                      key: "stype"
                      value { bytes_list: { value: ["3"] } }
                    }
                  """
              },
              "B": {
                  b"b1": """
                    feature {
                      key: "ftype"
                      value { float_list: { value: [1.0] } }
                    }
                  """,
                  b"b2": """
                    feature {
                      key: "ftype"
                      value { float_list: { value: [2.0] } }
                    }
                  """
              },
          },
          edge_sets={},
          expected_result="""
            features {
              feature {
                key: "nodes/A.stype"
                value { bytes_list: { value: ["2", "1", "3"] } }
              }
              feature {
                key: "nodes/A.#size"
                value { int64_list: { value: [3] } }
              }
              feature {
                key: "nodes/B.ftype"
                value { float_list: { value: [1.0, 2.0] } }
              }
              feature {
                key: "nodes/B.#size"
                value { int64_list: { value: [2] } }
              }
            }
          """),
      dict(
          testcase_name="edges_only",
          schema="""
            node_sets {
              key: "A"
              value {}
            }
            node_sets {
              key: "B"
              value {}
            }
            edge_sets {
              key: "A->B"
              value {
                source: "A"
                target: "B"
                features { key: "i" value {dtype: DT_INT64} }
              }
            }
          """,
          seeds={},
          context="",
          node_sets={},
          edge_sets={
              "A->B": [
                  """
                    id: "a2"
                    outgoing_edges {
                      neighbor_id: "b3"
                      features {
                        feature { key: "i" value { int64_list { value: 23 } } }
                      }
                    }
                    outgoing_edges {
                      neighbor_id: "b1"
                      features {
                        feature { key: "i" value { int64_list { value: 21 } } }
                      }
                    }
                  """,
                  """
                    id: "a1"
                    outgoing_edges {
                      neighbor_id: "b2"
                      features {
                        feature { key: "i" value { int64_list { value: 12 } } }
                      }
                    }
                  """,
              ]
          },
          expected_result="""
            features {
              feature {
                key: "nodes/A.#size" value { int64_list: { value: [2] } }
              }
              feature {
                key: "nodes/B.#size" value { int64_list: { value: [3] } }
              }
              feature {
                key: "edges/A->B.#size" value { int64_list: { value: [3] } }
              }
              feature {
                key: "edges/A->B.#source"
                value { int64_list: { value: [0, 1, 1] } }
              }
              feature {
                key: "edges/A->B.#target"
                value { int64_list: { value: [1, 0, 2] } }
              }
              feature {
                key: "edges/A->B.i"
                value {int64_list: { value: [12, 21, 23] } }
              }
            }
          """),
      dict(
          testcase_name="homogeneous",
          schema="""
            node_sets {
              key: "node"
              value {
                features { key: "id" value { dtype: DT_STRING } }
              }
            }
            edge_sets {
              key: "edge"
              value {
                source: "node"
                target: "node"
              }
            }
          """,
          seeds={},
          context="",
          node_sets={
              "node": {
                  b"3": """
                    feature {
                      key: "id"
                      value { bytes_list: { value: ["3"] } }
                    }
                  """,
                  b"2": """
                    feature {
                      key: "id"
                      value { bytes_list: { value: ["2"] } }
                    }
                  """,
                  b"1": """
                    feature {
                      key: "id"
                      value { bytes_list: { value: ["1"] } }
                    }
                  """
              },
          },
          edge_sets={
              "edge": [
                  """
                    id: "3"
                    outgoing_edges { neighbor_id: "1" }
                    outgoing_edges { neighbor_id: "2" }
                  """,
                  """
                    id: "1"
                    outgoing_edges { neighbor_id: "3" }
                    outgoing_edges { neighbor_id: "1" }
                  """,
                  """
                    id: "2"
                    outgoing_edges { neighbor_id: "1" }
                    outgoing_edges { neighbor_id: "3" }
                  """,
              ]
          },
          expected_result="""
            features {
              feature {
                key: "nodes/node.#size" value { int64_list: { value: [3] } }
              }
              feature {
                key: "nodes/node.id"
                value { bytes_list: { value: ["1", "2", "3"] } }
              }
              feature {
                key: "edges/edge.#size" value { int64_list: { value: [6] } }
              }
              feature {
                key: "edges/edge.#source"
                value { int64_list: { value: [0, 0, 1, 1, 2, 2] } }
              }
              feature {
                key: "edges/edge.#target"
                value { int64_list: { value: [0, 2, 0, 2, 0, 1] } }
              }
            }
          """))
  def test_logic(self, schema: str,
                 seeds: Mapping[tfgnn.NodeSetName, List[bytes]],
                 context: str,
                 node_sets: Mapping[tfgnn.NodeSet, Mapping[bytes, str]],
                 edge_sets: Mapping[tfgnn.NodeSet, List[str]],
                 expected_result: str):
    schema = text_format.Parse(schema, tfgnn.GraphSchema())
    context = text_format.Parse(context, tf.train.Features())
    node_sets = tf.nest.map_structure(
        lambda v: text_format.Parse(v, tf.train.Features()), node_sets)
    edge_sets = tf.nest.map_structure(
        lambda v: text_format.Parse(v, subgraph_pb2.Node()), edge_sets)

    actual_result = subgraph.encode_subgraph_pieces_to_example(
        schema, seeds, context, node_sets, edge_sets)

    expected_result = text_format.Parse(expected_result, tf.train.Example())
    self.assertProtoEquals(actual_result, expected_result)


if __name__ == "__main__":
  tf.test.main()
