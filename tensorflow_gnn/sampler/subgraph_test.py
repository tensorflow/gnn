import copy
import random

import tensorflow as tf
import tensorflow_gnn as gnn
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
          key: "stype" value {dtype: DT_STRING}
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
    """, gnn.GraphSchema())

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


if __name__ == "__main__":
  tf.test.main()
