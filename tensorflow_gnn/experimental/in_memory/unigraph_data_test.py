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
"""Tests for unigraph_data."""

import tensorflow as tf
from tensorflow_gnn.data import unigraph
from tensorflow_gnn.experimental.in_memory import unigraph_data
from tensorflow_gnn.utils import test_utils

from google.protobuf import text_format

Example = tf.train.Example


class UnigraphDataHomogeneousTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.resource_dir = test_utils.get_resource_dir('testdata/homogeneous')
    self.graph_schema_file = unigraph.find_schema_filename(self.resource_dir)
    self.graph_schema = unigraph.read_schema(self.graph_schema_file)

  def test_raw_io(self):
    in_mem_unigraph = unigraph_data.UnigraphData(
        self.graph_schema, keep_intermediate_examples=True)

    self.assertSameElements(['fruits'], in_mem_unigraph.node_features.keys())
    self.assertSameElements([
        b'amanatsu', b'daidai', b'hassaku', b'kiyomi', b'komikan', b'lumia',
        b'mandora', b'reikou', b'tangelo'
    ], in_mem_unigraph.node_features['fruits'])

    self.assertSameElements(['tastelike'],
                            in_mem_unigraph.flat_edge_list.keys())

    adjacency_list = in_mem_unigraph.get_adjacency_list()['tastelike']
    self.assertProtoEquals(
        adjacency_list[b'amanatsu'][b'daidai'],
        text_format.Parse(
            """features {
                feature {
                  key: "#source"
                  value {
                    bytes_list {
                        value: "amanatsu"
                    }
                  }
                }
                feature {
                    key: "#target"
                    value {
                        bytes_list {
                            value: "daidai"
                        }
                    }
                }
                feature {
                    key: "weight"
                    value {
                        float_list {
                            value: 0.1
                        }
                    }
                }
              }""", Example()))
    self.assertProtoEquals(
        adjacency_list[b'amanatsu'][b'lumia'],
        text_format.Parse(
            """features {
                feature {
                    key: "#source"
                    value {
                        bytes_list {
                            value: "amanatsu"
                        }
                    }
                }
                feature {
                    key: "#target"
                    value {
                        bytes_list {
                            value: "lumia"
                        }
                    }
                }
                feature {
                    key: "weight"
                    value {
                        float_list {
                            value: 0.2
                        }
                    }
                }
              }""", Example()))

    self.assertProtoEquals(
        adjacency_list[b'kiyomi'][b'komikan'],
        text_format.Parse(
            """features {
                  feature{
                      key: "#source"
                      value {
                          bytes_list{
                              value: "kiyomi"
                          }
                      }
                  }
                  feature {
                      key: "#target"
                      value {
                          bytes_list {
                              value: "komikan"
                          }
                      }
                  }
                  feature {
                      key: "weight"
                      value {
                          float_list {
                              value: 0.3
                          }
                      }
                  }
              }
              """, tf.train.Example()))

    self.assertProtoEquals(
        adjacency_list[b'mandora'][b'komikan'],
        text_format.Parse(
            """features {
                feature {
                    key: "#source"
                    value {
                        bytes_list {
                          value: "mandora"
                        }
                    }
                }
                feature {
                    key: "#target"
                    value {
                        bytes_list {
                            value: "komikan"
                        }
                    }
                }
                feature {
                    key: "weight"
                    value {
                        float_list {
                            value: 0.4
                        }
                    }
                }
                }""", tf.train.Example()))

    self.assertProtoEquals(
        adjacency_list[b'mandora'][b'tangelo'],
        text_format.Parse(
            """features {
                  feature {
                    key: "#source"
                    value {
                        bytes_list {
                            value: "mandora"
                        }
                    }
                  }
                  feature {
                      key: "#target"
                      value {
                          bytes_list {
                              value: "tangelo"
                          }
                      }
                  }
                  feature {
                      key: "weight"
                      value {
                          float_list {
                              value: 0.5
                          }
                      }
                  }
              }
            """, tf.train.Example()))


class DatasetsUnigraphHeterogeneousTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.resource_dir = test_utils.get_resource_dir('testdata/heterogeneous')
    self.graph_schema_file = unigraph.find_schema_filename(self.resource_dir)
    self.graph_schema = unigraph.read_schema(self.graph_schema_file)

  def test_raw_io(self):
    in_mem_unigraph = unigraph_data.UnigraphData(
        self.graph_schema, keep_intermediate_examples=True)
    self.assertSameElements([
        'transaction',
        'customer',
        'creditcard',
    ], in_mem_unigraph.node_features.keys())

    # Only check a few node features from each node set.
    self.assertProtoEquals(
        in_mem_unigraph.node_features['transaction'][b'5488583952'],
        text_format.Parse(
            """features {
                feature {
                    key: "#id"
                    value {
                        bytes_list {
                            value: "5488583952"
                        }
                    }
                }
                feature {
                  key: "merchant"
                  value {
                    bytes_list {
                      value: "Ice Records"
                    }
                  }
                }
                feature {
                    key: "amount"
                    value {
                      float_list {
                          value: 67.77
                      }
                    }
                }
              }""", Example()))

    self.assertProtoEquals(
        in_mem_unigraph.node_features['customer'][b'1974494'],
        text_format.Parse(
            """
      features {
        feature {
          key: "#id"
          value {
            bytes_list {
              value: "1974494"
            }
          }
        }
        feature {
          key: "address"
          value {
            bytes_list {
              value: "909 Vermont St. Livonia, MI 48150"
            }
          }
        }
        feature {
          key: "name"
          value {
            bytes_list {
              value: "Adriana Mcburney"
            }
          }
        }
        feature {
          key: "score"
          value {
            float_list {
              value: 0.34326043725013733
            }
          }
        }
        feature {
          key: "zipcode"
          value {
            int64_list {
              value: 48150
            }
          }
        }
      }
      """, Example()))

    self.assertProtoEquals(
        in_mem_unigraph.node_features['creditcard'][b'14844931107602160'],
        text_format.Parse(
            """
            features {
              feature {
                key: "#id"
                value {
                  bytes_list {
                    value: "14844931107602160"
                  }
                }
              }
              feature {
                key: "issuer"
                value {
                  bytes_list {
                    value: "BellsGarbo"
                  }
                }
              }
              feature {
                key: "number"
                value {
                  int64_list {
                    value: 14844931107602160
                  }
                }
              }
            }""", Example()))

    self.assertSameElements(['owns_card', 'paid_with'],
                            in_mem_unigraph.flat_edge_list.keys())

    # Only check a few examples from each edge set.
    adjacency_list = in_mem_unigraph.get_adjacency_list()
    self.assertProtoEquals(
        adjacency_list['owns_card'][b'1876448'][b'16827485386298040'],
        text_format.Parse(
            """features {
                  feature {
                      key: "#source"
                      value {
                          bytes_list {
                              value: "1876448"
                          }
                      }
                  }
                  feature {
                      key: "#target"
                      value {
                          bytes_list {
                              value: "16827485386298040"
                          }
                      }
                  }
              }""", Example()))

    print(f"paid_with: {adjacency_list['paid_with']}")
    self.assertProtoEquals(
        adjacency_list['paid_with'][b'14844931107602160'][b'4077264491'],
        text_format.Parse(
            """features {
                feature {
                    key: "#source"
                    value {
                        bytes_list {
                            value: "14844931107602160"
                        }
                    }
                }
                feature {
                    key: "#target"
                    value {
                        bytes_list {
                            value: "4077264491"
                        }
                    }
                }
                feature {
                    key: "retries"
                    value {
                        int64_list {
                            value: 0
                        }
                    }
                }
            }
            """, Example()))


if __name__ == '__main__':
  tf.test.main()
