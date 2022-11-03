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
import os
from os import path
import tempfile

import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing import util
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.data import unigraph
from tensorflow_gnn.utils import test_utils

from google.protobuf import text_format


class TestUnigraph(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.schema_filename = test_utils.get_resource(
        "testdata/homogeneous/citrus.pbtxt")
    self.testdata = path.dirname(self.schema_filename)

  def test_guess_file_format(self):
    # pylint: disable=invalid-name
    def assertFormat(expected_format, filename):
      self.assertEqual(expected_format, unigraph.guess_file_format(filename))

    # Placeholder for Google-internal output formats
    assertFormat("tfrecord", "/path/to/file.tfr")
    assertFormat("tfrecord", "/path/to/file.tfr@10")
    assertFormat("tfrecord", "/path/to/file.tfr-?????-of-00010")
    assertFormat("tfrecord", "/path/to/file_tfrecord")
    assertFormat("tfrecord", "/path/to/file_tfrecord-?????-of-00010")
    assertFormat("csv", "/path/to/file.csv")
    assertFormat("csv", "/path/to/file.csv@10")
    assertFormat("csv", "/path/to/file.csv-?????-of-00010")

_EXPECTED_CSV_SIZES = {"creditcard": 36,
                       "customer": 24,
                       "owns_card": 24,
                       "paid_with": 48,
                       "transaction": 48}

_CUSTOMER_IDS = b"""
  1876448 1372437 1368305 1974494 1257724 1758057 1531660 1489311 1407706
  196838 1195675 1659366 1499004 1344333 1443888 1108778 175583 1251872
  1493851 1599418 1768701 1549489 1879799 125454
""".split()

_OWNS_CARDS_SRC_IDS = b"""
  1876448 1372437 1368305 1974494 1257724 1758057 1531660 1489311 1407706
  196838 1195675 1659366 1499004 1344333 1443888 1108778 175583 1251872
  1493851 1599418 1768701 1549489 1879799 125454
""".split()

_OWNS_CARDS_TGT_IDS = b"""
  16827485386298040 11470379189154620 11163838768727470 16011471358128450
  18569067217418250 17396883707513070 14844931107602160 1238474857489384
  11290312140467510 17861046738135650 8878522895102384 13019350102369400
  11470379189154620 16283233487191600 9991040399813057 14912408563871390
  11290312140467510 12948957000457930 3549061668422198 9991040399813057
  18362223127059380 1238474857489384 18569067217418250 18526138896540830
""".split()


class TestReadGraph(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.resource_dir = test_utils.get_resource_dir("testdata/heterogeneous")

  def test_read_graph_and_schema(self):
    self.assertTrue({
        "creditcard.csv", "customer.csv", "graph.pbtxt", "owns_card.csv",
        "paid_with.csv", "transactions.csv", "invalid_customer.csv",
        "two_customers.csv", "one_customer.csv"}.issubset(
            set(os.listdir(self.resource_dir))))
    pipeline = test_pipeline.TestPipeline()
    schema, colls = unigraph.read_graph_and_schema(
        path.join(self.resource_dir, "graph.pbtxt"), pipeline)

    expected_sizes = dict(_EXPECTED_CSV_SIZES)
    expected_sizes[""] = 1  # Context set.

    for stype, sname, _ in tfgnn.iter_sets(schema):
      util.assert_that(
          (colls[stype][sname]
           | f"Size.{sname}" >> beam.combiners.Count.Globally()),
          util.equal_to([expected_sizes[sname]]), f"AssertSize.{sname}")
    pipeline.run()

  def test_read_node_set(self):
    filename = path.join(self.resource_dir, "customer.csv")
    pipeline = test_pipeline.TestPipeline()
    pcoll = (pipeline
             | unigraph.ReadTable(filename, "csv")
             | beam.Map(unigraph.get_node_ids)
             | beam.Keys())
    util.assert_that(pcoll, util.equal_to(_CUSTOMER_IDS))
    pipeline.run()

  def test_read_edge_set(self):
    filename = path.join(self.resource_dir, "owns_card.csv")
    pipeline = test_pipeline.TestPipeline()
    pcoll = (pipeline
             | unigraph.ReadTable(filename, "csv")
             | beam.Map(unigraph.get_edge_ids))
    source_coll = (pcoll | beam.Map(lambda item: item[0]))
    target_coll = (pcoll | beam.Map(lambda item: item[1]))
    util.assert_that(source_coll, util.equal_to(_OWNS_CARDS_SRC_IDS),
                     label="AssertSource")
    util.assert_that(target_coll, util.equal_to(_OWNS_CARDS_TGT_IDS),
                     label="AssertTarget")
    pipeline.run()

  def test_sharded_patterns(self):
    test_data = [
        ("filename", dict(
            file_path_prefix="filename",
            num_shards=None,
            shard_name_template="")),
        ("filename@10", dict(
            file_path_prefix="filename",
            num_shards=10,
            shard_name_template="-SSSSS-of-NNNNN")),
        ("filename@1", dict(
            file_path_prefix="filename",
            num_shards=1,
            shard_name_template="-SSSSS-of-NNNNN")),
        ("filename-?????-of-00030", dict(
            file_path_prefix="filename",
            num_shards=30,
            shard_name_template="-SSSSS-of-NNNNN")),
        ("filename-00012-of-00030", dict(
            file_path_prefix="filename",
            num_shards=30,
            shard_name_template="-SSSSS-of-NNNNN")),
    ]
    for filename, kwargs in test_data:
      self.assertDictEqual(kwargs, unigraph.get_sharded_pattern_args(filename),
                           filename)

  def test_read_write(self):
    filename = path.join(self.resource_dir, "owns_card.csv")
    with tempfile.TemporaryDirectory() as tmpdir:
      with beam.Pipeline() as pipeline:
        outfile = path.join(tmpdir, "output.tfrecords")
        _ = (pipeline
             | unigraph.ReadTable(filename)
             | unigraph.WriteTable(outfile, "tfrecord"))
      self.assertTrue(tf.io.gfile.exists(outfile))

  def test_bigquery_table_spec_args_from_proto(self):
    bq = text_format.Parse(
        """
      table_spec {
        project: "test_project"
        dataset: "test_dataset"
        table: "test_table"
      }""", tfgnn.proto.graph_schema_pb2.BigQuery())
    self.assertEqual(
        unigraph.ReadUnigraphPieceFromBigQuery.bigquery_args_from_proto(bq), {
            "table": "test_project:test_dataset.test_table",
            "method": beam.io.ReadFromBigQuery.Method.EXPORT
        })

  def test_bigquery_row_to_keyed_example_node_set(self):
    node_set = text_format.Parse(
        """
        features {
          key: "id"
          value: {
              dtype: DT_STRING
          }
        }
        features {
          key: "int_feature"
          value: {
            dtype: DT_INT64
          }
        }
        features {
          key: "float_feature"
          value {
              dtype: DT_FLOAT
          }
        }
        features {
            key: "string_feature"
            value {
                dtype: DT_STRING
            }
        }
        features {
            key: "bool_feature"
            value {
                dtype: DT_BOOL
            }
        }
        metadata {
            bigquery {
                table_spec {
                  project: "test_project"
                  dataset: "test_dataset"
                  table: "test_table"
                }
            }
        } """, tfgnn.proto.graph_schema_pb2.NodeSet())

    # Quick test of the suffix generation
    self.assertEqual(
        unigraph.ReadUnigraphPieceFromBigQuery.stage_name_suffix(
            "fake_node_set", node_set),
        ("ReadFromBigQuery/NodeSet/fake_node_set/test_project:"
         "test_dataset.test_table"))

    # Mock a source that returns fake BQ rows.
    def fake_bq_reader(**unused_kwargs):
      del unused_kwargs
      return beam.Create([{
          "id": "id1",
          "string_feature": "a",
          "float_feature": 1.0,
          "int_feature": 2,
          "bool_feature": True
      }, {
          "id": "id2",
          "string_feature": "b",
          "int_feature": 3,
          "float_feature": 4.0,
          "bool_feature": False
      }])

    with test_pipeline.TestPipeline() as pipeline:
      rows = (
          pipeline | unigraph.ReadUnigraphPieceFromBigQuery(
              "fake_node_set", node_set, bigquery_reader=fake_bq_reader))

      result = pipeline.run()
      result.wait_until_finish()

      util.assert_that(
          rows,
          util.equal_to([(b"id1",
                          text_format.Parse(
                              """features {
                                  feature {
                                    key: "#id"
                                    value {
                                      bytes_list {
                                        value: "id1"
                                      }
                                    }
                                  }
                                  feature {
                                    key: "float_feature"
                                    value {
                                      float_list {
                                        value: 1.0
                                      }
                                    }
                                  }
                                  feature {
                                    key: "int_feature"
                                    value {
                                      int64_list {
                                        value: 2
                                      }
                                    }
                                  }
                                  feature {
                                    key: "string_feature"
                                    value {
                                      bytes_list {
                                        value: "a"
                                      }
                                    }
                                  }
                                  feature {
                                      key: "bool_feature"
                                      value {
                                        int64_list {
                                          value: 1
                                        }
                                      }
                                  }
                              }""", tf.train.Example())),
                         (b"id2",
                          text_format.Parse(
                              """features {
                                  feature {
                                    key: "#id"
                                    value {
                                      bytes_list {
                                        value: "id2"
                                      }
                                    }
                                  }
                                  feature {
                                    key: "float_feature"
                                    value {
                                      float_list {
                                        value: 4.0
                                      }
                                    }
                                  }
                                  feature {
                                    key: "int_feature"
                                    value {
                                      int64_list {
                                        value: 3
                                      }
                                    }
                                  }
                                  feature {
                                    key: "string_feature"
                                    value {
                                      bytes_list {
                                        value: "b"
                                      }
                                    }
                                  }
                                  feature {
                                      key: "bool_feature"
                                      value {
                                          int64_list {
                                              value: 0
                                          }
                                      }
                                  }
                                }""", tf.train.Example()))]))

  def test_bigquery_row_to_keyed_example_edge_set(self):
    edge_set = text_format.Parse(
        """
        features {
          key: "int_feature"
          value: {
            dtype: DT_INT64
          }
        }
        features: {
          key: "float_feature"
          value {
              dtype: DT_FLOAT
          }
        }
        features: {
            key: "string_feature"
            value {
                dtype: DT_STRING
            }
        }
        metadata {
            bigquery {
                table_spec {
                  project: "test_project"
                  dataset: "test_dataset"
                  table: "test_table"
                }
            }
        }""", tfgnn.proto.graph_schema_pb2.EdgeSet())

    self.assertEqual(
        unigraph.ReadUnigraphPieceFromBigQuery.stage_name_suffix(
            "fake_edge_set", edge_set),
        ("ReadFromBigQuery/EdgeSet/fake_edge_set/"
         "test_project:test_dataset.test_table"))

    # Mock a source that returns fake BQ rows.
    def fake_bq_reader(**unused_kwargs):
      del unused_kwargs
      return beam.Create([{
          "source": "s1",
          "target": "t1",
          "string_feature": "a",
          "float_feature": 1.0,
          "int_feature": 2
      }, {
          "source": "s2",
          "target": "t2",
          "string_feature": "b",
          "int_feature": 3,
          "float_feature": 4.0
      }])

    with test_pipeline.TestPipeline() as pipeline:
      rows = (
          pipeline | unigraph.ReadUnigraphPieceFromBigQuery(
              "fake_edge_set", edge_set, bigquery_reader=fake_bq_reader))

      result = pipeline.run()
      result.wait_until_finish()

      util.assert_that(
          rows,
          util.equal_to([(b"s1", b"t1",
                          text_format.Parse(
                              """
                              features {
                                feature {
                                  key: "#source"
                                  value {
                                    bytes_list {
                                      value: "s1"
                                    }
                                  }
                                }
                                feature {
                                  key: "#target"
                                  value {
                                    bytes_list {
                                        value: "t1"
                                    }
                                  }
                                }
                                feature {
                                  key: "float_feature"
                                  value {
                                    float_list {
                                      value: 1.0
                                    }
                                  }
                                }
                                feature {
                                  key: "int_feature"
                                  value {
                                    int64_list {
                                      value: 2
                                    }
                                  }
                                }
                                feature {
                                  key: "string_feature"
                                  value {
                                    bytes_list {
                                      value: "a"
                                    }
                                  }
                                }
                            }""", tf.train.Example())),
                         (b"s2", b"t2",
                          text_format.Parse(
                              """features {
                                  feature {
                                    key: "#source"
                                    value {
                                      bytes_list {
                                        value: "s2"
                                      }
                                    }
                                  }
                                  feature {
                                    key: "#target"
                                    value {
                                      bytes_list {
                                        value: "t2"
                                      }
                                    }
                                  }
                                  feature {
                                    key: "float_feature"
                                    value {
                                      float_list {
                                        value: 4.0
                                      }
                                    }
                                  }
                                  feature {
                                    key: "int_feature"
                                    value {
                                      int64_list {
                                        value: 3
                                      }
                                    }
                                  }
                                  feature {
                                    key: "string_feature"
                                    value {
                                      bytes_list {
                                        value: "b"
                                      }
                                    }
                                  }
                                }""", tf.train.Example()))]))

  def test_bigquery_row_to_keyed_example_edge_set_reversed(self):
    edge_set = text_format.Parse(
        """
        features {
          key: "int_feature"
          value: {
            dtype: DT_INT64
          }
        }
        features: {
          key: "float_feature"
          value {
              dtype: DT_FLOAT
          }
        }
        features: {
            key: "string_feature"
            value {
                dtype: DT_STRING
            }
        }
        metadata {
            extra {
              key: "edge_type"
              value: "reversed"
            }
            bigquery {
                table_spec {
                  project: "test_project"
                  dataset: "test_dataset"
                  table: "test_table"
                }
            }
        }""", tfgnn.proto.graph_schema_pb2.EdgeSet())

    # Mock a source that returns fake BQ rows.
    def fake_bq_reader(**unused_kwargs):
      del unused_kwargs
      return beam.Create([{
          "source": "s1",
          "target": "t1",
          "string_feature": "a",
          "float_feature": 1.0,
          "int_feature": 2
      }, {
          "source": "s2",
          "target": "t2",
          "string_feature": "b",
          "int_feature": 3,
          "float_feature": 4.0
      }])

    with test_pipeline.TestPipeline() as pipeline:
      rows = (
          pipeline | unigraph.ReadUnigraphPieceFromBigQuery(
              "fake_node_set", edge_set, bigquery_reader=fake_bq_reader))

      result = pipeline.run()
      result.wait_until_finish()

      util.assert_that(
          rows,
          util.equal_to([(b"t1", b"s1",
                          text_format.Parse(
                              """
                              features {
                                feature {
                                  key: "#source"
                                  value {
                                    bytes_list {
                                      value: "t1"
                                    }
                                  }
                                }
                                feature {
                                  key: "#target"
                                  value {
                                    bytes_list {
                                        value: "s1"
                                    }
                                  }
                                }
                                feature {
                                  key: "float_feature"
                                  value {
                                    float_list {
                                      value: 1.0
                                    }
                                  }
                                }
                                feature {
                                  key: "int_feature"
                                  value {
                                    int64_list {
                                      value: 2
                                    }
                                  }
                                }
                                feature {
                                  key: "string_feature"
                                  value {
                                    bytes_list {
                                      value: "a"
                                    }
                                  }
                                }
                            }""", tf.train.Example())),
                         (b"t2", b"s2",
                          text_format.Parse(
                              """features {
                                  feature {
                                    key: "#source"
                                    value {
                                      bytes_list {
                                        value: "t2"
                                      }
                                    }
                                  }
                                  feature {
                                    key: "#target"
                                    value {
                                      bytes_list {
                                        value: "s2"
                                      }
                                    }
                                  }
                                  feature {
                                    key: "float_feature"
                                    value {
                                      float_list {
                                        value: 4.0
                                      }
                                    }
                                  }
                                  feature {
                                    key: "int_feature"
                                    value {
                                      int64_list {
                                        value: 3
                                      }
                                    }
                                  }
                                  feature {
                                    key: "string_feature"
                                    value {
                                      bytes_list {
                                        value: "b"
                                      }
                                    }
                                  }
                                }""", tf.train.Example()))]))

  def test_read_graph_bigquery(self):
    schema = text_format.Parse(
        """
      node_sets {
        key: "customers"
        value {
          features {
            key: "occupation"
            value {
              dtype: DT_STRING
            }
          }
          metadata {
            bigquery {
              sql: "SELECT customer_uid AS id, occupation FROM [test_project:test_dataset.customers] LIMIT 100"
            }
          }
        }
      }
      edge_sets {
        key: "transactions"
        value {
          features {
            key: "amount"
            value {
              dtype: DT_FLOAT
            }
          }
          features {
            key: "post_date"
            value {
              dtype: DT_STRING
            }
          }
          metadata {
            bigquery  {
              sql: "SELECT source, destination AS target, amount, post_date FROM [test_project:test_dataset.transactions] LIMIT 100"
            }
          }
        }
      }""", tfgnn.GraphSchema())

    def fake_bq_reader(**kwargs):
      if kwargs["query"] == ("SELECT customer_uid AS id, occupation FROM "
                             "[test_project:test_dataset.customers] LIMIT 100"):
        return beam.Create([{
            "id": "c1",
            "occupation": "SWE"
        }, {
            "id": "c2",
            "occupation": "SRE"
        }])
      elif kwargs["query"] == (
          "SELECT source, destination AS target, amount, post_date"
          " FROM [test_project:test_dataset.transactions] LIMIT "
          "100"):
        return beam.Create([{
            "source": "c1",
            "target": "c2",
            "amount": 42.0,
            "post_date": "2022/09/06",
            "int_feature": 2
        }, {
            "source": "c2",
            "target": "c1",
            "amount": 24.0,
            "post_date": "2022/09/07"
        }])
      else:
        raise ValueError(f"No query matches {kwargs['query']}")

    expected_graph = {
        "edges": {
            "transactions": [(b"c1", b"c2",
                              text_format.Parse(
                                  """features {
              feature {
                key: "#source"
                value {
                  bytes_list {
                    value: "c1"
                  }
                }
              }
              feature {
                key: "#target"
                value {
                  bytes_list {
                    value: "c2"
                  }
                }
              }
              feature {
                key: "amount"
                value {
                  float_list {
                    value: 42.0
                  }
                }
              }
              feature {
                key: "post_date"
                value {
                  bytes_list {
                    value: "2022/09/06"
                  }
                }
              }}""", tf.train.Example())),
                             (b"c2", b"c1",
                              text_format.Parse(
                                  """features {
              feature {
                key: "#source"
                value {
                  bytes_list {
                    value: "c2"
                  }
                }
              }
              feature {
                key: "#target"
                value {
                  bytes_list {
                    value: "c1"
                  }
                }
              }
              feature {
                key: "amount"
                value {
                  float_list {
                    value: 24.0
                  }
                }
              }
              feature {
                key: "post_date"
                value {
                  bytes_list {
                    value: "2022/09/07"
                  }
                }
              }
            }""", tf.train.Example()))]
        },
        "nodes": {
            "customers": [(b"c1",
                           text_format.Parse(
                               """features {
                          feature {
                            key: "#id"
                            value {
                              bytes_list {
                                value: "c1"
                              }
                            }
                          }
                          feature {
                            key: "occupation"
                            value {
                              bytes_list {
                                value: "SWE"
                              }
                            }
                          }} """, tf.train.Example())),
                          (b"c2",
                           text_format.Parse(
                               """features {
                          feature {
                              key: "#id"
                              value {
                                bytes_list {
                                  value: "c2"
                                }
                              }
                            }
                            feature {
                              key: "occupation"
                              value {
                                bytes_list {
                                  value: "SRE"
                                }
                              }
                            }}""", tf.train.Example()))]
        }
    }

    with test_pipeline.TestPipeline() as pipeline:
      graph = unigraph.read_graph(
          schema, "", pipeline, bigquery_reader=fake_bq_reader)
      result = pipeline.run()
      result.wait_until_finish()
      util.assert_that(
          graph["edges"]["transactions"],
          util.equal_to(expected_graph["edges"]["transactions"]),
          label="transactions")
      util.assert_that(
          graph["nodes"]["customers"],
          util.equal_to(expected_graph["nodes"]["customers"]),
          label="customers")


def deep_dict_value_map(fn, d, depth=2):
  if depth == 0:
    return fn(d)
  return {k: deep_dict_value_map(fn, v, depth - 1) for k, v in d.items()}


class TestDictStreams(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.resource_dir = test_utils.get_resource_dir("testdata/heterogeneous")

  def test_read_csv(self):
    graph_streams = unigraph.DictStreams.iter_graph_via_path(
        path.join(self.resource_dir, "graph.pbtxt"))
    graph_lists = deep_dict_value_map(list, graph_streams)

    for node_set_name, items in graph_lists["nodes"].items():
      self.assertLen(items, _EXPECTED_CSV_SIZES[node_set_name])
    for edge_set_name, items in graph_lists["edges"].items():
      self.assertLen(items, _EXPECTED_CSV_SIZES[edge_set_name])

  def test_read_node_set(self):
    filename = path.join(self.resource_dir, "customer.csv")
    stream = unigraph.DictStreams.iter_records_from_filepattern(filename)
    stream = map(unigraph.get_node_ids, stream)

    actual_keys = [key for key, value in stream]
    self.assertSetEqual(set(actual_keys), set(_CUSTOMER_IDS))

  def test_read_edge_set(self):
    filename = path.join(self.resource_dir, "owns_card.csv")
    stream = unigraph.DictStreams.iter_records_from_filepattern(filename)
    stream = map(unigraph.get_edge_ids, stream)
    actual_src_ids, actual_tgt_ids, unused_examples = zip(*stream)
    set_ids = set([(src, tgt)
                   for src, tgt in zip(actual_src_ids, actual_tgt_ids)])
    expected_set_ids = set(
        [(src, tgt)
         for src, tgt in zip(_OWNS_CARDS_SRC_IDS, _OWNS_CARDS_TGT_IDS)])
    self.assertSetEqual(set_ids, expected_set_ids)


if __name__ == "__main__":
  tf.test.main()
