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

    expected_sizes = {"creditcard": 36,
                      "customer": 24,
                      "": 1,  # Context set.
                      "owns_card": 24,
                      "paid_with": 48,
                      "transaction": 48}
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
    customer_ids = b"""
      1876448 1372437 1368305 1974494 1257724 1758057 1531660 1489311 1407706
      196838 1195675 1659366 1499004 1344333 1443888 1108778 175583 1251872
      1493851 1599418 1768701 1549489 1879799 125454
    """.split()
    util.assert_that(pcoll, util.equal_to(customer_ids))
    pipeline.run()

  def test_read_edge_set(self):
    filename = path.join(self.resource_dir, "owns_card.csv")
    pipeline = test_pipeline.TestPipeline()
    pcoll = (pipeline
             | unigraph.ReadTable(filename, "csv")
             | beam.Map(unigraph.get_edge_ids))
    source_coll = (pcoll | beam.Map(lambda item: item[0]))
    target_coll = (pcoll | beam.Map(lambda item: item[1]))
    source_ids = b"""
      1876448 1372437 1368305 1974494 1257724 1758057 1531660 1489311 1407706
      196838 1195675 1659366 1499004 1344333 1443888 1108778 175583 1251872
      1493851 1599418 1768701 1549489 1879799 125454
    """.split()
    target_ids = b"""
      16827485386298040 11470379189154620 11163838768727470 16011471358128450
      18569067217418250 17396883707513070 14844931107602160 1238474857489384
      11290312140467510 17861046738135650 8878522895102384 13019350102369400
      11470379189154620 16283233487191600 9991040399813057 14912408563871390
      11290312140467510 12948957000457930 3549061668422198 9991040399813057
      18362223127059380 1238474857489384 18569067217418250 18526138896540830
    """.split()
    util.assert_that(source_coll, util.equal_to(source_ids),
                     label="AssertSource")
    util.assert_that(target_coll, util.equal_to(target_ids),
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
        unigraph.ReadFromBigQuery.bigquery_args_from_proto(bq), {
            "table": "test_project:test_dataset.test_table",
            "read_method": "EXPORT"
        })

  def test_bigquery_row_to_keyed_example_node_set(self):
    node_set = text_format.Parse(
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
        } """, tfgnn.proto.graph_schema_pb2.NodeSet())

    # Quick test of the suffix generation
    self.assertEqual(
        unigraph.ReadFromBigQuery.stage_name_suffix("fake_node_set", node_set),
        ("ReadFromBigQuery\\NodeSet\\fake_node_set\\test_project:"
         "test_dataset.test_table"))

    # Mock a source that returns fake BQ rows.
    def fake_bq_reader(**unused_kwargs):
      del unused_kwargs
      return beam.Create([{
          "id": "id1",
          "string_feature": "a",
          "float_feature": 1.0,
          "int_feature": 2
      }, {
          "id": "id2",
          "string_feature": "b",
          "int_feature": 3,
          "float_feature": 4.0
      }])

    with test_pipeline.TestPipeline() as pipeline:
      rows = (
          pipeline | unigraph.ReadFromBigQuery(
              "fake_node_set", node_set, bigquery_reader=fake_bq_reader))

      result = pipeline.run()
      result.wait_until_finish()

      util.assert_that(
          rows,
          util.equal_to([("id1",
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
                              }""", tf.train.Example())),
                         ("id2",
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
        unigraph.ReadFromBigQuery.stage_name_suffix("fake_edge_set", edge_set),
        ("ReadFromBigQuery\\EdgeSet\\fake_edge_set\\"
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
          pipeline | unigraph.ReadFromBigQuery(
              "fake_edge_set", edge_set, bigquery_reader=fake_bq_reader))

      result = pipeline.run()
      result.wait_until_finish()

      util.assert_that(
          rows,
          util.equal_to([("s1", "t1",
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
                         ("s2", "t2",
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
          pipeline | unigraph.ReadFromBigQuery(
              "fake_node_set", edge_set, bigquery_reader=fake_bq_reader))

      result = pipeline.run()
      result.wait_until_finish()

      util.assert_that(
          rows,
          util.equal_to([("t1", "s1",
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
                         ("t2", "s2",
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


if __name__ == "__main__":
  tf.test.main()
