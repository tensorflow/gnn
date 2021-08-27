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
        "paid_with.csv", "transactions.csv", "month.csv",
        "invalid_customer.csv",
        "sampler_golden.ascii",
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


if __name__ == "__main__":
  tf.test.main()
