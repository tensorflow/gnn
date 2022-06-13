"""Tests for datasets."""
import os
from typing import Optional, Sequence

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.runner import orchestration
from tensorflow_gnn.runner.input import datasets


class DatasetsTest(tf.test.TestCase, parameterized.TestCase):

  def get_testdata(
      self,
      contents: Sequence[str],
      filename_prefix: Optional[str] = "file") -> str:
    tempdir = self.create_tempdir()
    for shard, content in enumerate(contents):
      filename = os.path.join(
          tempdir,
          f"{filename_prefix}-{shard:05}-of-{len(contents):05}")
      with open(filename, "w") as fd:
        fd.write(content)
    return f"{os.path.join(tempdir, filename_prefix + '*')}"

  @parameterized.named_parameters([
      dict(
          testcase_name="num_input_pipelines=1",
          num_input_pipelines=1,
          input_pipeline_id=0,
          contents=["8191", "8192"],
          expected_len=2,
          expected_values=[b"8191", b"8192"],
      ),
      dict(
          testcase_name="input_pipeline_id=0",
          num_input_pipelines=2,
          input_pipeline_id=0,
          contents=["8191", "8192"],
          expected_len=1,
          expected_values=[b"8191"],
      ),
      dict(
          testcase_name="input_pipeline_id=1",
          num_input_pipelines=2,
          input_pipeline_id=1,
          contents=["8191", "8192"],
          expected_len=1,
          expected_values=[b"8192"],
      ),
  ])
  def test_get_dataset(
      self,
      num_input_pipelines: int,
      input_pipeline_id: int,
      contents: Sequence[str],
      expected_len: int,
      expected_values: Sequence[int]):
    actual_dataset_pattern = self.get_testdata(contents)
    ds = datasets._get_dataset(
        actual_dataset_pattern,
        num_shards=num_input_pipelines,
        index=input_pipeline_id,
        interleave_fn=tf.data.TextLineDataset)
    self.assertEqual(ds.cardinality(), tf.data.UNKNOWN_CARDINALITY)
    self.assertLen(list(ds), expected_len)
    self.assertCountEqual(ds.as_numpy_iterator(), expected_values)

  @parameterized.named_parameters([
      dict(
          testcase_name="fixed_cardinality=True",
          extra_file_contents=[["8191"]] * 4,
          extra_weights=None,
          fixed_cardinality=True,
          principal_file_content=["8191"],
          principal_cardinality=1,
          principal_weight=None,
          expected_cardinality=5,
      ),
      dict(
          testcase_name="fixed_cardinality=False",
          extra_file_contents=[["8191"]] * 4,
          extra_weights=None,
          fixed_cardinality=False,
          principal_file_content=["8191"],
          principal_cardinality=None,
          principal_weight=None,
          expected_cardinality=tf.data.UNKNOWN_CARDINALITY,
      ),
      dict(
          testcase_name="fixed_cardinality=True;with weights",
          extra_file_contents=[["8191"]] * 4,
          extra_weights=[.25] * 4,
          fixed_cardinality=True,
          principal_file_content=["8191"],
          principal_cardinality=1,
          principal_weight=.4,
          expected_cardinality=2,
      ),
      dict(
          testcase_name="fixed_cardinality=False;with weights",
          extra_file_contents=[["8191"]] * 4,
          extra_weights=[.25] * 4,
          fixed_cardinality=False,
          principal_file_content=["8191"],
          principal_cardinality=None,
          principal_weight=.4,
          expected_cardinality=tf.data.UNKNOWN_CARDINALITY,
      ),
  ])
  def test_get_sampled_dataset(
      self,
      extra_file_contents: Sequence[Sequence[str]],
      extra_weights: Optional[Sequence[float]],
      fixed_cardinality: bool,
      principal_file_content: str,
      principal_cardinality: Optional[int],
      principal_weight: Optional[float],
      expected_cardinality: Optional[int]):
    extra_file_patterns = [
        self.get_testdata(content) for content in extra_file_contents
    ]
    principal_file_pattern = self.get_testdata(principal_file_content)
    ds = datasets._get_sampled_dataset(
        principal_file_pattern,
        extra_file_patterns,
        principal_weight,
        extra_weights,
        num_shards=1,
        index=0,
        fixed_cardinality=fixed_cardinality,
        principal_cardinality=principal_cardinality,
        interleave_fn=tf.data.TextLineDataset)

    self.assertEqual(ds.cardinality(), expected_cardinality)

  @parameterized.named_parameters([
      dict(
          testcase_name="Missing extra_weights",
          extra_file_contents=[["8191"]] * 4,
          extra_weights=None,
          principal_file_content=["8191"],
          principal_cardinality=None,
          fixed_cardinality=True,
          principal_weight=.4,
          expected_error="`extra_weights` required a `principal_weight`",
      ),
      dict(
          testcase_name="Missing principal_weight",
          extra_file_contents=[["8191"]] * 4,
          extra_weights=[.25] * 4,
          principal_file_content=["8191"],
          principal_cardinality=None,
          fixed_cardinality=True,
          principal_weight=None,
          expected_error="`principal_weight` is required with `extra_weights`",
      ),
      dict(
          testcase_name="Missing principal_cardinality",
          extra_file_contents=[["8191"]] * 4,
          extra_weights=None,
          principal_file_content=["8191"],
          principal_cardinality=None,
          fixed_cardinality=True,
          principal_weight=None,
          expected_error="`principal_cardinality` is required with `fixed_cardinality`",
      ),
  ])
  def test_get_sampled_dataset_error(
      self,
      extra_file_contents: Sequence[Sequence[str]],
      extra_weights: Optional[Sequence[float]],
      fixed_cardinality: Optional[bool],
      principal_file_content: str,
      principal_cardinality: Optional[int],
      principal_weight: Optional[float],
      expected_error: str):
    extra_file_patterns = [
        self.get_testdata(content) for content in extra_file_contents
    ]
    principal_file_pattern = self.get_testdata(principal_file_content)
    with self.assertRaisesRegex(ValueError, expected_error):
      _ = datasets._get_sampled_dataset(
          principal_file_pattern,
          extra_file_patterns,
          principal_weight,
          extra_weights,
          num_shards=1,
          index=0,
          fixed_cardinality=fixed_cardinality,
          principal_cardinality=principal_cardinality,
          interleave_fn=tf.data.TextLineDataset)

  @parameterized.named_parameters([
      dict(
          testcase_name="TFRecordDatasetProvider",
          klass=datasets.TFRecordDatasetProvider,
      ),
      dict(
          testcase_name="SampleTFRecordDatasetsProvider",
          klass=datasets.SampleTFRecordDatasetsProvider,
      ),
  ])
  def test_protocol(self, klass: object):
    self.assertIsInstance(klass, orchestration.DatasetProvider)


if __name__ == "__main__":
  tf.test.main()
