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
"""Tests for datasets."""
from typing import Any, Optional, Sequence

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.runner.input import datasets


def dataset(x: int = 0) -> tf.data.Dataset:
  return tf.data.Dataset.from_tensors(tf.constant([x + 8191])).repeat(8191)


def interleave_fn(x: Any) -> tf.data.Dataset:
  return tf.data.Dataset.from_tensors(x)


class DatasetsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="num_input_pipelines=1",
          num_input_pipelines=1,
          input_pipeline_id=0,
          shuffle_dataset=False,
          actual_dataset=dataset(0).take(1).concatenate(dataset(1).take(1)),
          expected_len=2,
          expected_values=[8191, 8192],
      ),
      dict(
          testcase_name="input_pipeline_id=0",
          num_input_pipelines=2,
          input_pipeline_id=0,
          shuffle_dataset=True,
          actual_dataset=dataset(0).take(1).concatenate(dataset(1).take(1)),
          expected_len=1,
          expected_values=[8191],
      ),
      dict(
          testcase_name="input_pipeline_id=1",
          num_input_pipelines=2,
          input_pipeline_id=1,
          shuffle_dataset=True,
          actual_dataset=dataset(0).take(1).concatenate(dataset(1).take(1)),
          expected_len=1,
          expected_values=[8192],
      ),
  ])
  def test_process_dataset(
      self,
      num_input_pipelines: int,
      input_pipeline_id: int,
      shuffle_dataset: bool,
      actual_dataset: tf.data.Dataset,
      expected_len: int,
      expected_values: Sequence[int]):
    ds = datasets._process_dataset(
        actual_dataset,
        num_shards=num_input_pipelines,
        index=input_pipeline_id,
        shuffle_dataset=shuffle_dataset,
        interleave_fn=interleave_fn)

    self.assertEqual(ds.cardinality(), tf.data.UNKNOWN_CARDINALITY)
    self.assertLen(list(ds), expected_len)

    self.assertCountEqual(ds.as_numpy_iterator(), expected_values)

  @parameterized.named_parameters([
      dict(
          testcase_name="fixed_cardinality=True",
          extra_datasets=[dataset()] * 4,
          extra_weights=None,
          fixed_cardinality=True,
          shuffle_dataset=False,
          principal_dataset=dataset(),
          principal_cardinality=1,
          principal_weight=None,
          expected_cardinality=5,
          ),
      dict(
          testcase_name="fixed_cardinality=False",
          extra_datasets=[dataset()] * 4,
          extra_weights=None,
          fixed_cardinality=False,
          shuffle_dataset=True,
          principal_dataset=dataset(),
          principal_cardinality=None,
          principal_weight=None,
          expected_cardinality=tf.data.UNKNOWN_CARDINALITY,
          ),
      dict(
          testcase_name="fixed_cardinality=True;with weights",
          extra_datasets=[dataset()] * 4,
          extra_weights=[.25] * 4,
          fixed_cardinality=True,
          shuffle_dataset=True,
          principal_dataset=dataset(),
          principal_cardinality=1,
          principal_weight=.4,
          expected_cardinality=2,
          ),
      dict(
          testcase_name="fixed_cardinality=False;with weights",
          extra_datasets=[dataset()] * 4,
          extra_weights=[.25] * 4,
          fixed_cardinality=False,
          shuffle_dataset=False,
          principal_dataset=dataset(),
          principal_cardinality=None,
          principal_weight=.4,
          expected_cardinality=tf.data.UNKNOWN_CARDINALITY,
          ),
  ])
  def test_process_sampled_dataset(
      self,
      extra_datasets: Sequence[tf.data.Dataset],
      extra_weights: Optional[Sequence[float]],
      fixed_cardinality: bool,
      shuffle_dataset: bool,
      principal_dataset: tf.data.Dataset,
      principal_cardinality: Optional[int],
      principal_weight: Optional[float],
      expected_cardinality: Optional[int]):
    ds = datasets._process_sampled_dataset(
        principal_dataset,
        extra_datasets,
        principal_weight,
        extra_weights,
        num_shards=1,
        index=0,
        fixed_cardinality=fixed_cardinality,
        shuffle_dataset=shuffle_dataset,
        principal_cardinality=principal_cardinality,
        interleave_fn=interleave_fn)

    self.assertEqual(ds.cardinality(), expected_cardinality)

  @parameterized.named_parameters([
      dict(
          testcase_name="Missing extra_weights",
          extra_datasets=[dataset()] * 4,
          extra_weights=None,
          principal_dataset=dataset(),
          principal_cardinality=None,
          fixed_cardinality=True,
          principal_weight=.4,
          expected_error="`extra_weights` required a `principal_weight`",
          ),
      dict(
          testcase_name="Missing principal_weight",
          extra_datasets=[dataset()] * 4,
          extra_weights=[.25] * 4,
          principal_dataset=dataset(),
          principal_cardinality=None,
          fixed_cardinality=True,
          principal_weight=None,
          expected_error="`principal_weight` is required with `extra_weights`",
          ),
      dict(
          testcase_name="Missing principal_cardinality",
          extra_datasets=[dataset()] * 4,
          extra_weights=None,
          principal_dataset=dataset(),
          principal_cardinality=None,
          fixed_cardinality=True,
          principal_weight=None,
          expected_error="`principal_cardinality` is required with `fixed_cardinality`",
          ),
  ])
  def test_process_sampled_dataset_error(
      self,
      extra_datasets: Sequence[tf.data.Dataset],
      extra_weights: Optional[Sequence[float]],
      fixed_cardinality: Optional[bool],
      principal_dataset: tf.data.Dataset,
      principal_cardinality: Optional[int],
      principal_weight: Optional[float],
      expected_error: str):
    with self.assertRaisesRegex(ValueError, expected_error):
      _ = datasets._process_sampled_dataset(
          principal_dataset,
          extra_datasets,
          principal_weight,
          extra_weights,
          num_shards=1,
          index=0,
          fixed_cardinality=fixed_cardinality,
          principal_cardinality=principal_cardinality,
          interleave_fn=interleave_fn)

  @parameterized.named_parameters([
      dict(
          testcase_name="Missing file_pattern and filenames argument",
          file_pattern=None,
          filenames=None,
          expected_error="Please provide either `_file_pattern` or `_filenames`"
          " argument, but not both.",
          ),
      dict(
          testcase_name="Both file_pattern and filenames argument",
          file_pattern="foo",
          filenames=["bar"],
          expected_error="Please provide either `_file_pattern` or `_filenames`"
          " argument, but not both.",
          )
  ])
  def test_simple_dataset_provider_initialization_error(
      self,
      file_pattern: Optional[str],
      filenames: Optional[Sequence[str]],
      expected_error: str):
    with self.assertRaisesRegex(ValueError, expected_error):
      _ = datasets.SimpleDatasetProvider(
          file_pattern,
          filenames=filenames,
          interleave_fn=interleave_fn)

  @parameterized.named_parameters([
      dict(
          testcase_name="Missing principal_file_pattern and principal_filename argument",
          principal_file_pattern=None,
          principal_filenames=None,
          expected_error="Please provide either `_principal_file_pattern` or "
          "`_principal_filenames` argument, but not both.",
          ),
      dict(
          testcase_name="Both principal_file_pattern and principal_filename argument",
          principal_file_pattern="foo",
          principal_filenames=["bar"],
          expected_error="Please provide either `_principal_file_pattern` or "
          "`_principal_filenames` argument, but not both.",
          )
  ])
  def test_simple_sample_datasets_provider_initialization_error(
      self,
      principal_file_pattern: Optional[str],
      principal_filenames: Optional[Sequence[str]],
      expected_error: str):
    with self.assertRaisesRegex(ValueError, expected_error):
      _ = datasets.SimpleSampleDatasetsProvider(
          principal_file_pattern=principal_file_pattern,
          principal_filenames=principal_filenames,
          interleave_fn=interleave_fn)

if __name__ == "__main__":
  tf.test.main()
