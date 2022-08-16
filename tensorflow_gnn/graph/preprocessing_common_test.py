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
"""Tests for preprocessing_common."""

import math
from typing import List
from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import preprocessing_common

as_tensor = tf.convert_to_tensor


class ReduceMeanTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('Tensor', as_tensor([1., 2.])),
      ('Tuple', (as_tensor([1.]), as_tensor([[2.]]))),
      ('Dict', {
          'm': as_tensor([[1], [2]]),
          'v': as_tensor([1]),
          's': as_tensor(2.),
      }),
  )
  def testEmpty(self, value):
    ds = tf.data.Dataset.from_tensors(value)
    ds = ds.take(0)
    result = preprocessing_common.compute_basic_stats(ds)
    tf.nest.map_structure(
        self.assertAllEqual, result.mean,
        tf.nest.map_structure(lambda t: tf.fill(tf.shape(t), 0.0), value))
    tf.nest.map_structure(
        self.assertAllEqual, result.minimum,
        tf.nest.map_structure(lambda t: tf.fill(tf.shape(t), t.dtype.max),
                              value))
    tf.nest.map_structure(
        self.assertAllEqual, result.maximum,
        tf.nest.map_structure(lambda t: tf.fill(tf.shape(t), t.dtype.min),
                              value))

  @parameterized.named_parameters(
      ('Tensor', as_tensor([1., 2.])),
      ('Tuple', as_tensor([1.])),
      ('Dict', {
          'm': as_tensor([[1], [2]]),
          'v': as_tensor([1]),
          's': as_tensor(2.),
      }),
  )
  def testSingleElement(self, value):
    ds = tf.data.Dataset.from_tensors(value)
    result = preprocessing_common.compute_basic_stats(ds)
    tf.nest.map_structure(
        self.assertAllEqual, result.minimum,
        tf.nest.map_structure(lambda t: tf.cast(t, tf.float32), value))
    tf.nest.map_structure(self.assertAllEqual, result.maximum, value)
    tf.nest.map_structure(self.assertAllEqual, result.minimum, value)

  def testMeanComputation(self):

    def generator(x):
      return {
          'x': x,
          '2x': 2 * x,
      }

    ds = tf.data.Dataset.range(100).map(generator)
    result = preprocessing_common.compute_basic_stats(ds)
    self.assertAllClose(result.mean['x'], (99.0 + 0.0) * 0.5)
    self.assertAllClose(result.mean['2x'], 2.0 * (99.0 + 0.0) * 0.5)
    self.assertAllClose(result.minimum['x'], 0)
    self.assertAllClose(result.minimum['2x'], 0)
    self.assertAllClose(result.maximum['x'], 99)
    self.assertAllClose(result.maximum['2x'], 99 * 2)


class DatasetFilterWithSummaryTest(tf.test.TestCase, parameterized.TestCase):

  def assertSummary(self,
                    events_dir: str,
                    expected_steps: List[int],
                    expected_values: List[float],
                    expected_summary_name: str = 'dataset_removed_fraction',
                    values_tol: float = 0.0):
    files = tf.io.gfile.glob(f'{events_dir}/*.v2')
    self.assertLen(files, 1)
    steps, values = [], []
    for event in tf.compat.v1.train.summary_iterator(files[0]):
      for value_proto in event.summary.value:
        self.assertEndsWith(value_proto.tag, expected_summary_name)
        steps.append(event.step)
        values.append(
            tf.io.parse_tensor(value_proto.tensor.SerializeToString(),
                               tf.float32))
    self.assertAllEqual(steps, expected_steps)
    self.assertAllClose(values, expected_values, atol=values_tol)

  def testWithNoRemoved(self):
    dataset = tf.data.Dataset.from_tensors(True)
    dataset = dataset.repeat(100)
    events_dir = self.create_tempdir().full_path
    with tf.summary.create_file_writer(events_dir).as_default():
      dataset = preprocessing_common.dataset_filter_with_summary(
          dataset, lambda g: g, summary_steps=10, summary_decay=0.9)
      self.assertLen(list(dataset), 100)
    self.assertSummary(
        events_dir,
        expected_steps=list(range(9, 100, 10)),
        expected_values=[0.] * 10)

  def testWithAllRemoved(self):
    dataset = tf.data.Dataset.from_tensors(False)
    dataset = dataset.repeat(100)
    events_dir = self.create_tempdir().full_path
    with tf.summary.create_file_writer(events_dir).as_default():
      dataset = preprocessing_common.dataset_filter_with_summary(
          dataset, lambda g: g, summary_steps=10, summary_decay=0.9)
      self.assertEmpty(list(dataset))
    self.assertSummary(
        events_dir,
        expected_steps=[0] * 10,
        expected_values=[1.] * 10)

  def testSummaryName(self):
    dataset = tf.data.Dataset.from_tensors(True)
    events_dir = self.create_tempdir().full_path
    with tf.summary.create_file_writer(events_dir).as_default():
      dataset = preprocessing_common.dataset_filter_with_summary(
          dataset, lambda g: g, summary_steps=1, summary_name='test_name')
      self.assertLen(list(dataset), 1)
    self.assertSummary(
        events_dir,
        expected_steps=[0],
        expected_values=[0.],
        expected_summary_name='test_name')

  @parameterized.product(
      summary_decay=[0.95, 0.99, 0.999],
      sample_size=[1_000, 10_000],
      remove_fraction=[0.01, 0.1, 0.5, 0.9, 0.99])
  def testOnRandomData(self, summary_decay, sample_size, remove_fraction):
    dataset = tf.data.Dataset.from_tensor_slices(
        tf.random.stateless_uniform([sample_size], seed=[42, 42]))
    events_dir = self.create_tempdir().full_path
    err0 = 6. * math.sqrt(remove_fraction * (1. - remove_fraction))
    with tf.summary.create_file_writer(events_dir).as_default():
      dataset = preprocessing_common.dataset_filter_with_summary(
          dataset,
          lambda g: g > remove_fraction,
          summary_steps=sample_size,
          summary_decay=summary_decay)
      final_size = len(list(dataset))
      self.assertNear(
          final_size,
          sample_size * (1. - remove_fraction),
          err=err0 * math.sqrt(sample_size))
    ema_window_size = 1. / (1. - summary_decay)
    self.assertSummary(
        events_dir,
        expected_steps=[final_size-1],
        expected_values=[remove_fraction],
        values_tol=err0 / math.sqrt(ema_window_size))

  def testSensitiveToDataShift(self):
    tp = tf.data.Dataset.from_tensors(True).repeat(1000)
    fp = tf.data.Dataset.from_tensors(False).repeat(1000)
    dataset = tp
    dataset = dataset.concatenate(fp)
    dataset = dataset.concatenate(tp)
    events_dir = self.create_tempdir().full_path
    with tf.summary.create_file_writer(events_dir).as_default():
      dataset = preprocessing_common.dataset_filter_with_summary(
          dataset, lambda g: g, summary_steps=1000, summary_decay=0.95)
      self.assertLen(list(dataset), 1000 * 2)
    self.assertSummary(
        events_dir,
        expected_steps=[999, 999, 1_999],
        expected_values=[0., 1., 0.],
        values_tol=1.0e-7)

  def testSupportsGraphTensor(self):
    dataset = tf.data.Dataset.from_tensors(
        gt.GraphTensor.from_pieces(
            context=gt.Context.from_fields(
                features={'f': tf.constant([True])})))
    dataset = dataset.repeat(10)
    events_dir = self.create_tempdir().full_path
    with tf.summary.create_file_writer(events_dir).as_default():
      dataset = preprocessing_common.dataset_filter_with_summary(
          dataset,
          lambda g: g.context.features['f'][0],
          summary_steps=1,
          summary_decay=0.9)
      self.assertLen(list(dataset), 10)
    self.assertSummary(
        events_dir,
        expected_steps=list(range(10)),
        expected_values=[0.] * 10)

  def testRaisesOnInvalidArguments(self):
    # pylint:disable=g-long-lambda
    dataset = tf.data.Dataset.from_tensors(True)
    self.assertRaisesRegex(
        ValueError, r'summary_steps > 0, actual summary_steps=0',
        lambda: preprocessing_common.dataset_filter_with_summary(
            dataset, lambda g: g, summary_steps=0))
    self.assertRaisesRegex(
        ValueError, r'0 < summary_decay < 1, actual summary_decay=1.0',
        lambda: preprocessing_common.dataset_filter_with_summary(
            dataset, lambda g: g, summary_steps=1, summary_decay=1.))
    self.assertRaisesRegex(
        ValueError, r'0 < summary_decay < 1, actual summary_decay=0.0',
        lambda: preprocessing_common.dataset_filter_with_summary(
            dataset, lambda g: g, summary_steps=1, summary_decay=0.))


if __name__ == '__main__':
  tf.test.main()
