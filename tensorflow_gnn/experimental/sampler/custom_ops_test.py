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
"""Tests for bulk_inference."""

from typing import Tuple
from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.experimental.sampler import gen_custom_ops as ops

rt = tf.ragged.constant

Tuple3 = Tuple[int, int, int]


class RaggedChoice(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([(1, 10), (2, 20), (3, 30)])
  def testWithRetries(self, num_samples: int, range_size: int):
    results = set()
    for _ in range(1_000):
      choice = ops.ragged_choice(
          [num_samples], [range_size], global_indices=False
      )
      results.update(choice.numpy())
    self.assertLen(results, range_size)

  @parameterized.parameters([([10, 5, 3],), ([3, 2, 1],), ([3, 3, 3],)])
  def testAlgorithmR(self, range_sizes: Tuple3):
    results1 = set()
    results2 = set()
    results3 = set()
    for _ in range(100):
      choice = ops.ragged_choice(
          tf.convert_to_tensor([3, 2, 1]),
          tf.convert_to_tensor(range_sizes),
          global_indices=False,
      )
      results1.update(choice[0:3].numpy())
      results2.update(choice[3:5].numpy())
      results3.update(choice[5:6].numpy())
    self.assertLen(results1, range_sizes[0])
    self.assertLen(results2, range_sizes[1])
    self.assertLen(results3, range_sizes[2])

  @parameterized.product(
      num_samples_dtype=[tf.int32, tf.int64],
      row_lengths_dtype=[tf.int32, tf.int64],
  )
  def testLocalIndices(
      self, num_samples_dtype: tf.DType, row_lengths_dtype: tf.DType
  ):
    for _ in range(100):
      choice = ops.ragged_choice(
          tf.convert_to_tensor([1, 2, 3], num_samples_dtype),
          tf.convert_to_tensor([5, 10, 15], row_lengths_dtype),
          global_indices=False,
      )
      self.assertContainsSubset(choice[0:1].numpy(), range(0, 5))
      self.assertContainsSubset(choice[1:3].numpy(), range(0, 10))
      self.assertContainsSubset(choice[3:6].numpy(), range(0, 15))

  @parameterized.product(
      num_samples_dtype=[tf.int32, tf.int64],
      row_lengths_dtype=[tf.int32, tf.int64],
  )
  def testGlobalIndices(
      self, num_samples_dtype: tf.DType, row_lengths_dtype: tf.DType
  ):
    for _ in range(100):
      choice = ops.ragged_choice(
          tf.convert_to_tensor([1, 2, 3], num_samples_dtype),
          tf.convert_to_tensor([5, 10, 15], row_lengths_dtype),
          global_indices=True,
      )
      self.assertContainsSubset(choice[0:1].numpy(), range(0, 5))
      self.assertContainsSubset(choice[1:3].numpy(), range(5, 15))
      self.assertContainsSubset(choice[3:6].numpy(), range(15, 30))


class RaggedUnique(tf.test.TestCase, parameterized.TestCase):

  @parameterized.product(
      values_dtype=[tf.int32, tf.int64, tf.float32, tf.double, tf.string],
      splits_dtype=[tf.int32, tf.int64],
  )
  def testTypes(self, values_dtype, splits_dtype):
    input_value = rt(
        [], dtype=values_dtype, ragged_rank=1, row_splits_dtype=splits_dtype
    )
    actual_values, actual_row_splits = ops.ragged_unique(
        input_value.values, input_value.row_splits
    )
    actual = tf.RaggedTensor.from_row_splits(
        actual_values, actual_row_splits, validate=True
    )

    expected = rt(
        [], dtype=values_dtype, ragged_rank=1, row_splits_dtype=splits_dtype
    )
    self.assertAllEqual(actual.values, expected.values)
    self.assertAllEqual(actual.row_splits, expected.row_splits)

  @parameterized.named_parameters([
      ("single_int", rt([[1]]), rt([[1]])),
      ("single_string", rt([["a"]]), rt([["a"]])),
      (
          "preserves_order",
          rt([[], [2.0, 1.0], [3.0, 1.0, 2.0], [3.0, 2.0, 1.0, 4.0]]),
          rt([[], [2.0, 1.0], [3.0, 1.0, 2.0], [3.0, 2.0, 1.0, 4.0]]),
      ),
      ("unique1", rt([["a"] * 10, [], ["c"] * 5]), rt([["a"], [], ["c"]])),
      (
          "unique2",
          rt([[], [2, 1] * 10, [5, 3, 1, 1, 2] * 5, [], [], [7] * 10, [], []]),
          rt([[], [2, 1], [5, 3, 1, 2], [], [], [7], [], []]),
      ),
  ])
  def testSorting(
      self, input_value: tf.RaggedTensor, expected: tf.RaggedTensor
  ):
    actual_values, actual_row_splits = ops.ragged_unique(
        input_value.values, input_value.row_splits
    )
    actual = tf.RaggedTensor.from_row_splits(
        actual_values, actual_row_splits, validate=True
    )
    self.assertAllEqual(actual.values, expected.values)
    self.assertAllEqual(actual.row_splits, expected.row_splits)


class RaggedLookup(tf.test.TestCase, parameterized.TestCase):

  @parameterized.product(
      values_dtype=[tf.int32, tf.int64, tf.float32, tf.double, tf.string],
      row_splits_dtype=[tf.int32, tf.int64],
  )
  def testTypes(
      self,
      values_dtype,
      row_splits_dtype,
  ):
    values = rt(
        [],
        dtype=values_dtype,
        ragged_rank=1,
        row_splits_dtype=row_splits_dtype,
    )
    vocab = rt(
        [],
        dtype=values_dtype,
        ragged_rank=1,
        row_splits_dtype=row_splits_dtype,
    )
    actual = ops.ragged_lookup(
        values.values,
        values.row_splits,
        vocab.values,
        vocab.row_splits,
        global_indices=True,
    )

    expected = tf.constant([], dtype=values_dtype)
    self.assertAllEqual(actual, expected)

  @parameterized.named_parameters([
      (
          "single_string",
          rt([["a"], ["b"], ["c"], ["d"]]),
          rt([
              ["a", "b"],
              ["a", "b", "c"],
              ["a", "b", "c", "d"],
              ["a", "b", "c", "d"],
          ]),
          False,
          rt([[0], [1], [2], [3]]),
      ),
      (
          "multiple_ints",
          rt([[0, 2, 0, 1, 1], [], [1] * 4, [2, 1, 0, 0], [], []]),
          rt([
              [1, 2, 0],
              [1, 2, 3],
              [1],
              [0, 1, 2, 3, 4, 5],
              [],
              [1, 2, 3, 4, 5, 6],
          ]),
          False,
          rt([[2, 1, 2, 0, 0], [], [0] * 4, [2, 1, 0, 0], [], []]),
      ),
      (
          "global",
          rt([[0, 2, 0, 1, 1], [], [2, 1, 0, 0], [], [5]]),
          rt([
              [0, 1, 2],
              [1, 2, 3],
              [1, 2, 3, 4, 5, 0],
              [],
              [1, 2, 3, 4, 5, 6],
          ]),
          True,
          rt([
              [0, 2, 0, 1, 1],
              [],
              [1 + 6, 0 + 6, 5 + 6, 5 + 6],
              [],
              [4 + 12],
          ]),
      ),
  ])
  def testImplementation(
      self, values, vocabulary, global_indices, expected_result
  ):
    result = ops.ragged_lookup(
        values.values,
        values.row_splits,
        vocabulary.values,
        vocabulary.row_splits,
        global_indices=global_indices,
    )
    result = values.with_values(result)
    self.assertAllEqual(result.values, expected_result.values)
    self.assertAllEqual(result.row_splits, expected_result.row_splits)


if __name__ == "__main__":
  tf.test.main()
