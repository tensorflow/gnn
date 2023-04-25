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
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_gnn.experimental.sampler import ext_ops as ops


rt = tf.ragged.constant


class ExtOpsTestBase(tf.test.TestCase):
  IMPLEMENTATION = 'vectorized'

  def setUp(self):
    super().setUp()
    ops.set_ops_implementation(self.IMPLEMENTATION)


class RaggedChoiceTest(ExtOpsTestBase, parameterized.TestCase):

  @parameterized.parameters([True, False])
  def testRandomness1(
      self, global_indices: bool
  ):
    results = set()
    for _ in range(100):
      choice = ops.ragged_choice(
          [1], [0, 10], global_indices=global_indices, seed=42
      )
      results.update(choice.values.numpy())
    self.assertLen(results, 10)

  @parameterized.parameters([True, False])
  def testRandomness2(self, global_indices: bool):
    results1 = set()
    results2 = set()
    results3 = set()
    for _ in range(100):
      choice = ops.ragged_choice(
          [3, 2, 1], [0, 10, 15, 18], global_indices=global_indices, seed=42
      )
      results1.update(choice[0, :].numpy())
      results2.update(choice[1, :].numpy())
      results3.update(choice[2, :].numpy())
    self.assertLen(results1, 10)
    self.assertLen(results2, 5)
    self.assertLen(results3, 3)

  def testGlobalIndices(self):
    for _ in range(100):
      choice = ops.ragged_choice(
          [1, 2, 3], [0, 5, 15, 30], global_indices=True, seed=42
      )
      self.assertContainsSubset(choice[0, :].numpy(), range(0, 5))
      self.assertContainsSubset(choice[1, :].numpy(), range(5, 15))
      self.assertContainsSubset(choice[2, :].numpy(), range(15, 30))

  def testLocalIndices(self):
    for _ in range(100):
      choice = ops.ragged_choice(
          [1, 2, 3], [0, 5, 15, 30], global_indices=False, seed=42
      )
      self.assertContainsSubset(choice[0, :].numpy(), range(0, 5))
      self.assertContainsSubset(choice[1, :].numpy(), range(0, 10))
      self.assertContainsSubset(choice[2, :].numpy(), range(0, 15))

  def testNumSampleCapping(self):
    choice = ops.ragged_choice(
        [1, 2, 2, 10], [0, 0, 2, 2, 7], global_indices=True, seed=42
    )
    self.assertSetEqual(set(choice[0, :].numpy()), set())
    self.assertSetEqual(set(choice[1, :].numpy()), {0, 1})
    self.assertSetEqual(set(choice[2, :].numpy()), set())
    self.assertSetEqual(set(choice[3, :].numpy()), {2, 3, 4, 5, 6})


class ParallelRaggedChoiceTest(RaggedChoiceTest):
  IMPLEMENTATION = 'parallel'


# copybara:uncomment_begin (test for internal implementation of ext_ops)
# class CustomRaggedChoiceTest(RaggedChoiceTest):
#   IMPLEMENTATION = 'custom'
# copybara:uncomment_end


class RaggedUniqueTest(ExtOpsTestBase, parameterized.TestCase):

  @parameterized.parameters(
      # pylint: disable=g-complex-comprehension
      (v, s)
      for v in [tf.int32, tf.int64, tf.float32, tf.double, tf.string]
      for s in [tf.int32, tf.int64]
      # pylint: enable=g-complex-comprehension
  )
  def testSupportedTypes(self, values_dtype, splits_dtype):
    input_value = tf.RaggedTensor.from_row_lengths(
        tf.zeros([10], dtype=values_dtype),
        tf.convert_to_tensor([5, 5], dtype=splits_dtype),
    )
    result: tf.RaggedTensor = ops.ragged_unique(input_value)
    expected = tf.RaggedTensor.from_row_lengths(
        tf.zeros([2], dtype=values_dtype),
        tf.convert_to_tensor([1, 1], dtype=splits_dtype),
    )
    self.assertAllEqual(expected.values, result.values)
    self.assertAllEqual(expected.row_splits, result.row_splits)

  @parameterized.named_parameters([
      ('empty', rt([], tf.int32, 1), rt([], tf.int32, 1)),
      ('empty_rows', rt([[], [], []]), rt([[], [], []])),
      ('single_int', rt([[1]]), rt([[1]])),
      ('single_string', rt([['a']]), rt([['a']])),
      (
          'single_value',
          rt([['a'] * 10, [], ['c'] * 5]),
          rt([['a'], [], ['c']]),
      ),
      (
          'unchanged',
          rt([[], [2.0, 1.0], [3.0, 1.0, 2.0], [3.0, 2.0, 1.0, 4.0]]),
          rt([[], [2.0, 1.0], [3.0, 1.0, 2.0], [3.0, 2.0, 1.0, 4.0]]),
      ),
      (
          'unique_int',
          rt([
              [],
              [2, 1] * 10,
              [5, 3, 1, 1, 2] * 10,
              [],
              [],
              [7] * 10,
              [],
              [],
          ]),
          rt([[], [2, 1], [5, 3, 1, 2], [], [], [7], [], []]),
      ),
      (
          'unique_string',
          rt([['a', 'a', 'b'], ['a', 'b', 'a', 'b', 'c'], ['b', 'c', 'b']]),
          rt([['a', 'b'], ['a', 'b', 'c'], ['b', 'c']]),
      ),
  ])
  def testImplementation(self, input_value, expected_result):
    result: tf.RaggedTensor = ops.ragged_unique(input_value)
    result = tf.RaggedTensor.from_row_splits(
        result.values, result.row_splits, validate=True
    )
    self.assertAllEqual(expected_result.values, result.values)
    self.assertAllEqual(expected_result.row_splits, result.row_splits)


class ParallelRaggedUniqueTest(RaggedUniqueTest):
  IMPLEMENTATION = 'parallel'


# copybara:uncomment_begin (test for internal implementation of ext_ops)
# class CustomRaggedUniqueTest(RaggedUniqueTest):
#   IMPLEMENTATION = 'custom'
# copybara:uncomment_end


class RaggedLookupTest(ExtOpsTestBase, parameterized.TestCase):

  @parameterized.parameters(
      # pylint: disable=g-complex-comprehension
      (v, s)
      for v in [tf.int32, tf.int64, tf.float32, tf.double, tf.string]
      for s in [tf.int32, tf.int64]
      # pylint: enable=g-complex-comprehension
  )
  def testSupportedTypes(self, values_dtype, splits_dtype):
    values = tf.RaggedTensor.from_row_lengths(
        tf.zeros([1], dtype=values_dtype),
        tf.convert_to_tensor([1], dtype=splits_dtype),
    )
    result: tf.RaggedTensor = ops.ragged_lookup(values, values)
    expected = tf.RaggedTensor.from_row_lengths(
        tf.zeros([1], dtype=splits_dtype),
        tf.convert_to_tensor([1], dtype=splits_dtype),
    )
    self.assertAllEqual(expected.values, result.values)
    self.assertAllEqual(expected.row_splits, result.row_splits)

  @parameterized.named_parameters([
      ('empty', rt([], tf.int32, 1), rt([], tf.int32, 1), rt([], tf.int32, 1)),
      ('empty_rows', rt([[], [], []]), rt([[], [], []]), rt([[], [], []])),
      (
          'empty_values',
          rt([[], []], tf.int32, 1),
          rt([[1, 2], [2, 3]], tf.int32, 1),
          rt([[], []], tf.int32, 1),
      ),
      ('single_int', rt([[1]]), rt([[1]]), rt([[0]])),
      ('single_string', rt([['a']]), rt([['a']]), rt([[0]])),
      (
          'single_row',
          rt([['a', 'a', 'b', 'a', 'b']]),
          rt([['b', 'c', 'a']]),
          rt([[2, 2, 0, 2, 0]]),
      ),
      (
          'multiple_rows1',
          rt([[2, 2, 4, 1, 3], [], [], [5, 4, 3, 2], []]),
          rt([[1, 2, 3, 4], [1], [1, 2], [2, 3, 4, 5], []]),
          rt([[1, 1, 3, 0, 2], [], [], [3, 2, 1, 0], []]),
      ),
      (
          'multiple_rows2',
          rt([['a', 'b'], ['b', 'c'], ['d', 'e'], ['f', 'g']]),
          rt([['a', 'b', 'c'], ['a', 'b', 'c'], ['e', 'd', 'c'], ['g', 'f']]),
          rt([[0, 1], [1, 2], [1, 0], [1, 0]]),
      ),
      (
          'multiple_rows3',
          rt([[], [], [], [2, 2, 4, 1, 3], [1, 1, 1], [], [5, 4, 3, 2]]),
          rt([[], [], [1], [1, 4, 2, 3], [1], [1, 2], [2, 4, 3, 5]]),
          rt([[], [], [], [2, 2, 1, 0, 3], [0, 0, 0], [], [3, 1, 2, 0]]),
      ),
  ])
  def testImplementation(self, values, vocabulary, expected_row_based):
    result = ops.ragged_lookup(values, vocabulary, global_indices=False)
    self.assertAllEqual(expected_row_based.values, result.values)
    self.assertAllEqual(expected_row_based.row_splits, result.row_splits)

    result = ops.ragged_lookup(values, vocabulary, global_indices=True)
    expected_global = expected_row_based + tf.expand_dims(
        tf.cast(vocabulary.row_starts(), expected_row_based.dtype), axis=-1
    )
    self.assertAllEqual(expected_global.values, result.values)
    self.assertAllEqual(expected_global.row_splits, result.row_splits)

  @parameterized.named_parameters([
      ('single_int', rt([[1]], tf.int32, 1), rt([[]], tf.int32, 1)),
      ('single_string', rt([['x']]), rt([['b']])),
      (
          'single_row1',
          rt([['x']]),
          rt([['b', 'c', 'a']]),
      ),
      (
          'single_row2',
          rt([['x', 'a', 'b', 'a', 'b', 'x']]),
          rt([['b', 'c', 'a']]),
      ),
      (
          'multiple_rows1',
          rt([[2, 2, 4, 1, 3], [], [], [5, 4, 3, -1], []]),
          rt([[1, 2, 3, 4], [1], [1, 2], [2, 3, 4, 5], []]),
      ),
      (
          'multiple_rows2',
          rt([['a', 'b'], ['b', 'c'], ['x', 'e'], ['f', 'g']]),
          rt([['a', 'b', 'c'], ['a', 'b', 'c'], ['e', 'd', 'c'], ['g', 'f']]),
      ),
      (
          'multiple_rows3',
          rt([['x', 'b'], ['b', 'c'], ['d', 'e'], ['f', 'g']]),
          rt([['a', 'b', 'c'], ['a', 'b', 'c'], ['e', 'd', 'c'], ['g', 'f']]),
      ),
  ])
  def testRaisesOnOOV(self, values, vocabulary):
    for global_indices in [True, False]:
      with self.assertRaisesRegex(
          tf.errors.InvalidArgumentError, 'Out of vocabulary values'
      ):
        ops.ragged_lookup(values, vocabulary, global_indices=global_indices)

  @parameterized.named_parameters([
      ('single_int', rt([[]], tf.int32, 1), rt([[1, 1]], tf.int32, 1)),
      ('single_string', rt([['x']]), rt([['x', 'x']])),
      (
          'single_row1',
          rt([['b']]),
          rt([['x', 'b', 'x']]),
      ),
      (
          'single_row2',
          rt([[]], dtype=tf.string, ragged_rank=1),
          rt([['x', 'c', 'a', 'x']]),
      ),
      (
          'multiple_rows1',
          rt([[2], [], [], [5], []]),
          rt([[-1, 2, 3, 4, -1], [1], [1, 2], [2, 3, 4, 5], []]),
      ),
      (
          'multiple_rows2',
          rt([['a', 'b'], ['b', 'c'], ['e'], []]),
          rt([['a', 'b', 'c'], ['a', 'b', 'c'], ['x', 'e', 'x', 'x'], []]),
      ),
      (
          'multiple_rows3',
          rt([['a', 'b'], ['b', 'c'], ['d', 'e'], []]),
          rt([['a', 'b', 'c'], ['a', 'b', 'c'], ['e', 'd', 'c'], ['x', 'x']]),
      ),
  ])
  def testRaisesOnInvalidVocabulary(self, values, vocabulary):
    for global_indices in [True, False]:
      with self.assertRaisesRegex(
          tf.errors.InvalidArgumentError,
          'Vocabulary has repeated values in rows',
      ):
        ops.ragged_lookup(values, vocabulary, global_indices=global_indices)


class ParallelRaggedLookupTest(RaggedLookupTest):
  IMPLEMENTATION = 'parallel'


# copybara:uncomment_begin (test for internal implementation of ext_ops)
# class CustomRaggedLookupTest(RaggedLookupTest):
#   IMPLEMENTATION = 'custom'
# copybara:uncomment_end

if __name__ == '__main__':
  tf.test.main()
