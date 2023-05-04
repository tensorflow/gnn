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
"""Tests for utils."""

from typing import List

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util

import numpy as np
import tensorflow as tf

from tensorflow_gnn.experimental.sampler.beam import utils

PCollection = beam.PCollection

rt = tf.ragged.constant


class TestSafeLookupJoin(parameterized.TestCase):

  @parameterized.named_parameters(
      ("empty", [], [], []),
      ("empty_values", [("a", 1)], [], [("a", (1, None))]),
      ("empty_queries", [], [("a", 2)], []),
      ("single_value", [("a", 1)], [("a", 2)], [("a", (1, 2))]),
      ("no_value", [(b"a", 1)], [(b"b", 2)], [(b"a", (1, None))]),
      (
          "left_join",
          [(2, "y"), (3, "z")],
          [(1, "a"), (2, "b")],
          [(2, ("y", "b")), (3, ("z", None))],
      ),
      (
          "all_present",
          [(1, "x"), (2, "y")],
          [(1, "a"), (2, "b")],
          [(1, ("x", "a")), (2, ("y", "b"))],
      ),
      (
          "repeated_queries",
          [(1, "X"), (1, "X"), (2, "Y"), (2, "Z")],
          [(1, 1.0), (2, 2.0)],
          [(1, ("X", 1.0)), (1, ("X", 1.0)), (2, ("Y", 2.0)), (2, ("Z", 2.0))],
      ),
      (
          "composite_values",
          [(1, ["x", "Qx"]), (2, ["y", "Qy"]), (2, ["y", "Qy"])],
          [(1, ["x", "V"])],
          [
              (1, (["x", "Qx"], ["x", "V"])),
              (2, (["y", "Qy"], None)),
              (2, (["y", "Qy"], None)),
          ],
      ),
  )
  def test_logic(self, queries, values, expected_result):
    with beam.Pipeline() as root:
      queries = root | "Queries" >> beam.Create(queries)
      values = root | "Values" >> beam.Create(values)
      actual_result = (queries, values) | "Join" >> utils.SafeLeftLookupJoin()

      util.assert_that(actual_result, util.equal_to(expected_result))


class RaggedSliceTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      ("empty_vector", np.zeros([0]), 0, 0, np.zeros([0])),
      ("empty_vector_slice", np.array([1, 2]), 0, 0, np.zeros([0], np.int32)),
      ("vector1", np.array([1, 2, 3, 4]), 0, 1, np.array([1])),
      ("vector2", np.array([1, 2, 3, 4]), 1, 3, np.array([2, 3])),
      ("vector3", np.array(["a", "b", "c"]), 1, 3, np.array(["b", "c"])),
      ("vector4", np.array([1.0, 2.0, 3.0]), 0, 3, np.array([1.0, 2.0, 3.0])),
      ("empty_matrix", np.zeros([0, 3]), 0, 0, np.zeros([0, 3])),
      (
          "empty_matrix_slice",
          np.array([[1, 2], [2, 3]]),
          1,
          1,
          np.zeros([0, 2], np.int32),
      ),
      ("matrix1", np.array([[1, 2], [3, 4], [5, 6]]), 0, 1, np.array([[1, 2]])),
      (
          "matrix2",
          np.array([[1, 2], [3, 4], [5, 6]]),
          0,
          2,
          np.array([[1, 2], [3, 4]]),
      ),
      ("matrix3", np.array([[1, 2], [3, 4], [5, 6]]), 2, 3, np.array([[5, 6]])),
      ("matrix4", np.array([[1, 2], [3, 4]]), 0, 3, np.array([[1, 2], [3, 4]])),
      (
          "large",
          np.array([[1, 2, 3]] * 100),
          50,
          100,
          np.array([[1, 2, 3]] * 50),
      ),
  ])
  def test_dense_slice(
      self,
      value: np.ndarray,
      start: int,
      limit: int,
      expected: np.ndarray,
  ):
    value = [value]
    expected = [expected]
    actual = utils.ragged_slice(value, start, limit)
    tf.nest.map_structure(self.assertAllEqual, actual, expected)

  @parameterized.named_parameters([
      ("empty", rt([], ragged_rank=1), 0, 0, rt([], ragged_rank=1)),
      ("empty_slice", rt([[1, 2], [3]]), 1, 1, rt([], ragged_rank=1)),
      ("rank1_1", rt([[], [1], [2, 3]]), 0, 1, rt([[]], ragged_rank=1)),
      ("rank1_2", rt([[], [1], [2, 3]]), 1, 2, rt([[1]])),
      ("rank1_3", rt([[], [1], [2, 3]]), 1, 3, rt([[1], [2, 3]])),
      ("rank1_4", rt([[], [1], [2, 3]]), 0, 3, rt([[], [1], [2, 3]])),
      (
          "rank1_5",
          rt([[[1.0, 2.0]], [[2.0, 3.0], [3.0, 4.0]]], ragged_rank=1),
          0,
          1,
          rt([[[1.0, 2.0]]], ragged_rank=1),
      ),
      (
          "rank2_1",
          rt([[["a", "b"], ["c"]], [], [], [["c"]]]),
          0,
          1,
          rt([[["a", "b"], ["c"]]]),
      ),
      (
          "rank2_2",
          rt([[["a", "b"], ["c"]], [], [], [["c"]]]),
          2,
          4,
          rt([[], [["c"]]]),
      ),
      (
          "rank3",
          rt([[[[1], [2]], [[3]]], [[[4], [5, 6]]]], ragged_rank=3),
          1,
          2,
          rt([[[[4], [5, 6]]]], ragged_rank=3),
      ),
      (
          "large",
          rt([[["x", "y"]]] * 100),
          50,
          100,
          rt([[["x", "y"]]] * 50),
      ),
  ])
  def test_ragged_slice(
      self,
      value: tf.RaggedTensor,
      start: int,
      limit: int,
      expected: tf.RaggedTensor,
  ):
    def as_value(r: tf.RaggedTensor) -> List[np.ndarray]:
      return tf.nest.map_structure(
          lambda t: t.numpy(), [r.flat_values, *r.nested_row_lengths()]
      )

    value = as_value(value)
    expected = as_value(expected)
    actual = utils.ragged_slice(value, start, limit)
    tf.nest.map_structure(self.assertAllEqual, actual, expected)


if __name__ == "__main__":
  absltest.main()
