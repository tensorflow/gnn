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

from typing import List, Union

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_gnn.experimental.sampler import proto as pb
from tensorflow_gnn.experimental.sampler.beam import utils

from google.protobuf import text_format

PCollection = beam.PCollection

rt = tf.ragged.constant
dt = tf.convert_to_tensor


def to_numpy(value: Union[tf.Tensor, tf.RaggedTensor]) -> utils.Value:
  result = []
  if isinstance(value, tf.Tensor):
    result.append(value.numpy())
  elif isinstance(value, tf.RaggedTensor):
    result.append(value.flat_values.numpy())
    for row_length, dim in zip(value.nested_row_lengths(), value.shape[1:]):
      if dim is None:
        result.append(row_length.numpy())
  else:
    raise ValueError("Unsupported type: %s" % type(value))
  return result


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


class StackingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      (
          "dense",
          [dt([], tf.int32), dt([1]), dt([2, 3]), dt([4, 5, 6])],
          rt([[], [1], [2, 3], [4, 5, 6]], row_splits_dtype=tf.int32),
      ),
      (
          "dense-rank-2",
          [dt([[1.0, 2.0], [3.0, 4.0]]), dt([[5.0, 6.0], [7.0, 8.0]])],
          rt(
              [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
              ragged_rank=1,
          ),
      ),
      (
          "ragged-rank-1",
          [rt([[1, 2], [3]]), rt([[], []]), rt([[4, 5, 6]])],
          rt([[[1, 2], [3]], [[], []], [[4, 5, 6]]]),
      ),
      (
          "ragged-rank-2",
          [rt([[[1], [2]], [[3]]]), rt([[[4, 5], [6]]])],
          rt([[[[1], [2]], [[3]]], [[[4, 5], [6]]]]),
      ),
  ])
  def test_stack_ragged(self, values, expected_result):
    result = utils.stack_ragged(
        tf.nest.map_structure(to_numpy, values),
        utils.get_np_dtype(expected_result.row_splits.dtype),
    )
    tf.nest.map_structure(
        self.assertAllEqual, result, to_numpy(expected_result)
    )

  @parameterized.named_parameters([
      (
          "single_value",
          [dt(1)],
          dt([1]),
      ),
      (
          "scalar",
          [dt(1), dt(2), dt(3)],
          dt([1, 2, 3]),
      ),
      (
          "vector",
          [dt([1.0, 2.0]), dt([3.0, 4.0])],
          dt([[1.0, 2.0], [3.0, 4.0]]),
      ),
      (
          "matrix",
          [dt([[b"a"]]), dt([[b"b"]])],
          dt([[[b"a"]], [[b"b"]]]),
      ),
  ])
  def test_stack_dense(self, values, expected_result):
    result = utils.stack(
        tf.nest.map_structure(to_numpy, values),
    )
    tf.nest.map_structure(
        self.assertAllEqual, result, to_numpy(expected_result)
    )


class ParseTfExampleTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      (
          "scalar",
          """
          features {
            feature { key: 'i' value { int64_list {  value: [42] } } }
          }
          """,
          "i",
          """
          tensor { dtype: DT_INT32 }
          """,
          dt(42),
      ),
      (
          "vector",
          """
          features {
            feature { key: 'f' value { float_list {  value: [1., 2., 3.] } } }
          }
          """,
          "f",
          """
          tensor {
              dtype: DT_FLOAT
              shape { dim { size: 3 } }
          }
          """,
          dt([1.0, 2.0, 3.0]),
      ),
      (
          "var-size",
          """
          features {
            feature { key: 'f' value { float_list {  value: [1., 2.] } } }
          }
          """,
          "f",
          """
          tensor {
              dtype: DT_FLOAT
              shape { dim { size: -1 } }
          }
          """,
          dt([1.0, 2.0]),
      ),
      (
          "matrix",
          """
          features {
            feature { key: 'b' value { bytes_list {  value: ['a', 'b'] } } }
            feature { key: 'f' value { float_list {  value: [1., 2., 3.] } } }
          }
          """,
          "b",
          """
          tensor {
              dtype: DT_STRING
              shape { dim { size: 1 }  dim { size: 2 } }
          }
          """,
          dt([[b"a", b"b"]]),
      ),
  ])
  def test_dense(
      self, example_pbtxt: str, name: str, spec_pbtxt: str, expected_result
  ):
    example = tf.train.Example()
    text_format.Parse(example_pbtxt, example)
    spec = pb.ValueSpec()
    text_format.Parse(spec_pbtxt, spec)

    result = utils.parse_tf_example(example, name, spec)
    self.assertAllEqual(result, to_numpy(expected_result))

  @parameterized.named_parameters([
      (
          "rank-1",
          """
          features {
            feature {
              key: 'i'
              value {
                int64_list {
                  value: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                }
              }
            }
            feature { key: 'i.d1' value { int64_list {  value: [2, 1, 1] } } }
          }
          """,
          "i",
          """
          ragged_tensor {
              dtype: DT_INT32
              shape { dim { size: -1 } dim { size: -1 } dim { size: 3 } }
              ragged_rank: 1
              row_splits_dtype: DT_INT32
          }
          """,
          rt(
              [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]],
              ragged_rank=1,
              row_splits_dtype=tf.int32,
          ),
      ),
      (
          "rank-2-uniform",
          """
          features {
            feature {
              key: 'i'
              value { float_list {  value: [1., 2., 3., 4., 5., 6.] } }
            }
            feature {
              key: 'i.d2'
              value { int64_list {  value: [1, 2, 2, 1] } }
            }
          }
          """,
          "i",
          """
          ragged_tensor {
              dtype: DT_FLOAT
              shape { dim { size: 2 } dim { size: 2 } dim { size: -1 }}
              ragged_rank: 2
              row_splits_dtype: DT_INT64
          }
          """,
          tf.RaggedTensor.from_uniform_row_length(
              rt([[1.0], [2.0, 3.0], [4.0, 5.0], [6.0]], ragged_rank=1), 2
          ),
      ),
  ])
  def test_ragged(
      self, example_pbtxt: str, name: str, spec_pbtxt: str, expected_result
  ):
    example = tf.train.Example()
    text_format.Parse(example_pbtxt, example)
    spec = pb.ValueSpec()
    text_format.Parse(spec_pbtxt, spec)

    result = utils.parse_tf_example(example, name, spec)
    tf.nest.map_structure(
        self.assertAllEqual, result, to_numpy(expected_result)
    )

  def test_raises_on_invalid_name(self):
    example = tf.train.Example()
    text_format.Parse(
        """
          features {
            feature { key: 'foo' value { int64_list {  value: [42] } } }
          }
          """,
        example,
    )
    spec = pb.ValueSpec()
    text_format.Parse("tensor { dtype: DT_INT64 }", spec)

    with self.assertRaisesRegex(
        ValueError, 'Expected feature "bar" is missing'
    ):
      utils.parse_tf_example(example, "bar", spec)

  def test_raises_on_invalid_type(self):
    example = tf.train.Example()
    text_format.Parse(
        """
          features {
            feature { key: 'foo' value { int64_list {  value: [1, 2] } } }
          }
          """,
        example,
    )
    spec = pb.ValueSpec()
    text_format.Parse("tensor { dtype: DT_FLOAT }", spec)

    with self.assertRaisesRegex(ValueError, 'Expected float feature for "foo"'):
      utils.parse_tf_example(example, "foo", spec)

  def test_raises_on_invalid_shape(self):
    example = tf.train.Example()
    text_format.Parse(
        """
          features {
            feature { key: 'foo' value { float_list {  value: [1., 2.] } } }
          }
          """,
        example,
    )
    spec = pb.ValueSpec()
    text_format.Parse(
        "tensor { dtype: DT_FLOAT shape { dim { size: 64 } } }", spec
    )

    with self.assertRaisesRegex(
        ValueError, 'requires 64 elements for "foo", actual 2'
    ):
      utils.parse_tf_example(example, "foo", spec)


if __name__ == "__main__":
  absltest.main()
