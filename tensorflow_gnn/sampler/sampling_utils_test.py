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
"""Tests for sampling_utils."""

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
from tensorflow_gnn.sampler import sampling_utils as utils

PCollection = beam.PCollection


class TestBalancedLookupJoin(parameterized.TestCase):

  @parameterized.named_parameters(
      ("empty", [], [], []),
      ("left_empty", [("a", 1)], [], []),
      ("right_empty", [], [("a", 2)], []),
      ("single_value", [("a", 1)], [("a", 2)], [("a", (1, 2))]),
      ("no_match", [("a", 1)], [("b", 2)], []),
      ("inner_join", [(2, "y"), (3, "z")], [(1, "a"),
                                            (2, "b")], [(2, ("y", "b"))]),
      ("multiple", [(1, "x"), (2, "y")], [(1, "a"),
                                          (2, "b")], [(1, ("x", "a")),
                                                      (2, ("y", "b"))]),
      ("composite_keys", [((1, "x"), "Q")], [((1, "x"), "W")], [((1, "x"),
                                                                 ("Q", "W"))]),
      ("composite_values", [(1, ["x", "Q"])], [(1, ["x", "W"])], [
          (1, (["x", "Q"], ["x", "W"]))
      ]),
  )
  def test_logic(self, queries, values, expected_result):
    with beam.Pipeline() as root:
      queries = root | "Queries" >> beam.Create(queries)
      values = root | "Values" >> beam.Create(values)
      actual_result = utils.balanced_inner_lookup_join("test", queries, values)

      util.assert_that(actual_result, util.equal_to(expected_result))

  @parameterized.parameters(1, 2, 3, 5, 10, 1000)
  def test_sharding(self, num_shards: int):
    with beam.Pipeline() as root:
      n_samples = 100
      queries = [(i % 7, i) for i in range(n_samples)]
      queries.append((-1, -1))
      values = [(i, str(i)) for i in range(7)]
      values.append((-2, str(-2)))
      queries = root | "Queries" >> beam.Create(queries)
      values = root | "Values" >> beam.Create(values)
      actual_result = utils.balanced_inner_lookup_join(
          "test", queries, values, num_shards=num_shards)
      expected_result = [(i % 7, (i, str(i % 7))) for i in range(n_samples)]
      util.assert_that(actual_result, util.equal_to(expected_result))


class TestUniqueValuesCombiner(parameterized.TestCase):

  def _matcher(self, expected):

    def _sorted(values):
      result = [(k, sorted(v)) for k, v in values]
      return sorted(result, key=lambda item: item[0])

    sorted_expected = _sorted(expected)

    def _equal(actual):
      sorted_actual = _sorted(actual)
      if sorted_expected != sorted_actual:
        raise util.BeamAssertException("Failed assert: %r == %r" %
                                       (sorted_expected, sorted_actual))

    return _equal

  @parameterized.named_parameters(
      ("empty", [], []),
      ("distinct1", [(1, ["1"])], [(1, ["1"])]),
      ("distinct2", [(1, ["1"]), (2, ["1", "2"])], [(1, "1"), (2, ["1", "2"])]),
      ("equal", [(1, ["1"]), (1, ["1"])], [(1, ["1"])]),
      ("merging", [("1", [1]), ("2", [1]), ("1", [1, 2]), ("2", [2]),
                   ("1", [2, 3]), ("2", [1])], [("1", [1, 2, 3]),
                                                ("2", [1, 2])]),
  )
  def test_logic(self, values, expected_result):

    with beam.Pipeline() as root:
      values = root | "Values" >> beam.Create(values)
      actual_result = values | "Combine" >> beam.CombinePerKey(
          utils.unique_values_combiner)

      util.assert_that(actual_result, self._matcher(expected_result))

  @parameterized.parameters(5, 10, 100)
  def test_max_result_size_satisfied(self, max_result_size: int):
    input_values = [("x", [i % 5]) for i in range(100)] + [("y", [1])]

    with beam.Pipeline() as root:
      values = root | "Values" >> beam.Create(input_values)
      actual_result = values | "Combine" >> beam.CombinePerKey(
          utils.unique_values_combiner, max_result_size=max_result_size)

      util.assert_that(actual_result,
                       self._matcher([("x", [0, 1, 2, 3, 4])] + [("y", [1])]))

  @parameterized.parameters(0, 1, 2, 5)
  def test_max_result_size_violated(self, max_result_size: int):
    input_values = [("x", [i]) for i in range(6)] + [("y", [1])]

    def run():
      with beam.Pipeline() as root:
        values = root | "Values" >> beam.Create(input_values)
        _ = values | "Combine" >> beam.CombinePerKey(
            utils.unique_values_combiner, max_result_size=max_result_size)

    self.assertRaisesRegex(ValueError, f"larger than {max_result_size}", run)


if __name__ == "__main__":
  absltest.main()
