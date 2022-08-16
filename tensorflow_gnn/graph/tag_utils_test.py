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
"""Tests for tag_utils."""

from absl.testing import absltest
from absl.testing import parameterized

from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import tag_utils


class ReverseTagTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("Source", const.SOURCE, const.TARGET),
      ("Target", const.TARGET, const.SOURCE))
  def test(self, tag, expected):
    actual = tag_utils.reverse_tag(tag)
    self.assertEqual(expected, actual)

  def testError(self):
    with self.assertRaisesRegex(ValueError, r"Expected tag .*got: 3"):
      _ = tag_utils.reverse_tag(3)


if __name__ == "__main__":
  absltest.main()
