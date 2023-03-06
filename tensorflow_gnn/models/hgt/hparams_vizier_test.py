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
"""Tests for hparams_vizier."""

from absl.testing import absltest
from tensorflow_gnn.models.hgt import hparams_vizier

from vizier.service import pyvizier as vz


class HparamsVizierTest(absltest.TestCase):

  def test_regularization(self):
    problem = vz.ProblemStatement()
    hparams_vizier.add_params_regularization(
        problem.search_space, prefix="foo."
    )
    self.assertCountEqual(
        [p.name for p in problem.search_space.parameters], ["foo.dropout_rate"]
    )

  def test_hgt_attention(self):
    problem = vz.ProblemStatement()
    hparams_vizier.add_params_attention(
        problem.search_space, prefix="foo."
    )
    self.assertCountEqual(
        [p.name for p in problem.search_space.parameters], ["foo.num_heads"]
    )


if __name__ == "__main__":
  absltest.main()
