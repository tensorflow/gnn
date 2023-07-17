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
"""Tests for unsupervised metrics."""
from __future__ import annotations
from typing import Callable
from absl.testing import parameterized

import tensorflow as tf
from tensorflow_gnn.models.contrastive_losses import metrics


class MetricsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      metrics.self_clustering,
      metrics.pseudo_condition_number,
      metrics.numerical_rank,
  )
  def test_incorrect_rank(self, metric: Callable[[tf.Tensor], tf.Tensor]):
    with self.assertRaisesRegex(ValueError, r"Expected 2D tensor \(got .*\)"):
      _ = metric(tf.ones((1, 1, 1)))

  @parameterized.named_parameters([
      dict(
          testcase_name="self_clustering_precomupted",
          # In this case, batch_size = 4, feature_dim = 2.
          # expected value is 4 + 4 * 3 / 2 = 10.
          # The correlation matrix is:
          # [1, 0, 0, -1]
          # [0, 1, -1, 0]
          # [0, -1, 1, 0]
          # [-1, 0, 0, 1]
          # with the sum of the squares equal to 8. Thus, the metric value shoud
          # be close to (8 - 10) / (4 * 4 - 10) = -1/3.
          metric_fn=metrics.self_clustering,
          inputs=tf.convert_to_tensor(
              [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
          ),
          expected=-1 / 3.0,
      ),
      dict(
          testcase_name="pseudo_condition_number_constructed",
          # Verify a hand-constructed example. In this case, right singular
          # vectors are [1, 0] and [0, 1] respectively.
          metric_fn=metrics.pseudo_condition_number,
          inputs=tf.convert_to_tensor(
              [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
          ),
          expected=1.0,
      ),
      dict(
          testcase_name="numerical_rank_constructed",
          # In this case, both singular values equal to 2.
          # X @ X^T equals to
          # [2, 0, 0, -2]
          # [0, 2, -2, 0]
          # [0, -2, 2, 0]
          # [-2, 0, 0, 2]
          # with the trace equal to 8.
          # Thus, the metric value shoud be close to 8 / 2 / 2 = 2.
          metric_fn=metrics.numerical_rank,
          inputs=tf.convert_to_tensor(
              [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
          ),
          expected=2.0,
      ),
      dict(
          testcase_name="rankme_constructed",
          # In this case, singular values equal to [2, 2].
          # p_ks equal to 1 / 2, meaning rankme = 2.
          metric_fn=metrics.rankme,
          inputs=tf.convert_to_tensor(
              [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
          ),
          expected=2.0,
      ),
      dict(
          testcase_name="coherence_constructed",
          # In this case, both singular values equal to 2.
          # Left singular vectors of the inputs are
          # [-1/2, 1/2]
          # [-1/2, -1/2]
          # [1/2, 1/2]
          # [1/2, -1/2]
          # with the maximum row norm equal to sqrt(2)/2.
          # Thus, the metric value shoud be close to 1/2 * 4 / 2 = 1.
          metric_fn=metrics.coherence,
          inputs=tf.convert_to_tensor(
              [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
          ),
          expected=1.0,
      ),
      dict(
          testcase_name="self_clustering_collapsed",
          metric_fn=metrics.self_clustering,
          inputs=tf.ones((2, 4)),
          expected=1.0,
      ),
      dict(
          testcase_name="pseudo_condition_number_collapsed",
          metric_fn=metrics.pseudo_condition_number,
          inputs=tf.ones((2, 2)),
          expected=0.0,
      ),
      dict(
          testcase_name="numerical_rank_collapsed",
          metric_fn=metrics.numerical_rank,
          inputs=tf.ones((4, 2)),
          expected=1.0,
      ),
      dict(
          testcase_name="rankme_collapsed",
          metric_fn=metrics.rankme,
          inputs=tf.ones((4, 2)),
          expected=1.0,
      ),
      dict(
          testcase_name="coherence_collapsed",
          metric_fn=metrics.coherence,
          inputs=tf.ones((20, 2)),
          expected=10.0,
      ),
      dict(
          testcase_name="numerical_rank_allzero",
          metric_fn=metrics.numerical_rank,
          inputs=tf.zeros((2, 2)),
          expected=0.0,
      ),
      dict(
          testcase_name="rankme_allzero",
          metric_fn=metrics.rankme,
          inputs=tf.zeros((2, 2)),
          expected=1.0,
      ),
  ])
  def test_value(self, metric_fn, inputs, expected):
    actual = metric_fn(inputs)
    self.assertAllClose(actual, expected)


if __name__ == "__main__":
  tf.test.main()
