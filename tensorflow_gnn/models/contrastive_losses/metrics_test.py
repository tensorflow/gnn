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

import tensorflow as tf
from tensorflow_gnn.models.contrastive_losses import metrics


class MetricsTest(tf.test.TestCase):

  def test_self_clustering_incorrect_rank(self):
    with self.assertRaisesRegex(ValueError, r"Expected 2D tensor \(got .*\)"):
      _ = metrics.self_clustering(tf.ones((1, 1, 1)))

  def test_self_clustering_value(self):
    """Verifies outputs in a simple case."""
    x = tf.convert_to_tensor(
        [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
    )
    metric_value = metrics.self_clustering(x)
    # In this case, batch_size = 4, feature_dim = 2.
    # expected value is 4 + 4 * 3 / 2 = 10.
    # The correlation matrix is:
    # [1, 0, 0, -1]
    # [0, 1, -1, 0]
    # [0, -1, 1, 0]
    # [-1, 0, 0, 1]
    # with the sum of the squares equal to 8.
    # Thus, the metric value shoud be close to (8 - 10) / (4 * 4 - 10) = -1/3.
    self.assertAllClose(metric_value, -1 / 3.0)

    # Test the collapsed case.
    x = tf.ones((2, 4))
    metric_value = metrics.self_clustering(x)
    self.assertAllClose(metric_value, 1.0)

  def test_pseudo_condition_number_value(self):
    # Check metric value for collapsed representations.
    x = tf.convert_to_tensor([[1.0, 1.0], [1.0, 1.0]])
    metric_value = metrics.pseudo_condition_number(x)
    self.assertAllClose(metric_value, 0.0)

    # Verify a hand-constructed example. In this case, right singular vectors
    # are [1, 0] and [0, 1] respectively.
    x = tf.convert_to_tensor(
        [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
    )
    metric_value = metrics.pseudo_condition_number(x)
    self.assertAllClose(metric_value, 1.0)


if __name__ == "__main__":
  tf.test.main()
