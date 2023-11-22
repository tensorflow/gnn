# Copyright 2022 The TensorFlow GNN Authors. All Rights Reserved.
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
"""Tests for item_dropout.py."""

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.keras.layers import item_dropout
from tensorflow_gnn.utils import tf_test_utils as tftu


class ItemDropoutTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("Rank0", 0, tftu.ModelReloading.SKIP),
      ("Rank1", 1, tftu.ModelReloading.SKIP),
      ("Rank2", 2, tftu.ModelReloading.SKIP),
      ("Rank1Restored", 1, tftu.ModelReloading.SAVED_MODEL),
      ("Rank1RestoredKeras", 1, tftu.ModelReloading.KERAS),
      ("Rank1Seeded", 1, tftu.ModelReloading.SKIP, 123))
  def test(self, feature_rank, model_reloading, seed=None):
    # Avoid flakiness.
    tf.random.set_seed(42)

    rate = 1/3
    inputs = tf.keras.layers.Input([None] * feature_rank)
    seed_kwarg = dict(seed=seed) if seed is not None else {}
    outputs = item_dropout.ItemDropout(rate=rate, **seed_kwarg)(inputs)
    model = tf.keras.Model(inputs, outputs)

    model = tftu.maybe_reload_model(self, model, model_reloading,
                                    "item-dropout-model")

    num_items = 30
    x = tf.ones([num_items] + [5] * feature_rank, dtype=tf.float32)

    # Non-training behavior.
    y = model(x)
    self.assertAllEqual(x, y)
    y = model(x, training=False)
    self.assertAllEqual(x, y)

    # Training behavior.
    y = model(x, training=True)
    self.assertShapeEqual(x, y)
    # In each row, all entries have the same constant value.
    feature_axes = list(range(1, feature_rank + 1))
    row_mins = tf.reduce_min(y, axis=feature_axes)
    row_maxs = tf.reduce_max(y, axis=feature_axes)
    self.assertAllEqual(row_mins, row_maxs)
    # Some rows have value 0.0 (dropped out), some rows have 1.5 (scaled up).
    # The risk of not seeing any dropouts is (1-rate)**num_items < 1e-5,
    # each time a new seed is fixed above or TensorFlow changes its RNG.
    global_min = tf.reduce_min(row_mins)
    global_max = tf.reduce_max(row_maxs)
    self.assertEqual(0.0, global_min)
    self.assertEqual(1.5, global_max)

    # Seed arg is forwarded properly (after call and build).
    if tftu.is_keras_model_reloading(model_reloading):
      self.assertEqual(seed, model.get_layer(index=1)._dropout.seed)


if __name__ == "__main__":
  tf.test.main()
