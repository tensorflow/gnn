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

import enum
import os

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.keras.layers import item_dropout


class ReloadModel(int, enum.Enum):
  """Controls how to reload a model for further testing after saving."""
  SKIP = 0
  SAVED_MODEL = 1
  KERAS = 2


class ItemDropoutTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("Rank0", 0, ReloadModel.SKIP),
      ("Rank1", 1, ReloadModel.SKIP),
      ("Rank2", 2, ReloadModel.SKIP),
      ("Rank1Restored", 1, ReloadModel.SAVED_MODEL),
      ("Rank1RestoredKeras", 1, ReloadModel.KERAS),
      ("Rank1Seeded", 1, ReloadModel.SKIP, 123))
  def test(self, feature_rank, reload_model, seed=None):
    # Avoid flakiness.
    tf.random.set_seed(42)

    rate = 1/3
    inputs = tf.keras.layers.Input([None] * feature_rank)
    seed_kwarg = dict(seed=seed) if seed is not None else {}
    outputs = item_dropout.ItemDropout(rate=rate, **seed_kwarg)(inputs)
    model = tf.keras.Model(inputs, outputs)

    model = self._maybe_reload_model(model, reload_model, "item-dropout-model")

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
    if reload_model != ReloadModel.SAVED_MODEL:
      self.assertEqual(seed, model.get_layer(index=1)._dropout.seed)

  def _maybe_reload_model(self, model: tf.keras.Model,
                          reload_model: ReloadModel, subdir_name: str):
    if reload_model == ReloadModel.SKIP:
      return model
    export_dir = os.path.join(self.get_temp_dir(), subdir_name)
    model.save(export_dir, include_optimizer=False)
    if reload_model == ReloadModel.SAVED_MODEL:
      return tf.saved_model.load(export_dir)
    elif reload_model == ReloadModel.KERAS:
      restored = tf.keras.models.load_model(export_dir)
      # Check that from_config() worked, no fallback to a function trace, see
      # https://www.tensorflow.org/guide/keras/save_and_serialize#how_savedmodel_handles_custom_objects
      for i in range(len(model.layers)):
        self.assertIsInstance(restored.get_layer(index=i),
                              type(model.get_layer(index=i)))
      return restored


if __name__ == "__main__":
  tf.test.main()
