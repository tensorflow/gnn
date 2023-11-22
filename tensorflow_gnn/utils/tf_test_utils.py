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
"""Test utilities for TensorFlow code."""

import enum
import os
from typing import Any

import tensorflow as tf


class ModelReloading(int, enum.Enum):
  """Controls how to save and reload a model for further testing.

  This enum parameterizes the behavior of `maybe_reload_model()` and unit tests
  that use it.

  The available values are:

    * `SKIP`: The model object is returned unchanged. Testing it validates the
      expectations on the model as a baseline for all other forms of reloading.
    * `SAVED_MODEL`: The Keras model is saved with traced tf.functions and
      restored by `tf.saved_model.load()` as a low-level TensorFlow object.
      This has become uncommon with end users, but this utility supports it
      to maintain existing test coverage for it.
    * `KERAS`: The Keras model is saved in `save_format="tf"` (using
      `save_traces=False`) and restored with `tf.keras.models.load_model()`
      to recreate an actual Keras model. This requires all layers to be
      registered as serializable and to implement `Layer.get_config()`.
      This utility function asserts that all Layers have been recreated with
      their exact type (no fallback to traces).
      Unit tests commonly use this way of reloading to test the end-to-end
      effect of `get_config()` and `from_config()` for the Layers in the model.
    * `KERAS_V3`: The Keras model is saved in `save_format="keras"` and
      restored with `tf.keras.models.load_model()`. At this time (Nov 2023),
      this does mostly **not** work for TF-GNN. If running under TensorFlow
      older than 2.13, this is unsupported and the test is skipped.
  """
  SKIP = 0
  SAVED_MODEL = 1
  KERAS = 2
  KERAS_V3 = 3


def is_keras_model_reloading(model_reloading: ModelReloading) -> bool:
  """Returns whether maybe_reload_model() will return a tf.keras.Model."""
  if model_reloading in [ModelReloading.SKIP, ModelReloading.KERAS,
                         ModelReloading.KERAS_V3]:
    return True
  if model_reloading in [ModelReloading.SAVED_MODEL]:
    return False
  raise ValueError(f"Unknown ModelReloading enum: {model_reloading}")


def maybe_reload_model(test_case: tf.test.TestCase, model: tf.keras.Model,
                       model_reloading: ModelReloading, basename: str) -> Any:
  """Returns the model round-tripped through serialization.

  This is a utility for writing unit tests derived from `tf.test.TestCase` in
  the following style to check that a model restored into Python behaves like
  the original model in certain ways.

  NOTE: Do not use this to test exporting to SavedModel for inference.
  Restoring into Python like this is an experimental approach intended to
  support transfer learning.

  TODO(b/312176241): Cite the forthcoming guide on model saving.

  ```
  from absl.testing import parameterized
  import tensorflow as tf
  from tensorflow_gnn.utils import tf_test_utils as tftu

  class MyModelTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        (tftu.ModelReloading.SKIP,),
        (tftu.ModelReloading.SAVED_MODEL,),
        (tftu.ModelReloading.KERAS,))
    def testModelSaving(self, reload_model):
      model = tf.keras.Model(...)
      model = tftu.maybe_reload_model(self, model, reload_model, 'my-model')
      actual = model(...)
      expected = ...
      self.assertAllClose(expected, actual)
  ```

  Args:
    test_case: The TestCase object from which this utility is called.
    model: A tf.keras.Model.
    model_reloading: A `ModelReloading` enum value, see its docstring for
      the available values and their meaning.
    basename: A string that becomes part of the saved model's filename, to
      make it more discoverable when inspecting the test's temp_dir.

  Returns:
    A callable with a signature like the original `model`. Details vary
    according to the `reload_model` type.
  """

  if model_reloading == ModelReloading.SKIP:
    return model

  filepath = os.path.join(test_case.get_temp_dir(), basename)
  if model_reloading == ModelReloading.SAVED_MODEL:
    model.save(filepath, save_format="tf", include_optimizer=False,
               save_traces=True)
    return tf.saved_model.load(filepath)

  if model_reloading == ModelReloading.KERAS:
    model.save(filepath, save_format="tf", include_optimizer=False,
               save_traces=False)
    restored = tf.keras.models.load_model(filepath)
    # Check that from_config() worked without fallback to a function trace.
    test_case.assertSequenceEqual([type(l) for l in model.layers],
                                  [type(l) for l in restored.layers])
    return restored

  if model_reloading == ModelReloading.KERAS_V3:
    if tf.__version__.startswith("2.12."):
      test_case.skipTest("Skipping test of save_format='keras_v3' for TF <2.13")
    filepath += ".keras"
    model.save(filepath, save_format="keras_v3")
    return tf.keras.models.load_model(filepath)

  raise ValueError(f"Unknown ModelReloading enum: {model_reloading}")
