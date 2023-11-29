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
"""Tests for loading a TensorFlow GNN saved model."""
import sys

import tensorflow as tf


def get_export_dir() -> str:
  """Return the export directory."""
  export_dir = "runner/utils/saved_model_testdata/"
  return export_dir  # copybara:comment(Retrieves data deps in Bazel/OSS only)
  # Placeholder for Google-internal directory fetch


class SavedModelTest(tf.test.TestCase):

  def test_saved_model(self):
    # This test expects TF-GNN to have _not_ been imported: it verifies the
    # ability to load TF-GNN models without importing `tensorflow_gnn`.
    self.assertNotIn("tensorflow_gnn", sys.modules)

    saved_model = tf.saved_model.load(get_export_dir())

    source = tf.random.uniform([16], 0, 4, dtype=tf.int32)
    target = tf.random.uniform([16], 0, 4, dtype=tf.int32)
    hidden_state = tf.constant([
        [8.0, 1.0, 9.0, 1.0],
        [0.8, 0.1, 0.9, 0.1],
        [8.0, 1.0, 9.0, 1.0],
        [0.8, 0.1, 0.9, 0.1],
    ])

    results = saved_model.signatures["serving_default"](
        source=source,
        target=target,
        hidden_state=hidden_state)
    self.assertLen(results, 1)

    [actual] = results.values()
    self.assertAllClose(actual, [[10.450001, 10.450001]])

if __name__ == "__main__":
  tf.test.main()
