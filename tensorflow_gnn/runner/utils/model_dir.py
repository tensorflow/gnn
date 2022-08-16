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
"""Model directory methods."""
import os

import tensorflow as tf


def incrementing_model_dir(dirname: str, start: int = 0) -> str:
  """Create, given some `dirname`, an incrementing model directory.

  Args:
    dirname: The base directory name.
    start: The starting integer.

  Returns:
    A model directory `dirname/n` where 'n' is the maximum integer in `dirname`.
  """
  if not tf.io.gfile.isdir(dirname):
    return os.path.join(dirname, str(start))
  files = tf.io.gfile.listdir(dirname)
  integers = [int(f) for f in files if f.isdigit()]
  return os.path.join(dirname, str(max(integers) + 1 if integers else start))
