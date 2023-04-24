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
"""Hyperparameter search spaces for Vizier studies.

This file defines search spaces for hyperparameter tuning of the GATv2 model
architecture with https://github.com/google/vizier. End-to-end models built with
GATv2 can use this to configure and launch a Vizier study and the training runs
for its trials. It's up to them how to forward Vizier params to the training
script and its use of GATv2. The parameter names set here for Vizier match the
keyword arguments in the Python modeling code.

For each search space definition, this file has a function

```
add_params_<name>(search_space)
```

that modifies `search_space` in-place by adding parameters and returns `None`.
"""

from vizier.service import pyvizier as vz


def add_params_regularization(search_space: vz.SearchSpace,
                              *, prefix: str = "")-> None:
  """Adds params for a study of regularization strength.

  Args:
    search_space: a `pyvizier.SearchSpace` that is changed in-place by adding
      `state_dropout_rate`, `edge_dropout_rate` and `l2_regularization`.
    prefix: a prefix added to param names.
  """
  # The params in `root` apply to all trials in the Vizier study.
  # go/pyvizier also lets you add conditional params.
  root = search_space.root
  root.add_discrete_param(
      prefix + "state_dropout_rate", [.1, .2, .3],
      scale_type=vz.ScaleType.LINEAR)
  root.add_discrete_param(
      prefix + "edge_dropout_rate", [.1, .2, .3],
      scale_type=vz.ScaleType.LINEAR)
  root.add_float_param(
      prefix + "l2_regularization", 1e-6, 1e-4,
      scale_type=vz.ScaleType.LOG)


def add_params_attention(search_space: vz.SearchSpace,
                         *, prefix: str = "")-> None:
  """Adds params for a study of attention configurations.

  Args:
    search_space: a `pyvizier.SearchSpace` that is changed in-place by adding
      `num_heads`.
    prefix: a prefix added to param names.
  """
  # The params in `root` apply to all trials in the Vizier study.
  # go/pyvizier also lets you add conditional params.
  root = search_space.root
  root.add_discrete_param(
      prefix + "num_heads", [4, 8, 16, 32],
      scale_type=vz.ScaleType.LINEAR)
