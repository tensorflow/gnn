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
"""Hyperparameter search spaces for Vizier studies.

This file defines search spaces for hyperparameter tuning of the Model Template
"Albis" with https://github.com/google/vizier. End-to-end models
built with `MtAlbisGraphUpdate` can use this to configure and launch a Vizier
study and the training runs for its trials. It's up to them how to forward
Vizier params to the training script and its use of MtAlbis. The parameter names
set here for Vizier match the keyword arguments in the Python modeling code.

For each search space definition, this file has a function

```
add_params_<name>(search_space)
```

that modifies `search_space` in-place by adding parameters and returns `None`.
"""

from vizier.service import pyvizier as vz


def add_params_regularization(
    search_space: vz.SearchSpace, *, prefix: str = "") -> None:
  """Adds params for a study of regularization strength.

  Args:
    search_space: a `pyvizier.SearchSpace` that is changed in-place by adding
      `dropout_rate` and `l2_regularization`.
    prefix: a prefix added to param names.
  """
  # The params in `root` apply to all trials in the Vizier study.
  # go/pyvizier also lets you add conditional params.
  root = search_space.root
  root.add_discrete_param(
      prefix + "state_dropout_rate", [.1, .2, .3],
      scale_type=vz.ScaleType.LINEAR)
  root.add_float_param(
      prefix + "l2_regularization", 1e-6, 1e-4,
      scale_type=vz.ScaleType.LOG)


def add_params_mt_albis(
    search_space: vz.SearchSpace, *, prefix: str = "",
    use_attention=False) -> None:
  """Adds params for the Model Template without attention.

  Args:
    search_space: a `pyvizier.SearchSpace` that is changed in-place.
    prefix: a prefix added to param names.
    use_attention: if true, fixes param value `attention_type="multi_head"`,
      instead of the default `"none"`.
  """
  root = search_space.root
  if not use_attention:
    root.add_categorical_param(
        prefix + "attention_type", ["none"])
    root.add_categorical_param(
        prefix + "simple_conv_reduce_type",
        ["mean", "mean|sum", "mean|max", "mean|sum|max"],
        default_value="mean|sum")
  else:
    root.add_categorical_param(
        prefix + "attention_type", ["multi_head"])
    root.add_discrete_param(
        prefix + "attention_num_heads", [2, 4, 8],
        default_value=8, scale_type=vz.ScaleType.LINEAR)
  root.add_discrete_param(
      prefix + "edge_dropout_rate", [.0, .1, .2, .3, .5, .8],
      default_value=.0, scale_type=vz.ScaleType.LINEAR)
  root.add_discrete_param(
      prefix + "state_dropout_rate", [.0, .1, .2, .3, .5, .8],
      default_value=.1, scale_type=vz.ScaleType.LINEAR)
  root.add_discrete_param(
      prefix + "l2_regularization", [1e-6, 3e-6, 1e-5, 3e-5, 1e-4],
      scale_type=vz.ScaleType.LOG)
  root.add_categorical_param(
      prefix + "next_state_type", ["dense", "residual"],
      default_value="dense")
