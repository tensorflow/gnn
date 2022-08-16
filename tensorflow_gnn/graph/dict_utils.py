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
"""Utilities for Python dictionaries."""

from typing import Any, Dict, Mapping, MutableMapping


def with_key_prefix(d: Mapping[str, Any], key_prefix: str) -> Dict[str, Any]:
  """Returns {key_prefix+k: v for k, v in d.items()}."""
  return {key_prefix+k: v for k, v in d.items()}


def pop_by_prefix(
    d: MutableMapping[str, Any], key_prefix: str) -> Dict[str, Any]:
  """Returns {k: v for key_prefix+k, v in d.items()} and removes them from d."""
  popped = {}
  for key in list(d.keys()):
    if key.startswith(key_prefix):
      popped[key[len(key_prefix):]] = d.pop(key)
  return popped
