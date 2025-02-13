# Copyright 2025 The TensorFlow GNN Authors. All Rights Reserved.
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
"""Utils for writing unit tests with the Runner."""

import dataclasses
from unittest import mock

from tensorflow_gnn.runner import interfaces


def mock_run_result(**kwargs):
  """Returns a mock RunResult with (possibly a subset of) fields set by kwargs.

  Unit tests of code that consumes a `runner.RunResult` can use this helper to
  avoid setting irrelevant fields (incl. future ones) to objects of proper type.

  Args:
    **kwargs: Field names and values to set on the result. Field names must be
      valid fields of `runner.RunResult`, but not all fields need to be set.
      Accessing unset fields of the mock will raise an AssertionError.
  """
  all_fields = set(field.name
                   for field in dataclasses.fields(interfaces.RunResult))
  if unknown_fields := kwargs.keys() - all_fields:
    raise TypeError(
        "Bad mock set-up: got keyword arguments not corresponding to fields of "
        "RunResult: " + ", ".join(f"'{name}'" for name in unknown_fields))

  result = mock.NonCallableMock(spec_set=list(all_fields))
  result.configure_mock(**kwargs)
  for name in all_fields - kwargs.keys():
    setattr(type(result), name, mock.PropertyMock(side_effect=AssertionError(
        f"Unexpected access to field RunResult.{name}. "
        "(If it is expected, give it a value in the mock RunResult.)"
    )))
  return result
