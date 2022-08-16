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
"""Test utilities.
"""

from os import path
from typing import Any

from google.protobuf import text_format
# Google-internal import(s).


def get_proto_resource(filename: str, message: Any) -> Any:
  """Parse a relative filename's contents as text-proto."""
  with open(get_resource(filename)) as pbfile:
    return text_format.Parse(pbfile.read(), message)


def get_resource(filename: str) -> str:
  """Return a local resource filename for testing.

  This function abstracts away between a Bazel build with resources and direct
  file access.

  Args:
    filename: A path relative to the root of this directory whose contents to
      fetch.
  Returns:
    An absolute path to a filename.
  Raises:
    OSError: If the resource filename does not exist.
  """
  return filename  # copybara:comment(Retrieves data deps in Bazel/OSS only)
  # Placeholder for Google-internal file fetch


def get_resource_dir(dirname: str) -> str:
  """Return a local resource directory for testing.

  This function abstracts away between a Bazel build with resources and direct
  file access.

  Args:
    dirname: A path relative to the root of this directory whose contents to
      fetch.
  Returns:
    An absolute path to a dirname.
  Raises:
    OSError: If the resource filename does not exist.
  """
  return dirname  # copybara:comment(Retrieves data deps in Bazel/OSS only)
  # Placeholder for Google-internal directory fetch
