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
