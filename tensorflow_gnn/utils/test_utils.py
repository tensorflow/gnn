"""Test utilities.
"""

from os import path
from typing import Any

from google.protobuf import text_format


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
  # pylint: disable=unreachable
  filename = path.join(find_root(__file__), filename)
  if not path.exists(filename):
    raise OSError("Resource {} does not exist.".format(filename))
  return filename


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
  # pylint: disable=unreachable
  res_dirname = path.join(find_root(dirname), dirname)
  if not path.isdir(res_dirname):
    raise OSError(
        "Resource path {} does not exist or is not a directory".format(
            res_dirname))
  return res_dirname


def find_root(start_filename: str) -> str:
  """Return root directory of repository."""
  dirname = start_filename
  while not path.exists(path.join(dirname, "LICENSE")):
    prevname = dirname
    dirname = path.dirname(dirname)
    if dirname == prevname:
      raise ValueError(f"Could not find root at {start_filename}")
  return dirname
