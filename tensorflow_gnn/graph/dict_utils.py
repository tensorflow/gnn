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
