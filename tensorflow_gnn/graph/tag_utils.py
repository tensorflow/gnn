"""Utilities related to the IncidentNodeTag values."""

from tensorflow_gnn.graph import graph_constants as const


def reverse_tag(tag):
  """Flips tfgnn.SOURCE to tfgnn.TARGET and vice versa."""
  if tag == const.TARGET:
    return const.SOURCE
  elif tag == const.SOURCE:
    return const.TARGET
  else:
    raise ValueError(
        f"Expected tag tfgnn.SOURCE ({const.SOURCE}) "
        f"or tfgnn.TARGET ({const.TARGET}), got: {tag}")
