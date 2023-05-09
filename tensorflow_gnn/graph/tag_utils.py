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
"""Utilities related to the IncidentNodeTag values."""

from __future__ import annotations
from typing import Optional, Sequence, Union

from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt


NodeSetName = const.NodeSetName
EdgeSetName = const.EdgeSetName
IncidentNodeTag = const.IncidentNodeTag
IncidentNodeOrContextTag = const.IncidentNodeOrContextTag
GraphTensorSpec = gt.GraphTensorSpec


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


def get_edge_or_node_set_name_args_for_tag(
    spec: GraphTensorSpec,
    tag: IncidentNodeOrContextTag,
    *,
    edge_set_name: Union[Sequence[EdgeSetName], EdgeSetName, None] = None,
    node_set_name: Union[Sequence[NodeSetName], NodeSetName, None] = None,
    function_name: str = "This operation"
) -> tuple[Optional[Sequence[EdgeSetName]],
           Optional[Sequence[NodeSetName]],
           bool]:
  """Returns canonicalized edge/node args for broadcast() and pool().
  
  Args:
    spec: The spec of the `GraphTensor` the args are for.
    tag: An `IncidentNodeTag` or the special value `tfgnn.CONTEXT`.
    edge_set_name: An edge set name or a sequence of these. Must be set if
        `node_set_name` is unset.
    node_set_name: A edge set name or a sequence of these. Must only be set
       with `tag=tfgnn.CONTEXT`. Mutually exclusive with `edge_set_name`.
    function_name: Optionally, the user-visible name of the function whose args
      are processed.

  Returns:
    Tuple `(edge_set_names, node_set_names, got_sequence_args)`
    with exactly one of `edge_set_names, node_set_names` being a list and
    the other being `None`. `got_sequence_args` is set to False if original
    non-sequence args have been converted to lists of length 1.

  Raises:
    ValueError: if not exactly one of edge_set_name, node_set_name is set.
    ValueError: if node_set_name is set for a `tag != tfgnn.CONTEXT`.
    ValueError: if the given edge_set_names have different endpoints at
      the given `tag != tfgnn.CONTEXT`.
    ValueError: if not exactly one of feature_value, feature_name is set.
  """
  if tag == const.CONTEXT:
    num_names = bool(edge_set_name is None) + bool(node_set_name is None)
    if num_names != 1:
      raise ValueError(
          f"{function_name} with tag tfgnn.CONTEXT requires to pass exactly "
          f"one of edge_set_name, node_set_name but got {num_names}.")
  else:
    if not isinstance(tag, const.IncidentNodeTag):
      raise ValueError(
          f"{function_name} got tag {tag} but requires either "
          "the special value tfgnn.CONTEXT "
          "or a valid IndicentNodeTag, that is, tfgnn.SOURCE, tfgnn.TARGET, "
          "or another valid integer in case of hypergraphs.")
    if edge_set_name is None or node_set_name is not None:
      raise ValueError(
          f"{function_name} requires to pass edge_set_name, "
          "not node_set_name, unless tag is set to tfgnn.CONTEXT.")

  got_sequence_args = not (isinstance(node_set_name, str) or
                           isinstance(edge_set_name, str))
  edge_set_names = _get_nonempty_name_list_or_none(
      function_name, "edge_set_name", edge_set_name)
  node_set_names = _get_nonempty_name_list_or_none(
      function_name, "node_set_name", node_set_name)

  if tag != const.CONTEXT:
    incident_node_set_names = {
        spec.edge_sets_spec[e].adjacency_spec.node_set_name(tag)
        for e in edge_set_names}
    if len(incident_node_set_names) > 1:
      raise ValueError(
          f"{function_name} requires the same endpoint for all named edge sets "
          f"but got node sets {incident_node_set_names} from tag={tag} and "
          f"edge_set_name={edge_set_name}")

  return edge_set_names, node_set_names, got_sequence_args


def _get_nonempty_name_list_or_none(
    function_name: str,
    arg_name: str,
    name: Union[Sequence[str], str, None]) -> Optional[Sequence[str]]:
  if name is None:
    return None
  if isinstance(name, str):
    return [name]
  if len(name) > 0:  # Crash here if len() doesn't work.  pylint: disable=g-explicit-length-test
    return name
  raise ValueError(
      f"{function_name} requires {arg_name} to be a non-empty Sequence or None")
