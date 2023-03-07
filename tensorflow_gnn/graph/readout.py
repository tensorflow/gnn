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
"""Defines `tfgnn.readout_named()` and supporting functions."""

import re
from typing import Dict, Optional, Sequence, Tuple

import tensorflow as tf

from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import graph_tensor_ops as ops


def validate_graph_tensor_spec_for_readout(
    graph_spec: gt.GraphTensorSpec,
    required_keys: Optional[Sequence[str]] = None,
    *,
    readout_node_set: const.NodeSetName = "_readout") -> None:
  """Checks `graph_spec` supports `readout_named()` from `required_keys`.

  This function checks that the `graph.spec` of a `tfgnn.GraphTensor` contains
  correctly connected auxiliary graph pieces (edge sets and node sets) such that
  subsequent calls to `tfgnn.readout_named(graph, key)` can work for all
  readout keys encoded in the spec. The argument `required_keys` can
  be set to check that these particular values of `key` are present.

  This function only considers the `tfgnn.GraphTensorSpec`, which means it can
  be handled completely at the time of tracing a `tf.function` for non-eager
  execution. To also check the index values found in actual `tfgnn.GraphTensor`
  values, call `tfgnn.validate_graph_tensor_for_readout()`; preferably during
  dataset preprocessing, as it incurs a runtime cost for every input.

  Args:
    graph_spec: The graph tensor spec to check. Must be scalar, that is, have
      shape [].
    required_keys: Can be set to a list of readout keys that are required to be
      provided by the spec.
    readout_node_set: The name of the auxiliary node set for readout, which is
      "_readout" by default. This name is also used as a prefix for the
      auxiliary edge sets connected to it.

  Raises:
    ValueError: if the auxiliary graph pieces for readout are malformed.
    KeyError: if any of the `required_keys` is missing.
  """
  gt.check_scalar_graph_tensor(graph_spec,
                               "tfgnn.validate_graph_tensor_spec_for_readout()")
  _ = get_validated_edge_set_map_for_readout(
      graph_spec, required_keys, readout_node_set=readout_node_set)


def validate_graph_tensor_for_readout(
    graph: gt.GraphTensor,
    required_keys: Optional[Sequence[str]] = None,
    *,
    readout_node_set: const.NodeSetName = "_readout") -> gt.GraphTensor:
  """Checks `graph` supports `readout_named()` from `required_keys`.

  This function checks that a `tfgnn.GraphTensor` contains correctly connected
  auxiliary graph pieces (edge sets and node sets) for `tfgnn.readout_named()`.
  It does all the checks of `tfgnn.validate_graph_tensor_spec_for_readout()`.
  Additionally, it checks that the actual tensor values (esp. node indices)
  are valid for `tfgnn.readout_named(graph, key)` for each `key` in
  `required_keys`. If `required_keys` is unset, all keys provided in the graph
  structure are checked.

  Args:
    graph: The graph tensor to check.
    required_keys: Can be set to a list of readout keys to check. If unset,
      checks all keys provided by the graph.
    readout_node_set: A non-standard name passed to `tfgnn.readout_named()`,
      if any.

  Returns:
    The input GraphTensor, unchanged. This helps to put the tf.debugging.assert*
    ops in this function into a dependency chain.

  Raises:
    ValueError: if the auxiliary graph pieces for readout are malformed in the
      GraphTensorSpec.
    KeyError: if any of the `required_keys` is missing.
    tf.errors.InvalidArgumentError: If values in the GraphTensor, notably
      node indices of auxiliary edge sets, are incorrect.
  """
  gt.check_scalar_graph_tensor(graph.spec,
                               "tfgnn.validate_graph_tensor_for_readout()")
  edge_set_map = get_validated_edge_set_map_for_readout(
      graph.spec, required_keys, readout_node_set=readout_node_set)
  readout_size = graph.node_sets[readout_node_set].total_size
  if required_keys is not None:
    keys = required_keys
  else:
    keys = edge_set_map.keys()

  assert_ops = []
  for key in keys:
    # Validate target indices of aux edge sets.
    target_indices_list = []
    for edge_set_name in edge_set_map[key].values():
      target_indices = graph.edge_sets[edge_set_name].adjacency[const.TARGET]
      tf.debugging.assert_less(
          target_indices[:-1], target_indices[1:],
          f"Not strictly sorted by target: '{edge_set_name}'")
      target_indices_list.append(target_indices)
    all_target_indices = tf.sort(tf.concat(target_indices_list, axis=0))
    assert_ops.append(tf.debugging.assert_equal(
        all_target_indices, tf.range(0, readout_size),
        "Target indices not equal to range(readout_size)",
        summarize=100))
    # Validate shadow node sets for readout from edge sets.
    for set_type, set_name in edge_set_map[key].keys():
      if set_type == const.EDGES:
        source_edge_set = graph.edge_sets[set_name]
        shadow_node_set = graph.node_sets[_SHADOW_PREFIX + set_name]
        assert_ops.append(tf.debugging.assert_equal(source_edge_set.sizes,
                                                    shadow_node_set.sizes))

  with tf.control_dependencies(assert_ops):
    return tf.identity(graph)


def readout_named(
    graph: gt.GraphTensor,
    key: str,
    *,
    feature_name: str,
    readout_node_set: const.NodeSetName = "_readout",
    validate: bool = True) -> const.Field:
  """Reads out a feature value from select nodes (or edges) in a graph.

  This helper function addresses the need to read out final hidden states
  from a GNN computation to make predictions for some nodes (or edges)
  of interest. Its typical usage looks as follows:

  ```python
  graph = ...  # Run your GNN here.
  seed_node_states = tfgnn.readout_named(graph, "seed",
                                         feature_name=tfgnn.HIDDEN_STATE)
  ```

  ...where `"seed"` is an arbitrary key. There can be multiple of those. For
  example, a link prediction model could read out `"source"` and `"target"` node
  states from the graph.

  The graph uses auxiliary edge sets to encode the subset of nodes from which
  values are read out. It is the responsibility of the graph data creation code
  to put these into place.

  Suppose all `"seed"` nodes come from node set `"users"`. Then this example
  requires an auxiliary edge set called `"_readout/seed"` with source node set
  `"users"` and target node set `"_readout"`, such that the target node indices
  form a sorted(!) sequence `[0, 1, ..., n-1]` up to the size `n` of node set
  `"_readout"`. The `seed_node_states` returned are the result of passing the
  `tfgnn.HIDDEN_STATE` features along the edges from `"users"` nodes to distinct
  `"_readout"` nodes. The number of readout results per graph component is
  given by the sizes field of the `"_readout"` node set, but can vary between
  graphs.

  Advanced users may need to vary the source node set of readout. That is
  possible by adding multiple auxiliary edge sets with names
  `"_readout/seed/" + unique_suffix`. In each node set, the target node ids
  must be sorted, and together they reach each `"_readout"` node exactly once.

  Very advanced users may need to read out features from edge sets instead of
  node sets. To read out from edge set `"links"`, create an auxiliary node set
  `"_shadow/links"` with the same sizes field as the edge set but no features
  of its own. When `"_shadow/links"` occurs as the source node set of an
  auxiliary node set like `"_readout/seed"`, features are taken from edge set
  `"links"` instead.

  Args:
    graph: A scalar GraphTensor with the auxiliary graph pieces described above.
    key: A string key to select between possibly multiple named readouts
      (such as `"source"` and `"target"` for link prediction).
    feature_name: The name of a feature that is present on the node set(s)
      (or edge set(s)) referenced by the auxiliary edge sets. The feature
      must have shape `[num_items, *feature_dims]` with the same `feature_dims`
      on all graph pieces, and the same dtype.
    readout_node_set: A string, defaults to `"_readout"`. This is used as the
      name for the readout node set and as a name prefix for its edge sets.
    validate: Setting this to false disables the validity checks for the
      auxiliary edge sets. This is stronlgy discouraged, unless great care is
      taken to run `tfgnn.validate_graph_tensor_for_readout()` earlier on
      structurally unchanged GraphTensors.

  Returns:
    A tensor of shape `[readout_size, *feature_dims]` with the read-out feature
    values.
  """
  gt.check_scalar_graph_tensor(graph.spec, "tfgnn.readout_named()")
  if validate:
    graph = validate_graph_tensor_for_readout(
        graph, [key], readout_node_set=readout_node_set)

  # Broadcast from sources onto the readout edge sets.
  edge_set_map = get_validated_edge_set_map_for_readout(
      graph.spec, [key], readout_node_set=readout_node_set)
  broadcast_values = {}
  for (set_type, set_name), edge_set_name in edge_set_map[key].items():
    if set_type == const.EDGES:
      source_value = graph.edge_sets[set_name][feature_name]
    else:  # NODES
      source_value = graph.node_sets[set_name][feature_name]
    broadcast_values[edge_set_name] = ops.broadcast_node_to_edges(
        graph, edge_set_name, const.SOURCE, feature_value=source_value)

  # Special case: single readout edge set.
  if len(broadcast_values) == 1:
    [single_value] = broadcast_values.values()
    # The target node ids form a range [0, 1, ..., n-1] (validated above).
    # Hence the values on the edge set are indexed the same as on the node set.
    return single_value

  # General case: multiple readout edge sets, which require pooling to the
  # readout node set.
  # The code below uses sum-pooling along each edge set and adds the results,
  # which aligns with standard GNN operations but requires a numeric dtype.
  # TODO(b/269076334): Support all dtypes, notably string. Options include:
  # - Concat target indices, argsort, and gather.
  # - Chain tf.tensor_scatter_nd_update() calls.
  dtypes = {value.dtype for value in broadcast_values.values()}
  if len(dtypes) > 1:
    raise ValueError(
        "Conflicing feature dtypes found by readout_named(..., '{key}', "
        f"feature_name='{feature_name}', readout_node_set='{readout_node_set}')"
        f": expected just one, found {dtypes}.")
  dtype = dtypes.pop()
  if not (dtype.is_floating or dtype.is_integer):
    raise NotImplementedError(
        "b/269076334: readout_named() from multiple sources does not support "
        "non-numeric dtypes yet, because it relies on sum-pooling.")

  pooled_values = [
      ops.pool_edges_to_node(
          graph, edge_set_name, const.TARGET, "sum", feature_value=value)
      for edge_set_name, value in broadcast_values.items()]
  result = tf.math.add_n(pooled_values)
  return result


# _SHADOW_PREFIX + edge_set_name is used as the name of an auxiliary node set
# that has the same sizes as the named edge set and appears as the source
# of an auxiliary edge set for readout from the named edge set.
_SHADOW_PREFIX = "_shadow/"


def get_validated_edge_set_map_for_readout(
    graph_spec: gt.GraphTensorSpec,
    required_keys: Optional[Sequence[str]] = None,
    *,
    readout_node_set: const.NodeSetName = "_readout",
) -> Dict[str, Dict[Tuple[str, const.SetName], const.EdgeSetName]]:
  """Returns a nested dict of edge set names for readout, after validating it.

  The `edge_set_map` returned by this function is indexed like

  ```
  edge_set_name = edge_set_map[key][(set_type, set_name)]
  ```

  where

    * `key` is the readout key for use in `tfgnn.readout_named(..., key)`
      for which the named edge set facilitates readout,
    * `set_type` is either `tfgnn.NODES` or `tfgnn.EDGES`, and
    * `set_name` is the name of the node set or edge set from which
      the named edge set facilitates readout.

  If this function returns instead of raising a ValueError, it is guaranteed
  that

    * there is a readout node set with the name `readout_node_set`,
    * all edge set names starting with `readout_node_set` followed by `"/"`
      can be parsed into a readout key followed by an optional unique suffix,
    * their target node set is `readout_node_set`, and
    * the readout keys found in the spec contain all `required_keys` if set,
      or else at least one readout key.

  Args:
    graph_spec: A scaler graph tensor spec.
    required_keys: Can be set to a list of readout keys that are required to be
      provided by the spec.
    readout_node_set: A non-standard name prefix passed to
      `tfgnn.readout_named()`, if any.

  Returns:
    A dict of dicts, as described above.

  Raises:
    ValueError: if the auxiliary graph pieces for readout are malformed.
    KeyError: if any of the `required_keys` is missing.
  """
  if readout_node_set not in graph_spec.node_sets_spec:
    raise ValueError(
        f"GraphTensor lacks auxiliary node set '{readout_node_set}'. "
        f"Existing node sets are: {', '.join(graph_spec.node_sets_spec)}")

  edge_set_map = {}
  edge_set_prefix = f"{readout_node_set}/"
  for edge_set_name in graph_spec.edge_sets_spec:
    if not edge_set_name.startswith(edge_set_prefix):
      continue
    match = re.fullmatch(r"(?P<key>\w+)(/.+)?",
                         edge_set_name[len(edge_set_prefix):])
    if match is None:
      raise ValueError(f"Malformed auxiliary edge set name: '{edge_set_name}'")
    key = match["key"]
    adjacency_spec = graph_spec.edge_sets_spec[edge_set_name].adjacency_spec
    target_node_set = adjacency_spec.node_set_name(const.TARGET)
    if target_node_set != readout_node_set:
      raise ValueError(
          f"Malformed auxiliary edge set '{edge_set_name}': Expected "
          f"target node set '{readout_node_set}' but found {target_node_set}.")
    source_node_set = adjacency_spec.node_set_name(const.SOURCE)
    if source_node_set.startswith(_SHADOW_PREFIX):
      set_type = const.EDGES
      set_name = source_node_set[len(_SHADOW_PREFIX):]
      if set_name not in graph_spec.edge_sets_spec:
        raise ValueError(
            f"GraphTensor lacks edge set '{set_name}' referenced by "
            f"auxiliary edge set '{edge_set_name}'")
    else:
      set_type = const.NODES
      set_name = source_node_set

    if key not in edge_set_map:
      edge_set_map[key] = {}
    if (set_type, set_name) in edge_set_map[key]:
      raise ValueError(
          f"Duplicate readout edge set from {set_type} '{set_name}': "
          f"'{edge_set_name}' vs '{edge_set_map[key][(set_type, set_name)]}'")
    edge_set_map[key][(set_type, set_name)] = edge_set_name

  if required_keys is not None:
    missing_keys = [key for key in required_keys if key not in edge_set_map]
    if missing_keys:
      raise KeyError(
          f"Unknown readout keys {missing_keys}: No edge sets found with "
          f"name prefix '{readout_node_set}/{missing_keys[0]}' and so on.")
  else:
    if not edge_set_map:
      raise ValueError(
          f"Found node set '{readout_node_set}' but no edge sets for readout "
          "into it")

  return edge_set_map
