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
"""Defines `tfgnn.structured_readout()` and supporting functions."""

import re
from typing import Dict, Optional, Sequence, Tuple

import tensorflow as tf

from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import broadcast_ops
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import pool_ops


def validate_graph_tensor_spec_for_readout(
    graph_spec: gt.GraphTensorSpec,
    required_keys: Optional[Sequence[str]] = None,
    *,
    readout_node_set: const.NodeSetName = "_readout") -> None:
  """Checks `graph_spec` supports `structured_readout()` from `required_keys`.

  This function checks that the `graph.spec` of a `tfgnn.GraphTensor` contains
  correctly connected auxiliary graph pieces (edge sets and node sets) such that
  subsequent calls to `tfgnn.structured_readout(graph, key)` can work for all
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
  """Checks `graph` supports `structured_readout()` from `required_keys`.

  This function checks that a `tfgnn.GraphTensor` contains correctly connected
  auxiliary graph pieces (edge sets and node sets) for structured readout.
  It does all the checks of `tfgnn.validate_graph_tensor_spec_for_readout()`.
  Additionally, it checks that the actual tensor values (esp. node indices)
  are valid for `tfgnn.structured_readout(graph, key)` for each `key` in
  `required_keys`. If `required_keys` is unset, all keys provided in the graph
  structure are checked.

  Args:
    graph: The graph tensor to check.
    required_keys: Can be set to a list of readout keys to check. If unset,
      checks all keys provided by the graph.
    readout_node_set: Optionally, a non-default name for use as
      `tfgnn.structured_readout(..., readout_node_set=...)`.

  Returns:
    The input GraphTensor, unchanged. This helps to put `tf.debugging.assert*`
    ops from this function into a dependency chain.

  Raises:
    ValueError: if the auxiliary graph pieces for readout are malformed in the
      `tfgnn.GraphTensorSpec`.
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


def structured_readout(
    graph: gt.GraphTensor,
    key: str,
    *,
    feature_name: str,
    readout_node_set: const.NodeSetName = "_readout",
    validate: bool = True) -> const.Field:
  """Reads out a feature value from select nodes (or edges) in a graph.

  This function implements "structured readout", that is, the readout of final
  hidden states of a GNN computation from a distinguished subset of nodes
  (or edges) in a `tfgnn.GraphTensor` into a freestanding `tf.Tensor`, by
  moving them along one (or more) auxiliary edge sets and collecting them
  in one auxiliary node set stored in the `tfgnn.GraphTensor`. Collectively,
  that auxiliary node set and the edge sets into it are called a "readout
  structure". It is the responsibility of the graph data creation code
  to create this structure.

  A typical usage of structured readout looks as follows:

  ```python
  graph = ...  # Run your GNN here.
  seed_node_states = tfgnn.structured_readout(graph, "seed",
                                              feature_name=tfgnn.HIDDEN_STATE)
  ```

  ...where `"seed"` is a key defined by the readout structure. There can be
  multiple of those. For example, a link prediction model could read out
  `"source"` and `"target"` node states from the graph. It is on the dataset
  to document which keys it provides.

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

  Note that this function returns a tensor shaped like a feature of the
  `"_readout"` node set, not a modified GraphTensor.
  See `tfgnn.structured_readout_into_feature()` for a function that returns a
  GraphTensor with the readout result stored as a feature on the `"_readout"`
  node set. To retrieve a feature `"ft"` that is stored on the `"_readout"`
  node set, do not use either of these, but access it as usual with
  `GraphTensor.node_sets["_readout"]["ft"]`.

  Args:
    graph: A scalar GraphTensor with a readout structure composed of auxiliary
      graph pieces as described above.
    key: A string key to select between possibly multiple named readouts
      (such as `"source"` and `"target"` for link prediction).
    feature_name: The name of a feature that is present on the node set(s)
      (or edge set(s)) referenced by the auxiliary edge sets. The feature
      must have shape `[num_items, *feature_dims]` with the same `feature_dims`
      on all graph pieces, and the same dtype.
    readout_node_set: The name for the readout node set and the name prefix for
      its edge sets. Permissible values are `"_readout"` (the default) and
      `f"_readout:{tag}"` where `tag` matches `[a-zA-Z0-9_]+`.
      Setting this to a different value allows to select between multiple
      independent readout structures in the same graph.
    validate: Setting this to false disables the validity checks for the
      auxiliary edge sets. This is stronlgy discouraged, unless great care is
      taken to run `tfgnn.validate_graph_tensor_for_readout()` earlier on
      structurally unchanged GraphTensors.

  Returns:
    A tensor of shape `[readout_size, *feature_dims]` with the read-out feature
    values.
  """
  gt.check_scalar_graph_tensor(graph.spec, "tfgnn.structured_readout()")
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
    broadcast_values[edge_set_name] = broadcast_ops.broadcast_node_to_edges(
        graph, edge_set_name, const.SOURCE, feature_value=source_value)

  # Special case: single readout edge set.
  if len(broadcast_values) == 1:
    [single_value] = broadcast_values.values()  # pylint: disable=unbalanced-dict-unpacking
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
        "Conflicing feature dtypes found by structured_readout(..., '{key}', "
        f"feature_name='{feature_name}', readout_node_set='{readout_node_set}')"
        f": expected just one, found {dtypes}.")
  dtype = dtypes.pop()
  if not (dtype.is_floating or dtype.is_integer):
    raise NotImplementedError(
        "b/269076334: structured_readout() from multiple sources does not "
        "support non-numeric dtypes yet, because it relies on sum-pooling.")

  pooled_values = [
      pool_ops.pool_edges_to_node(
          graph, edge_set_name, const.TARGET, "sum", feature_value=value)
      for edge_set_name, value in broadcast_values.items()]
  result = tf.math.add_n(pooled_values)
  return result


def structured_readout_into_feature(
    graph: gt.GraphTensor,
    key: str,
    *,
    feature_name: const.FieldName,
    new_feature_name: Optional[const.FieldName] = None,
    remove_input_feature: bool = False,
    overwrite: bool = False,
    readout_node_set: const.NodeSetName = "_readout",
    validate: bool = True,
) -> gt.GraphTensor:
  """Reads out a feature value from select nodes (or edges) in a graph.

  This helper function works like `tfgnn.structured_readout()` (see there),
  except that it does not return the readout result itself but a modified
  `GraphTensor` in which the readout result is stored as a feature on
  the `readout_node_set`.

  Args:
    graph: A scalar GraphTensor with the auxiliary graph pieces required by
      `tfgnn.structured_readout()`.
    key: A string key to select between possibly multiple named readouts
      (such as `"source"` and `"target"` for link prediction).
    feature_name: The name of a feature to read out from, as with
      `tfgnn.structured_readout()`.
    new_feature_name: The name of the feature to add to `readout_node_set`
      for storing the readout result. If unset, defaults to `feature_name`.
      It is an error if the added feature already exists on `readout_node_set`
      in the input `graph`, unless `overwrite=True` is set.
    remove_input_feature: If set, the given `feature_name` is removed from the
      node (or edge) set(s) that supply the input to
      `tfgnn.structured_readout()`.
    overwrite: If set, allows overwriting a potentially already existing
      feature `graph.node_sets[readout_node_set][new_feature_name]`.
    readout_node_set: A string, defaults to `"_readout"`. This is used as the
      name for the readout node set and as a name prefix for its edge sets.
      See `tfgnn.structured_readout()` for more.
    validate: Setting this to false disables the validity checks of
      `tfgnn.structured_readout()`. This is strongly discouraged, unless
      great care is taken to run `tfgnn.validate_graph_tensor_for_readout()`
      earlier on structurally unchanged GraphTensors.

  Returns:
    A `GraphTensor` like `graph`, with the readout result stored as
    `.node_sets[readout_node_set][new_feature_name]` and possibly the
    readout inputs removed (see `remove_input_feature`).
  """
  gt.check_scalar_graph_tensor(graph.spec,
                               "tfgnn.structured_readout_into_feature()")
  if new_feature_name is None:
    new_feature_name = feature_name

  edge_sets_features = {}
  node_sets_features = {}

  node_sets_features[readout_node_set] = graph.node_sets[
      readout_node_set].get_features_dict()
  if not overwrite and new_feature_name in node_sets_features[readout_node_set]:
    raise ValueError(
        f"Output feature '{new_feature_name}' already exists on node set "
        f"'{readout_node_set}'. "
        "Pass tfgnn.structured_readout_into_feature(..., overwrite=True) "
        "to discard the old value."
    )
  node_sets_features[readout_node_set][new_feature_name] = structured_readout(
      graph, key, feature_name=feature_name,
      readout_node_set=readout_node_set, validate=validate)

  if remove_input_feature:
    edge_set_map = get_validated_edge_set_map_for_readout(
        graph.spec, [key], readout_node_set=readout_node_set)
    for (set_type, set_name) in edge_set_map[key].keys():
      if set_type == const.EDGES:
        features = graph.edge_sets[set_name].get_features_dict()
        del features[feature_name]
        edge_sets_features[set_name] = features
      else:  # NODES
        features = graph.node_sets[set_name].get_features_dict()
        del features[feature_name]
        node_sets_features[set_name] = features

  return graph.replace_features(node_sets=node_sets_features,
                                edge_sets=edge_sets_features)


def context_readout_into_feature(
    graph: gt.GraphTensor,
    *,
    feature_name: const.FieldName,
    new_feature_name: Optional[const.FieldName] = None,
    remove_input_feature: bool = False,
    overwrite: bool = False,
    readout_node_set: const.NodeSetName = "_readout",
) -> gt.GraphTensor:
  """Reads a feature value from context and stores it on readout_node_set.

  This helper function copies a context feature to the `"_readout"` node set.
  If the `"_readout"` node set does not exist, it is created with 1 node
  per component; if it exists already, it must have that size already.

  This function exists for symmetry with
  `tfgnn.structured_readout_into_feature()`, except that it reads a feature
  unchanged from `graph.context` instead of performing a
  `tfgnn.structured_readout()` operation.

  Args:
    graph: A scalar `GraphTensor`. If it contains the `readout_node_set`
      already, its size in each graph component must be 1. If not, it gets
      created with those sizes.
    feature_name: The name of a context feature to read out.
    new_feature_name: The name of the feature to add to `readout_node_set`
      for storing the readout result. If unset, defaults to feature_name.
      It is an error if the added feature already exists on `readout_node_set`
      in the input `graph`, unless `overwrite=True` is set.
    remove_input_feature: If set, the given `feature_name` is removed from the
      `graph.context`.
    overwrite: If set, allows overwriting a potentially already existing
      feature `graph.node_sets[readout_node_set][new_feature_name]`.
    readout_node_set: A string, defaults to `"_readout"`. This is used as the
      name for the readout node set.

  Returns:
    A `GraphTensor` like `graph`, with the named context feature stored as
    `.node_sets[readout_node_set][new_feature_name]` and possibly removed from
    `.context`.
  """
  gt.check_scalar_graph_tensor(graph.spec,
                               "tfgnn.context_readout_into_feature()")
  if new_feature_name is None:
    new_feature_name = feature_name

  context_features = graph.context.get_features_dict()
  if remove_input_feature:
    input_feature = context_features.pop(feature_name)
  else:
    input_feature = context_features.get(feature_name)

  graph = _add_readout_node_set(graph, sizes=graph.context.sizes,
                                readout_node_set=readout_node_set)
  readout_features = graph.node_sets[readout_node_set].get_features_dict()
  if not overwrite and new_feature_name in readout_features:
    raise ValueError(
        f"Output feature '{new_feature_name}' already exists on node set "
        f"'{readout_node_set}'. "
        "Pass tfgnn.context_readout_into_feature(..., overwrite=True) "
        "to discard the old value."
    )
  readout_features[new_feature_name] = input_feature

  return graph.replace_features(node_sets={readout_node_set: readout_features},
                                context=context_features)


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

    * `key` is the readout key for use in `tfgnn.structured_readout(..., key)`
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
      `tfgnn.structured_readout()`, if any.

  Returns:
    A dict of dicts, as described above.

  Raises:
    ValueError: if the auxiliary graph pieces for readout are malformed.
    KeyError: if any of the `required_keys` is missing.
  """
  if not re.fullmatch(r"_readout(:[a-zA-Z0-9_]+)?", readout_node_set):
    raise ValueError(
        "Malformed name of readout_node_set. Expected '_readout' or "
        "'_readout:'+tag, where tag consists of letters, digits and "
        f"underscores; got '{readout_node_set}'.")

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
          f"Malformed auxiliary edge set '{edge_set_name}': Expected target "
          f"node set '{readout_node_set}' but found '{target_node_set}'.")
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


def add_readout_from_first_node(
    graph: gt.GraphTensor,
    key: str,
    *,
    node_set_name: const.NodeSetName,
    readout_node_set: const.NodeSetName = "_readout",
) -> gt.GraphTensor:
  """Adds a readout structure equivalent to `tfgnn.gather_first_node()`.

  Args:
    graph: A scalar `GraphTensor`. If it contains the `readout_node_set`
      already, its size in each graph component must be 1.
    key: A key, for use with `tfgnn.structured_readout()`. The input graph
      must not already contain auxiliary edge sets for readout with this key.
    node_set_name: The name of the node set from which values are to be read
      out.
    readout_node_set: The name of the auxiliary node set for readout,
      as in `tfgnn.structured_readout()`.

  Returns:
    A modified GraphTensor such that `tfgnn.structured_readout(..., key)` works
    like `tfgnn.gather_first_node(...)` on the input graph.
  """
  gt.check_scalar_graph_tensor(graph, "tfgnn.add_readout_from_first_node()")

  # Get the source node set and check it has the nodes to read out from.
  if node_set_name not in graph.node_sets:
    raise ValueError(f"Unknown node set name '{node_set_name}'")
  node_set_sizes = graph.node_sets[node_set_name].sizes
  assert_positive_node_set_sizes = tf.debugging.assert_positive(
      node_set_sizes,
      f"tfgnn.add_readout_from_first_node(..., node_set_name='{node_set_name}')"
      " called for a graph in which one or more components contain no nodes.")
  with tf.control_dependencies([assert_positive_node_set_sizes]):
    first_node_indices = tf.math.cumsum(node_set_sizes, exclusive=True)

  # Get or make the readout node set with appropriate sizes.
  readout_sizes = tf.ones_like(node_set_sizes)
  graph = _add_readout_node_set(graph, sizes=readout_sizes,
                                readout_node_set=readout_node_set)

  # Make a new and unique readout edge set for the given key.
  edge_sets = dict(graph.edge_sets)
  readout_edge_set = f"{readout_node_set}/{key}"
  for edge_set_name in edge_sets:
    if re.fullmatch(rf"{re.escape(readout_edge_set)}(/.+)?", edge_set_name):
      raise ValueError("Requested readout key already exists: "
                       f"found edge set name '{edge_set_name}'")
  num_components = tf.size(readout_sizes, out_type=readout_sizes.dtype)
  edge_sets[readout_edge_set] = gt.EdgeSet.from_fields(
      sizes=readout_sizes,
      adjacency=adj.Adjacency.from_indices(
          (node_set_name, first_node_indices),
          (readout_node_set, tf.range(num_components))))

  return gt.GraphTensor.from_pieces(context=graph.context,
                                    node_sets=graph.node_sets,
                                    edge_sets=edge_sets)


def _add_readout_node_set(
    graph: gt.GraphTensor,
    *,
    sizes: gt.Field,
    readout_node_set: const.NodeSetName = "_readout",
) -> gt.GraphTensor:
  """Adds readout node set of given sizes, or verifies it exists.

  Args:
    graph: A scalar `GraphTensor`.
    sizes: A Tensor for use as the `NodeSet.sizes` of the readout node set.
      It must have shape `[num_components]` and dtype `graph.indices_dtype`.
    readout_node_set: The name of the auxiliary node set for readout,
      as in `tfgnn.structured_readout()`.

  Returns:
    A `GraphTensor` that contains a `readout_node_set` with the given `sizes`.
  """
  gt.check_scalar_graph_tensor(graph, "tfgnn.add_readout_node_set()")
  if sizes.dtype != graph.indices_dtype:
    raise ValueError(f"Got sizes.dtype = {sizes.dtype} but expected "
                     f"graph.indices_dtype = {graph.indices_dtype}")
  if sizes.shape != graph.context.sizes.shape:
    raise ValueError(f"Got sizes.shape = {sizes.shape} but expected "
                     f"{graph.context.sizes.shape}")

  node_sets = dict(graph.node_sets)
  if readout_node_set in node_sets:
    assert_same_sizes = tf.debugging.assert_equal(
        node_sets[readout_node_set].sizes, sizes,
        f"Existing aux node set '{readout_node_set}' does not match "
        f"requested sizes {sizes}")
    with tf.control_dependencies([assert_same_sizes]):
      sizes = tf.identity(sizes, name="checked_sizes")
    features = node_sets[readout_node_set].features
  else:
    features = None

  node_sets[readout_node_set] = gt.NodeSet.from_fields(sizes=sizes,
                                                       features=features)
  return gt.GraphTensor.from_pieces(context=graph.context,
                                    node_sets=node_sets,
                                    edge_sets=graph.edge_sets)
