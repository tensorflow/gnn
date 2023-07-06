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
"""Keras layer types for fundamental graph ops: Broadcast, Pool and Readout."""

from __future__ import annotations
from typing import Any, Mapping, Optional, Sequence, Union

import tensorflow as tf

from tensorflow_gnn.graph import broadcast_ops
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import graph_tensor_ops as ops
from tensorflow_gnn.graph import pool_ops
from tensorflow_gnn.graph import readout

Field = const.Field
FieldName = const.FieldName
NodeSetName = const.NodeSetName
EdgeSetName = const.EdgeSetName
IncidentNodeTag = const.IncidentNodeTag
IncidentNodeOrContextTag = const.IncidentNodeOrContextTag
GraphTensor = gt.GraphTensor


@tf.keras.utils.register_keras_serializable(package="GNN")
class Readout(tf.keras.layers.Layer):
  """Reads a feature out of a GraphTensor.

  The `Readout` layer is a convenience wrapper for indexing into a
  `tfgnn.GraphTensor` and retrieving a feature tensor from one of its edge sets,
  node sets, or the context. It is intended for use in places such as
  `tf.keras.Sequential` that require a Keras Layer and do not allow
  subscrpting syntax like `graph_tensor.node_sets["user"]["name"]`.

  A location in the graph is selected by setting exactly one of the keyword
  arguments `edge_set_name=...`, `node_set_name=...` or `from_context=True`.
  From there, the keyword argument `feature_name=...` selects the feature.

  Both the initialization of and the call to this layer accept arguments to
  select the feature location and the feature name. The call arguments take
  effect for that call only and can supply missing values, but they are not
  allowed to contradict initialization arguments. The feature name can be left
  unset to select tfgnn.HIDDEN_STATE.

  For example:

  ```python
  readout = tfgnn.keras.layers.Readout(feature_name="value")
  value = readout(graph_tensor, edge_set_name="edges")
  assert value == graph_tensor.edge_sets["edge"]["value"]
  ```

  Besides this direct readout of a full feature tensor, the library also
  supports readout that gathers feature values only from the nodes (or edges)
  that matter for a particular task, see `tfgnn.keras.layers.StructuredReadout`.
  See also `tfgnn.keras.layers.AddReadoutFromFirstNode` for adding the
  necessary readout structure to handle the legacy cases that were previously
  handled by `tfgnn.keras.layers.ReadoutFirstNode`.

  Init args:
    edge_set_name: If set, the feature will be read from this edge set.
      Mutually exclusive with node_set_name and from_context.
    node_set_name: If set, the feature will be read from this node set.
      Mutually exclusive with edge_set_name and from_context.
    from_context: If true, the feature will be read from the context.
      Mutually exclusive with edge_set_name and node_set_name.
    feature_name: The name of the feature to read. If unset (also in call),
      tfgnn.HIDDEN_STATE will be read.

  Call args:
    graph: The GraphTensor to read from.
    edge_set_name, node_set_name, from_context: Same meaning as for init. One of
      them must be passed to init, or to call, or to both (with the same value).
    feature_name: Same meaning as for init. If passed to both, the value must
      be the same. If passed to neither, tfgnn.HIDDEN_STATE is used.

  Returns:
    The tensor with the selected feature.
  """

  def __init__(self,
               *,
               edge_set_name: Optional[EdgeSetName] = None,
               node_set_name: Optional[NodeSetName] = None,
               from_context: bool = False,
               feature_name: Optional[FieldName] = None,
               **kwargs):
    super().__init__(**kwargs)
    self._location = self._get_location(node_set_name=node_set_name,
                                        edge_set_name=edge_set_name,
                                        from_context=from_context)
    self._feature_name = feature_name

  def get_config(self):
    config = super().get_config().copy()
    config.update(self._location)
    config["feature_name"] = self._feature_name
    return config

  def call(self,
           graph: GraphTensor,
           *,
           edge_set_name: Optional[EdgeSetName] = None,
           node_set_name: Optional[NodeSetName] = None,
           from_context: bool = False,
           feature_name: Optional[FieldName] = None) -> Field:
    location = self._get_location(node_set_name=node_set_name,
                                  edge_set_name=edge_set_name,
                                  from_context=from_context)
    _check_init_call_kwargs_consistency("Readout",
                                        self._location, location)
    if not location:
      location = self._location
    if not location:
      raise ValueError(
          "The Readout layer requires one of edge_set_name, node_set_name or "
          "from_context to be set at init or call time")

    _check_init_call_arg_consistency("Readout", "feature_name",
                                     self._feature_name, feature_name)
    if feature_name is None:
      feature_name = self._feature_name
    if feature_name is None:
      feature_name = const.HIDDEN_STATE

    if location.get("from_context"):
      return graph.context[feature_name]
    node_set_name = location.get("node_set_name")
    if node_set_name is not None:
      return graph.node_sets[node_set_name][feature_name]
    else:
      return graph.edge_sets[location["edge_set_name"]][feature_name]

  @property
  def location(self) -> Mapping[str, Any]:
    """Returns a dict with the kwarg to init that selected the feature location.

    The result contains the keyword argument and value passed to `__init__()`
    that selects the location from which the layer's output feature is read,
     that is, one of `edge_set_name=...`, `node_set_name=...` or
    `from_context=True`. If none of these has been set, the result is
    empty, and one of them must be set at call time.
    """
    return self._location

  @property
  def feature_name(self) -> Optional[FieldName]:
    """Returns the feature_name argument to init, or None if unset."""
    return self._feature_name

  def _get_location(self, *, node_set_name, edge_set_name, from_context):
    """Returns dict of non-None kwargs for selecting the feature location."""
    result = dict()
    if node_set_name is not None: result.update(node_set_name=node_set_name)
    if edge_set_name is not None: result.update(edge_set_name=edge_set_name)
    if from_context: result.update(from_context=from_context)
    if len(result) > 1:
      raise ValueError(
          "The Readout layer allows at most one of "
          "edge_set_name, node_set_name and from_context to be set "
          f"but was passed {_format_as_kwargs(result)}")
    return result


def _check_init_call_arg_consistency(layer_name, arg_name,
                                     init_value, call_value):
  """Raises ValueError if init and call values are non-None and different."""
  if init_value is None or call_value is None:
    return
  if init_value != call_value:
    raise ValueError(f"The {layer_name} layer was "
                     f"initialized with {arg_name}={init_value} "
                     f"but called with {arg_name}={call_value}")


# Same for the slice of kwargs stored as feature location.
def _check_init_call_kwargs_consistency(layer_name, init_kwargs, call_kwargs):
  """Raises ValueError if init and call kwargs are different."""
  if not init_kwargs or not call_kwargs:
    return
  if init_kwargs != call_kwargs:
    raise ValueError(f"The {layer_name} layer was "
                     f"initialized with {_format_as_kwargs(init_kwargs)} "
                     f"but called with {_format_as_kwargs(call_kwargs)}")


def _format_as_kwargs(kwargs_dict):
  return ", ".join([f"{k}={repr(v)}" for k, v in kwargs_dict.items()])


@tf.keras.utils.register_keras_serializable(package="GNN")
class AddSelfLoops(tf.keras.layers.Layer):
  """Adds self-loops to scalar graphs.

  The edge_set_name is expected to be a homogeneous edge (connects a node pair
  of the node set). NOTE: Self-connections will always be added, regardless if
  if self-connections already exist or not.
  """

  def __init__(self, edge_set_name):
    super().__init__()
    self._edge_set_name = edge_set_name

  def call(self, graph_tensor: GraphTensor) -> GraphTensor:
    return ops.add_self_loops(graph_tensor, self._edge_set_name)

  def get_config(self):
    config = super().get_config().copy()
    config["edge_set_name"] = self._edge_set_name
    return config


@tf.keras.utils.register_keras_serializable(package="GNN")
class ReadoutFirstNode(tf.keras.layers.Layer):
  """Reads a feature from the first node of each graph component.

  Given a particular node set (identified by `node_set_name`), this layer
  will gather the given feature from the first node of each graph component.

  This is often used for rooted graphs created by sampling around the
  neighborhoods of seed nodes in a large graph: by convention, each seed node is
  the first node of its component in the respective node set, and this layer
  reads out the information it has accumulated there. (In other node sets, the
  first node may be arbitrary -- or nonexistant, in which case this operation
  must not be used and may raise an error at runtime.)

  This implicit convention is inflexible and hard to validate. New models are
  encouraged to use `tfgnn.keras.layers.StructuredReadout` instead. The
  necessary readout structure should preferably be present in the sampled
  dataset itself; if not, it can be added after the fact by
  `tfgnn.keras.layers.AddReadoutFromFirstNode`.

  Init args:
    node_set_name: If set, the feature will be read from this node set.
    feature_name: The name of the feature to read. If unset (also in call),
      tfgnn.HIDDEN_STATE will be read.

  Call args:
    graph: The scalar GraphTensor to read from.
    node_set_name: Same meaning as for init. Must be passed to init, or to call,
      or to both (with the same value).
    feature_name: Same meaning as for init. If passed to both, the value must
      be the same. If passed to neither, tfgnn.HIDDEN_STATE is used.

  Returns:
    A tensor of gathered feature values, one for each graph component, like a
    context feature.
  """

  def __init__(self,
               *,
               node_set_name: Optional[NodeSetName] = None,
               feature_name: Optional[FieldName] = None,
               **kwargs):
    super().__init__(**kwargs)
    self._node_set_name = node_set_name
    self._feature_name = feature_name

  def get_config(self):
    config = super().get_config().copy()
    config["node_set_name"] = self._node_set_name
    config["feature_name"] = self._feature_name
    return config

  def call(self,
           graph: GraphTensor,
           *,
           node_set_name: Optional[NodeSetName] = None,
           feature_name: Optional[FieldName] = None) -> Field:

    _check_init_call_arg_consistency("ReadoutFirstNode", "node_set_name",
                                     self._node_set_name, node_set_name)
    if node_set_name is None:
      node_set_name = self._node_set_name
    if node_set_name is None:
      raise ValueError("The ReadoutFirstNode layer requires node_set_name "
                       "to be set at init or call time")

    _check_init_call_arg_consistency("ReadoutFirstNode", "feature_name",
                                     self._feature_name, feature_name)
    if feature_name is None:
      feature_name = self._feature_name
    if feature_name is None:
      feature_name = const.HIDDEN_STATE

    gt.check_scalar_graph_tensor(graph, "ReadoutFirstNode")
    return ops.gather_first_node(
        graph, node_set_name, feature_name=feature_name)

  @property
  def location(self) -> Mapping[str, Any]:
    """Returns a dict with the kwarg to init that selected the feature location.
    """
    if self._node_set_name is not None:
      return dict(node_set_name=self._node_set_name)
    else:
      return dict()

  @property
  def feature_name(self) -> Optional[FieldName]:
    """Returns the feature_name argument to init, or None if unset."""
    return self._feature_name


@tf.keras.utils.register_keras_serializable(package="GNN")
class StructuredReadout(tf.keras.layers.Layer):
  """Reads out a feature value from select nodes (or edges) in a graph.

  This Keras layer wraps the `tfgnn.structured_readout()` function. It addresses
  the need to read out final hidden states from a GNN computation to make
  predictions for some nodes (or edges) of interest. Its typical usage looks
  as follows:

  ```python
  input_graph = tf.keras.Input(type_spec=graph_tensor_spec)
  graph = SomeGraphUpdate(...)(input_graph)  # Run your GNN here.
  seed_node_states = tfgnn.keras.layers.StructuredReadout("seed")(graph)
  logits = tf.keras.layers.Dense(num_classes)(seed_node_states)
  model = tf.keras.Model(inputs, logits)
  ```

  ...where `"seed"` is a key defined by the readout structure. There can be
  multiple of those. For example, a link prediction model could read out
  `"source"` and `"target"` node states from the graph.

  Please see the documentation of `tfgnn.structured_readout()` for the auxiliary
  node set and edge sets that make up the readout structure which encodes how
  values are read out from the graph. Whenever these are available, it is
  strongly recommended to make use of them with this layer and avoid the older
  `tfgnn.keras.layers.ReadoutFirstNode`.

  Note that this layer returns a tensor shaped like a feature of the
  `"_readout"` node set but not actually stored on it. To store it there,
  see `tfgnn.keras.layers.StructuredReadoutIntoFeature`. To retrieve a
  feature unchanged, see ``tfgnn.keras.layers.Readout`.

  Init args:
    key: A string key to select between possibly multiple named readouts
      (such as `"source"` and `"target"` for link prediction). Can be fixed
      in init, or selected for each call.
    feature_name: The name of the feature to read. If unset (also in call),
      `tfgnn.HIDDEN_STATE` will be read.
    readout_node_set: A string, defaults to `"_readout"`. This is used as the
      name for the readout node set and as a name prefix for its edge sets.
    validate: Setting this to false disables the validity checks for the
      auxiliary edge sets. This is stronlgy discouraged, unless great care is
      taken to run `tfgnn.validate_graph_tensor_for_readout()` earlier on
      structurally unchanged GraphTensors.

  Call args:
    graph: The scalar GraphTensor to read from.
    key: Same meaning as for init. Must be passed to init, or to call,
      or to both (with the same value).

  Returns:
    A tensor of read-out feature values, shaped like a feature of the
    `readout_node_set`.
  """

  def __init__(self,
               key: Optional[str] = None,
               *,
               feature_name: str = const.HIDDEN_STATE,
               readout_node_set: NodeSetName = "_readout",
               validate: bool = True,
               **kwargs):
    super().__init__(**kwargs)
    self._key = key
    self._feature_name = feature_name
    self._readout_node_set = readout_node_set
    self._validate = validate

  def get_config(self):
    return dict(
        key=self._key,
        feature_name=self._feature_name,
        readout_node_set=self._readout_node_set,
        validate=self._validate,
        **super().get_config())

  def call(self,
           graph: GraphTensor,
           *,
           key: Optional[str] = None) -> Field:
    _check_init_call_arg_consistency("StructuredReadout", "key",
                                     self._key, key)
    if key is None:
      key = self._key
    if key is None:
      raise ValueError("The StructuredReadout layer requires a readout key "
                       "to be set at init or call time")

    gt.check_scalar_graph_tensor(graph, "StructuredReadout")
    return readout.structured_readout(
        graph,
        key=key,
        feature_name=self._feature_name,
        readout_node_set=self._readout_node_set,
        validate=self._validate)


@tf.keras.utils.register_keras_serializable(package="GNN")
class StructuredReadoutIntoFeature(tf.keras.layers.Layer):
  """Reads out a feature value from select nodes (or edges) in a graph.

  This Keras Layer wraps the `tfgnn.structured_readout_into_feature()` function.
  It is is similar to `tfgnn.keras.layers.StructuredReadout` (see there),
  except that it does not return the readout result itself but a modified
  `GraphTensor` in which the readout result is stored as a feature on
  the auxiliary `readout_node_set`. (See `tfgnn.structured_readout()` for more
  information on that.)

  For example, this layer can be used in data processing before training
  to move training labels out of the input graph seen my the model (see
  `remove_input_feature`) onto the auxiliary `readout_node_set`.

  Init args:
    key: A string key to select between possibly multiple named readouts
      (such as `"source"` and `"target"` for link prediction). Can be fixed
      in init, or selected for each call.
    feature_name: The name of a feature to read out from, as with
      `tfgnn.structured_readout()`.
    new_feature_name: The name of the feature to add to `readout_node_set`
      for storing the readout result. If unset, defaults to `feature_name`.
      It is an error if the added feature already exists on `readout_node_set`
      in the input `graph`, unless `overwrite=True` is set.
    remove_input_feature: If set, the given `feature_name` is removed from the
      node (or edge) set(s) that contain the value to be read out in the input
      `GraphTensor`.
    overwrite: If set, allows overwriting a potentially already existing
      feature `graph.node_sets[readout_node_set][new_feature_name]`.
    readout_node_set: A string, defaults to `"_readout"`. This is used as the
      name for the readout node set and as a name prefix for its edge sets.
    validate: Setting this to false disables the validity checks of
      `tfgnn.structured_readout()`. This is strongly discouraged, unless great
      care is taken to run `tfgnn.validate_graph_tensor_for_readout()` earlier
      on structurally unchanged GraphTensors.

  Call args:
    graph: The scalar `GraphTensor` to read from.
    key: Same meaning as for init. Must be passed to init, or to call,
      or to both (with the same value).

  Returns:
    A `GraphTensor` like `graph`, with the readout result stored as
    `.node_sets[readout_node_set][new_feature_name]` and possibly the
    readout input feature removed (see `remove_input_feature`).
  """

  def __init__(
      self,
      key: Optional[str] = None,
      *,
      feature_name: FieldName,
      new_feature_name: Optional[FieldName] = None,
      remove_input_feature: bool = False,
      overwrite: bool = False,
      readout_node_set: NodeSetName = "_readout",
      validate: bool = True,
      **kwargs):
    super().__init__(**kwargs)
    self._key = key
    self._feature_name = feature_name
    self._new_feature_name = new_feature_name
    self._remove_input_feature = remove_input_feature
    self._overwrite = overwrite
    self._readout_node_set = readout_node_set
    self._validate = validate

  def get_config(self):
    return dict(
        key=self._key,
        feature_name=self._feature_name,
        new_feature_name=self._new_feature_name,
        remove_input_feature=self._remove_input_feature,
        overwrite=self._overwrite,
        readout_node_set=self._readout_node_set,
        validate=self._validate,
        **super().get_config())

  def call(self,
           graph: GraphTensor,
           *,
           key: Optional[str] = None) -> GraphTensor:
    _check_init_call_arg_consistency("StructuredReadoutIntoFeature", "key",
                                     self._key, key)
    if key is None:
      key = self._key
    if key is None:
      raise ValueError("The StructuredReadoutIntoFeature layer requires "
                       "a readout key to be set at init or call time")

    gt.check_scalar_graph_tensor(graph, "StructuredReadoutIntoFeature")
    return readout.structured_readout_into_feature(
        graph,
        key=key,
        feature_name=self._feature_name,
        new_feature_name=self._new_feature_name,
        remove_input_feature=self._remove_input_feature,
        overwrite=self._overwrite,
        readout_node_set=self._readout_node_set,
        validate=self._validate)


@tf.keras.utils.register_keras_serializable(package="GNN")
class AddReadoutFromFirstNode(tf.keras.layers.Layer):
  """Adds readout node set equivalent to `tfgnn.keras.layers.ReadoutFirstNode`.

  Init args:
    key: A key, for use with `tfgnn.keras.layers.StructuredReadout`. The input
      graph must not already contain auxiliary edge sets for readout with this
      key.
    node_set_name: The name of the node set from which values are to be read
      out.
    readout_node_set: The name of the auxiliary node set for readout,
      as in `tfgnn.structured_readout()`.

  Call args:
    graph: A scalar `GraphTensor`. If it contains the readout_node_set already,
      its size in each graph component must be 1.

  Returns:
    A modified `GraphTensor` so that `tfgnn.keras.layers.StructuredReadout(key)`
    acts like `tfgnn.keras.layers.ReadoutFirstNode(node_set_name=node_set_name)`
    on the input graph.
  """

  def __init__(
      self,
      key: str,
      *,
      node_set_name: NodeSetName,
      readout_node_set: NodeSetName = "_readout",
      **kwargs):
    super().__init__(**kwargs)
    self._key = key
    self._node_set_name = node_set_name
    self._readout_node_set = readout_node_set

  def get_config(self):
    return dict(
        key=self._key,
        node_set_name=self._node_set_name,
        readout_node_set=self._readout_node_set,
        **super().get_config())

  def call(self, graph: GraphTensor) -> GraphTensor:
    gt.check_scalar_graph_tensor(graph, "AddReadoutFromFirstNode")
    return readout.add_readout_from_first_node(
        graph,
        key=self._key,
        node_set_name=self._node_set_name,
        readout_node_set=self._readout_node_set)


class BroadcastPoolBase(tf.keras.layers.Layer):
  """Base class to Broadcast and Pool.

  Broadcast and Pool work on the same "many-to-one" relationships in a
  GraphTensor, just with different directions of data flow. This base class
  provides their common handling of init and call args that specify the
  relationship:

    * An edge_set_name or node_set_name (or a sequence thereof) specifies the
      "many" things that are being broadcast to or pooled from. Collectively,
      the edge or node set name is called the location.
    * An incident node tag SOURCE or TARGET or the special tag CONTEXT
      specifies the "one" thing that is being broadcast from or pooled to.
      The tag is understood relative to each edge (or node): the SOURCE or
      TARGET node incident to each edge, or the CONTEXT of the component
      to which the edge or node belongs.
      It is an error to use node_set_name with a tag other than CONTEXT.

  This base class also manages the feature_name used to select a feature
  at the origin of the Broadcast or Pool operation.

  This base class manages tag, edge_set_name, node_set_name and feature_name
  for init, get_config and call but leaves the actual computation and
  user-visible documentation to concrete subclasses Broadcast and Pool.
  """

  def __init__(
      self,
      *,
      tag: Optional[IncidentNodeOrContextTag] = None,
      edge_set_name: Union[Sequence[EdgeSetName], EdgeSetName, None] = None,
      node_set_name: Union[Sequence[NodeSetName], NodeSetName, None] = None,
      feature_name: Optional[FieldName] = None,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self._check_tag(tag)
    self._tag = tag
    self._location = self._get_location(node_set_name=node_set_name,
                                        edge_set_name=edge_set_name)
    self._check_location(self._location, tag, required=False)
    self._feature_name = feature_name

  def get_config(self):
    config = super().get_config().copy()
    config["tag"] = self._tag
    config.update(self._location)
    config["feature_name"] = self._feature_name
    return config

  def _fixup_call_args(
      self,
      tag: Optional[IncidentNodeOrContextTag] = None,
      edge_set_name: Union[Sequence[EdgeSetName], EdgeSetName, None] = None,
      node_set_name: Union[Sequence[NodeSetName], NodeSetName, None] = None,
      feature_name: Optional[FieldName] = None,
  ):
    self._check_tag(tag)
    _check_init_call_arg_consistency(self._layer_name, "tag", self._tag, tag)
    if tag is None:
      tag = self._tag
    if tag is None:
      raise ValueError(f"The {self._layer_name} layer requires `tag=` "
                       "to be set at init or call time")

    location = self._get_location(node_set_name=node_set_name,
                                  edge_set_name=edge_set_name)
    _check_init_call_kwargs_consistency(self._layer_name,
                                        self._location, location)
    if not location:
      location = self._location
    self._check_location(location, tag, required=True)
    node_set_name = location.get("node_set_name")
    edge_set_name = location.get("edge_set_name")

    _check_init_call_arg_consistency(self._layer_name, "feature_name",
                                     self._feature_name, feature_name)
    if feature_name is None:
      feature_name = self._feature_name
    if feature_name is None:
      feature_name = const.HIDDEN_STATE

    return tag, edge_set_name, node_set_name, feature_name

  @property
  def tag(self) -> Optional[IncidentNodeOrContextTag]:
    """Returns the tag argument to init, or None if unset."""
    return self._tag

  @property
  def location(self) -> Mapping[str, Union[str, Sequence[str]]]:
    """Returns dict of kwarg to init with the node or edge set name."""
    return self._location

  @property
  def feature_name(self) -> Optional[FieldName]:
    """Returns the feature_name argument to init, or None if unset."""
    return self._feature_name

  @property
  def _layer_name(self) -> str:
    """The user-visible name of the Layer class in logged messages."""
    return self.__class__.__name__

  def _check_tag(self, tag):
    if tag not in [None, const.SOURCE, const.TARGET, const.CONTEXT]:
      raise ValueError(f"The {self._layer_name} layer requires tag to be "
                       "one of tfgnn.SOURCE, tfgnn.TARGET or tfgnn.CONTEXT.")

  def _get_location(self, *, node_set_name, edge_set_name):
    """Returns dict of non-None kwargs for selecting the node or edge set."""
    def clone_sequence(x):
      if isinstance(x, Sequence) and not isinstance(x, str):
        # Make a copy to avoid references to user-visible mutable lists.
        return tuple(x)
      return x

    result = dict()
    if node_set_name is not None:
      result.update(node_set_name=clone_sequence(node_set_name))
    if edge_set_name is not None:
      result.update(edge_set_name=clone_sequence(edge_set_name))

    if len(result) > 1:
      raise ValueError(f"The {self._layer_name} layer allows at most one of "
                       "edge_set_name and node_set_name to be set.")
    return result

  def _check_location(self, location, tag, required=False):
    """Raises ValueError for bad location. May be None if not required."""
    if tag is None:  # Not set in init.
      assert not required, "Internal error: required unexpected without tag"
      # Nothing left to check.
    elif tag == const.CONTEXT:
      if required and not location:
        raise ValueError(
            f"The {self._layer_name} layer with tag CONTEXT ""requires "
            "exactly one of edge_set_name and node_set_name")
    else:  # SOURCE or TARGET
      assert tag in (const.SOURCE, const.TARGET), f"Internal error: tag={tag}"
      if required and not location or "node_set_name" in location:
        raise ValueError(
            f"The {self._layer_name} layer with tag SOURCE or TARGET "
            "requires edge_set_name but not node_set_name")


@tf.keras.utils.register_keras_serializable(package="GNN")
class Broadcast(BroadcastPoolBase):
  """Broadcasts a GraphTensor feature.

  This layer accepts a complete GraphTensor and returns a tensor (or tensors)
  with the broadcast feature value.

  There are two kinds of broadcast that this layer can be used for:

    * From a node set to an edge set (or multiple edge sets). This is selected
      by specifying the receiver edge set(s) as `edge_set_name=...` and the
      sender by tag `tgnn.SOURCE` or `tfgnn.TARGET` relative to the edge set(s).
      The node set name is implied. (In case of multiple edge sets, it must
      agree between all of them.)
      The result is a tensor (or list of tensors) shaped like an edge feature
      in which each edge has a copy of the feature that is present at its
      `SOURCE` or `TARGET` node. From a node's point of view, `SOURCE` means
      broadcast to outgoing edges, and `TARGET` means broadcast to incoming
      edges.
    * From the context to one (or more) node sets or one (or more) edge sets.
      This is selected by specifying the receiver(s) as either
      `node_set_name=...` or `edge_set_name=...` and the sender by tag
      `tfgnn.CONTEXT`.
      The result is a tensor (or list of tensors) shaped like a node/edge
      feature in which each node/edge has a copy of the context feature from
      the graph component it belongs to. (For more on components, see
      `tfgnn.GraphTensor.merge_batch_to_components()`.)

  Both the initialization of and the call to this layer accept arguments to
  set the `tag`, `node_set_name`/`edge_set_name`, and the `feature_name`. The
  call arguments take effect for that call only and can supply missing values,
  but they are not allowed to contradict initialization arguments.
  The feature name can be left unset to select `tfgnn.HIDDEN_STATE`.

  Init args:
    tag: Can be set to one of `tfgnn.SOURCE`, `tfgnn.TARGET` or `tfgnn.CONTEXT`
      to select the sender from which feature values are broadcast.
    edge_set_name: If set, the feature will be broadcast to this edge set
      (or this sequence of edge sets) from the sender given by `tag`.
      Mutually exclusive with `node_set_name`.
    node_set_name: If set, the feature will be broadcast to this node set
      (or sequence of node sets). The sender must be selected as
      `tag=tfgn.CONTEXT`. Mutually exclusive with `edge_set_name`.
    feature_name: The name of the feature to read. If unset (also in call),
      the `tfgnn.HIDDEN_STATE` feature will be read.

  Call args:
    graph: The scalar `tfgnn.GraphTensor` to read from.
    tag: Same meaning as for init. Must be passed to init, or to call,
      or to both (with the same value).
    edge_set_name, node_set_name: Same meaning as for init. One of them must
      be passed to init, or to call, or to both (with the same value).
    feature_name: Same meaning as for init. If passed to both, the value must
      be the same. If passed to neither, `tfgnn.HIDDEN_STATE` is used.

  Returns:
    A tensor (or list of tensors) with the feature values broadcast to the
    requested receivers.
  """

  def __init__(
      self,
      tag: Optional[IncidentNodeOrContextTag] = None,
      *,
      edge_set_name: Union[Sequence[EdgeSetName], EdgeSetName, None] = None,
      node_set_name: Union[Sequence[NodeSetName], NodeSetName, None] = None,
      feature_name: Optional[FieldName] = None,
      **kwargs,
  ):
    super().__init__(
        tag=tag, edge_set_name=edge_set_name, node_set_name=node_set_name,
        feature_name=feature_name, **kwargs)

  def call(
      self,
      graph: GraphTensor,
      *,
      tag: Optional[IncidentNodeOrContextTag] = None,
      edge_set_name: Union[Sequence[EdgeSetName], EdgeSetName, None] = None,
      node_set_name: Union[Sequence[NodeSetName], NodeSetName, None] = None,
      feature_name: Optional[FieldName] = None,
  ) -> Union[list[Field], Field]:
    gt.check_scalar_graph_tensor(graph, "Broadcast")
    tag, edge_set_name, node_set_name, feature_name = self._fixup_call_args(
        tag, edge_set_name, node_set_name, feature_name)

    return broadcast_ops.broadcast_v2(
        graph, tag, edge_set_name=edge_set_name, node_set_name=node_set_name,
        feature_name=feature_name)


@tf.keras.utils.register_keras_serializable(package="GNN")
class Pool(BroadcastPoolBase):
  """Pools a GraphTensor feature.

  This layer accepts a complete GraphTensor and returns a tensor with the
  result of pooling some feature.

  There are two kinds of pooling that this layer can be used for:

    * From an edge set (or multiple edge sets) to a single node set. This is
      selected by specifying the sender as `edge_set_name=...` and the receiver
      with tag `tgnn.SOURCE` or `tfgnn.TARGET`; the corresponding node set name
      is implied. (In case of multiple edge sets, it must be the same for all.)
      The result is a tensor shaped like a node feature in which each node
      has the aggregated feature values from the edges of the edge set(s) that
      have it as their `SOURCE` or `TARGET`, resp.; that is, the outgoing or
      incoming edges of the node.
    * From one (or more) node sets or one (or more) edge sets to the context.
      This is selected by specifying the sender as either `node_set_name=...`
      or `edge_set_name=...` and the receiver with tag `tfgnn.CONTEXT`.
      The result is a tensor shaped like a context feature in which the entry
      for each graph component has the aggregated feature values from its
      nodes/edges in the selected node or edge set(s). (For more on components,
      see `tfgnn.GraphTensor.merge_batch_to_components()`.)

  Feature values are aggregated into a single value by a reduction function
  like `"sum"` or `"mean|max_no_inf"` as described for `tfgnn.pool()`; see there
  for more details.

  Both the initialization of and the call to this layer accept arguments for
  the receiver `tag`, the `node_set_name`/`edge_set_name`, the `reduce_type` and
  the `feature_name`. The call arguments take effect for that call only and can
  supply missing values, but they are not allowed to contradict initialization
  arguments.
  The feature name can be left unset to select `tfgnn.HIDDEN_STATE`.

  Init args:
    tag: Can be set to one of `tfgnn.SOURCE`, `tfgnn.TARGET` or `tfgnn.CONTEXT`
      to select the receiver.
    reduce_type: Can be set to any `reduce_type` understood by `tfgnn.pool()`.
    edge_set_name: If set, the feature will be pooled from this edge set
      (or this sequence of edge sets) to the receiver given by `tag`.
      Mutually exclusive with `node_set_name`.
    node_set_name: If set, the feature will be pooled from this node set
      (or sequence of node sets). The receiver must be selected as
      `tag=tfgnn.CONTEXT`. Mutually exclusive with `edge_set_name`.
    feature_name: The name of the feature to read. If unset (also in call),
      the `tfgnn.HIDDEN_STATE` feature will be read.

  Call args:
    graph: The scalar `tfgnn.GraphTensor` to read from.
    reduce_type: Same meaning as for init. Must be passed to init, or to call,
      or to both (with the same value).
    tag: Same meaning as for init. Must be passed to init, or to call,
      or to both (with the same value).
    edge_set_name, node_set_name: Same meaning as for init. One of them must
      be passed to init, or to call, or to both (with the same value).
    feature_name: Same meaning as for init. If passed to both, the value must
      be the same. If passed to neither, `tfgnn.HIDDEN_STATE` is used.

  Returns:
    A tensor with the pooled feature value.
  """

  def __init__(
      self,
      tag: Optional[IncidentNodeOrContextTag] = None,
      reduce_type: Optional[str] = None,
      *,
      edge_set_name: Union[Sequence[EdgeSetName], EdgeSetName, None] = None,
      node_set_name: Union[Sequence[NodeSetName], NodeSetName, None] = None,
      feature_name: Optional[FieldName] = None,
      **kwargs,
  ):
    super().__init__(
        tag=tag, edge_set_name=edge_set_name, node_set_name=node_set_name,
        feature_name=feature_name, **kwargs)
    self._reduce_type = reduce_type

  def get_config(self):
    config = super().get_config()  # Our base class returns a private copy.
    config["reduce_type"] = self._reduce_type
    return config

  def call(
      self,
      graph: GraphTensor,
      *,
      tag: Optional[IncidentNodeOrContextTag] = None,
      reduce_type: Optional[str] = None,
      edge_set_name: Union[Sequence[EdgeSetName], EdgeSetName, None] = None,
      node_set_name: Union[Sequence[NodeSetName], NodeSetName, None] = None,
      feature_name: Optional[FieldName] = None,
  ) -> Field:
    gt.check_scalar_graph_tensor(graph, "Pool")
    tag, edge_set_name, node_set_name, feature_name = self._fixup_call_args(
        tag, edge_set_name, node_set_name, feature_name)

    _check_init_call_arg_consistency(self._layer_name, "reduce_type",
                                     self._reduce_type, reduce_type)
    if reduce_type is None:
      reduce_type = self._reduce_type
    if reduce_type is None:
      raise ValueError("The Pool layer requires reduce_type "
                       "to be set at init or call time")

    return pool_ops.pool_v2(
        graph, tag, edge_set_name=edge_set_name, node_set_name=node_set_name,
        reduce_type=reduce_type, feature_name=feature_name)

  @property
  def reduce_type(self) -> str:
    """Returns the reduce_type argument to init, or None if unset."""
    return self._reduce_type
