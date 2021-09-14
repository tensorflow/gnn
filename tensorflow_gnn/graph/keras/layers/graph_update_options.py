"""Dataclass GraphUpdateOptions and its pieces."""

import collections
import copy
import dataclasses
import enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from tensorflow_gnn.graph import graph_constants as const

# Keras allows tensors, list of tensors or dicts of tensors, but no deeper
# nesting. This is the type for correspondingly nested feature names,
FieldNames = Union[const.FieldName,
                   List[const.FieldName],
                   Dict[str, const.FieldName]]


class UpdateInputEnabled(str, enum.Enum):
  """Controls when to use an input in a GraphUpdate.

  Values:
    NEVER: The input is not used.
    ALWAYS: The input is used unconditionally.
    ON_UPDATE: The input is used if it has already been computed during the
      same GraphUpdate. In other words, older values are ignored.
  """
  NEVER = "never"
  ALWAYS = "always"
  ON_UPDATE = "on_update"


# TODO(b/193496101): Proper types for sublayers. pylint: disable=g-bare-generic
@dataclasses.dataclass
class GraphUpdateEdgeSetOptions:
  """The per-EdgeSet subobject of GraphUpdateOptions.

  GraphUpdate and its constituent layers get EdgeSet-related options from
  objects of this class held by a GraphUpdateOptions object.

  For the EdgeSetUpdate layer (see there for a detailed explanation):
    update_input_fn_factories: Can be set to a list [f1, f2, ...] of input
      factories, for use as EdgeSetUpdate(..., input_fns=[f1(), f2(), ...]).
    update_use_node_tags: If input_fns are not set, the default input_fns
      contain the node states from precisely the incident node sets whose tags
      are listed here.
    update_use_recurrent_state: If input_fns are not set, this boolean controls
      whether the previous edge state is part of the default input_fns.
    update_use_context: If input_fns are not set, this boolean controls
      whether the context state is part of the default input_fns.
    update_combiner_fn: If set, the EdgeSetUpdate(..., combiner_fn=...).
      For no combiner, use `"none"`; literal `None` means unset.
    update_fn_factory: If set, called without arguments for use as
      EdgeSetUpdate(..., update_fn=update_fn_factory()).
    update_output_feature: If set, the EdgeSetUpdate(..., output_feature=...).

  For the NodeSetUpdate layer (see there for a detailed explanation):
    node_pool_enabled: Can be set to an enum value from UpdateInputEnabled
      to control in which cases the state of this edge set is used as input
      to a NodeSetUpdate.
    node_pool_tags: Can be a list of incident node sets. A NodeSetUpdate
      for any of them without explicit input_fns receives as input the
      pooled features from each edge set.
    node_pool_factory: Can be set to a function that accepts arguments
      (tag=..., edge_set_name=...) with any `tag` from `node_pool_tags`
      and returns the input_fn that does the pooling.

  For the ContextUpdate (see there for a detailed explanation):
    context_pool_enabled: Can be set to an enum value from UpdateInputEnabled
      to control in which cases the state of this edge set is used as input
      to a ContextUpdate.
    context_pool_factory: Can be set to a function that accepts arguments
      (edge_set_name=...) and returns the input_fn that does the pooling.
  """
  update_input_fn_factories: Optional[List[Callable[[], Callable]]] = None
  update_use_node_tags: Optional[List[const.IncidentNodeTag]] = None
  update_use_recurrent_state: Optional[bool] = None
  update_use_context: Optional[bool] = None
  update_combiner_fn: Optional[Union[str, Callable]] = None
  update_fn_factory: Optional[Callable[[], Callable]] = None
  update_output_feature: Optional[FieldNames] = None
  node_pool_enabled: Optional[UpdateInputEnabled] = None
  node_pool_tags: Optional[List[const.IncidentNodeTag]] = None
  node_pool_factory: Optional[Callable[..., Callable]] = None
  context_pool_enabled: Optional[UpdateInputEnabled] = None
  context_pool_factory: Optional[Callable[..., Callable]] = None


@dataclasses.dataclass
class GraphUpdateNodeSetOptions:
  """The per-NodeSet subobject of GraphUpdateOptions.

  GraphUpdate and its constituent layers get NodeSet-related options from
  objects of this class held by a GraphUpdateOptions object.

  For the NodeSetUpdate layer (see there for a detailed explanation):
    update_input_fn_factories: Can be set to a list [f1, f2, ...] of input
      factories, for use as NodeSetUpdate(..., input_fns=[f1(), f2(), ...]).
      Otherwise, a default applies that may involve the features of
      incoming edges pooled per GraphUpdateEdgeSetOptions.node_pool_factory().
    update_use_recurrent_state: If input_fns are not set, this boolean controls
      whether the previous node state is part of the default input_fns.
    update_use_context: If input_fns are not set, this boolean controls
      whether the context state state is part of the default input_fns.
    update_combiner_fn: If set, the NodeSetUpdate(..., combiner_fn=...).
      For no combiner, use `"none"`; literal `None` means unset.
    update_fn_factory: If set, called without arguments for use as
      NodeSetUpdate(..., update_fn=update_fn_factory()).
    update_output_feature: the NodeSetUpdate(..., output_feature=...).

  For the ContextUpdate (see there for a detailed explanation):
    context_pool_enabled: Can be set to an enum value from UpdateInputEnabled
      to control in which cases the state of this node set is used as input
      to a ContextUpdate.
    context_pool_factory: Can be set to a function that accepts arguments
      (node_set_name=...) and returns the input_fn that does the pooling.
  """
  update_input_fn_factories: Optional[List[Callable]] = None
  update_use_recurrent_state: Optional[bool] = None
  update_use_context: Optional[bool] = None
  update_combiner_fn: Optional[Union[str, Callable]] = None
  update_fn_factory: Optional[Callable[[], Callable]] = None
  update_output_feature: Optional[FieldNames] = None
  context_pool_enabled: Optional[UpdateInputEnabled] = None
  context_pool_factory: Optional[Callable[..., Callable]] = None


@dataclasses.dataclass
class GraphUpdateContextOptions:
  """The Context-related subobject of GraphUpdateOptions.

  GraphUpdate and its constituent ContextUpdate layer get Context-related
  options from an object of this class held by a GraphUpdateOptions object.

  For the ContextUpdate (see there for a detailed explanation):
    update_input_fn_factories: Can be set to a list [f1, f2, ...] of input
      factories, for use as ContextUpdate(..., input_fns=[f1(), f2(), ...]).
      Otherwise, a default applies that may involve the features of
      node sets and edge sets, each one pooled per context_pool_factory()
      from the respective EdgeSetOptions and NodeSetOptions.
    update_use_recurrent_state: If input_fns are not set, this boolean controls
      whether the previous context state is part of the default input_fns.
    update_combiner_fn: If set, the ContextUpdate(..., combiner_fn=...).
      For no combiner, use `"none"`; literal `None` means unset.
    update_fn_factory: If set, called without arguments for use as
      ContextUpdate(..., update_fn=update_fn_factory()).
    update_output_feature: If set, the ContextUpdate(..., output_feature=...).
  """
  update_input_fn_factories: Optional[List[Callable]] = None
  update_use_recurrent_state: Optional[bool] = None
  update_combiner_fn: Optional[Union[str, Callable]] = None
  update_fn_factory: Optional[Callable[[], Callable]] = None
  update_output_feature: Optional[FieldNames] = None


# The dataclass fields of GraphUpdateOptions are defined in this private
# base class so that GraphUpdateOptions.__eq__ can call super().__eq__ etc.
@dataclasses.dataclass()
class _GraphUpdateOptionsData:  # pylint: disable=missing-class-docstring
  context: GraphUpdateContextOptions = dataclasses.field(
      default_factory=GraphUpdateContextOptions)
  node_set_default: GraphUpdateNodeSetOptions = dataclasses.field(
      default_factory=GraphUpdateNodeSetOptions)
  edge_set_default: GraphUpdateEdgeSetOptions = dataclasses.field(
      default_factory=GraphUpdateEdgeSetOptions)
  graph_tensor_spec: Any = None
  update_context: Optional[bool] = None


class GraphUpdateOptions(_GraphUpdateOptionsData):
  """Provides structured initializer args for GraphUpdate and its sublayers.

  Constructing a GraphUpdate layer for a GraphTensor with multiple NodeSets and
  EdgeSets can involve a large number of options. This class organizes them in
  a structured way and applies them while building the multiple EdgeSetUpdate,
  NodeSetUpdate and possibly ContextUpdate sub-layers of the GraphUpdate
  layer.

  Users who construct EdgeSetUpdate, NodeSetUpdate or ContextUpdate layers
  directly can also use this class to configure options centrally and pass
  an options object instead of long initializer lists. See the documentation
  of each class for its particular initializer arguments, how defaults are
  computed from the options stored in this class, and which ultimate
  (usually naive) defaults apply if this class leaves an option unset as well.

  Global options:
    graph_tensor_spec: If set, the graph_tensor_spec=... argument used for
      the connection between edge sets and node sets.
    update_context: If set to True, a GraphUpdate includes a ContextUpdate step.

  Context options:
    context: a GraphUpdateContextOptions object, see its documentation.

  NodeSet options:
    node_set_default: A GraphUpdateNodeSetOptions object with default values
      for all node sets.
    node_sets[node_set_name]: A defaultdict of GraphUpdateNodeSetOptions;
      see that class for the available fields. Accessing a node_set_name
      for the first time creates a new GraphUpdateNodeSetOptions object with
      all fields set to None.
  The method `node_set_with_defaults(node_set_name)` returns the effective
  node set options with default values put in (where available) but does not
  mutate the options stored on this object.

  EdgeSet options:
    edge_set_default: An GraphUpdateEdgeSetOptions object with default values
      for all edge sets.
    edge_sets[edge_set_name]: A map of GraphUpdateEdgeSetOptions object,
      with automatic insertion as described above for node sets.
  The method `edge_set_with_defaults(node_set_name)` returns the effective
  edge set options with default values put in (where available) but does not
  mutate the options stored on this object.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._node_sets = collections.defaultdict(GraphUpdateNodeSetOptions)
    self._edge_sets = collections.defaultdict(GraphUpdateEdgeSetOptions)

  @property
  def node_sets(self) -> Mapping[const.NodeSetName, GraphUpdateNodeSetOptions]:
    return self._node_sets

  @property
  def edge_sets(self) -> Mapping[const.EdgeSetName, GraphUpdateEdgeSetOptions]:
    return self._edge_sets

  def node_set_with_defaults(self, node_set_name: str):
    return _lookup_with_defaults(self._node_sets, node_set_name,
                                 self.node_set_default)

  def edge_set_with_defaults(self, edge_set_name: str):
    return _lookup_with_defaults(self._edge_sets, edge_set_name,
                                 self.edge_set_default)

  def __eq__(self, other) -> bool:
    return (super().__eq__(other) and
            _trim_ddict(self._node_sets) == _trim_ddict(other._node_sets) and
            _trim_ddict(self._edge_sets) == _trim_ddict(other._edge_sets))

  def __repr__(self):
    base_repr = super().__repr__()
    assert base_repr[-1] == ")", "Internal error: dataclass repr() changed"
    return (f"{base_repr[:-1]}, "  # Strip trailing ")".
            f"node_sets={_mapping_repr(self._node_sets)}, "
            f"edge_sets={_mapping_repr(self._edge_sets)}"
            ")")  # Add trailing ")".


def _lookup_with_defaults(mapping, key, defaults):
  """Returns value from mapping with defaults for missing value or fields."""
  value = mapping.get(key)
  if value is None:
    return copy.copy(defaults)
  else:
    return _with_defaults(value, defaults)


def _with_defaults(data, defaults):
  """Returns copy of data with unset fields replaced by defaults."""
  updates = {}
  for field in dataclasses.fields(data):
    if getattr(data, field.name) is None:
      update = getattr(defaults, field.name)
      if update is not None:
        updates[field.name] = update
  return dataclasses.replace(data, **updates)


def _trim_ddict(ddict):
  default = ddict.default_factory()
  return {k: v for k, v in ddict.items() if v != default}


def _mapping_repr(mapping):
  """Returns dict-like repr for a mapping."""
  return "{%s}" % ", ".join([f"{repr(k)}: {repr(v)}"
                             for k, v in mapping.items()])


# Some helper functions accept any of these.
GraphUpdatePieceOptions = Union[GraphUpdateEdgeSetOptions,
                                GraphUpdateNodeSetOptions,
                                GraphUpdateContextOptions]
