"""The GraphUpdate layer and its pieces."""

from typing import Any, Callable, Mapping, Optional, Sequence

import tensorflow as tf

from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import graph_tensor_ops as ops
from tensorflow_gnn.graph.keras.layers import next_state as next_state_lib


# This file defines the canonical implementations of EdgeSetUpdate,
# NodeSetUpdate, and ContextUpdate. However, users are free to pass objects
# of their own compatible reimplementations as long as they adhere to
# the following interfaces:
#
# class AbstractEdgeSetUpdate(tf.keras.layers.Layer):
#   def call(self, graph: GraphTensor, *, edge_set_name: EdgeSetName
#   ) -> FieldOrFields:  # Results shaped like edge features.
#     raise NotImplementedError()
#
# class AbstractNodeSetUpdate(tf.keras.layers.Layer):
#   def call(self, graph: GraphTensor, *, node_set_name: NodeSetName
#   ) -> FieldOrFields:  # Results shaped like node features.
#     raise NotImplementedError()
#
# class AbstractContextUpdate(tf.keras.layers.Layer):
#   def call(self, graph: GraphTensor
#   ) -> FieldOrFields:  # Results shaped like context features.
#     raise NotImplementedError()
#
# For now, we don't want to require that custom classes of these kinds inherit
# from abstract interfaces. The best we can do in type annotations is to use
# the names above, reminding the programmer of the interfaces described there.
AbstractEdgeSetUpdate = tf.keras.layers.Layer
AbstractNodeSetUpdate = tf.keras.layers.Layer
AbstractContextUpdate = tf.keras.layers.Layer


# The NodeSetUpdate and ContextUpdate layers are initialized with maps of
# input layers from those graph pieces that are in a many-to-one relation
# with the updated graph piece (e.g., many incoming edges per node, many
# nodes per graph component). There is a variety of such input layers,
# including user-defined ones, and they are expected to provide the
# following interfaces:
#
# class EdgesToNodePooling(tf.keras.layers.Layer):
#   def call(self, graph: GraphTensor, *, edge_set_name: EdgeSetName
#   ) -> FieldOrFields:  # Results shaped like node features.
#     raise NotImplementedError()
#
# class NodesToContextPooling(tf.keras.layers.Layer):
#   def call(self, graph: GraphTensor, *, node_set_name: NodeSetName
#   ) -> FieldOrFields:  # Results shaped like context features.
#     raise NotImplementedError()
#
# class EdgesToContextPooling(tf.keras.layers.Layer):
#   def call(self, graph: GraphTensor, *, edge_set_name: EdgeSetName
#   ) -> FieldOrFields:  # Results shaped like context features.
#     raise NotImplementedError()
#
# For now, we don't want to require that classes of these kinds inherit
# from abstract interfaces. The best we can do in type annotations is to use
# the names above, reminding the programmer of the interfaces described there.
EdgesToNodePooling = tf.keras.layers.Layer
NodesToContextPooling = tf.keras.layers.Layer
EdgesToContextPooling = tf.keras.layers.Layer


@tf.keras.utils.register_keras_serializable(package="GNN")
class GraphUpdate(tf.keras.layers.Layer):
  """Applies one round of updates to EdgeSets, NodeSets and Context.

  The updates of EdgeSets, NodeSets and Context can either be passed as
  init arguments, or constructed later by passing a deferred_init_callback,
  which allows advanced users to adjust the updates to the GraphTensorSpec
  of the input (which EdgeSets and NodeSets even exist).

  Init args:
    edge_sets: A dict `{edge_set_name: edge_set_update, ...}` of EdgeSetUpdate
      layers (or custom reimplementations). They are run on the input graph
      tensor as `edge_set_update(graph, edge_set_name=edge_set_name)`.
      Their results are merged into the feature map of the respective edge set.
      This argument can be omitted, which is common in models with node set
      updates that use convolutions (i.e., read from adjacent nodes without
      computing explicit edge states).
    node_sets: A dict `{node_set_name: node_set_update, ...}` of NodeSetUpdate
      layers (or custom reimplementations). They are run on the graph tensor
      with edge set updates (if any) as
      `node_set_update(graph, node_set_name=node_set_name)`,
      Their results are merged into the feature map of the respective node set.
      This argument can be omitted (but that is uncommon).
    context: A ContextUpdate that is run on the graph tensor with edge set and
      node set updates (if any). Its results are merged back into the context
      feature map. This argument can be omitted, which is common in models
      without a context state.
    deferred_init_callback: Can be set to a function that accepts a
      GraphTensorSpec and returns a dictionary with the kwargs
      edge_sets=..., node_sets=... and context=... that would otherwise be
      passed directly at initialization time. If this argument is set,
      edge_sets, node_sets and context must all be unset.
      The object is initialized upon its first call from the results of
      the callback on the spec of the input. Before that, the object cannot
      be saved.

  Call result:
    A graph tensor with feature maps that have all configured updates merged in:
    If an update returns a str-keyed dict, it gets merged into respective
    feature map with the given names. If an update returns a single tensor,
    the tfgnn.DEFAULT_STATE_NAME is used.
  """

  def __init__(self,
               *,
               edge_sets: Optional[Mapping[const.EdgeSetName,
                                           AbstractEdgeSetUpdate]] = None,
               node_sets: Optional[Mapping[const.NodeSetName,
                                           AbstractNodeSetUpdate]] = None,
               context: Optional[AbstractContextUpdate] = None,
               deferred_init_callback: Optional[
                   Callable[[gt.GraphTensorSpec], Mapping[str, Any]]] = None,
               **kwargs):
    super().__init__(**kwargs)
    if not deferred_init_callback:
      self._deferred_init_callback = None
      self._init_from_updates(edge_sets, node_sets, context)
      assert self._is_initialized
    else:
      self._deferred_init_callback = deferred_init_callback
      self._is_initialized = False
      if (edge_sets, node_sets, context) != (None, None, None):
        raise ValueError(
            "GraphUpdate(deferred_init_callback=...) is mutually exclusive "
            "with any of edge_sets=..., node_sets=..., context=...")
      self._edge_set_updates = None
      self._node_set_updates = None
      self._context_update = None

  def _init_from_updates(self, edge_sets=None, node_sets=None, context=None):
    self._edge_set_updates = dict(edge_sets or {})
    self._node_set_updates = dict(node_sets or {})
    self._context_update = context
    self._is_initialized = True

  def get_config(self):
    if not self._is_initialized:
      raise ValueError(
          "GraphUpdate(deferred_init_callback=...) must be called "
          "to trigger deferred initialization before it can be saved.")
    return dict(
        edge_sets=self._edge_set_updates,
        node_sets=self._node_set_updates,
        context=self._context_update,
        **super().get_config())

  def call(self, graph: gt.GraphTensor) -> gt.GraphTensor:
    if not self._is_initialized:
      with tf.init_scope():
        self._init_from_updates(**self._deferred_init_callback(graph.spec))
        self._deferred_init_callback = None  # Enable garbage collection.
    assert self._is_initialized

    if self._edge_set_updates:
      edge_set_features = {}
      for edge_set_name, update_fn in sorted(self._edge_set_updates.items()):
        features = graph.edge_sets[edge_set_name].get_features_dict()
        features.update(_ensure_dict(
            update_fn(graph, edge_set_name=edge_set_name)))
        edge_set_features[edge_set_name] = features
      graph = graph.replace_features(edge_sets=edge_set_features)

    if self._node_set_updates:
      node_set_features = {}
      for node_set_name, update_fn in sorted(self._node_set_updates.items()):
        features = graph.node_sets[node_set_name].get_features_dict()
        features.update(_ensure_dict(
            update_fn(graph, node_set_name=node_set_name)))
        node_set_features[node_set_name] = features
      graph = graph.replace_features(node_sets=node_set_features)

    if self._context_update:
      context_features = graph.context.get_features_dict()
      context_features.update(_ensure_dict(self._context_update(graph)))
      graph = graph.replace_features(context=context_features)

    return graph


@tf.keras.utils.register_keras_serializable(package="GNN")
class EdgeSetUpdate(tf.keras.layers.Layer):
  """Computes the new state of an EdgeSet from select input features.

  Init args:
    next_state: The NextState layer to apply.
    edge_input_feature: The feature name(s) of inputs from the edge set to
      `next_state`, defaults to `tfgnn.DEFAULT_STATE_NAME`.
      If set to a single feature name, a single tensor is passed.
      If set to `None` or an empty sequence, an empty dict is passed.
      Otherwise, a dict of tensors keyed by feature names is passed.
    node_input_tags: The incident nodes of each edge whose states are used
      as an input, specified as IncidentNodeTags (tfgnn.SOURCE and tfgnn.TARGET
      by default).
    node_input_feature: The feature name of the input from node sets to
      `next_state`, defaults to `tfgnn.DEFAULT_STATE_NAME`.
      Setting this to `None` passes an empty dict of node inputs.
      This class supports only a single input feature from nodes. For more
      complex settings, you need to write your own, or start a design discussion
      about a node_input_map from tags to the respective features for each.
    context_input_feature: The feature name(s) of inputs from the context to
      `next_state`. Defaults to `None`, which passes an empty dict.
      If set to a single feature name, a single tensor is passed.
      Otherwise, a dict of tensors keyed by feature names is passed.
      To pass the default state tensor of the context, set this to
      `tfgnn.DEFAULT_STATE_NAME`.

  Call returns:
    The result of next_state called on the configured inputs.
  """

  def __init__(self,
               next_state: next_state_lib.NextStateForEdgeSet,
               *,
               edge_input_feature: Optional[const.FieldNameOrNames]
               = const.DEFAULT_STATE_NAME,
               node_input_tags: Sequence[const.IncidentNodeTag] = (
                   const.SOURCE, const.TARGET),
               node_input_feature: Optional[
                   const.FieldName] = const.DEFAULT_STATE_NAME,
               context_input_feature: Optional[const.FieldNameOrNames] = None,
               **kwargs):
    super().__init__(**kwargs)
    self._next_state = next_state
    self._edge_input_feature = _copy_if_sequence(edge_input_feature)
    self._node_input_tags = list(node_input_tags)
    if isinstance(node_input_feature, (list, tuple, set)):
      raise ValueError(
          "EdgeSetUpdate does not support multiple node_input_features "
          f"but got: {node_input_feature}")
    self._node_input_feature = node_input_feature
    self._context_input_feature = _copy_if_sequence(context_input_feature)

  def get_config(self):
    return dict(
        next_state=self._next_state,
        edge_input_feature=self._edge_input_feature,
        node_input_tags=self._node_input_tags,
        node_input_feature=self._node_input_feature,
        context_input_feature=self._context_input_feature,
        **super().get_config())

  def call(self, graph: gt.GraphTensor,
           edge_set_name: const.EdgeSetName) -> gt.GraphTensor:
    next_state_inputs = []
    # Input from the edges themselves.
    next_state_inputs.append(
        _get_feature_or_features(graph.edge_sets[edge_set_name],
                                 self._edge_input_feature))
    # Input from incident nodes.
    input_from_incident_nodes = {}
    if self._node_input_feature is not None:
      for node_tag in self._node_input_tags:
        input_from_incident_nodes[node_tag] = ops.broadcast_node_to_edges(
            graph, edge_set_name, node_tag,
            feature_name=self._node_input_feature)
    next_state_inputs.append(input_from_incident_nodes)
    # Input from context.
    next_state_inputs.append(tf.nest.map_structure(
        ops.broadcast_context_to_nodes,
        _get_feature_or_features(graph.context, self._context_input_feature)))

    next_state_inputs = tuple(next_state_inputs)
    assert len(next_state_inputs) == 3, "Internal error"
    return self._next_state(next_state_inputs)


@tf.keras.utils.register_keras_serializable(package="GNN")
class NodeSetUpdate(tf.keras.layers.Layer):
  """A node state update with input from convolutions or other edge set inputs.

  Init args:
    edge_set_inputs: A dict `{edge_set_name: edge_set_input, ...}` of Keras
      layers (such as convolutions) that return values shaped like node features
      with information aggregated from the given edge set.
      They are run in parallel on the input graph tensor as
      `edge_set_input(graph, edge_set_name=edge_set_name)`.
    next_state: A Keras layer to compute the new node state from a tuple of
      inputs that contains, in this order:
        - the `node_input_feature` (see there),
        - a dict `{edge_set_name: input}` with the results of `edge_set_inputs`,
          in which each result is a tensor or dict of tensors,
        - if context_input_feature is not None, those feature(s).
    node_input_feature: The feature name(s) of inputs from the node set to
      `next_state`, defaults to `tfgnn.DEFAULT_STATE_NAME`.
      If set to a single feature name, a single tensor is passed.
      If set to `None` or an empty sequence, an empty dict is passed.
      Otherwise, a dict of tensors keyed by feature names is passed.
    context_input_feature: The feature name(s) of inputs from the context to
      `next_state`. Defaults to `None`, which passes an empty dict.
      If set to a single feature name, a single tensor is passed.
      Otherwise, a dict of tensors keyed by feature names is passed.
      To pass the default state tensor of the context, set this to
      `tfgnn.DEFAULT_STATE_NAME`.

  Call result:
    The tensor or dict of tensors with the new node state, as returned by
    next_state.
  """

  def __init__(self,
               edge_set_inputs: Mapping[const.EdgeSetName, EdgesToNodePooling],
               next_state: next_state_lib.NextStateForNodeSet,
               *,
               node_input_feature: Optional[const.FieldNameOrNames]
               = const.DEFAULT_STATE_NAME,
               context_input_feature: Optional[const.FieldNameOrNames] = None,
               **kwargs):
    super().__init__(**kwargs)
    self._edge_set_inputs = dict(edge_set_inputs)
    self._next_state = next_state
    self._node_input_feature = _copy_if_sequence(node_input_feature)
    self._context_input_feature = _copy_if_sequence(context_input_feature)

  def get_config(self):
    return dict(
        edge_set_inputs=self._edge_set_inputs,
        next_state=self._next_state,
        node_input_feature=self._node_input_feature,
        context_input_feature=self._context_input_feature,
        **super().get_config())

  def call(self, graph: gt.GraphTensor,
           node_set_name: const.NodeSetName) -> gt.GraphTensor:
    next_state_inputs = []
    # Input from the nodes themselves.
    next_state_inputs.append(
        _get_feature_or_features(graph.node_sets[node_set_name],
                                 self._node_input_feature))
    # Input from edge sets.
    input_from_edge_sets = {}
    for edge_set_name, input_fn in sorted(self._edge_set_inputs.items()):
      input_from_edge_sets[edge_set_name] = input_fn(
          graph, edge_set_name=edge_set_name)
    next_state_inputs.append(input_from_edge_sets)
    # Input from context.
    next_state_inputs.append(tf.nest.map_structure(
        ops.broadcast_context_to_nodes,
        _get_feature_or_features(graph.context, self._context_input_feature)))

    next_state_inputs = tuple(next_state_inputs)
    assert len(next_state_inputs) == 3, "Internal error"
    return self._next_state(next_state_inputs)


@tf.keras.utils.register_keras_serializable(package="GNN")
class ContextUpdate(tf.keras.layers.Layer):
  """A context update with input from node sets and/or edge sets.

  Init args:
    node_set_inputs: A dict `{node_set_name: node_set_input, ...}` of Keras
      layers that return values shaped like context features with information
      aggregated from the given edge set. They are run on the input graph tensor
      as `node_set_input(graph, node_set_name=node_set_name)`.
    edge_set_inputs: A dict `{edge_set_name: edge_set_input, ...}` of Keras
      layers that return values shaped like context features with information
      aggregated from the given edge set. They are run on the input graph tensor
      as `edge_set_input(graph, edge_set_name=edge_set_name)`.
    next_state: A Keras layer to compute the new node state from a tuple of
      inputs that contains, in this order:
        - the `context_input_feature` (see there),
        - a dict `{node_set_name: input}` with the results of `node_set_inputs`,
          in which each result is a tensor or dict of tensors,
        - a dict `{edge_set_name: input}` with the results of `edge_set_inputs`,
          in which each result is a tensor or dict of tensors, if there are any.
    context_input_feature: The feature name(s) of inputs from the context to
      `next_state`, defaults to `tfgnn.DEFAULT_STATE_NAME`.
      If set to a single feature name, a single tensor is passed.
      If set to `None` or an empty sequence, an empty dict is passed.
      Otherwise, a dict of tensors keyed by feature names is passed.

  Call result:
    The tensor or dict of tensors with the new node state, as returned by
    next_state.
  """

  def __init__(self,
               node_set_inputs: Mapping[const.NodeSetName,
                                        NodesToContextPooling],
               next_state: next_state_lib.NextStateForContext,
               *,
               edge_set_inputs: Optional[Mapping[const.EdgeSetName,
                                                 EdgesToContextPooling]] = None,
               context_input_feature: Optional[const.FieldNameOrNames]
               = const.DEFAULT_STATE_NAME,
               **kwargs):
    super().__init__(**kwargs)
    self._node_set_inputs = dict(node_set_inputs)
    self._next_state = next_state
    if edge_set_inputs is not None:
      self._edge_set_inputs = dict(edge_set_inputs)
    else:
      self._edge_set_inputs = None
    self._context_input_feature = _copy_if_sequence(context_input_feature)

  def get_config(self):
    return dict(
        node_set_inputs=self._node_set_inputs,
        next_state=self._next_state,
        edge_set_inputs=self._edge_set_inputs,
        context_input_feature=self._context_input_feature,
        **super().get_config())

  def call(self, graph: gt.GraphTensor) -> gt.GraphTensor:
    next_state_inputs = []
    # Input from the context itself.
    next_state_inputs.append(_get_feature_or_features(
        graph.context, self._context_input_feature))
    # Input from node sets.
    input_from_node_sets = {}
    for node_set_name, input_fn in sorted(self._node_set_inputs.items()):
      input_from_node_sets[node_set_name] = input_fn(
          graph, node_set_name=node_set_name)
    next_state_inputs.append(input_from_node_sets)
    # Input from edge sets.
    input_from_edge_sets = {}
    if self._edge_set_inputs:
      for edge_set_name, input_fn in sorted(self._edge_set_inputs.items()):
        input_from_edge_sets[edge_set_name] = input_fn(
            graph, edge_set_name=edge_set_name)
    next_state_inputs.append(input_from_edge_sets)

    next_state_inputs = tuple(next_state_inputs)
    assert len(next_state_inputs) == 3, "Internal error"
    return self._next_state(next_state_inputs)


def _ensure_dict(features):
  if not isinstance(features, Mapping):
    features = {const.DEFAULT_STATE_NAME: features}
  return features


def _copy_if_sequence(names):
  if names is None:
    return None
  elif isinstance(names, const.FieldName):
    return names
  else:
    return list(names)


def _get_feature_or_features(features, names):
  if isinstance(names, const.FieldName):
    return features[names]
  elif not names:  # None or empty.
    return {}
  else:
    return {name: features[name] for name in names}
