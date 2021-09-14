"""The GraphUpdate layer and its pieces."""

import abc
import functools
from typing import Callable, Dict, List, Optional, Union

import tensorflow as tf

from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph.keras.layers import graph_ops
from tensorflow_gnn.graph.keras.layers import graph_update_options as opt


class _GraphPieceUpdateBase(tf.keras.layers.Layer, metaclass=abc.ABCMeta):
  """Abstract base class of EdgeSetUpdate, NodeSetUpdate and ContextUpdate.

  This class manages the storage and some shared logic around the attributes
  `input_fns`, `combiner_fn`, `update_fn` and `output_feature`.
  Everything specific to node sets, edge sets or context is delegated to the
  concrete subclasses.

  Init args:
    input_fns, combiner_fn, update_fn, output_feature, options: as publicly
      documented for EdgeSetUpdate, NodeSetUpdate and ContextUpdate.
    piece_options: An options subobject (with defaults merged in) that provides
      the applicable update_* options for the respective graph piece.
    get_default_input_fns: A callback that returns the value for `input_fns`
      in case neither `input_fns` nor `piece_options.update_input_fn_factories`
      is set.
  """

  # TODO(b/193496101): Proper types for sublayers. pylint: disable=g-bare-generic
  def __init__(self,
               *,
               input_fns: Optional[List[Callable]] = None,
               combiner_fn: Optional[Union[str, Callable]] = None,
               update_fn: Optional[Callable] = None,
               output_feature: Optional[opt.FieldNames] = None,
               options: opt.GraphUpdateOptions,
               piece_options: opt.GraphUpdatePieceOptions,
               get_default_input_fns: Callable[[], List[Callable]],
               **kwargs):
    super().__init__(**kwargs)
    self._input_fns = _get_option_from_factory(
        input_fns,
        piece_options.update_input_fn_factories,
        get_default_input_fns)
    self._combiner_fn = _get_combiner_fn(_get_option(
        combiner_fn,
        # NOTE: No factory here, because we're not expecting stateful combiners.
        piece_options.update_combiner_fn,
        lambda: "concatenate"))
    self._update_fn = _get_option_from_factory(
        update_fn,
        piece_options.update_fn_factory,
        else_raise=ValueError(
            "Must provide update_fn as argument or in options"))
    self._output_feature = _get_option(
        output_feature,
        piece_options.update_output_feature,
        lambda: const.DEFAULT_STATE_NAME)

  def get_config(self):
    config = super().get_config().copy()
    config["input_fns"] = self._input_fns
    config["combiner_fn"] = self._combiner_fn
    config["update_fn"] = self._update_fn
    config["output_feature"] = self._output_feature
    return config

  def _apply_update_fn(self,
                       graph: gt.GraphTensor,
                       features: Dict[const.FieldName, const.Field]) -> None:
    """In-place update of `features` with the result of update_fn."""
    net = graph
    if self._input_fns is not None:
      net = tf.nest.map_structure(
          lambda input_fn: self._call_input_fn(input_fn, net),
          self._input_fns)
    if self._combiner_fn is not None:
      net = self._combiner_fn(net)
    if self._update_fn is not None:
      net = self._update_fn(net)

    def _put_feature(k, v):
      features[k] = v
    tf.nest.map_structure(_put_feature, self._output_feature, net)

  @abc.abstractmethod
  def _call_input_fn(self,
                     input_fn: Callable,
                     graph: gt.GraphTensor) -> const.Field:
    """Returns the result of `input_fn` on `graph`."""
    raise NotImplementedError


@tf.keras.utils.register_keras_serializable(package="GNN")
class EdgeSetUpdate(_GraphPieceUpdateBase):
  """Updates the state of an EdgeSet.

  This layer maps a GraphTensor to a GraphTensor in which one EdgeSet has had
  the given output_feature updated.

  The final configuration of this layer consists of edge_set_name, input_fns,
  combiner_fn, update_fn and output_feature. When initializing, they can be set
  explicitly or be inferred from options and defaulting rules explained below
  per argument.

  Init args:
    edge_set_name: The name of the EdgeSet to update. Required.
    input_fns: If set, the input features for the update, specified as a list of
      Keras Layer objects that are called on this layer's input GraphTensor to
      each produce a tensor shaped like an edge feature.
        * Use Readout(feature_name=...) for a feature from the edge to update.
          The edge_set_name will be passed automatically when calling.
        * Use Broadcast(tfgnn.SOURCE, feature_name=...) for a feature from the
          source node, or analogously for a feature from the TARGET node or from
          the CONTEXT. The edge_set_name will be passed automatically.
      If unset, the edge set option update_input_fn_factories is inspected.
      If also unset in options, a default value is formed to obtain
      the following input features:
        * The tfgnn.DEFAULT_STATE_NAME feature from all incident nodes
          whose tags are listed in edge set option update_use_node_tags,
          or [tfgnn.SOURCE, tfgnn.TARGET] if that option is unset.
        * The tfgnn.DEFAULT_STATE_NAME feature from the target node.
        * If edge set option update_use_recurrent_state is set, the
          tfgnn.DEFAULT_STATE_NAME feature of the edge itself.
        * If edge set option update_use_context is set, the
          tfgnn.DEFAULT_STATE_NAME feature from the graph component's context.
    combiner_fn: Specifies how to combine the list of results from the input_fns
      for input to the update_fn. Can be set to a Keras layer accepting that
      list or one of the strings
      "concatenate" (short for tf.keras.layers.Concatenate()) or
      "none" (the list is passed through to update_fn).
      If unset in args, options.edge_sets[edge_set_name].update_combiner_fn is
      inspected. If also unset in options, defaults to "concatenate".
    update_fn: A Keras Layer to compute the `output_feature` from the combined
      inputs. If unset in args, the edge set option update_fn_factory must be
      set and its result is used instead.
    output_feature: The feature name (or list of names) to which the output
      (or list of outputs, resp.) from update_fn is stored in the edge set
      of the output GraphTensor.
      If unset in args, the edge set option update_output_feature is inspected.
      If also unset in options, the tfgnn.DEFAULT_STATE_NAME feature name is
      used.
    options: Optionally, a GraphUpdateOptions object, to control default values
      for the other arguments in a centralized way. The options object itself
      is only used during initialization and does not become part of this
      layer's state.
  """

  # TODO(b/192858913): Debug printing for the effective args (like get_config).
  # TODO(b/193496101): Proper types for sublayers. pylint: disable=g-bare-generic
  def __init__(self,
               edge_set_name: const.EdgeSetName,
               *,
               input_fns: Optional[List[Callable]] = None,
               combiner_fn: Optional[Union[str, Callable]] = None,
               update_fn: Optional[Callable] = None,
               output_feature: Optional[opt.FieldNames] = None,
               options: Optional[opt.GraphUpdateOptions] = None,
               graph_tensor_spec: Optional[gt.GraphTensorSpec] = None,
               **kwargs):
    self._edge_set_name = edge_set_name
    if options is None: options = opt.GraphUpdateOptions()
    get_default_input_fns = functools.partial(
        _get_default_input_fns_for_edge_set,
        edge_set_name=edge_set_name, options=options)
    edge_set_options = options.edge_set_with_defaults(edge_set_name)
    super().__init__(input_fns=input_fns,
                     combiner_fn=combiner_fn,
                     update_fn=update_fn,
                     output_feature=output_feature,
                     options=options,
                     piece_options=edge_set_options,
                     get_default_input_fns=get_default_input_fns,
                     **kwargs)

  def get_config(self):
    config = super().get_config()  # Our base class returns a private copy.
    config["edge_set_name"] = self._edge_set_name
    return config

  def call(self, graph: gt.GraphTensor) -> gt.GraphTensor:
    features = graph.edge_sets[self._edge_set_name].get_features_dict()
    self._apply_update_fn(graph, features)
    return graph.replace_features(edge_sets={self._edge_set_name: features})

  def _call_input_fn(self,
                     input_fn: Callable,
                     graph: gt.GraphTensor) -> const.Field:
    if isinstance(input_fn, graph_ops.UpdateInputLayerExtended):
      # pylint: disable=protected-access
      return input_fn._call_for_edge_set(graph,
                                         edge_set_name=self._edge_set_name)
    else:
      return input_fn(graph)


def _get_default_input_fns_for_edge_set(*, edge_set_name, options):
  """Returns default value for EdgeSetUpdate(input_fns=...); see there."""
  edge_set_options = options.edge_set_with_defaults(edge_set_name)
  input_fns = []

  origin_tags = _get_option(None, edge_set_options.update_use_node_tags,
                            lambda: [const.SOURCE, const.TARGET])
  for tag in origin_tags:
    input_fns.append(graph_ops.Broadcast(tag))
  if edge_set_options.update_use_recurrent_state:  # Default `None` is false.
    input_fns += [graph_ops.Readout()]
  if edge_set_options.update_use_context:  # Default `None` is false.
    input_fns += [graph_ops.Broadcast(const.CONTEXT)]
  return input_fns


@tf.keras.utils.register_keras_serializable(package="GNN")
class NodeSetUpdate(_GraphPieceUpdateBase):
  """Updates the state of a NodeSet.

  This layer maps a GraphTensor to a GraphTensor in which one NodeSet has had
  the given output_feature updated.

  The final configuration of this layer consists of node_set_name, input_fns,
  combiner_fn, update_fn and output_feature. When initializing, they can be set
  explicitly or be inferred from options and defaulting rules explained below
  per argument.

  Init args:
    node_set_name: The name of the NodeSet to update. Required.
    input_fns: If set, the input features for the update, specified as a list of
      Keras Layer objects that are called on this layer's input GraphTensor to
      each produce tensor shaped like a node feature.
        * Use Readout(feature_name=...) for a feature from the node to update.
          The node_set_name will be passed automatically when calling.
        * Use Pool(tfgnn.TARGET, edge_set_name=...., feature_name=...) for a
          feature from incoming edges of a particular edge set, or analogously
          with SOURCE for outgoing edes.
        * Use Broadcast(tfgnn.CONTEXT, feature_name=...) for a context feature.
          The node_set_name will be passed automatically when calling.
      If unset, the node set option update_input_fn_factories is inspected.
      If also unset in options, a default value is formed to obtain the
      following input features:
        * The tfgnn.DEFAULT_STATE_NAME feature from the node, unless the
          node set option update_use_recurrent_state is explicitly set to False.
        * If the node set option update_use_context is set, the
          tfgnn.DEFAULT_STATE_NAME feature from the graph component's context.
        * The pooled tfgnn.DEFAULT_STATE_NAME features from all incident edge
          sets that list the the applicable IncidentNodeTag in their option
          node_pool_tags (if unset, node_pool_tags=[tfgnn.TARGET] is assumed)
          and that do not set node_pool_enabled=NEVER.
          For each, the pooler is built by calling the option node_pool_factory
          with arguments (tag=..., edge_set_name=...). If node_pool_factory
          is unset, it defaults to Pool(tag, "sum", edge_set_name=...).
          The option value node_pool_enabled=ON_UPDATE must not be set when
          passing options directly to a NodeSetUpdate (because it cannot tell
          updated from non-updated states).
    combiner_fn: Specifies how to combine the list of results from the input_fns
      for input to the update_fn. Can be set to a Keras layer accepting that
      list or one of the strings
      "concatenate" (short for tf.keras.layers.Concatenate()) or
      "none" (the list is passed through to update_fn).
      If unset in args, the node set option update_combiner_fn is inspected.
      If also unset in options, defaults to "concatenate".
    update_fn: A Keras Layer to compute the `output_feature` from the combined
      inputs. If unset in args, the node set option update_fn_factory must be
      set and its result is used instead.
    output_feature: The feature name (or list of names) to which the output
      (or list of outputs, resp.) from update_fn is stored in the node set
      of the output GraphTensor.
      If unset in args, the node set option update_output_feature is inspected.
      If also unset in options, the tfgnn.DEFAULT_STATE_NAME feature name is
      used.
    options: optionally, a GraphUpdateOptions object, to control default values
      for the other arguments in a centralized way. The options object itself
      is only used during initialization and does not become part of this
      layer's state.
    graph_tensor_spec: A spec for the GraphTensor to be passed in, to define
      how edge sets connect node sets. This is required as a direct argument
      or via options if input_fns are not passed explicitly as an argument
      or via options.
  """

  # TODO(b/193496101): Proper types for sublayers. pylint: disable=g-bare-generic
  def __init__(self,
               node_set_name: const.NodeSetName,
               *,
               input_fns: Optional[List[Callable]] = None,
               combiner_fn: Optional[Union[str, Callable]] = None,
               update_fn: Optional[Callable] = None,
               output_feature: Optional[opt.FieldNames] = None,
               options: Optional[opt.GraphUpdateOptions] = None,
               graph_tensor_spec: Optional[gt.GraphTensorSpec] = None,
               **kwargs):
    self._node_set_name = node_set_name
    if options is None: options = opt.GraphUpdateOptions()
    get_default_input_fns = functools.partial(
        _get_default_input_fns_for_node_set,
        node_set_name=node_set_name, options=options,
        graph_tensor_spec=graph_tensor_spec)
    node_set_options = options.node_set_with_defaults(node_set_name)
    super().__init__(input_fns=input_fns,
                     combiner_fn=combiner_fn,
                     update_fn=update_fn,
                     output_feature=output_feature,
                     options=options,
                     piece_options=node_set_options,
                     get_default_input_fns=get_default_input_fns,
                     **kwargs)

  def get_config(self):
    config = super().get_config()  # Our base class returns a private copy.
    config["node_set_name"] = self._node_set_name
    return config

  def call(self, graph: gt.GraphTensor) -> gt.GraphTensor:
    features = graph.node_sets[self._node_set_name].get_features_dict()
    self._apply_update_fn(graph, features)
    return graph.replace_features(node_sets={self._node_set_name: features})

  def _call_input_fn(self,
                     input_fn: Callable,
                     graph: gt.GraphTensor) -> const.Field:
    if isinstance(input_fn, graph_ops.UpdateInputLayerExtended):
      # pylint: disable=protected-access
      return input_fn._call_for_node_set(graph,
                                         node_set_name=self._node_set_name)
    else:
      return input_fn(graph)


def _get_default_input_fns_for_node_set(*, node_set_name, options,
                                        graph_tensor_spec,
                                        updated_edge_sets=None):
  """Returns default value for NodeSetUpdate(input_fns=...); see there.."""
  spec = _get_option(
      graph_tensor_spec,
      options.graph_tensor_spec,
      else_raise=ValueError("To use the default `input_fns` of NodeSetUpdate, "
                            "a `graph_tensor_spec` must be provided."))

  # Collect inputs.
  input_fns = []
  node_set_options = options.node_set_with_defaults(node_set_name)
  # From this node set itself.
  # Interpret default `None` as true.  pylint: disable=g-bool-id-comparison
  if node_set_options.update_use_recurrent_state is not False:
    input_fns += [graph_ops.Readout()]
  # From context.
  if node_set_options.update_use_context:  # Default `None` is false.
    input_fns += [graph_ops.Broadcast(const.CONTEXT)]
  # From edge sets.
  for edge_set_name, tag in _get_edge_set_name_and_tag_for_default_node_inputs(
      [node_set_name], options, spec):
    edge_set_options = options.edge_set_with_defaults(edge_set_name)
    if updated_edge_sets is not None:
      edge_input_enabled = _pool_input_enabled(
          value=edge_set_options.node_pool_enabled,
          default_value=opt.UpdateInputEnabled.ON_UPDATE,
          is_updated=edge_set_name in updated_edge_sets)
    else:
      edge_input_enabled = _pool_input_enabled(
          value=edge_set_options.node_pool_enabled,
          default_value=opt.UpdateInputEnabled.ALWAYS,
          is_updated=None)
    if not edge_input_enabled:
      continue  # Skip this input.
    if edge_set_options.node_pool_factory is not None:
      pool = edge_set_options.node_pool_factory(
          tag=tag, edge_set_name=edge_set_name)
    else:
      pool = graph_ops.Pool(tag, "sum", edge_set_name=edge_set_name)
    input_fns += [pool]

  return input_fns


def _pool_input_enabled(*,
                        value: Optional[opt.UpdateInputEnabled],
                        default_value: opt.UpdateInputEnabled,
                        is_updated: Optional[bool]) -> bool:
  """Returns whether a pooled input is enabled by options.

  Args:
    value: The applicable *_pool_enabled value from options, possibly unset.
    default_value: The value to use in lieu of an unset option.
    is_updated: A boolean whether the input has been updated in the same
      GraphUpdate, or None if used in a single *Update layer.

  Returns:
    True if this input is to be used.
  """
  if value is None:
    value = default_value

  if value == opt.UpdateInputEnabled.NEVER:
    return False
  if value == opt.UpdateInputEnabled.ALWAYS:
    return True
  if value == opt.UpdateInputEnabled.ON_UPDATE:
    if is_updated is None:
      # This case goes away when options are centralized in GraphUpdate.
      raise ValueError(
          "Option UpdateInputEnabled.ON_UPDATE only allowed for GraphUpdate")
    return is_updated
  assert False, "Missing an enum value of UpdateInputEnabled"


def _get_edge_set_name_and_tag_for_default_node_inputs(node_set_names, options,
                                                       graph_tensor_spec):
  """Returns (name, tag) pairs for default input edge sets to node sets."""
  result = []
  for edge_set_name in sorted(graph_tensor_spec.edge_sets_spec):
    edge_set_spec = graph_tensor_spec.edge_sets_spec[edge_set_name]
    edge_options = options.edge_set_with_defaults(edge_set_name)
    destination_tags = _get_option(None, edge_options.node_pool_tags,
                                   lambda: [const.TARGET])
    for tag in destination_tags:
      if edge_set_spec.adjacency_spec.node_set_name(tag) in node_set_names:
        result.append((edge_set_name, tag))
  return result


def _get_node_set_names_and_tags_for_default_edge_readers(
    edge_set_names, options, graph_tensor_spec):
  """Returns (name, tag) pairs of node sets that read from these edge sets."""
  result = []
  for edge_set_name in edge_set_names:
    edge_set_spec = graph_tensor_spec.edge_sets_spec[edge_set_name]
    edge_options = options.edge_set_with_defaults(edge_set_name)
    destination_tags = _get_option(None, edge_options.node_pool_tags,
                                   lambda: [const.TARGET])
    for tag in destination_tags:
      node_set_name = edge_set_spec.adjacency_spec.node_set_name(tag)
      result.append((node_set_name, tag))
  return result


@tf.keras.utils.register_keras_serializable(package="GNN")
class ContextUpdate(_GraphPieceUpdateBase):
  """Updates the graph context.

  This layer maps a GraphTensor to a GraphTensor in which the Context has had
  the given output_feature updated.

  The final configuration of this layer consists of input_fns, combiner_fn,
  update_fn and output_feature. When initializing, they can be set explicitly
  or be inferred from options and defaulting rules explained below per argument.

  Init args:
    input_fns: If set, the input features for the update, specified as a list of
      Keras Layer objects that are called on this layer's input GraphTensor to
      each produce a component-indexed tensor (shaped like a context feature).
        * Use Readout(feature_name=...) for a context feature.
          The flag from_context=True will be passed automatically when calling.
        * Use Pool(tfgnn.CONTEXT, node_set_name=...., feature_name=...) for a
          feature from all nodes of a particular node set in each component,
          or analogously with edge_set_name for an edge set.
      If unset, options.context.update_input_fn_factories is inspected.
      If also unset in options, a default value is formed to obtain the
      following input features:
        * The tfgnn.DEFAULT_STATE_NAME feature from the context, unless the
          context option update_use_recurrent_state is explicitly set to False.
        * The pooled features from all node sets with option
          context_pool_enabled=ALWAYS (which is the default).
          For each, the node set option context_pool_factory is called with
          argument (node_set_name=...) to build a pooler. If unset, the pooler
          defaults to tfgnn.Pool(tfgnn.CONTEXT, "sum", node_set_name=...).
        * The pooled features from all edge sets with option
          context_pool_enabled=ALWAYS set explicitly (the default is NEVER).
          For each, the edge set option context_pool_factory is called with
          argument (edge_set_name=...) to build a pooler. If unset, the pooler
          defaults to tfgnn.Pool(tfgnn.CONTEXT, "sum", edge_set_name=...).
      Note that context_pool_enabled=ON_UPDATE must not be set in edge set or
      node set options passed directly to a ContextUpdate (because it cannot
      tell updated from non-updated states).
    combiner_fn: Specifies how to combine the list of results from the input_fns
      for input to the update_fn. Can be set to a Keras layer accepting that
      list or one of the strings
      "concatenate" (short for tf.keras.layers.Concatenate()) or
      "none" (the list is passed through to update_fn).
      If unset in args, the context option update_combiner_fn is inspected.
      If also unset in options, defaults to "concatenate".
    update_fn: A Keras Layer to compute the `output_feature` from the combined
      inputs. If unset in args, the context option update_fn_factory must be
      set and its result is used instead.
    output_feature: The feature name (or list of names) to which the output
      (or list of outputs, resp.) from update_fn is stored in the context
      of the output GraphTensor.
      If unset in args, the context option update_output_feature is inspected.
      If also unset in options, the tfgnn.DEFAULT_STATE_NAME feature name is
      used.
    options: optionally, a GraphUpdateOptions object, to control default values
      for the other arguments in a centralized way. The options object itself
      is only used during initialization and does not become part of this
      layer's state.
    graph_tensor_spec: A spec for the GraphTensor to be passed in, to define the
      node sets and edge sets. This is required as an argument or via options
      if input_fns are not passed explicitly as an argument or via options.
  """

  # TODO(b/193496101): Proper types for sublayers. pylint: disable=g-bare-generic
  def __init__(self,
               *,
               input_fns: Optional[List[Callable]] = None,
               combiner_fn: Optional[Union[str, Callable]] = None,
               update_fn: Optional[Callable] = None,
               output_feature: Optional[opt.FieldNames] = None,
               options: Optional[opt.GraphUpdateOptions] = None,
               graph_tensor_spec: Optional[gt.GraphTensorSpec] = None,
               **kwargs):
    if options is None: options = opt.GraphUpdateOptions()
    get_default_input_fns = functools.partial(
        _get_default_input_fns_for_context,
        options=options, graph_tensor_spec=graph_tensor_spec)
    super().__init__(input_fns=input_fns,
                     combiner_fn=combiner_fn,
                     update_fn=update_fn,
                     output_feature=output_feature,
                     options=options,
                     piece_options=options.context,
                     get_default_input_fns=get_default_input_fns,
                     **kwargs)

  # get_config() is inherited unchanged from the base class.

  def call(self, graph: gt.GraphTensor) -> gt.GraphTensor:
    features = graph.context.get_features_dict()
    self._apply_update_fn(graph, features)
    return graph.replace_features(context=features)

  def _call_input_fn(self,
                     input_fn: Callable,
                     graph: gt.GraphTensor) -> const.Field:
    if isinstance(input_fn, graph_ops.UpdateInputLayerExtended):
      # pylint: disable=protected-access
      return input_fn._call_for_context(graph)
    else:
      return input_fn(graph)


def _get_default_input_fns_for_context(
    *, options, graph_tensor_spec,
    updated_edge_sets=None, updated_node_sets=None):
  """Returns default value for ContextUpdate(input_fns=...); see there.."""
  spec = _get_option(
      graph_tensor_spec,
      options.graph_tensor_spec,
      else_raise=ValueError("To use the default `input_fns` of ContextUpdate, "
                            "a `graph_tensor_spec` must be provided."))

  # Collect inputs.
  input_fns = []
  # From context iself.
  # Interpret default `None` as true.  pylint: disable=g-bool-id-comparison
  if options.context.update_use_recurrent_state is not False:
    input_fns += [graph_ops.Readout()]
  # From node sets.
  for node_set_name in sorted(spec.node_sets_spec):
    node_options = options.node_set_with_defaults(node_set_name)
    if updated_node_sets is not None:
      node_input_enabled = _pool_input_enabled(
          value=node_options.context_pool_enabled,
          default_value=opt.UpdateInputEnabled.ON_UPDATE,
          is_updated=node_set_name in updated_node_sets)
    else:
      node_input_enabled = _pool_input_enabled(
          value=node_options.context_pool_enabled,
          default_value=opt.UpdateInputEnabled.ALWAYS,
          is_updated=None)
    if node_input_enabled:
      if node_options.context_pool_factory is not None:
        pool = node_options.context_pool_factory(node_set_name=node_set_name)
      else:
        pool = graph_ops.Pool(const.CONTEXT, "sum", node_set_name=node_set_name)
      input_fns += [pool]
  # From edge sets.
  for edge_set_name in sorted(spec.edge_sets_spec):
    if updated_edge_sets is not None:
      is_updated = edge_set_name in updated_edge_sets
    else:
      is_updated = None
    edge_options = options.edge_set_with_defaults(edge_set_name)
    if _pool_input_enabled(value=edge_options.context_pool_enabled,
                           default_value=opt.UpdateInputEnabled.NEVER,
                           is_updated=is_updated):
      if edge_options.context_pool_factory is not None:
        pool = edge_options.context_pool_factory(edge_set_name=edge_set_name)
      else:
        pool = graph_ops.Pool(const.CONTEXT, "sum",
                              edge_set_name=edge_set_name)
      input_fns += [pool]
  return input_fns


def _get_combiner_fn(combiner_fn):
  """Returns combiner_fn, with str shorthands replaced by proper values."""
  if isinstance(combiner_fn, str):
    if combiner_fn == "concatenate":
      return tf.keras.layers.Concatenate()
    elif combiner_fn == "none":
      return None
    else:
      raise ValueError(f"Unknown string for combiner_fn: {combiner_fn}")
  else:
    return combiner_fn


@tf.keras.utils.register_keras_serializable(package="GNN")
class GraphUpdate(tf.keras.layers.Layer):
  """Updates the states of certain edge sets, node sets and possibly context.

  This layer maps a GraphTensor to a GraphTensor by applying, in this order:

    1. EdgeSetUpdate layers for the selected edge sets,
    2. NodeSetUpdate layers for the selected node sets,
    3. a ContextUpdate layer if update_context is true.

  The layers are constructed according to the `options` provided.

  By default, a GraphUpdate layer updates all edge sets and node sets.
  This avoids several complications of partial updates that are discussed
  in the next paragraphs. In particular, it completely solves the case of
  homogeneous graphs (i.e., those with just one node set and one edge set).

  In some cases, the model can be optimized by updating just some node sets or
  edge sets. For example, consider node sets "a", "b", "c" connected by
  edge sets "a->b" and "b->c" in a chain, and suppose the model's prediction
  is read out from c. For the last GraphUpdate, it suffices to update edge set
  "b->c" and node set "c". Conversely, for the first GraphUpdate it may make
  sense to update only edge set "a->b" and node set "b", say, if only "a" has
  a meaningful initial state set from input features.

  For such cases, the caller can set node_set_names to update
    - exactly these node sets,
    - prior to that, all edge sets from which these node sets take an input,
    - lastly, the context (if update_context is true).

  Alternatively, the caller can set edge_set_names to update
    - exactly these edge sets,
    - then, all node sets that take an input from any of these edge sets,
    - lastly, the context (if update_context is true).

  For maximum control, the caller can set both edge_set_names and node_set_names
  (and also update_context); precisely those updates will be done.

  With manually specified edge_set_names, it may happen that not all
  EdgeSetUpdates usually expected for each NodeSetUpdate have occurred.
  The NodeSetUpdates constructed as part of a GraphUpdate only read from
  edge sets whose states have been updated by the same GraphUpdate.
  The edge set option `node_pool_enabled` can be set to `ALWAYS` or `NEVER`
  to override that default `ON_UPDATE` behavior.

  Likewise, there are edge set and node set options `context_pool_enable`
  to control the inputs to ContextUpdates. By default, the ContextUpdate
  reads `ON_UPDATE` from node sets and `NEVER` from edge sets.

  All inputs configured by `options` according to the incidence of node sets
  and edge sets must be present in the GraphTensor when the respective update
  is executed. It falls on the caller to pass in dummy feature values (usually,
  tensors of appropriate shape filled with zeros) when no meaningful value is
  available for a state.

  With suitable options, a sequence of GraphUpdate layers can implement
    - the Message Passing Neural Network (MPNN) algorithm of Gilmer&al. (2017),
    - the GraphNetworks algorithm of Battaglia&al. (2018),
  generalized to heterogeneous graphs (i.e., those having multiple node sets
  and edge sets).

  For MPNN, a GraphUpdate layer performs one round of message passing along
  one or more edge sets, followed by a state update of the receiving node sets.
  The message of an edge is computed as its state by an EdgeSetUpdate.
  No other edge states (say, from earlier rounds) are meant to be received.
  This is achieved by the following edge set options:
    - update_use_recurrent_state=False (default),
    - node_pool_enabled=ON_UPDATE (default),
    - context_pool_enabled=NEVER (default).

  For GraphNetworks, a GraphUpdate layer performs one round of edge, node and
  context updates, with edges having a state that may get reused in the next
  round. To do this, set some or all of the following edge set options:
    - update_use_recurrent_state=True,
    - node_pool_enabled=ALWAYS,
    - context_pool_enabled=ALWAYS (both in edge set and node set options).
  The context state can be used as input by EdgeSetUpdates and NodeSetUpdates
  by setting the respective option update_use_context. To do a subsequent
  ContextUpdate, set update_context in options or as a direct argument.

  Init args:
    edge_set_names: If set, defines the edge_sets to update. If unset, the
      defaulting rules from above are applied (using node_set_names).
    node_set_names: If set, defines the edge_sets to update. If unset, the
      defaulting rules from above are applied (using edge_set_names).
    update_context: If set to true, a ContextUpdate is done.
      If unset, options.update_context is inspected instead.
      If also unset in options, defaults to false.
    options: A GraphUpdateOptions object, to configure the subordinate update
      layers a centralized way. The options object itself is only used during
      initialization and does not become part of this layer's state.
      The options must provide:
        * an update_fn_factory for each edge set, node set and context
          for which an update layer is constructed;
        * a graph_tensor_spec, if not passed as a direct argument.
    graph_tensor_spec: A spec for the GraphTensor to be passed in, to define
      how edge sets connect node sets. This is required as a direct argument
      or via options.
  """

  def __init__(self,
               *,
               edge_set_names: Optional[List[const.EdgeSetName]] = None,
               node_set_names: Optional[List[const.NodeSetName]] = None,
               update_context: Optional[bool] = None,
               options: Optional[opt.GraphUpdateOptions],  # Actually required.
               graph_tensor_spec: Optional[gt.GraphTensorSpec] = None,
               **kwargs):
    from_config = kwargs.pop("_from_config", None)
    sublayers = kwargs.pop("sublayers", None)
    super().__init__(**kwargs)

    if sublayers is not None:
      if not from_config:
        raise TypeError(
            "GraphUpdate.__init__() got an argument 'sublayers' "
            "that is only allowed for internal use by from_config()")
      self._sublayers = sublayers
      return

    if options is None: options = opt.GraphUpdateOptions()
    graph_tensor_spec = _get_option(
        graph_tensor_spec,
        options.graph_tensor_spec,
        else_raise=ValueError("GraphUpdate() requires a graph_tensor_spec."))
    update_context = _get_option(
        update_context,
        options.update_context,
        lambda: False)

    # Create sublayers after handling the various combinations of set or unset
    # args for edge_set_names and node_set_names.
    init_kwargs = dict(
        update_context=update_context, options=options,
        graph_tensor_spec=graph_tensor_spec)
    if edge_set_names is not None and node_set_names is not None:
      # Fully explicit.
      self._init_from_edge_sets_and_node_sets(
          edge_set_names=edge_set_names,
          node_set_names=node_set_names, **init_kwargs)
    elif edge_set_names is None and node_set_names is None:
      # Fully implicit: update all.
      self._init_from_edge_sets_and_node_sets(
          edge_set_names=list(graph_tensor_spec.edge_sets_spec),
          node_set_names=list(graph_tensor_spec.node_sets_spec), **init_kwargs)
    elif edge_set_names is not None:
      # Explicit edge_set_names, inferred node_set_names.
      node_inputs = _get_node_set_names_and_tags_for_default_edge_readers(
          edge_set_names, options, graph_tensor_spec)
      node_set_names = sorted(set(
          [node_set_name for node_set_name, tag in node_inputs]))
      self._init_from_edge_sets_and_node_sets(
          edge_set_names=edge_set_names,
          node_set_names=node_set_names, **init_kwargs)
    else:
      # Explicit node_set_names, inferred edge_set_names.
      assert node_set_names is not None
      edge_inputs = _get_edge_set_name_and_tag_for_default_node_inputs(
          node_set_names, options, graph_tensor_spec)
      edge_set_names = sorted(set(
          [edge_set_name for edge_set_name, tag in edge_inputs]))
      self._init_from_edge_sets_and_node_sets(
          edge_set_names=edge_set_names,
          node_set_names=node_set_names, **init_kwargs)

  def _init_from_edge_sets_and_node_sets(
      self, *, edge_set_names, node_set_names, update_context,
      options, graph_tensor_spec):
    self._sublayers = []

    # Update edge sets.
    for edge_set_name in edge_set_names:
      input_fns = _get_default_input_fns_for_edge_set(
          edge_set_name=edge_set_name, options=options)
      edge_set_options = options.edge_set_with_defaults(edge_set_name)
      if edge_set_options.update_fn_factory is None:
        raise ValueError(
            "GraphUpdate requires edge set option update_fn_factory; "
            f"found none for edge set '{edge_set_name}'.")
      self._sublayers.append(EdgeSetUpdate(
          edge_set_name, input_fns=input_fns,
          combiner_fn=edge_set_options.update_combiner_fn,
          update_fn=edge_set_options.update_fn_factory(),
          output_feature=edge_set_options.update_output_feature))

    # Update node sets.
    for node_set_name in node_set_names:
      input_fns = _get_default_input_fns_for_node_set(
          node_set_name=node_set_name, options=options,
          graph_tensor_spec=graph_tensor_spec,
          updated_edge_sets=edge_set_names)
      node_set_options = options.node_set_with_defaults(node_set_name)
      if node_set_options.update_fn_factory is None:
        raise ValueError(
            "GraphUpdate requires node set option update_fn_factory; "
            f"found none for node set '{node_set_name}'.")
      self._sublayers.append(NodeSetUpdate(
          node_set_name, input_fns=input_fns,
          combiner_fn=node_set_options.update_combiner_fn,
          update_fn=node_set_options.update_fn_factory(),
          output_feature=node_set_options.update_output_feature))

    # Update context.
    if update_context:
      input_fns = _get_default_input_fns_for_context(
          options=options, graph_tensor_spec=graph_tensor_spec,
          updated_edge_sets=edge_set_names, updated_node_sets=node_set_names)
      if options.context.update_fn_factory is None:
        raise ValueError(
            "GraphUpdate requires options.context.update_fn_factory; "
            "found none.")
      self._sublayers.append(ContextUpdate(
          input_fns=input_fns,
          combiner_fn=options.context.update_combiner_fn,
          update_fn=options.context.update_fn_factory(),
          output_feature=options.context.update_output_feature))

  def get_config(self):
    # This layer is serialized as its constituent sequence of sublayers,
    # not as the options that created them, some of which are not directly
    # serializable (e.g., consider the factories). Keras takes care of
    # serializing their configs and preserving the identity of shared objects.
    config = super().get_config().copy()
    config["sublayers"] = self._sublayers
    return config

  @classmethod
  def from_config(cls, config):
    sublayers = config["sublayers"]
    if not (isinstance(sublayers, list) and
            all(isinstance(s, tf.keras.layers.Layer) for s in sublayers)):
      raise ValueError(
          "GraphUpdate.from_config() expects config['sublayers'] "
          "to be a list of Keras Layers")
    return cls(_from_config=True, options=None, **config)

  def call(self, graph):
    for sublayer in self._sublayers:
      graph = sublayer(graph)
    return graph


def _get_option(value_from_args,
                value_from_options,
                default_fn=None,
                *,
                else_raise=None):
  """Gets a value from args, Options, a default rule, or raises."""
  if value_from_args is not None:
    return value_from_args
  elif value_from_options is not None:
    return value_from_options
  elif default_fn is not None:
    return default_fn()
  elif else_raise is not None:
    raise else_raise
  else:
    raise AssertionError("Internal error: set one of default_fn, else_raise.")


def _get_option_from_factory(value_from_arguments,
                             factory_from_options,
                             default_fn=None,
                             *,
                             else_raise=None):
  """Gets a value from args, a factory in Options, a default rule, or raises."""
  if value_from_arguments is not None:
    return value_from_arguments
  elif factory_from_options is not None:
    if isinstance(factory_from_options, list):
      return [f() for f in factory_from_options]
    else:
      if callable(factory_from_options):
        return factory_from_options()
      else:
        raise ValueError("Expected a callable as factory in options, "
                         f"got {repr(factory_from_options)}")
  elif default_fn is not None:
    return default_fn()
  elif else_raise is not None:
    raise else_raise
  else:
    raise AssertionError("Internal error: set one of default_fn, else_raise.")
