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
        * The tfgnn.DEFAULT_STATE_NAME feature from the source node.
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
               **kwargs):
    self._edge_set_name = edge_set_name
    if options is None: options = opt.GraphUpdateOptions()
    edge_set_options = options.edge_set_with_defaults(edge_set_name)
    get_default_input_fns = functools.partial(
        _get_default_input_fns_for_edge_set, edge_set_options=edge_set_options)
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


def _get_default_input_fns_for_edge_set(*, edge_set_options):
  """Returns default value for EdgeSetUpdate(input_fns=...); see there."""
  result = [graph_ops.Broadcast(const.SOURCE),
            graph_ops.Broadcast(const.TARGET)]
  if edge_set_options.update_use_recurrent_state:  # Default `None` is false.
    result += [graph_ops.Readout()]
  if edge_set_options.update_use_context:  # Default `None` is false.
    result += [graph_ops.Broadcast(const.CONTEXT)]
  return result


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
          node_pool_tags. (If unset, node_pool_tags=[tfgnn.TARGET] is assumed.)
          For each, the pooler is built by calling the option node_pool_factory
          with arguments (tag=..., edge_set_name=...). If node_pool_factory
          is unset, it defaults to Pool(tag, "sum", edge_set_name=...),
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
    node_set_options = options.node_set_with_defaults(node_set_name)
    get_default_input_fns = functools.partial(
        _get_default_input_fns_for_node_set,
        node_set_name=node_set_name, options=options,
        graph_tensor_spec=graph_tensor_spec)
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
                                        graph_tensor_spec):
  """Returns default value for NodeSetUpdate(input_fns=...); see there.."""
  spec = _get_option(
      graph_tensor_spec,
      options.graph_tensor_spec,
      else_raise=ValueError("To use the default `input_fns` of NodeSetUpdate, "
                            "a `graph_tensor_spec` must be provided."))

  node_set_options = options.node_set_with_defaults(node_set_name)
  result = []
  # Interpret default `None` as true.  pylint: disable=g-bool-id-comparison
  if node_set_options.update_use_recurrent_state is not False:
    result += [graph_ops.Readout()]
  if node_set_options.update_use_context:  # Default `None` is false.
    result += [graph_ops.Broadcast(const.CONTEXT)]
  # Get pooled results from edge sets.
  for edge_set_name in sorted(spec.edge_sets_spec):
    edge_set_spec = spec.edge_sets_spec[edge_set_name]
    edge_options = options.edge_set_with_defaults(edge_set_name)
    destination_tags = _get_option(None, edge_options.node_pool_tags,
                                   lambda: [const.TARGET])
    for tag in destination_tags:
      if node_set_name != edge_set_spec.adjacency_spec.node_set_name(tag):
        continue
      if edge_options.node_pool_factory is not None:
        pool = edge_options.node_pool_factory(
            tag=tag, edge_set_name=edge_set_name)
      else:
        pool = graph_ops.Pool(tag, "sum", edge_set_name=edge_set_name)
      result += [pool]
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
        * The pooled features from all node sets with option context_pool_enable
          (which is true by default).
          For each, the node set option context_pool_factory is called with
          argument (node_set_name=...) to build a pooler. If unset, the pooler
          defaults to tfgnn.Pool(tfgnn.CONTEXT, "sum", node_set_name=...).
        * The pooled features from all edge sets with option context_pool_enable
          (which is false by default, unless context_pool_factory is set).
          For each, the edge set option context_pool_factory is called with
          argument (edge_set_name=...) to build a pooler. If unset, the pooler
          defaults to tfgnn.Pool(tfgnn.CONTEXT, "sum", edge_set_name=...).
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


def _get_default_input_fns_for_context(*, options, graph_tensor_spec):
  """Returns default value for ContextUpdate(input_fns=...); see there.."""
  spec = _get_option(
      graph_tensor_spec,
      options.graph_tensor_spec,
      else_raise=ValueError("To use the default `input_fns` of ContextUpdate, "
                            "a `graph_tensor_spec` must be provided."))
  result = []
  # Interpret default `None` as true.  pylint: disable=g-bool-id-comparison
  if options.context.update_use_recurrent_state is not False:
    result += [graph_ops.Readout()]
  for node_set_name in sorted(spec.node_sets_spec):
    node_options = options.node_set_with_defaults(node_set_name)
    if not _context_pool_enabled(node_options, default=True):
      continue
    if node_options.context_pool_factory is not None:
      pool = node_options.context_pool_factory(node_set_name=node_set_name)
    else:
      pool = graph_ops.Pool(const.CONTEXT, "sum", node_set_name=node_set_name)
    result += [pool]
  for edge_set_name in sorted(spec.edge_sets_spec):
    edge_options = options.edge_set_with_defaults(edge_set_name)
    if not _context_pool_enabled(edge_options, default=False):
      continue
    if edge_options.context_pool_factory is not None:
      pool = edge_options.context_pool_factory(edge_set_name=edge_set_name)
    else:
      pool = graph_ops.Pool(const.CONTEXT, "sum", edge_set_name=edge_set_name)
    result += [pool]
  return result


def _context_pool_enabled(
    set_options: Union[opt.GraphUpdateEdgeSetOptions,
                       opt.GraphUpdateNodeSetOptions],
    *, default: bool) -> bool:
  """Returns option context_pool_enable with defaulting rules applied."""
  if set_options.context_pool_factory is None:
    if set_options.context_pool_enable is None:
      return default
    else:
      return set_options.context_pool_enable
  else:
    # Interpret `None` as true.
    return set_options.context_pool_enable is not False  # pylint: disable=g-bool-id-comparison


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
