"""Model-building code."""

import collections
import functools
from typing import Callable, Mapping, Optional, Tuple

import tensorflow as tf
import tensorflow_gnn as tfgnn


def vanilla_mpnn_model(
    graph_tensor_spec: tfgnn.GraphTensorSpec,
    *,
    init_states_fn: Callable[..., tfgnn.GraphTensor],
    pass_messages_fn: Callable[..., tfgnn.GraphTensor]) -> tf.keras.Model:
  """Creates a Keras Model for a Message Passing Neural Network.

  This helper function packs `y = pass_messages_fn(init_states_fn(x))` into
  a Keras model with input `x` and output `y` of type tfgnn.GraphTensor.

  Args:
    graph_tensor_spec: A GraphTensorSpec for the scalar input GraphTensor to the
      model for which the non-trainable preprocessing of features has already
      been done.
    init_states_fn: A function that maps an input GraphTensor matching
      `graph_tensor_spec` to a GraphTensor with a tfgnn.DEFAULT_STATE_NAME
      feature on each node set, suitable for use with `pass_messages_fn`.
      Typically, this performs the trainable transformations per node (like
      lookups in a trainable embedding table) before any message passing starts.
    pass_messages_fn: A function that maps a GraphTensor returned by
      `init_states_fn` to the output of the model, typically by applying
      several rounds of message passing in the graph.

  Returns:
    A Keras Model that maps a scalar input GraphTensor with processed features
    to an output GraphTensor with final hidden states. Reading out the relevant
    node states is left to the user of this model.
  """
  inputs = tf.keras.layers.Input(type_spec=graph_tensor_spec)
  outputs = pass_messages_fn(init_states_fn(inputs))
  return tf.keras.Model(inputs, outputs)


def pass_simple_messages(
    graph: tfgnn.GraphTensor,
    *,
    num_message_passing: int,
    receiver_tag: tfgnn.IncidentNodeTag,
    message_dim: int,
    h_next_dim: int,
    l2_regularization: float = 0.0,
    dropout_rate: float = 0.0,
    modeling_flavor: str = "gnn_builder") -> tfgnn.GraphTensor:
  """Performs message passing for simple (plain MPNN) messages.

  Args:
    graph: A scalar GraphTensor with `tfgnn.DEFAULT_STATE_NAME` features on all
       node sets.
    num_message_passing: The number of rounds of message passing along edges.
    receiver_tag: One of `tfgnn.TARGET` or `tfgnn.SOURCE`, to select the
      incident node of each edge that receives the message.
    message_dim: The dimension of messages computed on each edge.
    h_next_dim: The dimension of hidden states computed for each node.
    l2_regularization: The coefficient of L2 regularization for weights and
      biases.
    dropout_rate: The dropout rate applied to states at the end of feed-forward
      sub-networks of this model.
    modeling_flavor: One of the following, to choose between different ways of
      implementing the graph updates for message passing. (Should not matter
      for end-users, but allows experimental comparison of modeling APIs.)
        * "raw_gnn_ops": The model is implemented with broadcast and pool
          operations applied to GraphTensors directly.
        * "edge_node_updates": The model is implemented as a series of
          `num_message_passing` many GraphUpdate layers, each made of
          EdgeSetUpdates from nodes and NodeSetUpdates from edges.
        * "node_updates": The model is implemented as a series of
          `num_message_passing` many GraphUpdate layers, each made of
          NodeSetUpdates from neighbor nodes.
        * "gnn_builder": Like "node_updates", but built with the convenience
          methods of `tfgnn.keras.ConvGNNBuilder`.

  Returns:
    A scalar GraphTensor with `tfgnn.DEFAULT_STATE_NAME` features on all node
    sets that have been updated by the specified rounds of message passing.
  """
  # Bind hparam kwargs for subroutines.
  dense = functools.partial(
      _dense_layer,
      l2_regularization=l2_regularization,
      dropout_rate=dropout_rate)
  gnn_kwargs = dict(
      num_message_passing=num_message_passing,
      receiver_tag=receiver_tag,
      message_dim=message_dim,
      h_next_dim=h_next_dim,
      dense=dense)
  # Dispatch between modeling flavors.
  if modeling_flavor == "raw_gnn_ops":
    graph = _pass_messages_with_raw_gnn_ops(graph, **gnn_kwargs)
  elif modeling_flavor == "edge_node_updates":
    graph = _pass_messages_with_edge_node_updates(graph, **gnn_kwargs)
  elif modeling_flavor == "node_updates":
    graph = _pass_messages_with_node_updates(graph, **gnn_kwargs)
  elif modeling_flavor == "gnn_builder":
    graph = _pass_messages_with_gnn_builder(graph, **gnn_kwargs)
  else:
    raise ValueError(f"Unknown modeling='{modeling_flavor}'")
  return graph


def _dense_layer(units: int,
                 *,
                 l2_regularization: float,
                 dropout_rate: float) -> tf.keras.layers.Layer:
  """Returns a feed-forward network with the given number of output units."""
  regularizer = tf.keras.regularizers.l2(l2_regularization)
  return tf.keras.Sequential([
      tf.keras.layers.Dense(
          units,
          activation="relu",
          use_bias=True,
          kernel_initializer="glorot_uniform",
          bias_initializer="zeros",
          kernel_regularizer=regularizer,
          bias_regularizer=regularizer),
      tf.keras.layers.Dropout(dropout_rate)])


def _pass_messages_with_raw_gnn_ops(
    graph: tfgnn.GraphTensor,
    *,
    num_message_passing: int,
    receiver_tag: tfgnn.IncidentNodeTag,
    message_dim: int,
    h_next_dim: int,
    dense: Callable[..., tf.keras.layers.Layer]) -> tfgnn.GraphTensor:
  """Performs message passing, using TF-GNN GraphTensor operations."""

  def message_pass(graph):
    pooled_messages = collections.defaultdict(list)

    for edge_name in sorted(graph.edge_sets.keys()):
      values = [
          tfgnn.broadcast_node_to_edges(
              graph,
              edge_name,
              tfgnn.reverse_tag(receiver_tag),  # Sender.
              feature_name=tfgnn.DEFAULT_STATE_NAME),
          tfgnn.broadcast_node_to_edges(
              graph,
              edge_name,
              receiver_tag,
              feature_name=tfgnn.DEFAULT_STATE_NAME)
      ]
      messages = dense(message_dim)(tf.keras.layers.Concatenate()(values))
      pooled_message = tfgnn.pool_edges_to_node(
          graph,
          edge_name,
          receiver_tag,
          reduce_type="sum",
          feature_value=messages)
      source_name = graph.edge_sets[edge_name].adjacency.node_set_name(
          receiver_tag)
      pooled_messages[source_name].append(pooled_message)

    node_set_states = collections.defaultdict(dict)

    for source_node_name in sorted(pooled_messages.keys()):
      h_olds = [graph.node_sets[source_node_name][tfgnn.DEFAULT_STATE_NAME]]
      inputs = h_olds + pooled_messages[source_node_name]
      h_new = dense(h_next_dim)(tf.keras.layers.Concatenate()(inputs))
      node_set_states[source_node_name][tfgnn.DEFAULT_STATE_NAME] = h_new

    return graph.replace_features(node_sets=node_set_states)

  for _ in range(num_message_passing):
    graph = message_pass(graph)
  return graph


def _pass_messages_with_edge_node_updates(
    graph: tfgnn.GraphTensor,
    *,
    num_message_passing: int,
    receiver_tag: tfgnn.IncidentNodeTag,
    message_dim: int,
    h_next_dim: int,
    dense: Callable[..., tf.keras.layers.Layer]) -> tfgnn.GraphTensor:
  """Performs message passing, built with EdgeSetUpdates and NodeSetUpdates."""
  for _ in range(num_message_passing):
    incoming_edge_sets = collections.defaultdict(list)

    edge_set_updates = {}
    for edge_set_name in sorted(graph.edge_sets.keys()):
      source_name = graph.edge_sets[edge_set_name].adjacency.node_set_name(
          receiver_tag)
      incoming_edge_sets[source_name].append(edge_set_name)
      edge_set_updates[edge_set_name] = tfgnn.keras.layers.EdgeSetUpdate(
          tfgnn.keras.layers.NextStateFromConcat(dense(message_dim)),
          edge_input_feature=None)

    node_set_updates = {}
    for node_set_name in sorted(incoming_edge_sets.keys()):
      node_set_updates[node_set_name] = tfgnn.keras.layers.NodeSetUpdate(
          {edge_set_name: tfgnn.keras.layers.Pool(receiver_tag, "sum")
           for edge_set_name in incoming_edge_sets[node_set_name]},
          tfgnn.keras.layers.NextStateFromConcat(dense(h_next_dim)))

    graph = tfgnn.keras.layers.GraphUpdate(
        edge_sets=edge_set_updates, node_sets=node_set_updates)(graph)

  return graph


def _pass_messages_with_node_updates(
    graph: tfgnn.GraphTensor,
    *,
    num_message_passing: int,
    receiver_tag: tfgnn.IncidentNodeTag,
    message_dim: int,
    h_next_dim: int,
    dense: Callable[..., tf.keras.layers.Layer]) -> tfgnn.GraphTensor:
  """Performs message passing, built with NodeSetUpdates and convolutions."""
  for _ in range(num_message_passing):
    incoming_edge_sets = collections.defaultdict(list)
    for edge_set_name in sorted(graph.edge_sets.keys()):
      source_name = graph.edge_sets[edge_set_name].adjacency.node_set_name(
          receiver_tag)
      incoming_edge_sets[source_name].append(edge_set_name)

    node_set_updates = {}
    for node_set_name in sorted(incoming_edge_sets.keys()):
      node_set_updates[node_set_name] = tfgnn.keras.layers.NodeSetUpdate(
          {edge_set_name: tfgnn.keras.layers.SimpleConvolution(
              dense(message_dim), "sum", receiver_tag=receiver_tag)
           for edge_set_name in incoming_edge_sets[node_set_name]},
          tfgnn.keras.layers.NextStateFromConcat(dense(h_next_dim)))

    graph = tfgnn.keras.layers.GraphUpdate(node_sets=node_set_updates)(graph)

  return graph


def _pass_messages_with_gnn_builder(
    graph: tfgnn.GraphTensor,
    *,
    num_message_passing: int,
    receiver_tag: tfgnn.IncidentNodeTag,
    message_dim: int,
    h_next_dim: int,
    dense: Callable[..., tf.keras.layers.Layer]) -> tfgnn.GraphTensor:
  """Performs message passing, built with tfgnn.keras.ConvGNNBuilder."""
  # pylint: disable=g-long-lambda
  gnn_builder = tfgnn.keras.ConvGNNBuilder(
      lambda edge_set_name, receiver_tag: tfgnn.keras.layers.SimpleConvolution(
          dense(message_dim), "sum", receiver_tag=receiver_tag),
      lambda node_set_name: tfgnn.keras.layers.NextStateFromConcat(
          dense(h_next_dim)),
      receiver_tag=receiver_tag)

  for _ in range(num_message_passing):
    graph = gnn_builder.Convolve()(graph)
  return graph


def init_states_by_embed_and_transform(
    graph: tfgnn.GraphTensor,
    *,
    node_embeddings: Optional[Mapping[str, Tuple[int, int]]] = None,
    node_transformations: Optional[Mapping[str, int]] = None,
    l2_regularization: float = 0.0,
    dropout_rate: float = 0.0) -> tfgnn.GraphTensor:
  """An init_states_fn for vanilla_mpnn_model().

  Args:
    graph: A scalar input GraphTensor to the model for which the non-trainable
      preprocessing of features has already been done.
    node_embeddings: Optional tail node embedding dimensions, keyed by node set
      name. Node set features are assumed to be `tfgnn.DEFAULT_STATE_NAME`.
    node_transformations: Optional tail node transformation dimensions, keyed
      by node set name. Node set features are assumed to be
      `tfgnn.DEFAULT_STATE_NAME`.
    l2_regularization: The coefficient of L2 regularization for weights and
      biases of the node transformations (if any).
    dropout_rate: The dropout rate applied after the node transformations
      (if any).

  Returns:
    A scalar GraphTensor in which the specified node_embeddings and
    node_transformations have been applied.
  """
  if node_embeddings:
    graph = _embed_inputs(graph, configs=node_embeddings)
  if node_transformations:
    graph = _transform_inputs(
        graph,
        configs=node_transformations,
        l2_regularization=l2_regularization,
        dropout_rate=dropout_rate)
  return graph


def _embed_inputs(graph: tfgnn.GraphTensor,
                  *,
                  configs: Mapping[str, Tuple[int, int]]) -> tfgnn.GraphTensor:
  """Returns GraphTensor with categorical input features embedded as states."""
  node_sets = {}

  for node_set_name, dims in configs.items():
    features = graph.node_sets[node_set_name].features[tfgnn.DEFAULT_STATE_NAME]
    features = tf.keras.layers.Embedding(*dims)(features)

    if len(features.shape) > 2:
      # Maybe reduce embeddings for the case of a RaggedTensor input
      features = tf.math.reduce_mean(features, axis=-2)

    node_sets[node_set_name] = {
        tfgnn.DEFAULT_STATE_NAME: features
    }

  return graph.replace_features(node_sets=node_sets)


def _transform_inputs(graph: tfgnn.GraphTensor,
                      *,
                      configs: Mapping[str, int],
                      l2_regularization: float,
                      dropout_rate: float) -> tfgnn.GraphTensor:
  """Returns GraphTensor with numeric input features transformed."""
  node_sets = {}

  for node_set_name, dim in configs.items():
    features = graph.node_sets[node_set_name].features[tfgnn.DEFAULT_STATE_NAME]
    features = _dense_layer(
        dim,
        l2_regularization=l2_regularization,
        dropout_rate=dropout_rate)(features)

    node_sets[node_set_name] = {
        tfgnn.DEFAULT_STATE_NAME: features
    }

  return graph.replace_features(node_sets=node_sets)
