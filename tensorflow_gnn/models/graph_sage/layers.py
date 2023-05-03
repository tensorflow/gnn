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
"""Contains GraphSAGE convolution layer implementations."""

import copy
from typing import Any, Callable, Collection, Optional, Set, Union

import tensorflow as tf
import tensorflow_gnn as tfgnn


@tf.keras.utils.register_keras_serializable(package="GraphSAGE")
class GraphSAGEAggregatorConv(tfgnn.keras.layers.AnyToAnyConvolutionBase):
  """GraphSAGE: element-wise aggregation of neighbors and their linear transformation.

  For a complete GraphSAGE update on a node set, use this class in a
  NodeSetUpdate together with the `graph_sage.GraphSAGENextState` layer
  to handle the node state (see there for details).

  GraphSAGE and the mean aggregation method are from
  Hamilton et al.: ["Inductive Representation Learning
  on Large Graphs"](https://arxiv.org/abs/1706.02216), 2017.
  Following the authors' implementation, dropout is applied to the inputs
  of neighbor nodes (separately for each node-neighbor pair).

  This class supports the element-wise aggregations with different operator
  types besides "mean", see the reduce_type=... argument. For stateful
  transformation with a hidden layer, see `graph_sage.GraphSAGEPoolingConv`.
  """

  def __init__(
      self,
      *,
      receiver_tag: tfgnn.IncidentNodeTag,
      reduce_type: str = "mean",
      sender_node_feature: Optional[tfgnn.FieldName] = tfgnn.HIDDEN_STATE,
      units: int,
      dropout_rate: float = 0.,
      **kwargs):
    """Initializes the `GraphSAGEAggregatorConv` convolution layer.

    Args:
      receiver_tag: Either one of `tfgnn.SOURCE` or `tfgnn.TARGET`. The results
        of GraphSAGE convolution are aggregated for this graph piece. If set to
        `tfgnn.SOURCE` or `tfgnn.TARGET`, the layer will be called for an edge
        set and will aggregate results at the specified endpoint of the edges.
      reduce_type: An aggregation operation name. Supported list of aggregation
        operators can be found at
        `tfgnn.get_registered_reduce_operation_names()`.
      sender_node_feature: Can be set to specify the feature name for use as the
        input feature from sender nodes to GraphSAGE aggregation, defaults to
        `tfgnn.HIDDEN_STATE`.
      units: Number of output units for the linear transformation applied to
        sender node features.
      dropout_rate: Can be set to a dropout rate that will be applied to sender
        node features (independently on each edge).
      **kwargs: Additional arguments for the Layer.
    """
    kwargs.setdefault("name", "graph_sage_aggregator_conv")
    if kwargs.get("receiver_feature") is not None:
      raise ValueError(
          "receiver_feature has to be None for GraphSAGEAggregatorConv.")
    if kwargs.get("sender_edge_feature") is not None:
      raise ValueError(
          "sender_edge_feature has to be None for GraphSAGEAggregatorConv.")
    if sender_node_feature is None:
      raise ValueError(
          "sender_node_feature should be specified for GraphSAGEAggregatorConv."
      )
    super().__init__(
        receiver_tag=receiver_tag,
        receiver_feature=None,
        sender_node_feature=sender_node_feature,
        sender_edge_feature=None,
        **kwargs)

    self._units = units
    self._transform_neighbor_fn = tf.keras.layers.Dense(units, use_bias=False)
    self._dropout_rate = dropout_rate
    self._dropout = tf.keras.layers.Dropout(dropout_rate)
    self._reduce_type = reduce_type

  def get_config(self):
    """Returns the config for Aggregator Convolution."""
    config = super().get_config()
    assert config.pop("receiver_feature", None) is None, (
        "init should guarantee receiver_feature=None")
    assert config.pop("sender_edge_feature", None) is None, (
        "init should guarantee sender_edge_feature=None")
    return dict(
        **config,
        units=self._units,
        dropout_rate=self._dropout_rate,
        reduce_type=self._reduce_type)

  def convolve(self, *, sender_node_input: Optional[tf.Tensor],
               sender_edge_input: Optional[tf.Tensor],
               receiver_input: Optional[tf.Tensor],
               broadcast_from_sender_node: Callable[[tf.Tensor], tf.Tensor],
               broadcast_from_receiver: Callable[[tf.Tensor], tf.Tensor],
               pool_to_receiver: Callable[..., tf.Tensor],
               extra_receiver_ops: Any = None,
               training: bool) -> tf.Tensor:
    """Overridden internal method of the base class."""
    assert extra_receiver_ops is None, "Internal error: bad super().__init__()"
    assert sender_node_input is not None, "sender_node_input can't be None."
    result = broadcast_from_sender_node(sender_node_input)
    result = self._dropout(result, training=training)
    result = pool_to_receiver(result, reduce_type=self._reduce_type)
    result = self._transform_neighbor_fn(result)
    return result


@tf.keras.utils.register_keras_serializable(package="GraphSAGE")
class GraphSAGEPoolingConv(tfgnn.keras.layers.AnyToAnyConvolutionBase):
  """GraphSAGE: pooling aggregator transform of neighbors followed by linear transformation.

  For a complete GraphSAGE update on a node set, use a this class in a
  NodeSetUpdate together with the `graph_sage.GraphSAGENextState` layer
  to update the final node state (see there for details).

  GraphSAGE and the pooling aggregation are from
  Hamilton et al.: ["Inductive Representation Learning
  on Large Graphs"](https://arxiv.org/abs/1706.02216), 2017.
  Similar to `graph_sage.GraphSAGEAggregatorConv`, dropout is applied to the
  inputs of neighbor nodes (separately for each node-neighbor pair). Then,
  they are passed through a fully connected layer and aggregated by an
  element-wise maximum (or whichever reduce_type is specified), see Eq. (3)
  in paper. Finally, the result is multiplied with the final weights mapping
  it to output space of units dimension.

  The name of this class reflects the terminology of the paper, where "pooling"
  involves the aforementioned hidden layer. For element-wise aggregation (as in
  `tfgnn.pool_edges_to_node()`), see `graph_sage.GraphSAGEAggregatorConv`.
  """

  def __init__(
      self,
      *,
      receiver_tag: tfgnn.IncidentNodeTag,
      sender_node_feature: Optional[tfgnn.FieldName] = tfgnn.HIDDEN_STATE,
      units: int,
      hidden_units: int,
      reduce_type: str = "max_no_inf",
      use_bias: bool = True,
      dropout_rate: float = 0.,
      activation: Union[str, Callable[..., Any]] = "relu",
      **kwargs):
    """Initializes the `GraphSAGEPoolingConv` convolution layer.

    Args:
      receiver_tag: Either one of `tfgnn.SOURCE` or `tfgnn.TARGET`. The results
        of GraphSAGE are aggregated for this graph piece. If set to
        `tfgnn.SOURCE` or `tfgnn.TARGET`, the layer will be called for an edge
        set and will aggregate results at the specified endpoint of the edges.
      sender_node_feature: Can be set to specify the feature name for use as the
        input feature from sender nodes to GraphSAGE aggregation, defaults to
        `tfgnn.HIDDEN_STATE`.
      units: Number of output units for the final dimensionality of the output
        from the layer.
      hidden_units: Number of output units for the linear transformation applied
        to the sender node features.This specifies the output dimensions of the
        W_pool from Eq. (3) in
        [Hamilton et al., 2017](https://arxiv.org/abs/1706.02216).
      reduce_type: An aggregation operation name. Supported list of aggregation
        operators can be found at
        `tfgnn.get_registered_reduce_operation_names()`.
      use_bias: If true a bias term will be added to the linear transformations
        for the sender node features.
      dropout_rate: Can be set to a dropout rate that will be applied to sender
        node features (independently on each edge).
      activation: The nonlinearity applied to the concatenated or added node
        state and aggregated sender node features. This can be specified as a
        Keras layer, a tf.keras.activations.* function, or a string understood
        by `tf.keras.layers.Activation()`. Defaults to relu.
      **kwargs: Additional arguments for the Layer.
    """
    kwargs.setdefault("name", "graph_sage_pooling_conv")
    if kwargs.get("receiver_feature") is not None:
      raise ValueError(
          "receiver_feature has to be None for GraphSAGEPoolingConv.")
    if kwargs.get("sender_edge_feature") is not None:
      raise ValueError(
          "sender_edge_feature has to be None for GraphSAGEPoolingConv.")
    if sender_node_feature is None:
      raise ValueError(
          "sender_node_feature should be specified for GraphSAGEPoolingConv.")
    super().__init__(
        receiver_tag=receiver_tag,
        receiver_feature=None,
        sender_node_feature=sender_node_feature,
        sender_edge_feature=None,
        **kwargs)
    self._dropout_rate = dropout_rate
    self._dropout = tf.keras.layers.Dropout(self._dropout_rate)
    self._units = units
    self._hidden_units = hidden_units
    self._use_bias = use_bias
    self._activation = tf.keras.activations.get(activation)
    self._pooling_transform_fn = tf.keras.layers.Dense(
        self._hidden_units,
        use_bias=self._use_bias,
        activation=self._activation)
    self._transform_neighbor_fn = tf.keras.layers.Dense(
        self._units, use_bias=False)
    self._reduce_type = reduce_type

  def get_config(self):
    """Returns the config for Pooling Convolution."""
    config = super().get_config()
    assert config.pop("receiver_feature", None) is None, (
        "init should guarantee receiver_feature=None")
    if config.pop("sender_edge_feature", None) is not None:
      raise ValueError("init should guarantee sender_edge_feature=None")
    return dict(
        **config,
        activation=self._activation,
        units=self._units,
        hidden_units=self._hidden_units,
        dropout_rate=self._dropout_rate,
        use_bias=self._use_bias,
        reduce_type=self._reduce_type)

  def convolve(self, *, sender_node_input: Optional[tf.Tensor],
               sender_edge_input: Optional[tf.Tensor],
               receiver_input: Optional[tf.Tensor],
               broadcast_from_sender_node: Callable[[tf.Tensor], tf.Tensor],
               broadcast_from_receiver: Callable[[tf.Tensor], tf.Tensor],
               pool_to_receiver: Callable[..., tf.Tensor],
               extra_receiver_ops: Any = None,
               training: bool) -> tf.Tensor:
    """Overridden internal method of the base class."""
    assert extra_receiver_ops is None, "Internal error: bad super().__init__()"
    assert sender_node_input is not None, "sender_node_input can't be None."
    result = broadcast_from_sender_node(sender_node_input)
    result = self._dropout(result, training=training)
    # The "Pooling aggregator" from Eq. (3) of the paper, plus dropout.
    result = self._pooling_transform_fn(result)
    result = pool_to_receiver(result, reduce_type=self._reduce_type)
    result = self._transform_neighbor_fn(result)
    return result


@tf.keras.utils.register_keras_serializable(package="GraphSAGE")
class GCNGraphSAGENodeSetUpdate(tf.keras.layers.Layer):
  r"""GCNGraphSAGENodeSetUpdate is an extension of the mean aggregator operator.

  For a complete GraphSAGE update on a node set, use this layer in a
  `GraphUpdate` call as a `NodeSetUpdate` layer. An example update would look
  as below:

  ```python
  import tensorflow_gnn as tfgnn
  graph = tfgnn.keras.layers.GraphUpdate(
      node_sets={
          "paper":
              graph_sage.GCNGraphSAGENodeSetUpdate(
                  edge_set_names=["cites", "writes"],
                  receiver_tag=tfgnn.TARGET,
                  units=32)
      })(graph)
  ```

  This class extends Eq. (2) from
  [Hamilton et al., 2017](https://arxiv.org/abs/1706.02216) to multiple edge
  sets. For each node state pooled from the configured edge list
  and the self node states there's a separate weight vector learned which is
  mapping each to the same output dimensions. Also if specified a random dropout
  operation with given probability will be applied to all the node states. If
  share_weights is enabled, then it'll learn the same weights for self and
  sender node states, this is the implementation for homogeneous graphs from the
  paper. Note that enabling this requires both sender and receiver node states
  to have the same dimension. Below is the simplified summary of the applied
  transformations to generate new node states:

  ```
  h_v = activation(
            reduce(  {W_E * D_p[h_{N(v)}] for all edge sets E}
                   U {W_self * D_p[h_v]})
            + b)
  ```

  N(v) denotes the neighbors of node v, D_p denotes dropout with probability p
  which is applied independenly to self and sender node states, W_E and W_self
  denote the edge and self node transformation weight vectors and b is the bias.
  If add_self_loop is disabled then self node states won't be used during the
  reduce operation, instead only the sender node states will be accumulated
  based on the reduce_type specified. If share_weights is set to True, then
  single weight matrix will be used in place of W_E and W_self.
  """

  def __init__(self,
               *,
               edge_set_names: Set[str],
               receiver_tag: tfgnn.IncidentNodeTag,
               reduce_type: str = "mean",
               self_node_feature: str = tfgnn.HIDDEN_STATE,
               sender_node_feature: str = tfgnn.HIDDEN_STATE,
               units: int,
               dropout_rate: float = 0.0,
               activation: Union[str, Callable[..., Any]] = "relu",
               use_bias: bool = False,
               share_weights: bool = False,
               add_self_loop: bool = True,
               **kwargs):
    """Initializes GCNGraphSAGENodeSetUpdate node set update layer.

    Args:
      edge_set_names: A list of edge set names to broadcast sender node states.
      receiver_tag: Either one of `tfgnn.SOURCE` or `tfgnn.TARGET`. The results
        of GraphSAGE convolution are aggregated for this graph piece. If set to
        `tfgnn.SOURCE` or `tfgnn.TARGET`, the layer will be called for each edge
        set and will aggregate results at the specified endpoint of the edges.
        This should point at the node_set_name for each of the specified edge
        set name in the edge_set_name_dict.
      reduce_type: An aggregation operation name. Supported list of aggregation
        operators are sum or mean.
      self_node_feature: Feature name for the self node sets to be aggregated
        with the broadcasted sender node states. Default is
        `tfgnn.HIDDEN_STATE`.
      sender_node_feature: Feature name for the sender node sets. Default is
        `tfgnn.HIDDEN_STATE`.
      units: Number of output units for the linear transformation applied to
        sender node and self node features.
      dropout_rate: Can be set to a dropout rate that will be applied to both
        self node and the sender node states.
      activation: The nonlinearity applied to the update node states. This can
        be specified as a Keras layer, a tf.keras.activations.* function, or a
        string understood by tf.keras.layers.Activation(). Defaults to relu.
      use_bias: If true a bias term will be added to mean aggregated feature
        vectors before applying non-linear activation.
      share_weights: If left unset, separate weights are used to transform the
        inputs along each edge set and the input of previous node states (unless
        disabled by add_self_loop=False). If enabled, a single weight matrix is
        applied to all inputs.
      add_self_loop: If left at True (the default), each node state update takes
        the node's old state as an explicit input next to all the inputs along
        edge sets. Typically, this is done when the graph does not have loops.
        If set to False, each node state update uses only the inputs along the
        requested edge sets. Typically, this is done when loops are already
        contained among the edges.
      **kwargs:
    """
    kwargs.setdefault("name", "graph_sage_gcn_update")
    super().__init__(**kwargs)
    self._self_node_feature = self_node_feature
    self._edge_set_names = copy.deepcopy(edge_set_names)
    self._units = units
    self._activation = tf.keras.activations.get(activation)
    self._receiver_tag = receiver_tag
    self._reduce_type = reduce_type
    self._dropout_rate = dropout_rate
    self._dropout = tf.keras.layers.Dropout(dropout_rate)
    self._share_weights = share_weights
    self._add_self_loop = add_self_loop
    self._sender_node_feature = sender_node_feature
    if not self._share_weights:
      self._transform_edge_fn_dict = dict()
      for edge_set_name in edge_set_names:
        self._transform_edge_fn_dict[edge_set_name] = tf.keras.layers.Dense(
            self._units, use_bias=False)
    if self._add_self_loop or self._share_weights:
      self._node_transform_fn = tf.keras.layers.Dense(
          self._units, use_bias=False)
    self._use_bias = use_bias
    if self._use_bias:
      self._bias_term = self.add_weight(
          name="bias",
          shape=[self._units],
          trainable=True,
          initializer=tf.keras.initializers.Zeros())

  def get_config(self):
    """Returns the config for the convolution."""
    config = super().get_config()
    return dict(
        **config,
        edge_set_names=self._edge_set_names,
        receiver_tag=self._receiver_tag,
        reduce_type=self._reduce_type,
        self_node_feature=self._self_node_feature,
        sender_node_feature=self._sender_node_feature,
        units=self._units,
        dropout_rate=self._dropout_rate,
        activation=self._activation,
        use_bias=self._use_bias,
        share_weights=self._share_weights,
        add_self_loop=self._add_self_loop)

  def call(self,
           graph: tfgnn.GraphTensor,
           *,
           node_set_name: str,
           training: Optional[bool] = False) -> tfgnn.FieldOrFields:
    """Calls the layer on `node_set_name` and returns node feature tensors."""
    tfgnn.check_scalar_graph_tensor(graph, name="GCNGraphSAGENodeSetUpdate")
    if self._reduce_type not in ["sum", "mean"]:
      raise ValueError(
          f"{self._reduce_type} isn't supported, please instead use any of "
          "['sum', 'mean']")
    edge_set_in_degrees_list = []
    pooled_node_states_list = []
    for edge_set_name in self._edge_set_names:
      edge_set = graph.edge_sets[edge_set_name]
      if node_set_name != edge_set.adjacency.node_set_name(self._receiver_tag):
        raise ValueError(
            f"Incorrect {edge_set_name} that has a different node at "
            f"receiver_tag:{self._receiver_tag} other than {node_set_name}.")
      sender_node_set_name = edge_set.adjacency.node_set_name(
          tfgnn.reverse_tag(self._receiver_tag))
      sender_node_values = graph.node_sets[sender_node_set_name][
          self._sender_node_feature]
      sender_node_values = self._dropout(sender_node_values)
      if not self._share_weights:
        sender_node_values = self._transform_edge_fn_dict[edge_set_name](
            sender_node_values)
      else:
        sender_node_values = self._node_transform_fn(sender_node_values)
      broadcasted_sender_values = tfgnn.broadcast_node_to_edges(
          graph,
          edge_set_name,
          tfgnn.reverse_tag(self._receiver_tag),
          feature_value=sender_node_values)
      pooled_sender_values = tfgnn.pool_edges_to_node(
          graph,
          edge_set_name,
          self._receiver_tag,
          "sum",
          feature_value=broadcasted_sender_values)
      pooled_node_states_list.append(pooled_sender_values)
      if self._reduce_type == "mean":
        edge_set_ones = tf.ones([edge_set.total_size, 1])
        edge_set_in_degrees = tf.squeeze(
            tfgnn.pool_edges_to_node(
                graph,
                edge_set_name,
                self._receiver_tag,
                "sum",
                feature_value=edge_set_ones))
        edge_set_in_degrees_list.append(edge_set_in_degrees)
    total_size = graph.node_sets[node_set_name].total_size
    # aggregate with self node states only if add_self_loop is enabled.
    if self._add_self_loop:
      self_node_values = graph.node_sets[node_set_name][self._self_node_feature]
      self_node_values = self._dropout(self_node_values)
      self_node_values = self._node_transform_fn(self_node_values)
      pooled_node_states_list.append(self_node_values)
      if self._reduce_type == "mean":
        edge_set_in_degrees_list.append(tf.ones(total_size))
    summed_node_values = tf.math.add_n(pooled_node_states_list)
    if self._reduce_type == "mean":
      total_in_degrees = tf.math.add_n(edge_set_in_degrees_list)
      result = tf.math.divide_no_nan(summed_node_values,
                                     total_in_degrees[:, tf.newaxis])
    else:
      result = summed_node_values
    if self._use_bias:
      result += self._bias_term
    result = self._activation(result)
    return {self._self_node_feature: result}


@tf.keras.utils.register_keras_serializable(package="GraphSAGE")
def GraphSAGEGraphUpdate(  # To be called like a class initializer.  pylint: disable=invalid-name
    *,
    units: int,
    hidden_units: Optional[int] = None,
    receiver_tag: tfgnn.IncidentNodeTag,
    node_set_names: Optional[Collection[tfgnn.NodeSetName]] = None,
    reduce_type: str = "mean",
    use_pooling: bool = True,
    use_bias: bool = True,
    dropout_rate: float = 0.0,
    l2_normalize: bool = True,
    combine_type: str = "sum",
    activation: Union[str, Callable[..., Any]] = "relu",
    feature_name: str = tfgnn.HIDDEN_STATE,
    name: str = "graph_sage",
    **kwargs) -> tf.keras.layers.Layer:
  """Returns a GraphSAGE GraphUpdater layer for nodes in node_set_names.

  For more information on GraphSAGE algorithm please refer to
  [Hamilton et al., 2017](https://arxiv.org/abs/1706.02216).
  Returned layer applies only one step of GraphSAGE convolution over the
  incident nodes of the edge_set_name_list for the specified node_set_name node.

  Args:
    units: Number of output units of the linear transformation applied to both
      final aggregated sender node features as well as the self node feature.
    hidden_units: Number of output units to be configure for GraphSAGE pooling
      type convolution only.
    receiver_tag: Either one of `tfgnn.SOURCE` or `tfgnn.TARGET`. The results of
      GraphSAGE are aggregated for this graph piece. When set to `tfgnn.SOURCE`
      or `tfgnn.TARGET`, the layer is called for an edge set and will aggregate
      results at the specified endpoint of the edges.
    node_set_names: By default, this layer updates all node sets that receive
      from at least one edge set. Optionally, this argument can specify a subset
      of those node sets. It is not allowed to include node sets that do not
      receive messages from any edge set. It is also not allowed to include
      auxiliary node sets.
    reduce_type: An aggregation operation name. Supported list of aggregation
      operators can be found at `tfgnn.get_registered_reduce_operation_names()`.
    use_pooling: If enabled, `graph_sage.GraphSAGEPoolingConv` will be used,
      otherwise `graph_sage.GraphSAGEAggregatorConv` will be executed for the
      provided edges.
    use_bias: If true a bias term will be added to the linear transformations
      for the incident node features as well as for the self node feature.
    dropout_rate: Can be set to a dropout rate that will be applied to both
      incident node features as well as the self node feature.
    l2_normalize: If enabled l2 normalization will be applied to final node
      states.
    combine_type: Can be set to "sum" or "concat". If it's specified as concat
      node state will be concatenated with the sender node features, otherwise
      node state will be added with the sender node features.
    activation: The nonlinearity applied to the concatenated or added node state
      and aggregated sender node features. This can be specified as a Keras
      layer, a tf.keras.activations.* function, or a string understood by
      `tf.keras.layers.Activation()`. Defaults to relu.
    feature_name: The feature name of node states; defaults to
      `tfgnn.HIDDEN_STATE`.
    name: Optionally, a name for the layer returned.
    **kwargs: Any optional arguments to `graph_sage.GraphSAGEPoolingConv`,
      `graph_sage.GraphSAGEAggregatorConv` or `graph_sage.GraphSAGENextState`,
      see there.
  """
  if use_pooling != (hidden_units is not None):
    raise ValueError(
        "Either use_pooling or hidden_units has been configured without the "
        "other, please configure them together or disable them both.")

  def convolutions_factory(edge_set_name, receiver_tag):
    del edge_set_name  # Unused.
    if use_pooling:
      return GraphSAGEPoolingConv(
          receiver_tag=receiver_tag,
          sender_node_feature=feature_name,
          reduce_type=reduce_type,
          units=units,
          hidden_units=hidden_units,
          use_bias=use_bias,
          dropout_rate=dropout_rate,
          **kwargs)
    else:
      return GraphSAGEAggregatorConv(
          receiver_tag=receiver_tag,
          reduce_type=reduce_type,
          sender_node_feature=feature_name,
          units=units,
          dropout_rate=dropout_rate,
          **kwargs)

  def nodes_next_state_factory(node_set_name):
    del node_set_name  # Unused.
    return GraphSAGENextState(
        units=units,
        use_bias=use_bias,
        dropout_rate=dropout_rate,
        feature_name=feature_name,
        l2_normalize=l2_normalize,
        combine_type=combine_type,
        activation=activation,
        **kwargs)

  def node_set_update_factory(node_set_name, edge_set_inputs, next_state):
    del node_set_name  # Unused.
    return tfgnn.keras.layers.NodeSetUpdate(
        edge_set_inputs=edge_set_inputs,
        next_state=next_state,
        node_input_feature=feature_name)

  gnn_builder = tfgnn.keras.ConvGNNBuilder(
      convolutions_factory,
      nodes_next_state_factory,
      node_set_update_factory=node_set_update_factory,
      receiver_tag=receiver_tag)
  return gnn_builder.Convolve(node_set_names, name=name)


@tf.keras.utils.register_keras_serializable(package="GraphSAGE")
class GraphSAGENextState(tf.keras.layers.Layer):
  r"""GraphSAGENextState: compute new node states with GraphSAGE algorithm.

  This layer lets you compute a GraphSAGE update of node states from the
  outputs of a `graph_sage.GraphSAGEAggregatorConv` and/or a
  `graph_sage.GraphSAGEPoolingConv` on each of the specified end-point
  of edge sets.

  Usage example (with strangely mixed aggregations for demonstration):

  ```python
  import tensorflow_gnn as tfgnn
  from tensorflow_gnn.models import graph_sage
  graph = tfgnn.keras.layers.GraphUpdate(node_sets={
      "papers": tfgnn.keras.layers.NodeSetUpdate(
          {"citations": graph_sage.GraphSAGEAggregatorConv(
               units=32, receiver_tag=tfgnn.TARGET),
           "affiliations": graph_sage.GraphSAGEPoolingConv(
               units=32, hidden_units=16, receiver_tag=tfgnn.SOURCE)},
           graph_sage.GraphSAGENextState(units=32)),
      "...": ...,
  })(graph)
  ```

  The `units=...` parameter of the next-state layer and all convolutions must be
  equal, unless `combine_type="concat"`is set.

  GraphSAGE is Algorithm 1 in Hamilton et al.: ["Inductive Representation
  Learning on Large Graphs"](https://arxiv.org/abs/1706.02216), 2017.
  It computes the new hidden state h_v for each node v from a concatenation of
  the previous hiddden state with an aggregation of the neighbor states as

  $$h_v = \sigma(W \text{ concat}(h_v, h_{N(v)}))$$

  ...followed by L2 normalization. This implementation uses the mathematically
  equivalent formulation

  $$h_v = \sigma(W_{\text{self}} h_v + W_{\text{neigh}} h_{N(v)}),$$

  which transforms both inputs separately and then combines them by summation
  (assuming the default `combine_type="sum"`; see there for more).

  The GraphSAGE*Conv classes are in charge of computing the right-hand term
  W_{neigh} h_{N(v)} (for one edge set each, typically with separate weights).
  This class is in charge of computing the left-hand term W_{self} h_v from the
  old node state h_v, combining it with the results for each edge set and
  computing the new node state h_v from it.

  Beyond the original GraphSAGE, this class supports:

    * dropout, applied to the input h_v, analogous to the dropout provided
      by GraphSAGE*Conv for their inputs;
    * a bias term added just before the final nonlinearity;
    * a configurable combine_type (originally "sum");
    * additional options to influence normalization, activation, etc.
  """

  def __init__(self,
               *,
               units: int,
               use_bias: bool = True,
               dropout_rate: float = 0.0,
               feature_name: str = tfgnn.HIDDEN_STATE,
               l2_normalize: bool = True,
               combine_type: str = "sum",
               activation: Union[str, Callable[..., Any]] = "relu",
               **kwargs):
    """Initializes the GraphSAGENextState layer.

    Args:
      units: Number of output units for the linear transformation applied to the
        node feature.
      use_bias: If true a bias term will be added to the linear transformations
        for the self node feature.
      dropout_rate: Can be set to a dropout rate that will be applied to the
        node feature.
      feature_name: The feature name of node states; defaults to
        `tfgnn.HIDDEN_STATE`.
      l2_normalize: If enabled l2 normalization will be applied to node state
        vectors.
      combine_type: Can be set to "sum" or "concat". The default "sum" recovers
        the original behavior of GraphSAGE (as reformulated above): the results
        of transforming the old state and the neighborhood state are added.
        Setting this to "concat" concatenates the results of the transformations
        (not described in the paper).
      activation: The nonlinearity applied to the concatenated or added node
        state and aggregated sender node features. This can be specified as a
        Keras layer, a tf.keras.activations.* function, or a string understood
        by `tf.keras.layers.Activation()`. Defaults to relu.
      **kwargs: Forwarded to the base class tf.keras.layers.Layer.
    """
    super().__init__(**kwargs)
    self._use_bias = use_bias
    self._l2_normalize = l2_normalize
    self._combine_type = combine_type
    self._activation = tf.keras.activations.get(activation)
    self._units = units
    self._self_transform = tf.keras.layers.Dense(units, use_bias=False)
    self._feature_name = feature_name
    self._dropout_rate = dropout_rate
    self._dropout = tf.keras.layers.Dropout(self._dropout_rate)

  def get_config(self):
    """Returns the config for GraphSAGENextState."""
    return dict(
        units=self._units,
        use_bias=self._use_bias,
        dropout_rate=self._dropout_rate,
        feature_name=self._feature_name,
        l2_normalize=self._l2_normalize,
        combine_type=self._combine_type,
        activation=self._activation,
        **super().get_config())

  def build(self, input_shapes):
    """Creates the bias_term based on input shapes."""
    _, edge_inputs_shape_dict, _ = input_shapes
    if self._use_bias:
      if self._combine_type == "sum":
        shape = self._units
      elif self._combine_type == "concat":
        edge_input_shapes = list(edge_inputs_shape_dict.values())
        if any(s.rank != 2 or s[1] is None for s in edge_input_shapes):
          raise ValueError("Invalid shape for edge inputs.")
        shape = self._units + sum(s[1] for s in edge_input_shapes)
      else:
        raise ValueError(
            f"combine_type: {self._combine_type} isn't supported. Please"
            " instead specify 'concat' or 'sum'.")
      self._bias_term = self.add_weight(
          name="bias",
          shape=[shape],
          trainable=True,
          initializer=tf.keras.initializers.Zeros())

  def call(self, inputs, training):
    """Calls the layer on inputs and returns node feature tensors."""
    old_node_state, edge_inputs_dict, unused_context_state = inputs
    if unused_context_state:
      raise ValueError(
          "Input from context is not supported by GraphSAGENextState")
    result = self._dropout(old_node_state, training=training)
    result = self._self_transform(result)
    result = [result, *[v for _, v in sorted(edge_inputs_dict.items())]]
    result = tfgnn.combine_values(result, self._combine_type)
    if self._use_bias:
      result += self._bias_term
    result = self._activation(result)
    if self._l2_normalize:
      result = tf.math.l2_normalize(result, axis=1)
    return {self._feature_name: result}
