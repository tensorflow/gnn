"""Contains GraphSAGE convolution layer implementations."""

import collections
from typing import Any, Callable, Optional, Set, Union

import tensorflow as tf
import tensorflow_gnn as tfgnn


@tf.keras.utils.register_keras_serializable(package="GraphSAGE")
class GraphSAGEAggregatorConv(tfgnn.keras.layers.AnyToAnyConvolutionBase):
  """GraphSAGE: element-wise aggregation of neighbors and their linear transformation.

  For a complete GraphSAGE update on a node set, use this class in a
  NodeSetUpdate together with the `GraphSAGENextState` layer to handle the
  node state (see there for details).

  GraphSAGE and the mean aggregation method are from 'Inductive Representation
  Learning on Graphs',
  [Hamilton et al.,2017](https://arxiv.org/abs/1706.02216).
  Following the authors' implementation, dropout is applied to the inputs
  of neighbor nodes (separately for each node-neighbor pair).

  This class supports the element-wise aggregations with different operator
  types besides "mean", see the reduce_type=... argument. For stateful
  transformation with a hidden layer, see class `GraphSAGEPoolingConv`.
  """

  def __init__(
      self,
      *,
      receiver_tag: tfgnn.IncidentNodeTag,
      reduce_type: str = "mean",
      sender_node_feature: Optional[tfgnn.FieldName] = tfgnn.DEFAULT_STATE_NAME,
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
        `tfgnn.DEFAULT_STATE_NAME`.
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
    assert config.pop(
        "receiver_feature",
        None) is None, ("init should guarantee receiver_feature=None")
    assert config.pop(
        "sender_edge_feature",
        None) is None, ("init should guarantee sender_edge_feature=None")
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
               training: bool) -> tf.Tensor:
    """Returns convolution result."""
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
  NodeSetUpdate together with the `GraphSAGENextState` layer to update the final
  node state (see there for details).

  GraphSAGE and the pooling aggregation are from 'Inductive Representation
  Learning on Graphs', [Hamilton et al., 2017]
  (https://arxiv.org/abs/1706.02216), Eq (3). Similar to
  `GraphSAGEAggregatorConv`, dropout is applied to the inputs of neighbor nodes
  (separately for each node-neighbor pair). Then, they are passed through a
  fully connected layer and aggregated by an element-wise maximum (or whichever
  reduce_type is specified), see Eq. (3) in paper. Finally, the result is
  multiplied with the final weights mapping it to output space of units
  dimension.

  The name of this class reflects the terminology of the paper, where "pooling"
  involves the aforementioned hidden layer. For element-wise aggregation
  (as in `tfgnn.pool_edges_to_node()`), see class `GraphSAGEAggregatorConv`.
  """

  def __init__(
      self,
      *,
      receiver_tag: tfgnn.IncidentNodeTag,
      sender_node_feature: Optional[tfgnn.FieldName] = tfgnn.DEFAULT_STATE_NAME,
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
        `tfgnn.DEFAULT_STATE_NAME`.
      units: Number of output units for the final dimensionality of the output
        from the layer.
      hidden_units: Number of output units for the linear transformation applied
        to the sender node features.This specifies the output dimensions of the
        W_pool from Eq (3) in [Hamilton et al., 2017]
          (https://arxiv.org/abs/1706.02216).
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
    assert config.pop(
        "receiver_feature",
        None) is None, ("init should guarantee receiver_feature=None")
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
               training: bool) -> tf.Tensor:
    """Returns convolution result."""
    assert sender_node_input is not None, "sender_node_input can't be None."
    result = broadcast_from_sender_node(sender_node_input)
    result = self._dropout(result, training=training)
    # The "Pooling aggregator" from Eq. (3) of the paper, plus dropout.
    result = self._pooling_transform_fn(result)
    result = pool_to_receiver(result, reduce_type=self._reduce_type)
    result = self._transform_neighbor_fn(result)
    return result


@tf.keras.utils.register_keras_serializable(package="GraphSAGE")
def GraphSAGEGraphUpdate(*,
                         node_set_names: Set[str],
                         receiver_tag: tfgnn.IncidentNodeTag,
                         reduce_type: str = "mean",
                         use_pooling: bool = True,
                         use_bias: bool = True,
                         dropout_rate: float = 0.0,
                         units: int,
                         hidden_units: Optional[int] = None,
                         l2_normalize: bool = True,
                         combine_type: str = "sum",
                         activation: Union[str, Callable[..., Any]] = "relu",
                         feature_name: str = tfgnn.DEFAULT_STATE_NAME,
                         name: str = "graph_sage",
                         **kwargs):
  """Returns a GraphSAGE GraphUpdater layer for nodes in node_set_names.

  For more information on GraphSAGE algorithm please refer to the paper:
  [Hamilton et al., 2017](https://arxiv.org/abs/1706.02216).
  Returned layer applies only one step of GraphSAGE convolution over the
  incident nodes of the edge_set_name_list for the specified node_set_name node.

  Example: GraphSAGE aggregation on heterogenous incoming edges would look as
  below:

  ```
  graph = tfgnn.keras.layers.GraphUpdate(
      node_sets={"paper": tfgnn.keras.layers.NodeSetUpdate(
          { "cites": tfgnn.models.graphsage.GraphSAGEPoolingConv(
                      receiver_tag=tfgnn.TARGET,
                      units=32),
            "writes": tfgnn.models.graphsage.GraphSAGEPoolingConv(
                      receiver_tag=tfgnn.TARGET,
                      units=32,
                      hidden_units=16)},
          tfgnn.models.graphsage.GraphSAGENextState(units=32,
                                                    dropout_rate=0.05))}
  )(graph)
  ```

  Args:
    node_set_names: A set of node_set_names for which GraphSAGE graph update
      happens over each of their incident edges, where node_set_name is
      configured as the receiver_tag end.
    receiver_tag: Either one of `tfgnn.SOURCE` or `tfgnn.TARGET`. The results of
      GraphSAGE are aggregated for this graph piece. When set to `tfgnn.SOURCE`
      or `tfgnn.TARGET`, the layer is called for an edge set and will aggregate
      results at the specified endpoint of the edges.
    reduce_type: An aggregation operation name. Supported list of aggregation
      operators can be found at `tfgnn.get_registered_reduce_operation_names()`.
    use_pooling: If enabled  `GraghSAGEPoolingConv` will be used otherwise
      `GraphSAGEAggregatorConv` will be executed for the provided edges.
    use_bias: If true a bias term will be added to the linear transformations
      for the incident node features as well as for the self node feature.
    dropout_rate: Can be set to a dropout rate that will be applied to both
      incident node features as well as the self node feature.
    units: Number of output units of the linear transformation applied to both
      final aggregated sender node features as well as the self node feature.
    hidden_units: Number of output units to be configure for GraphSAGE pooling
      type convolution only.
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
      `tfgnn.DEFAULT_STATE_NAME`.
    name: Optionally, a name for the layer returned.
    **kwargs: Any optional arguments to `GraphSAGEPoolingConv`,
      `GraphSAGEAggregatorConv` or `GraphSAGENextState`, see there.
  """

  # graph_update_callback is deferred until we get the graph spec.
  def GraphUpdateCallback(spec: tfgnn.GraphTensorSpec):
    if use_pooling != (hidden_units is not None):
      raise ValueError(
          "Either use_pooling or hidden_units has been configured without the "
          "other, please configure them together or disable them both.")
    node_set_update_dict = collections.defaultdict(dict)
    for edge_set_name, edge_set_spec in spec.edge_sets_spec.items():
      node_set_name = edge_set_spec.adjacency_spec.node_set_name(receiver_tag)
      if node_set_name in node_set_names:
        if use_pooling:
          node_set_update_dict[node_set_name][
              edge_set_name] = GraphSAGEPoolingConv(
                  receiver_tag=receiver_tag,
                  sender_node_feature=feature_name,
                  reduce_type=reduce_type,
                  units=units,
                  hidden_units=hidden_units,
                  use_bias=use_bias,
                  dropout_rate=dropout_rate,
                  **kwargs)
        else:
          node_set_update_dict[node_set_name][
              edge_set_name] = GraphSAGEAggregatorConv(
                  receiver_tag=receiver_tag,
                  reduce_type=reduce_type,
                  sender_node_feature=feature_name,
                  units=units,
                  dropout_rate=dropout_rate,
                  **kwargs)
    node_set_updates = dict()
    for node_set_name, edge_set_update_dict in node_set_update_dict.items():
      node_set_updates[node_set_name] = tfgnn.keras.layers.NodeSetUpdate(
          edge_set_update_dict,
          next_state=GraphSAGENextState(
              units=units,
              use_bias=use_bias,
              dropout_rate=dropout_rate,
              feature_name=feature_name,
              l2_normalize=l2_normalize,
              combine_type=combine_type,
              activation=activation,
              **kwargs),
          node_input_feature=feature_name)
    return dict(node_sets=node_set_updates)

  return tfgnn.keras.layers.GraphUpdate(
      deferred_init_callback=GraphUpdateCallback, name=name)


@tf.keras.utils.register_keras_serializable(package="GraphSAGE")
class GraphSAGENextState(tf.keras.layers.Layer):
  """GraphSAGENextState: compute new node states with GraphSAGE algorithm.

  This layer lets you compute a GraphSAGE update of node states from the
  outputs of a `GraphSAGEAggregatorConv` and/or a `GraphSAGEPoolingConv` on
  each of the specified end-point of edge sets.

  Usage example (with strangely mixed aggregations for demonstration):
  ```
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

  The units=... parameter of the next-state layer and all convolutions must be
  equal, unless combine_type="concat" is set.

  GraphSAGE is from 'Inductive Representation Learning on Graphs',
  [Hamilton et al., 2017](https://arxiv.org/abs/1706.02216), Algorithm 1.

  The update of a node state from its neighbors is
  ```
  h_v = sigma(W combine(h_v, h_{N(v)}))  for all nodes v
  ```
  after some aggregation h_{N(v)} of the neighbors of v, followed by a
  combine operation (concat or sum), non-linear activation and L2 normalization.

  Mathematically, if combine_type="sum", the product with the weight matrix W
  is equal to the sum
  ```
  W_{self} h_v  +  W_{neigh} h_{N(v)}.
  ```
  of products with its left and right pieces. The GraphSAGE*Conv classes are
  in charge of computing W_{neigh} h_{N(v)} (for one edge set each, typically
  with separate weights). This class is in charge of computing W_{self} h_v
  from the old node state h_v, combining it with the results for each edge set
  and computing the new node state.

  Beyond the original GraphSAGE, this class supports:
   - dropout, applied to the input h_v, analogous to the dropout provided
     by GraphSAGE*Conv for their inputs;
   - a bias term added just before the final nonlinearity;
   - a configurable combine_type (originally "sum");
   - additional options to influence normalization, activation, etc.
  """

  def __init__(self,
               *,
               units: int,
               use_bias: bool = True,
               dropout_rate: float = 0.0,
               feature_name: str = tfgnn.DEFAULT_STATE_NAME,
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
        `tfgnn.DEFAULT_STATE_NAME`.
      l2_normalize: If enabled l2 normalization will be applied to node state
        vectors.
      combine_type: Can be set to "sum" or "concat". If it's specified as concat
        node state will be concatenated with the sender node features, otherwise
        node state will be added with the sender node features.
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
    x = self._dropout(old_node_state, training=training)
    x = self._self_transform(x)
    x = [x, *[v for _, v in sorted(edge_inputs_dict.items())]]
    x = tfgnn.combine_values(x, self._combine_type)
    if self._use_bias:
      x += self._bias_term
    x = self._activation(x)
    if self._l2_normalize:
      x = tf.math.l2_normalize(x, axis=1)
    return {self._feature_name: x}
