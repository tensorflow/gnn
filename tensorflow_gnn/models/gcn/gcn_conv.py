"""Keras layer for Graph Convolutional Network by Kipf&Welling (2016).

This file implements the fundamental transformation which can be wrapped in
NodeSetUpdate and EdgeSetUpdate.
"""
from typing import Optional, Union

import tensorflow as tf
import tensorflow_gnn as tfgnn

_RegularizerType = Union[tf.keras.regularizers.Regularizer, str]


@tf.keras.utils.register_keras_serializable(package='GNN>models>gcn')
class GCNConv(tf.keras.layers.Layer):
  """Implements the Graph Convolutional Network by Kipf&Welling (2016).

  This class implements a Graph Convolutional Network from
  https://arxiv.org/abs/1609.02907 as a Keras layer that can be used
  as a convolution on an edge set in a tfgnn.keras.layers.NodeSetUpdate.

  Init arguments:
    units: The number of output features (input features are inferred).
    receiver_tag: This layer's result is obtained by pooling the per-edge
      results at this endpoint of each edge. The default is `tfgnn.TARGET`,
      but it is perfectly reasonable to do a convolution towards the
      `tfgnn.SOURCE` instead. (Source and target are conventional names for
      the incident nodes of a directed edge, data flow in a GNN may happen
      in either direction.)
    activation: Keras activation to apply to the result, defaults to 'relu'.
    use_bias: Whether to add bias in the final transformation. The original
      paper doesn't use a bias, but this defaults to True to be consistent
      with Keras and other implementations.
    add_self_loops: Whether to compute the result with self-loops.
    normalize: Whether to normalize the node features by in-degree.
    kernel_initializer: initializer of type tf.keras.initializers .
    node_feature: Name of the node feature to transform.
    **kwargs: additional arguments for the Layer.

  Call arguments:
    graph: The GraphTensor on which to apply the layer.
    edge_set_name: Edge set of `graph` over which to apply the layer.

  Example:

  This example shows how to apply GCNConv to a graph with 2 discrete components.
  This graph has one edge set, named tfgnn.EDGES
  This returns a tf.Tensor of shape (4,3).
  In order to return a GraphTensor this should be wrapped in NodeSetUpdate/
  EdgeSetUpdate.
  ```
  >>> import tensorflow as tf
  >>> import tensorflow_gnn as tfgnn
  >>> from tensorflow_gnn.models.gcn import gcn_conv
  >>> graph = tfgnn.GraphTensor.from_pieces(
  ...    node_sets={
  ...        tfgnn.NODES: tfgnn.NodeSet.from_fields(
  ...            sizes=[2, 2],
  ...            features={tfgnn.HIDDEN_STATE: tf.constant(
  ...                          [[1., 0, 0], [0, 1, 0]]*2)})},
  ...    edge_sets={
  ...        tfgnn.EDGES: tfgnn.EdgeSet.from_fields(
  ...            sizes=[2, 2],
  ...            adjacency=tfgnn.Adjacency.from_indices(
  ...                source=(tfgnn.NODES, tf.constant([0, 1, 2, 3],
  ...                                                 dtype=tf.int64)),
  ...                target=(tfgnn.NODES, tf.constant([1, 0, 3, 2],
  ...                                                 dtype=tf.int64))))})
  >>> gcnconv = gcn_conv.GCNConv(3)
  >>> gcnconv(graph, edge_set_name=tfgnn.EDGES)
  <... shape=(4, 3) dtype=...>
  ```
  """

  def __init__(self,
               units: int,
               *,
               receiver_tag: tfgnn.IncidentNodeTag = tfgnn.TARGET,
               activation='relu',
               use_bias: bool = True,
               add_self_loops: bool = True,
               normalize: bool = True,
               kernel_initializer: bool = None,
               node_feature: Optional[str] = tfgnn.HIDDEN_STATE,
               kernel_regularizer: Optional[_RegularizerType] = None,
               **kwargs):

    super().__init__(**kwargs)
    self._filter = tf.keras.layers.Dense(
        units=units,
        activation=activation,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer)
    self._add_self_loops = add_self_loops
    self._normalize = normalize
    self._node_feature = node_feature
    self._receiver = receiver_tag
    self._sender = tfgnn.reverse_tag(receiver_tag)

  def get_config(self):
    return dict(
        receiver_tag=self._receiver,
        node_feature=self._node_feature,
        add_self_loops=self._add_self_loops,
        normalize=self._normalize,
        units=self._filter.get_config()['units'],
        activation=self._filter.get_config()['activation'],
        use_bias=self._filter.get_config()['use_bias'],
        kernel_initializer=self._filter.get_config()['kernel_initializer'],
        **super().get_config())

  def call(
      self,
      graph: tfgnn.GraphTensor,
      *,
      edge_set_name: Optional[tfgnn.EdgeSetName],
  ):
    # Calculate the diagonal of the degree matrix
    # Broadcasting this diagonal is more efficient than forming
    # the diagonal matrix
    edge_adj = graph.edge_sets[edge_set_name].adjacency
    sender_name = edge_adj.node_set_name(self._sender)
    if edge_adj.node_set_name(self._receiver) != sender_name:
      raise ValueError('source and target node sets must be the same '
                       f'for edge set {edge_set_name} ')

    nnodes = tf.cast(graph.node_sets[sender_name].total_size, tf.int64)
    float_type = graph.node_sets[sender_name][self._node_feature].dtype

    if self._normalize:
      edge_set = graph.edge_sets[edge_set_name]
      edge_ones = tf.ones([edge_set.total_size, 1])
      in_degree = tf.squeeze(tfgnn.pool_edges_to_node(
          graph,
          edge_set_name,
          self._receiver,
          'sum',
          feature_value=edge_ones), -1)
      # Degree matrix is the sum of rows of adjacency
      # Adding self-loops adds an identity matrix to the adjacency
      # This adds 1 to each diagonal element of the degree matrix
      if self._add_self_loops:
        in_degree += 1
      invsqrt_deg = tf.math.rsqrt(in_degree)
    else:
      invsqrt_deg = tf.ones(nnodes, dtype=float_type)

    # Calculate \hat{D^{-1/2}}X first
    normalized_values = (
        invsqrt_deg[:, tf.newaxis] *
        graph.node_sets[sender_name][self._node_feature])

    # Calculate A\hat{D^{-1/2}}X by broadcasting then pooling
    source_bcast = tfgnn.broadcast_node_to_edges(
        graph,
        edge_set_name,
        self._sender,
        feature_value=normalized_values,
    )
    pooled = tfgnn.pool_edges_to_node(
        graph, edge_set_name, self._receiver, 'sum', feature_value=source_bcast)

    # left-multiply the result by \hat{D^{-1/2}}
    pooled = invsqrt_deg[:, tf.newaxis] * pooled

    # Add \hat{D^{-1/2}} I \hat{D^{-1/2}} X
    # Since the right two factors are already computed,
    # we can remove I and just multiply by the normalizing matrix again
    if self._add_self_loops:
      pooled += invsqrt_deg[:, tf.newaxis] * normalized_values

    return self._filter(pooled)


def GCNConvGraphUpdate(*,  # To be called like a class initializer.  pylint: disable=invalid-name
                       units: int,
                       edge_set_name: str,
                       receiver_tag: tfgnn.IncidentNodeTag = tfgnn.TARGET,
                       feature_name: str = tfgnn.HIDDEN_STATE,
                       name: str = 'gcn',
                       **kwargs):
  """Returns a graph update layer for GCN convolution.

  The returned layer performs one update step of a Graph Convolutional Network
  (GCN) from https://arxiv.org/abs/1609.02907 on an edge set of a GraphTensor.
  It is best suited for graphs that have just that one edge set, which connects
  one node set to itself.
  For heterogeneous graphs with multiple edge sets connecting a single node set,
  users are advised to consider a GraphUpdate with one or more GCNConv objects
  instead.

  Args:
    units: The desired number of output node features.
    edge_set_name: A GCNConv update happens on this edge set and its incident
      node set of the input GraphTensor.
    receiver_tag: The default is `tfgnn.TARGET`,
      but it is perfectly reasonable to do a convolution towards the
      `tfgnn.SOURCE` instead. (Source and target are conventional names for
      the incident nodes of a directed edge, data flow in a GNN may happen
      in either direction.)
    feature_name: The feature name of node states; defaults to
      tfgnn.HIDDEN_STATE.
    name: Optionally, a name for the layer returned.
    **kwargs: Any optional arguments to GCNConv, see there.
  """

  # Build a GraphUpdate for the target node set of the given edge_set_name.
  # That needs to be deferred until we see a GraphTensorSpec that tells us
  # the node_set_name.
  def deferred_init_callback(spec: tfgnn.GraphTensorSpec):
    node_set_name = spec.edge_sets_spec[
        edge_set_name].adjacency_spec.node_set_name(tfgnn.TARGET)
    if spec.edge_sets_spec[edge_set_name].adjacency_spec.node_set_name(
        tfgnn.SOURCE) != node_set_name:
      raise ValueError('source and target node sets must be the same '
                       f'for edge set {edge_set_name} ')
    node_set_updates = {
        node_set_name: tfgnn.keras.layers.NodeSetUpdate(
            {edge_set_name: GCNConv(
                units=units,
                receiver_tag=receiver_tag,
                node_feature=feature_name,
                **kwargs)},
            next_state=NextStateForNodeSetFromSingleEdgeSetInput(),
            node_input_feature=feature_name)}
    return dict(node_sets=node_set_updates)
  return tfgnn.keras.layers.GraphUpdate(
      deferred_init_callback=deferred_init_callback, name=name)


# For use by GCNConvGraphUpdate().
@tf.keras.utils.register_keras_serializable(package='GNN>models>gcn')
class NextStateForNodeSetFromSingleEdgeSetInput(tf.keras.layers.Layer):

  def call(self, inputs):
    unused_node_input, edge_inputs, unused_context_input = inputs
    single_edge_set_input, = edge_inputs.values()  # Unpack.
    return single_edge_set_input
