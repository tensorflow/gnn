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
"""Keras layer for Graph Convolutional Network by Kipf&Welling (2016).

This file implements the fundamental transformation which can be wrapped in
NodeSetUpdate and EdgeSetUpdate.
"""
from typing import Any, Optional

import tensorflow as tf
import tensorflow_gnn as tfgnn


@tf.keras.utils.register_keras_serializable(package='GNN>models>gcn')
class GCNConv(tf.keras.layers.Layer):
  r"""Implements the Graph Convolutional Network by Kipf&Welling (2016).

  This class implements a Graph Convolutional Network from
  https://arxiv.org/abs/1609.02907 as a Keras layer that can be used
  as a convolution on an edge set in a tfgnn.keras.layers.NodeSetUpdate.
  The original algorithm proposed in the Graph Convolutional Network paper
  expects a symmetric graph as input. That is, if there is an edge from node i
  to node j, there is also an edge from node j to node i. This implementation,
  however, is able to take asymmetric graphs as input.

  Let $w_{ij}$ be the weight of the edge from sender i to receiver j.
  Let $\deg^{in}_i$ be the number of incoming edges to i (in the direction
  of message flow, see `receiver_tag`), and $\deg^{out}_i$ the number of
  outgoing edges from i. In a symmetric graphs, both are equal.

  In this implementation, we provide multiple approaches for normalizing an edge
  weight $w_{ij}$ in $v_{ij}$, namely `"none"`, `"in"`, `"out"`, `"in_out"`, and
  `"in_in"`. Setting normalization to `"none"` will end up in set $v_{ij} =
  w_{ij}$.
  The `"in"` normalization normalizes edge weights using the in-degree of the
  receiver node, that is:

  $$v_{ij} = w_{ij} / \deg^{in}_j.$$

  The `"out"` normalization normalizes edges using the out-degree of sender
  nodes that is:

  $$v_{ij} = w_{ij} / \deg^{out}_i.$$

  The `"in_out"` normalization normalizes edges as follows:

  $$v_{ij} = w_{ij} / (\sqrt{\deg^{out}_i} \sqrt{\deg^{in}_j}).$$

  The `"in_in"` normalization normalizes the edge weights as:

  $$v_{ij} = w_{ij} / (\sqrt{\deg^{in}_i} \sqrt{\deg^{in}_j}).$$

  For symmetric graphs (as in the original GCN paper), `"in_out"` and `"in_in"`
  are equal, but the latter needs to compute degrees just once.

  Init arguments:
    units: Number of output units for this transformation applied to sender
      node features.
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
    add_self_loops: Whether to compute the result as if a loop from each node
      to itself had been added to the edge set. The self-loop edges are added
      with an edge weight of one.
    kernel_initializer: Can be set to a `kernel_initializer` as understood
      by `tf.keras.layers.Dense` etc.
      An `Initializer` object gets cloned before use to ensure a fresh seed,
      if not set explicitly. For more, see `tfgnn.keras.clone_initializer()`.
    node_feature: Name of the node feature to transform.
    edge_weight_feature_name: Can be set to the name of a feature on the edge
      set that supplies a scalar weight for each edge. The GCN computation uses
      it as the edge's entry in the adjacency matrix, instead of the default 1.
    degree_normalization: Can be set to `"none"`, `"in"`, `"out"`, `"in_out"`,
      or `"in_in"`, as explained above.
    **kwargs: additional arguments for the Layer.

  Call arguments:
    graph: The GraphTensor on which to apply the layer.
    edge_set_name: Edge set of `graph` over which to apply the layer.

  Example:

  This example shows how to apply GCNConv to a graph with 2 discrete components.
  This graph has one edge set, named `tfgnn.EDGES`.
  This returns a tf.Tensor of shape (4,3).
  In order to return a GraphTensor this should be wrapped in NodeSetUpdate/
  EdgeSetUpdate.

  ```python
  import tensorflow as tf
  import tensorflow_gnn as tfgnn
  from tensorflow_gnn.models.gcn import gcn_conv
  graph = tfgnn.GraphTensor.from_pieces(
     node_sets={
         tfgnn.NODES: tfgnn.NodeSet.from_fields(
             sizes=[2, 2],
             features={tfgnn.HIDDEN_STATE: tf.constant(
                           [[1., 0, 0], [0, 1, 0]]*2)})},
     edge_sets={
         tfgnn.EDGES: tfgnn.EdgeSet.from_fields(
             sizes=[2, 2],
             adjacency=tfgnn.Adjacency.from_indices(
                 source=(tfgnn.NODES, tf.constant([0, 1, 2, 3],
                                                  dtype=tf.int64)),
                 target=(tfgnn.NODES, tf.constant([1, 0, 3, 2],
                                                  dtype=tf.int64))))})
  gcnconv = gcn_conv.GCNConv(3)
  gcnconv(graph, edge_set_name=tfgnn.EDGES)   # Has shape=(4, 3).
  ```
  """

  def __init__(
      self,
      units: int,
      *,
      receiver_tag: tfgnn.IncidentNodeTag = tfgnn.TARGET,
      activation='relu',
      use_bias: bool = True,
      add_self_loops: bool = False,
      kernel_initializer: Any = None,
      node_feature: Optional[str] = tfgnn.HIDDEN_STATE,
      kernel_regularizer: Any = None,
      edge_weight_feature_name: Optional[tfgnn.FieldName] = None,
      degree_normalization: str = 'in_out',
      **kwargs,
  ):
    super().__init__(**kwargs)
    self._filter = tf.keras.layers.Dense(
        units=units,
        activation=activation,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=tfgnn.keras.clone_initializer(kernel_initializer))
    self._add_self_loops = add_self_loops
    self._node_feature = node_feature
    self._receiver = receiver_tag
    self._sender = tfgnn.reverse_tag(receiver_tag)
    self._edge_weight_feature_name = edge_weight_feature_name
    self._degree_normalization = degree_normalization

  def get_config(self):
    filter_config = self._filter.get_config()
    return dict(
        receiver_tag=self._receiver,
        node_feature=self._node_feature,
        add_self_loops=self._add_self_loops,
        units=filter_config['units'],
        activation=filter_config['activation'],
        use_bias=filter_config['use_bias'],
        kernel_initializer=filter_config['kernel_initializer'],
        kernel_regularizer=filter_config['kernel_regularizer'],
        edge_weight_feature_name=self._edge_weight_feature_name,
        degree_normalization=self._degree_normalization,
        **super().get_config(),
    )

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

    edge_set = graph.edge_sets[edge_set_name]
    if self._edge_weight_feature_name is not None:
      try:
        edge_weights = graph.edge_sets[edge_set_name][
            self._edge_weight_feature_name
        ]
      except KeyError as e:
        raise ValueError(
            f'{self._edge_weight_feature_name} is not given '
            f'for edge set {edge_set_name} '
        ) from e
      if edge_weights.shape.rank != 1:
        # GraphTensor guarantees it is not None.
        raise ValueError(
            'Expecting vector for edge weights. Received rank '
            f'{edge_weights.shape.rank}.'
        )
      edge_weights = tf.expand_dims(
          edge_weights, axis=1
      )  # Align with state feature.
    else:
      edge_weights = tf.ones([edge_set.total_size, 1])

    def get_degree(node_tag: tfgnn.IncidentNodeTag):
      # If node_tag is receiver, this function computes the in_degree of nodes
      # and if node_tag is sender, it comptes the out_degree of nodes.
      # Shape of node_degree is [nnodes, 1]
      node_degree = tfgnn.pool_edges_to_node(
          graph,
          edge_set_name,
          node_tag,
          'sum',
          feature_value=edge_weights,
      )
      # Adding self-loops connects each node to itself.
      # This adds 1 to each diagonal element of the degree matrix
      if self._add_self_loops:
        node_degree += 1
      else:
        # Prevent division by zero.
        node_degree = tf.maximum(node_degree, 1)
      return node_degree

    if self._degree_normalization == 'none':
      sender_scale = receiver_scale = None
    elif self._degree_normalization == 'in':
      receiver_scale = 1 / get_degree(self._receiver)
      sender_scale = None
    elif self._degree_normalization == 'out':
      sender_scale = 1 / get_degree(self._sender)
      receiver_scale = None
    elif self._degree_normalization == 'in_out':
      sender_scale = tf.math.rsqrt(get_degree(self._sender))
      receiver_scale = tf.math.rsqrt(get_degree(self._receiver))
    elif self._degree_normalization == 'in_in':
      sender_scale = receiver_scale = tf.math.rsqrt(get_degree(self._receiver))
    else:
      raise ValueError(
          'Expecting degree_normalization to be `none`, `in`, `out`,'
          ' `in_out`, or `in_in`.'
      )

    if sender_scale is not None:
      normalized_values = (
          sender_scale * graph.node_sets[sender_name][self._node_feature]
      )
    else:
      normalized_values = graph.node_sets[sender_name][self._node_feature]

    source_bcast = tfgnn.broadcast_node_to_edges(
        graph,
        edge_set_name,
        self._sender,
        feature_value=normalized_values,
    )
    if self._edge_weight_feature_name is not None:
      source_bcast = source_bcast * edge_weights
    pooled = tfgnn.pool_edges_to_node(
        graph, edge_set_name, self._receiver, 'sum', feature_value=source_bcast)
    if receiver_scale is not None:
      pooled = receiver_scale * pooled

    if self._add_self_loops:
      if receiver_scale is not None:
        pooled += receiver_scale * normalized_values
      else:
        pooled += normalized_values

    return self._filter(pooled)


def GCNHomGraphUpdate(*,  # To be called like a class initializer.  pylint: disable=invalid-name
                      units: int,
                      receiver_tag: tfgnn.IncidentNodeTag = tfgnn.TARGET,
                      add_self_loops: bool = False,
                      feature_name: str = tfgnn.HIDDEN_STATE,
                      name: str = 'gcn',
                      **kwargs):
  """Returns a graph update layer for GCN convolution.

  The returned layer performs one update step of a Graph Convolutional Network
  (GCN) from https://arxiv.org/abs/1609.02907 on a GraphTensor that stores
  a homogeneous graph.
  For heterogeneous graphs with multiple edge sets connecting a single node set,
  users are advised to consider a GraphUpdate with one or more GCNConv objects
  instead.

  > IMPORTANT: By default, the graph convolution computed by this class takes
  > inputs only along edges that are explicitly stored in the input GraphTensor.
  > Including the old node state in the inputs for computing the new node state
  > requires having an explicit loop in the edge set, or setting
  > `add_self_loops=True`.

  Args:
    units: The dimension of output hidden states for each node.
    receiver_tag: The default is `tfgnn.TARGET`,
      but it is perfectly reasonable to do a convolution towards the
      `tfgnn.SOURCE` instead. (Source and target are conventional names for
      the incident nodes of a directed edge, data flow in a GNN may happen
      in either direction.)
    add_self_loops: Whether to compute the result as if a loop from each node
      to itself had been added to the edge set.
    feature_name: The feature name of node states; defaults to
      `tfgnn.HIDDEN_STATE`.
    name: Optionally, a name for the layer returned.
    **kwargs: Any optional arguments to GCNConv, see there.
  """

  # Build a GraphUpdate for the target node set of the given edge_set_name.
  # That needs to be deferred until we see a GraphTensorSpec that tells us
  # the node_set_name.
  def deferred_init_callback(spec: tfgnn.GraphTensorSpec):
    node_set_name, edge_set_name = tfgnn.get_homogeneous_node_and_edge_set_name(
        spec, 'GCNHomGraphUpdate')
    node_set_updates = {
        node_set_name: tfgnn.keras.layers.NodeSetUpdate(
            {edge_set_name: GCNConv(
                units=units,
                receiver_tag=receiver_tag,
                add_self_loops=add_self_loops,
                node_feature=feature_name,
                **kwargs,  # Any kernel_initializer gets cloned by GCNConv.
            )},
            next_state=tfgnn.keras.layers.SingleInputNextState(),
            node_input_feature=feature_name)}
    return dict(node_sets=node_set_updates)
  return tfgnn.keras.layers.GraphUpdate(
      deferred_init_callback=deferred_init_callback, name=name)
