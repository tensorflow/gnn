"""Example models demonstrating modeling functionality of tfgnn."""

import functools
from typing import Optional, Tuple, Callable, Union

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import gcn

MODEL_NAMES = ('GCN', 'Simple', 'JKNet')


def make_map_node_features_layer(
    layer: Callable[[tf.Tensor], tf.Tensor],
    node_set: str = tfgnn.NODES,
    feature_name: str = tfgnn.HIDDEN_STATE) -> tfgnn.keras.layers.MapFeatures:
  """Transforms `graph.node_sets[node_set][feature_name]` through `layer`.

  Example usage:
    transform = tf.keras.layers.Activation("relu")  # Or batchnorm, dropout, ...
    layer = make_map_node_features_layer(transform)
    graph = layer(graph)
  Args:
    layer: Function that maps tf.Tensor -> tf.Tensor. Likely, it is a keras
      trainable layer.
    node_set: name of node set to map. Features for all other node-sets are
      copied as-is without modification.
    feature_name: name of features of `node_set` to map through `layer`. Al
      other features are copied as-is without modification.

  Returns:
    tfgnn.keras.layers.MapFeatures that maps node features of its input
    GraphTensor as: `layer(graph_tensor.node_sets[node_set][feature_name])`
    but leaves all other features unmodified.
  """

  target_node_set = node_set
  target_feature = feature_name
  def _map_node_features(node_set, *, node_set_name):
    """Map feature `target_feature` of `target_node_set` but copy others."""
    if target_node_set != node_set_name:
      return node_set
    return {feat_name: layer(tensor) if feat_name == target_feature else tensor
            for feat_name, tensor in node_set.features.items()}

  return tfgnn.keras.layers.MapFeatures(node_sets_fn=_map_node_features)


def make_model_by_name(
    model_name: str, num_classes: int,
    l2_coefficient: float = 1e-5,
    model_kwargs=None) -> Tuple[bool, tf.keras.Model]:
  """Given a model in `MODEL_NAMES` returns a keras GNN model.

  Args:
    model_name: Must be one of `MODEL_NAMES`. Each name will utilize a different
      model constructor function.
    num_classes: int number of classes.
    l2_coefficient: L2 regularizer coefficient to be passed to
      the underlying Keras model. The regularizer will be applied to kernel
      parameters (not bias terms).
    model_kwargs: If given, must be dict will be passed as kwargs to the model
      constructor function.

  Returns:
    Tuple (bool, model) where the bool indicates if model expects undirected
    graph with single edge set ("edges") -- if False, model expects two edge
    sets ("edges", "rev_edges"). The model is a Keras model that maps
    `GraphTensor` onto model logits (shape=num_seed_nodes x num_classes).
  """
  if model_name not in MODEL_NAMES:
    raise ValueError('Model name should be one of: ' + ', '.join(MODEL_NAMES))
  model_kwargs = model_kwargs or {}
  regularizer = tf.keras.regularizers.L2(l2_coefficient)

  if model_name == 'GCN':
    return True, make_gcn_model(
        num_classes, kernel_regularizer=regularizer, **model_kwargs)
  elif model_name == 'Simple':
    return False, make_simple_model(
        num_classes, kernel_regularizer=regularizer, **model_kwargs)
  elif model_name == 'JKNet':
    return True, make_jknet_model(
        num_classes, kernel_regularizer=regularizer, **model_kwargs)
  else:
    raise ValueError('Invalid model name ' + model_name)


def make_simple_model(
    num_classes: int, hidden_units: int = 200, depth: int = 3,
    kernel_regularizer=None) -> tf.keras.Sequential:
  """Message Passing model show-casing `GraphUpdate`.

  Model expects edge sets "edges" and "rev_edges". Unlike GCN (Kipf & Welling),
  it models incoming VS outgoing edges with different parameter matrices.

  Args:
    num_classes: num of units at output layer.
    hidden_units: size of hidden layers.
    depth: number of graph layers.
    kernel_regularizer: Regularizer to kernel parameters.

  Returns:
    tf.Keras model
  """
  layers = []
  for i in range(depth):
    if i == depth - 1:
      # Output layer.
      next_state_transform = tf.keras.layers.Dense(
          num_classes, kernel_regularizer=kernel_regularizer)
    else:
      next_state_transform = tf.keras.layers.Dense(
          hidden_units, 'relu', kernel_regularizer=kernel_regularizer)

    layer = tfgnn.keras.layers.NodeSetUpdate(
        {
            'edges': tfgnn.keras.layers.SimpleConv(
                tf.keras.Sequential([
                    tf.keras.layers.Dropout(rate=0.5),
                    tf.keras.layers.Dense(
                        hidden_units, 'relu',
                        kernel_regularizer=kernel_regularizer),
                ]), 'mean', receiver_tag=tfgnn.SOURCE),
            'rev_edges': tfgnn.keras.layers.SimpleConv(
                tf.keras.Sequential([
                    tf.keras.layers.Dropout(rate=0.5),
                    tf.keras.layers.Dense(
                        hidden_units, 'relu',
                        kernel_regularizer=kernel_regularizer),
                ]), 'mean',
                receiver_tag=tfgnn.SOURCE),
        },
        tfgnn.keras.layers.NextStateFromConcat(next_state_transform))
    layer = tfgnn.keras.layers.GraphUpdate(node_sets={tfgnn.NODES: layer})
    layers.append(layer)

  return tf.keras.Sequential(layers)


# For conciseness.
_OptionalActivation = Optional[Union[str, Callable[[tf.Tensor], tf.Tensor]]]
_OptionalRegularizer = Optional[tf.keras.regularizers.Regularizer]


def make_gcn_model(
    num_classes: int, depth: int = 3, hidden_units: int = 200,
    hidden_activation: _OptionalActivation = 'relu',
    out_activation: _OptionalActivation = None,
    kernel_regularizer: _OptionalRegularizer = None,
    batchnorm: bool = False, dropout: float = 0) -> tf.keras.Sequential:
  """Implements GCN of Kipf & Welling (ICLR'17) as keras Model."""
  layers = []
  for i in range(depth):
    num_units = num_classes if i == depth - 1 else hidden_units
    activation = out_activation if i == depth - 1 else hidden_activation
    if dropout > 0:
      layers.append(make_map_node_features_layer(
          tf.keras.layers.Dropout(dropout)))

    layers.append(gcn.GCNHomGraphUpdate(
        units=num_units, receiver_tag=tfgnn.SOURCE, add_self_loops=True,
        name='gcn_layer_%i' % i, activation=None,
        kernel_regularizer=kernel_regularizer))

    if batchnorm is not None:
      layers.append(make_map_node_features_layer(
          tf.keras.layers.BatchNormalization(momentum=0.9)))

    if activation is not None:
      layers.append(make_map_node_features_layer(
          tf.keras.layers.Activation(activation)))

  return tf.keras.Sequential(layers)


def make_jknet_model(
    num_classes: int, depth: int = 3, hidden_units: int = 200,
    kernel_regularizer: _OptionalRegularizer = None) -> tf.keras.Model:
  gcn_fn = functools.partial(
      make_gcn_model, hidden_units, depth, hidden_units, out_activation='relu',
      kernel_regularizer=kernel_regularizer)
  return JKNetModel(num_classes, gcn_fn, kernel_regularizer)


class JKNetModel(tf.keras.Model):
  """Implements Jumping Knowledge Network (Xu et al, ICML 2018).

  The model runs a k-layer GCN, then combines all layer features (by
  concatenation) into a classification network.
  """

  def __init__(
      self, num_classes: int,
      make_gnn_fn: Callable[[], tf.keras.Sequential],
      kernel_regularizer: _OptionalRegularizer = None):
    super().__init__()
    self.gcn = make_gnn_fn()
    self.readout_dropout = tf.keras.layers.Dropout(rate=0.5)
    self.readout_layer = tf.keras.layers.Dense(
        num_classes, kernel_regularizer=kernel_regularizer)

  def call(self, graph: tfgnn.GraphTensor, training: bool = True):
    if not self.gcn.built:
      self.gcn(graph)
    # Will be partially over-written.
    out_features = graph.node_sets[tfgnn.NODES].get_features_dict()

    # Node features at all layers:
    all_features = [graph.node_sets[tfgnn.NODES][tfgnn.HIDDEN_STATE]]
    for layer in self.gcn.layers:
      graph = layer(graph)
      all_features.append(
          graph.node_sets[tfgnn.NODES][tfgnn.HIDDEN_STATE])

    # Output = all node feats --> concat --> dropout --> dense.
    out_features[tfgnn.HIDDEN_STATE] = self.readout_layer(
        self.readout_dropout(tf.concat(all_features, axis=1)))

    return graph.replace_features(node_sets={tfgnn.NODES: out_features})
