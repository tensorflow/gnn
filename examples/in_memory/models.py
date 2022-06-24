"""Example models demonstrating modeling functionality of tfgnn."""

import functools
from typing import Tuple, Callable

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import gcn

MODEL_NAMES = ('GCN', 'Simple', 'JKNet')


def make_model_by_name(
    model_name: str, num_classes: int,
    l2_coefficient: float = 1e-5) -> Tuple[bool, tf.keras.Model]:
  """Given a model in `MODEL_NAMES` returns a keras GNN model.

  Args:
    model_name: Must be one of `MODEL_NAMES`.
    num_classes: int number of classes.
    l2_coefficient: L2 regularizer coefficient to be passed to
      the underlying Keras model. The regularizer will be applied to kernel
      parameters (not bias terms).

  Returns:
    Tuple (bool, model) where the bool indicates if model expects undirected
    graph with single edge set ("edges") -- if False, model expects two edge
    sets ("edges", "rev_edges"). The model is a Keras model that maps
    `GraphTensor` onto model logits (shape=num_seed_nodes x num_classes).
  """
  if model_name not in MODEL_NAMES:
    raise ValueError('Model name should be one of: ' + ', '.join(MODEL_NAMES))
  regularizer = tf.keras.regularizers.L2(l2_coefficient)
  if model_name == 'GCN':
    return True, make_gcn_model(
        num_classes, hidden_units=200, depth=3, kernel_regularizer=regularizer)
  elif model_name == 'Simple':
    return False, make_simple_model(num_classes, kernel_regularizer=regularizer)
  elif model_name == 'JKNet':
    return True, make_jknet_model(
        num_classes, hidden_units=200, depth=3, kernel_regularizer=regularizer)
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


def make_gcn_model(
    num_classes, depth=2, hidden_units=200, hidden_activation='relu',
    out_activation=None, kernel_regularizer=None) -> tf.keras.Sequential:
  """Implements GCN of Kipf & Welling (ICLR'17) as keras Model."""
  layers = []
  for i in range(depth):
    num_units = num_classes if i == depth - 1 else hidden_units
    activation = out_activation if i == depth - 1 else hidden_activation
    layers.append(gcn.GCNHomGraphUpdate(
        units=num_units, receiver_tag=tfgnn.SOURCE, add_self_loops=True,
        name='gcn_layer_%i' % i, activation=activation,
        kernel_regularizer=kernel_regularizer))
  return tf.keras.Sequential(layers)


def make_jknet_model(
    num_classes: int, depth: int = 2, hidden_units: int = 128,
    kernel_regularizer=None) -> tf.keras.Model:
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
      kernel_regularizer=None):
    super().__init__()
    self.gcn = make_gnn_fn()
    self.readout_dropout = tf.keras.layers.Dropout(rate=0.5)
    self.readout_layer = tf.keras.layers.Dense(
        num_classes, kernel_regularizer=kernel_regularizer)

  def call(self, graph, training=True):
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
