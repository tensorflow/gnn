"""Methods for read-out functions from GraphTensor."""

import tensorflow as tf
import tensorflow_gnn as tfgnn


def readout_seed_node_features(
    graph_tensor: tfgnn.GraphTensor,
    seed_feature_name: str = 'seed', node_set_name: str = tfgnn.NODES,
    node_features_name: str = tfgnn.HIDDEN_STATE) -> tf.Tensor:
  """Read node features of multiple seed nodes identified by a context feature.

  Use this function instead of `tfgnn.gather_first_node()` if `graph_tensor`
  has one graph with one graph component but multiple seed nodes, which are
  identified by storing their indices in a context feature.

  Args:
    graph_tensor: A scalar GraphTensor with a single graph component.
    seed_feature_name: the name of a context feature with the indices of
      seed nodes. Must have shape `[1, num_seeds]`.
    node_set_name: Name of the node set containing the seed nodes.
    node_features_name: Name of the feature to be read and returned.
      Must be a tf.Tensor of shape [num_nodes, *feature_shape].

  Returns:
    A tf.Tensor of shape `[num_seeds, *feature_shape]`. The value at index i is
    the feature value for node `graph_tensor.context[seed_feature_name][i]`.
  """
  seed_node_positions = graph_tensor.context[seed_feature_name]
  features = graph_tensor.node_sets[node_set_name][node_features_name]
  return  tf.gather(features, seed_node_positions)


def readout_groundtruth_labels(
    num_classes: int, graph_tensor: tfgnn.GraphTensor,
    seed_feature_name: str = 'seed', node_set_name: str = tfgnn.NODES,
    node_label_features_name: str = 'label'
) -> tf.Tensor:
  """node features['label'] for node IDs listed in context.features['seed']."""
  labels = readout_seed_node_features(
      graph_tensor, seed_feature_name, node_set_name, node_label_features_name)
  return tf.one_hot(labels, num_classes)


def pair_graphs_with_labels(
    num_classes: int, graph_tensor: tfgnn.GraphTensor,
    node_set_name: str = tfgnn.NODES, node_label_features_name: str = 'label'):
  """Returns pair (`graph_tensor` without labels, seed nodes one-hot labels)."""
  seed_y = readout_groundtruth_labels(
      num_classes, graph_tensor, node_set_name=node_set_name,
      node_label_features_name=node_label_features_name)

  features = dict(graph_tensor.node_sets[node_set_name].features)
  features.pop(node_label_features_name, None)
  graph_nolabels = graph_tensor.replace_features(
      node_sets={node_set_name: features})

  return graph_nolabels, seed_y
