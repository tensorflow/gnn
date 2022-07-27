"""Layers for operating on node features without processing graph edges."""
import collections
from typing import Callable, Iterable, Tuple

import tensorflow as tf

from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt


def map_node_features(
    graph: gt.GraphTensor,
    nodeset_feature_mapfn_tuples: Iterable[
        Tuple[
            const.NodeSetName, const.FieldName,
            Callable[[tf.Tensor], tf.Tensor]]]) -> gt.GraphTensor:
  """Transforms node features on `graph` through custom functions.

  Args:
    graph: GraphTensor which its node features will be replaced.
    nodeset_feature_mapfn_tuples: list of triplets (node_set, feature_name, fn),
      where `graph.node_sets[node_set][feature_name]` will be mapped through
      `fn`. Node features that are not present will not be modified.

  Returns:
    `GraphTensor` with selective node features replaced, according to
    `nodeset_feature_mapfn_tuples`.
  """
  # Group-by node set.
  nodeset_to_feature_to_mapfn = collections.defaultdict(dict)
  for nodeset, feature, map_fn in nodeset_feature_mapfn_tuples:
    nodeset_to_feature_to_mapfn[nodeset][feature] = map_fn

  replacement_features = {}
  for node_set, feature_to_mapfn in nodeset_to_feature_to_mapfn.items():
    replacement_features[node_set] = {}
    for feat_name, feat_value in graph.node_sets[node_set].features.items():
      if feat_name in feature_to_mapfn:
        feat_value = feature_to_mapfn[feat_name](feat_value)
      replacement_features[node_set][feat_name] = feat_value

  return graph.replace_features(node_sets=replacement_features)


class MapNodeFeatures(tf.keras.layers.Layer):

  def __init__(self, nodeset_feature_mapfn_tuples):
    super().__init__()
    self.nodeset_feature_mapfn_tuples = nodeset_feature_mapfn_tuples

  def call(self, graph: gt.GraphTensor) -> gt.GraphTensor:
    return map_node_features(graph, self.nodeset_feature_mapfn_tuples)


class ActivateNodeFeatures(MapNodeFeatures):

  def __init__(
      self,
      nodeset_feature_activation_tuples=(
          (const.NODES, const.HIDDEN_STATE, 'relu'),)):
    super().__init__(
        [(n, f, tf.keras.layers.Activation(v))
         for (n, f, v) in nodeset_feature_activation_tuples])


class BatchNormalizeNodeFeatures(MapNodeFeatures):

  def __init__(
      self,
      nodeset_feature_bnargs_tuples=((const.NODES, const.HIDDEN_STATE, None),)):
    super().__init__(
        [(n, f, tf.keras.layers.BatchNormalization(**(kwargs or {})))
         for (n, f, kwargs) in nodeset_feature_bnargs_tuples])


class DropoutNodeFeatures(MapNodeFeatures):

  def __init__(
      self,
      nodeset_feature_dropout_tuples=((const.NODES, const.HIDDEN_STATE, 0.5),)):
    super().__init__(
        [(n, f, tf.keras.layers.Dropout(v))
         for (n, f, v) in nodeset_feature_dropout_tuples])
