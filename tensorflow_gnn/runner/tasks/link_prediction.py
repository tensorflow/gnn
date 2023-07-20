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
"""Link Prediction tasks."""

import abc
from typing import Callable, Optional, Sequence, Tuple

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.runner import interfaces


def _validate_readout_for_link_prediction(
    graph: tfgnn.GraphTensor, *,
    readout_node_set_name: tfgnn.NodeSetName = '_readout'):
  """Raises `ValueError` if any of `*_set_names` is absent from `graph`."""
  ns_name = readout_node_set_name  # for short.
  src_name = readout_node_set_name + '/source'
  tgt_name = readout_node_set_name + '/target'
  if (ns_name not in graph.node_sets
      or src_name not in graph.edge_sets
      or tgt_name not in graph.edge_sets
      or graph.edge_sets[src_name].adjacency.target_name != ns_name
      or graph.edge_sets[tgt_name].adjacency.target_name != ns_name):
    raise ValueError(
        f'GraphTensor for link prediction must contain node set "{ns_name}". '
        'In addition, GraphTensor must contain edge sets '
        f'"{src_name}" and "{tgt_name}": respectively, their `source` '
        'endpoint must name the source and target node set for link '
        'prediciton; Further, `target` of both must have point to node set '
        f'"{ns_name}".')


class _LinkPrediction(interfaces.Task):
  """Base class for implementing Link Prediction tasks on heterogeneous graphs.

  Suppose link prediction task from node set `"S"` to node set `"T"`, i.e.,
  `"S"` for `source` and `"T"` for `target`. Then, this task expects
  `GraphTensor` with:

  * node sets `"S"` and `"T"`.
  * node set `"_readout"`, with `sizes=[B]`, where `[B]` equals number of
    supervised edges (positive or negative). This node-set must have feature
    with name `label` with `dtype=tf.float32` and `shape=[B]`. It should contain
    the binary label for every labeled edge.
  * with edge set `"_readout/source"` that should connect `"S"` to
    `"_readout"`. There should be exactly `[B]` edges with source IDs set to the
    source end-points and target IDs to range(B).
  * with edge set `"_readout/target"` that should connect `"T"` to
    `"_readout"`. There should be exactly `[B]` edges with source IDs set to the
    target end-points and target IDs to `tf.range(B)`.
  * Lists must correspond: `graph.node_sets["_readout"]["label"]` must be the
    labels for connecting edges
    `graph.edge_sets["_readout/source"].adjacency.source` to
    `graph.edge_sets["_readout/target"].adjacency.source`.
  * NOTE: the names "_readout", "_readout/source", "_readout/target" are
    injected into `GraphTensor` instances can be created by
    `sampler/link_samplers.py`. If you create your own `GraphTensor` examples,
    you may override these names by supplying to constructor
    `readout_node_set_name` and `readout_label_feature_name`. You may also refer
    to function `tfgnn.structured_readout` for full documentation.

  Implementing classes must override `_compute_edge_scores(...)` to return
  tensor of scores (`shape=[B]`) given two matrices: source features, target
  features, both with shape `[B, d]` where `d` denotes hidden (latent) feature
  dimensions.
  """

  def __init__(
      self, *,
      node_feature_name: tfgnn.FieldName = tfgnn.HIDDEN_STATE,
      readout_label_feature_name: str = 'label',
      readout_node_set_name: tfgnn.NodeSetName = '_readout'):
    """Constructs link prediction task.

    The default readout node-set and label feature names are maintained by
    examples/sampling_runner.py and experimental/link_samplers.py.

    Args:
      node_feature_name: Name of feature where node state for link-prediction
        is read from. The final link prediction score will be:
        ```
          score(graph.node_sets[source][node_feature_name],
                graph.node_sets[target][node_feature_name])
        ```
        where `source` and `target`, respectively, are:
        `graph.edge_sets[readout_node_set_name+"/source"].adjacency.source_name`
        and
        `graph.edge_sets[readout_node_set_name+"/target"].adjacency.source_name`
      readout_label_feature_name: The labels for edge connections,
        source nodes
        `graph.edge_sets[readout_node_set_name+"/source"].adjacency.source` in
        node set `graph.node_sets[source]` against target nodes
        `graph.edge_sets[readout_node_set_name+"/target"].adjacency.source` in
        node set `graph.node_sets[source]`, must be stored in
        `graph.node_sets[readout_node_set_name][readout_label_feature_name]`.
      readout_node_set_name: Determines the readout node-set, which must have
        feature `readout_label_feature_name`, and must receive connections (at
        target endpoints) from edge-sets `readout_node_set_name+"/source"` and
        `readout_node_set_name+"/target"`.
    """
    super().__init__()
    self._node_feature_name = node_feature_name
    self._readout_label_feature_name = readout_label_feature_name
    self._readout_node_set_name = readout_node_set_name

  @abc.abstractmethod
  def _compute_edge_scores(
      self, src_features: tf.Tensor, tgt_features: tf.Tensor) -> tf.Tensor:
    """Accepts 2 tensors with equal leading dimension (`=B`) to output (`B`) scores.

    The function outputs `output` tf.Tensor with `output.shape[0] == B`, with
    `output[i]` containing score for pair `(src_features[i], tgt_features[i])`.

    Args:
      src_features: Tensor with shape (B, ...) containing features for source
        nodes, where B is batch size (number of links to be evaluated).
      tgt_features: Tensor with shape (B, ...) containing features for target
        nodes.
    """
    raise NotImplementedError()

  def preprocess(self, gt: tfgnn.GraphTensor) -> Tuple[
      tfgnn.GraphTensor, tfgnn.Field]:
    _validate_readout_for_link_prediction(
        gt, readout_node_set_name=self._readout_node_set_name)
    x = gt
    y = tfgnn.keras.layers.Readout(
        feature_name=self._readout_label_feature_name,
        node_set_name=self._readout_node_set_name)(gt)
    return x, y

  def predict(self, graph: tfgnn.GraphTensor) -> tfgnn.Field:
    tfgnn.check_scalar_graph_tensor(graph, name='LinkPrediction')
    src_features = tfgnn.keras.layers.StructuredReadout(
        key='source', feature_name=self._node_feature_name)(graph)
    tgt_features = tfgnn.keras.layers.StructuredReadout(
        key='target', feature_name=self._node_feature_name)(graph)
    scores = self._compute_edge_scores(src_features, tgt_features)

    return scores

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    """Binary cross-entropy."""
    return (tf.keras.losses.BinaryCrossentropy(from_logits=True),)

  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return (tf.keras.metrics.BinaryAccuracy(),)


class DotProductLinkPrediction(_LinkPrediction):
  """Implements edge score as dot product of features of endpoint nodes."""

  def _compute_edge_scores(
      self, src_features: tf.Tensor, tgt_features: tf.Tensor) -> tf.Tensor:
    if src_features.shape.rank != 2 or tgt_features.shape.rank != 2:
      raise ValueError('DotProductLinkPrediction is only supported for '
                       'matrix Tensors (batch size x feature size). Did you '
                       'mean to reshape?')
    return tf.expand_dims(tf.einsum('ij,ij->i', src_features, tgt_features), -1)


class HadamardProductLinkPrediction(_LinkPrediction):
  """Implements edge score as hadamard product of features of endpoint nodes.

  The hadamard product is followed by one layer with scalar output.
  """
  _dense_layer: Optional[tf.keras.layers.Layer] = None

  def _compute_edge_scores(
      self, src_features: tf.Tensor, tgt_features: tf.Tensor) -> tf.Tensor:
    if self._dense_layer is None:
      self._dense_layer = tf.keras.layers.Dense(1, name='linkpred')
    if src_features.shape.rank != 2 or tgt_features.shape.rank != 2:
      raise ValueError('HadamardProductLinkPrediction is only supported for '
                       'matrix Tensors (batch size x feature size). Did you '
                       'mean to reshape?')
    hadamard = src_features * tgt_features
    return self._dense_layer(hadamard)
