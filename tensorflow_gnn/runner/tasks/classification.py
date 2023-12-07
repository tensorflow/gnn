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
"""Classification tasks."""
from __future__ import annotations

import abc
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.runner import interfaces

Field = tfgnn.Field
GraphTensor = tfgnn.GraphTensor
# TODO(b/274672364): make this tuple[...] in Python 3.9 style
# when we drop py38 support.
LabelFn = Callable[[GraphTensor], Tuple[GraphTensor, Field]]


class _FromLogitsMixIn(tf.keras.metrics.Metric):
  """Mixin for `tf.keras.metrics.Metric` with a from_logits option."""

  def __init__(self, from_logits: bool, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._from_logits = from_logits

  def update_state(self,
                   y_true: tf.Tensor,
                   y_pred: tf.Tensor,
                   sample_weight: Optional[tf.Tensor] = None) -> None:
    return super().update_state(
        y_true,
        tf.nn.sigmoid(y_pred) if self._from_logits else y_pred,
        sample_weight)

  def get_config(self) -> Mapping[Any, Any]:
    return dict(from_logits=self._from_logits, **super().get_config())


@tf.keras.utils.register_keras_serializable(package="GNN")
class FromLogitsPrecision(_FromLogitsMixIn, tf.keras.metrics.Precision):
  pass


@tf.keras.utils.register_keras_serializable(package="GNN")
class FromLogitsRecall(_FromLogitsMixIn, tf.keras.metrics.Recall):
  pass


class _PerClassMetricMixIn(tf.keras.metrics.Metric):
  """Mixin for `tf.keras.metrics.Metric` with a sparse_class_id option.

  This Mixin is needed because ground truths come as class id integer, which is
  incompatible with tf.keras.metrics.Precision.
  """

  def __init__(self, sparse_class_id: int, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._sparse_class_id = sparse_class_id

  def update_state(self,
                   y_true: tf.Tensor,
                   y_pred: tf.Tensor,
                   sample_weight: Optional[tf.Tensor] = None) -> None:
    return super().update_state(
        (y_true == self._sparse_class_id),
        (tf.argmax(y_pred, -1) == self._sparse_class_id),
        sample_weight)

  def get_config(self) -> Mapping[Any, Any]:
    return dict(sparse_class_id=self._sparse_class_id, **super().get_config())


@tf.keras.utils.register_keras_serializable(package="GNN")
class PerClassPrecision(_PerClassMetricMixIn, tf.keras.metrics.Precision):
  pass


@tf.keras.utils.register_keras_serializable(package="GNN")
class PerClassRecall(_PerClassMetricMixIn, tf.keras.metrics.Recall):
  pass


class _Classification(interfaces.Task):
  """Abstract classification class.

  Any subclass must implement all of `gather_activations`, `losses` and
  `metrics`, usually by inheriting from the classes below.
  """

  def __init__(
      self,
      units: int,
      *,
      name: str = "classification_logits",
      label_fn: Optional[LabelFn] = None,
      label_feature_name: Optional[str] = None):
    """Sets `Task` parameters.

    Args:
      units: The units for the classification head. (Typically `1` for binary
        classification and `num_classes` for multiclass classification.)
      name: The classification head's layer name. To control the naming of saved
        model outputs see the runner model exporters (e.g.,
        `KerasModelExporter`).
      label_fn: A label extraction function. This function mutates the input
        `GraphTensor`. Mutually exclusive with `label_feature_name`.
      label_feature_name: A label feature name for readout from the auxiliary
        '_readout' node set. Readout does not mutate the input `GraphTensor`.
        Mutually exclusive with `label_fn`.
    """
    if (label_fn is None) == (label_feature_name is None):
      raise ValueError(
          "Exactly one of `label_fn` or `label_feature_name` may be specified"
          f" (got label_fn={label_fn} and"
          f" label_feature_name={label_feature_name})"
      )
    self._units = units
    self._name = name
    self._label_fn = label_fn
    self._label_feature_name = label_feature_name

  @abc.abstractmethod
  def gather_activations(self, inputs: GraphTensor) -> Field:
    raise NotImplementedError()

  def predict(self, inputs: tfgnn.GraphTensor) -> interfaces.Predictions:
    """Apply a linear head for classification.

    Args:
      inputs: A `tfgnn.GraphTensor` for classification.

    Returns:
      The classification logits.
    """
    tfgnn.check_scalar_graph_tensor(inputs, name="Classification")
    activations = self.gather_activations(inputs)
    logits = tf.keras.layers.Dense(
        self._units,
        name=self._name)(activations)
    return logits

  def preprocess(self, inputs: GraphTensor) -> tuple[GraphTensor, Field]:
    if self._label_fn is not None:
      return self._label_fn(inputs)
    x = inputs
    y = tfgnn.keras.layers.Readout(
        feature_name=self._label_feature_name,
        node_set_name="_readout")(inputs)
    return x, y

  @abc.abstractmethod
  def losses(self) -> interfaces.Losses:
    raise NotImplementedError()

  @abc.abstractmethod
  def metrics(self) -> interfaces.Metrics:
    raise NotImplementedError()


class _BinaryClassification(_Classification):
  """Binary classification."""

  def __init__(self, units: int = 1, **kwargs):
    super().__init__(units, **kwargs)

  def losses(self) -> interfaces.Losses:
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)

  def metrics(self) -> interfaces.Metrics:
    return (FromLogitsPrecision(from_logits=True),
            FromLogitsRecall(from_logits=True),
            tf.keras.metrics.AUC(from_logits=True, name="auc_roc"),
            tf.keras.metrics.AUC(curve="PR", from_logits=True, name="auc_pr"),
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.losses.BinaryCrossentropy(from_logits=True))


class _MulticlassClassification(_Classification):
  """Multiclass classification."""

  def __init__(self,
               *,
               num_classes: Optional[int] = None,
               class_names: Optional[Sequence[str]] = None,
               per_class_statistics: bool = False,
               **kwargs):
    if (num_classes is None) == (class_names is None):
      raise ValueError(
          "Exactly one of `num_classes` or `class_names` must be specified")
    if num_classes is None:
      num_classes = len(class_names)
    super().__init__(num_classes, **kwargs)
    if class_names is None:
      self._class_names = [f"class_{i}" for i in range(num_classes)]
    else:
      self._class_names = class_names
    self._per_class_statistics = per_class_statistics

  def losses(self) -> interfaces.Losses:
    """Sparse categorical crossentropy loss."""
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  def metrics(self) -> interfaces.Metrics:
    """Sparse categorical metrics."""
    metric_objs = [
        tf.keras.metrics.SparseCategoricalAccuracy(),
        tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)]

    if self._per_class_statistics:
      for i, class_name in enumerate(self._class_names):
        metric_objs.append(
            PerClassPrecision(
                sparse_class_id=i,
                name=f"precision_for_{class_name}"))
        metric_objs.append(
            PerClassRecall(
                sparse_class_id=i,
                name=f"recall_for_{class_name}"))
    return metric_objs


class _GraphClassification(_Classification):
  """Classification by the label provided in the graph context."""

  def __init__(self,
               node_set_name: str,
               *,
               state_name: str = tfgnn.HIDDEN_STATE,
               reduce_type: str = "mean",
               **kwargs):
    """Classification by the label provided in the graph context.

    This task defines classification of a graph by a label found in its unique
    context. The representations are mean-pooled from a single set of nodes.

    Labels are expected to be sparse, i.e.: scalars.

    Args:
      node_set_name: Node set to pool representations from.
      state_name: The feature name for node activations
        (typically: tfgnn.HIDDEN_STATE).
      reduce_type: The context pooling reduction type.
      **kwargs: Additional keyword arguments.
    """
    super().__init__(**kwargs)
    self._node_set_name = node_set_name
    self._state_name = state_name
    self._reduce_type = reduce_type

  def gather_activations(self, inputs: GraphTensor) -> Field:
    return tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT,
        self._reduce_type,
        node_set_name=self._node_set_name,
        feature_name=self._state_name)(inputs)


class _RootNodeClassification(_Classification):
  """Classification by root node label."""

  def __init__(self,
               node_set_name: str,
               *,
               state_name: str = tfgnn.HIDDEN_STATE,
               **kwargs):
    """Classification by root node label.

    This task defines classification of a rooted graph by a label found
    on its unique root node. By convention, root nodes are stored as the
    first node of each graph component in the respective node set.
    Typically, such graphs are created by go/graph-sampler by sampling the
    neighborhoods of root nodes (or "seed nodes") in large graphs to
    aggregate information relevant to the root node's classification.

    Any labels are expected to be sparse, i.e.: scalars.

    Args:
      node_set_name: The node set containing the root node.
      state_name: The feature name for activations
        (typically: tfgnn.HIDDEN_STATE).
      **kwargs: Additional keyword arguments.
    """
    super().__init__(**kwargs)
    self._node_set_name = node_set_name
    self._state_name = state_name

  def gather_activations(self, inputs: GraphTensor) -> Field:
    """Gather activations from root nodes."""
    return tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name=self._node_set_name,
        feature_name=self._state_name)(inputs)


class _NodeClassification(_Classification):
  """Classification by node(s) via structured readout."""

  def __init__(self,
               key: str = "seed",
               *,
               feature_name: str = tfgnn.HIDDEN_STATE,
               readout_node_set: tfgnn.NodeSetName = "_readout",
               validate: bool = True,
               **kwargs):
    """Classification of node(s) via structured readout.

    This task defines classification via structured readout (see
    `tfgnn.keras.layers.StructuredReadout`).  Structured readout addresses the
    need to read out final hidden states from a GNN computation to make
    predictions for some nodes (or edges) of interest. To add auxiliary node
    (and edge) sets for structured readout see, e.g.:
    `tfgnn.keras.layers.AddReadoutFromFirstNode`.

    Any labels are expected to be sparse, i.e.: scalars.

    Args:
      key: A string key to select between possibly multiple named readouts.
      feature_name: The name of the feature to read. If unset,
        `tfgnn.HIDDEN_STATE` will be read.
      readout_node_set: A string, defaults to `"_readout"`. This is used as the
        name for the readout node set and as a name prefix for its edge sets.
      validate: Setting this to false disables the validity checks for the
        auxiliary edge sets. This is stronlgy discouraged, unless great care is
        taken to run `tfgnn.validate_graph_tensor_for_readout()` earlier on
        structurally unchanged GraphTensors.
      **kwargs: Additional keyword arguments.
    """
    super().__init__(**kwargs)
    self._key = key
    self._feature_name = feature_name
    self._readout_node_set = readout_node_set
    self._validate = validate

  def gather_activations(self, inputs: GraphTensor) -> Field:
    """Gather activations from auxiliary node (and edge) sets."""
    try:
      return tfgnn.keras.layers.StructuredReadout(
          self._key,
          feature_name=self._feature_name,
          readout_node_set=self._readout_node_set,
          validate=self._validate)(inputs)
    except (KeyError, ValueError) as e:
      raise ValueError(
          "This NodeClassification task failed in StructuredReadout("
          f"{self._key}, feature_name={self._feature_name}, "
          f"readout_node_set={self._readout_node_set}).\n"
          "For a dataset of sampled subgraphs that does not provide a readout "
          "structure but follows the conventional placement of root nodes "
          "first in their node set, consider using a RootNodeClassification "
          "task or tfgnn.keras.layers.AddReadoutFromFirstNode."
      ) from e


class GraphBinaryClassification(_GraphClassification, _BinaryClassification):
  """Graph binary (or multi-label) classification from pooled node states."""

  def __init__(self,
               node_set_name: str,
               units: int = 1,
               *,
               state_name: str = tfgnn.HIDDEN_STATE,
               reduce_type: str = "mean",
               name: str = "classification_logits",
               label_fn: Optional[LabelFn] = None,
               label_feature_name: Optional[str] = None):
    """Graph binary (or multi-label) classification.

    This task performs binary classification (or multiple independent ones:
    often called multi-label classification).

    Args:
      node_set_name: The node set to pool.
      units: The units for the classification head. (Typically `1` for binary
        classification and the number of labels for multi-label classification.)
      state_name: The feature name for activations (e.g.: tfgnn.HIDDEN_STATE).
      reduce_type: The context pooling reduction type.
      name: The classification head's layer name. To control the naming of saved
        model outputs see the runner model exporters (e.g.,
        `KerasModelExporter`).
      label_fn: A label extraction function. This function mutates the input
        `GraphTensor`. Mutually exclusive with `label_feature_name`.
      label_feature_name: A label feature name for readout from the auxiliary
        '_readout' node set. Readout does not mutate the input `GraphTensor`.
        Mutually exclusive with `label_fn`.
    """
    super().__init__(
        node_set_name,
        units=units,
        state_name=state_name,
        reduce_type=reduce_type,
        name=name,
        label_fn=label_fn,
        label_feature_name=label_feature_name)


class GraphMulticlassClassification(_GraphClassification,
                                    _MulticlassClassification):
  """Graph multiclass classification from pooled node states."""

  def __init__(self,
               node_set_name: str,
               *,
               num_classes: Optional[int] = None,
               class_names: Optional[Sequence[str]] = None,
               per_class_statistics: bool = False,
               state_name: str = tfgnn.HIDDEN_STATE,
               reduce_type: str = "mean",
               name: str = "classification_logits",
               label_fn: Optional[LabelFn] = None,
               label_feature_name: Optional[str] = None):
    """Graph multiclass classification from pooled node states.

    Args:
      node_set_name: The node set to pool.
      num_classes: The number of classes. Exactly one of `num_classes` or
        `class_names` must be specified
      class_names: The class names. Exactly one of `num_classes` or
        `class_names` must be specified
      per_class_statistics: Whether to compute statistics per class.
      state_name: The feature name for activations (e.g.: tfgnn.HIDDEN_STATE).
      reduce_type: The context pooling reduction type.
      name: The classification head's layer name. To control the naming of saved
        model outputs see the runner model exporters (e.g.,
        `KerasModelExporter`).
      label_fn: A label extraction function. This function mutates the input
        `GraphTensor`. Mutually exclusive with `label_feature_name`.
      label_feature_name: A label feature name for readout from the auxiliary
        '_readout' node set. Readout does not mutate the input `GraphTensor`.
        Mutually exclusive with `label_fn`.
    """
    super().__init__(
        node_set_name,
        num_classes=num_classes,
        class_names=class_names,
        per_class_statistics=per_class_statistics,
        state_name=state_name,
        reduce_type=reduce_type,
        name=name,
        label_fn=label_fn,
        label_feature_name=label_feature_name)


class RootNodeBinaryClassification(_RootNodeClassification,
                                   _BinaryClassification):
  """Root node binary (or multi-label) classification."""

  def __init__(self,
               node_set_name: str,
               units: int = 1,
               *,
               state_name: str = tfgnn.HIDDEN_STATE,
               name: str = "classification_logits",
               label_fn: Optional[LabelFn] = None,
               label_feature_name: Optional[str] = None):
    """Root node binary (or multi-label) classification.

    This task performs binary classification (or multiple independent ones:
    often called multi-label classification).

    The task can be used on graph datasets without a readout structure.
    It requires that each input graph stores its unique root node as the
    first node of `node_set_name`.

    Args:
      node_set_name: The node set containing the root node.
      units: The units for the classification head. (Typically `1` for binary
        classification and the number of labels for multi-label classification.)
      state_name: The feature name for activations (e.g.: tfgnn.HIDDEN_STATE).
      name: The classification head's layer name. To control the naming of saved
        model outputs see the runner model exporters (e.g.,
        `KerasModelExporter`).
      label_fn: A label extraction function. This function mutates the input
        `GraphTensor`. Mutually exclusive with `label_feature_name`.
      label_feature_name: A label feature name for readout from the auxiliary
        '_readout' node set. Readout does not mutate the input `GraphTensor`.
        Mutually exclusive with `label_fn`.
    """
    super().__init__(
        node_set_name,
        units=units,
        state_name=state_name,
        name=name,
        label_fn=label_fn,
        label_feature_name=label_feature_name)


class RootNodeMulticlassClassification(_RootNodeClassification,
                                       _MulticlassClassification):
  """Root node multiclass classification."""

  def __init__(self,
               node_set_name: str,
               *,
               num_classes: Optional[int] = None,
               class_names: Optional[Sequence[str]] = None,
               per_class_statistics: bool = False,
               state_name: str = tfgnn.HIDDEN_STATE,
               name: str = "classification_logits",
               label_fn: Optional[LabelFn] = None,
               label_feature_name: Optional[str] = None):
    """Root node multiclass classification.

    This task can be used on graph datasets without a readout structure.
    It requires that each input graph stores its unique root node as the
    first node of `node_set_name`.

    Args:
      node_set_name: The node set containing the root node.
      num_classes: The number of classes. Exactly one of `num_classes` or
        `class_names` must be specified
      class_names: The class names. Exactly one of `num_classes` or
        `class_names` must be specified
      per_class_statistics: Whether to compute statistics per class.
      state_name: The feature name for activations (e.g.: tfgnn.HIDDEN_STATE).
      name: The classification head's layer name. To control the naming of saved
        model outputs see the runner model exporters (e.g.,
        `KerasModelExporter`).
      label_fn: A label extraction function. This function mutates the input
        `GraphTensor`. Mutually exclusive with `label_feature_name`.
      label_feature_name: A label feature name for readout from the auxiliary
        '_readout' node set. Readout does not mutate the input `GraphTensor`.
        Mutually exclusive with `label_fn`.
    """
    super().__init__(
        node_set_name,
        num_classes=num_classes,
        class_names=class_names,
        per_class_statistics=per_class_statistics,
        state_name=state_name,
        name=name,
        label_fn=label_fn,
        label_feature_name=label_feature_name)


class NodeBinaryClassification(_NodeClassification, _BinaryClassification):
  """Node binary (or multi-label) classification via structured readout."""

  def __init__(self,
               key: str = "seed",
               units: int = 1,
               *,
               feature_name: str = tfgnn.HIDDEN_STATE,
               readout_node_set: tfgnn.NodeSetName = "_readout",
               validate: bool = True,
               name: str = "classification_logits",
               label_fn: Optional[LabelFn] = None,
               label_feature_name: Optional[str] = None):
    """Node binary (or multi-label) classification.

    This task performs binary classification (or multiple independent ones:
    often called multi-label classification).

    Args:
      key: A string key to select between possibly multiple named readouts.
      units: The units for the classification head. (Typically `1` for binary
        classification and the number of labels for multi-label classification.)
      feature_name: The name of the feature to read. If unset,
        `tfgnn.HIDDEN_STATE` will be read.
      readout_node_set: A string, defaults to `"_readout"`. This is used as the
        name for the readout node set and as a name prefix for its edge sets.
      validate: Setting this to false disables the validity checks for the
        auxiliary edge sets. This is stronlgy discouraged, unless great care is
        taken to run `tfgnn.validate_graph_tensor_for_readout()` earlier on
        structurally unchanged GraphTensors.
      name: The classification head's layer name. To control the naming of saved
        model outputs see the runner model exporters (e.g.,
        `KerasModelExporter`).
      label_fn: A label extraction function. This function mutates the input
        `GraphTensor`. Mutually exclusive with `label_feature_name`.
      label_feature_name: A label feature name for readout from the auxiliary
        '_readout' node set. Readout does not mutate the input `GraphTensor`.
        Mutually exclusive with `label_fn`.
    """
    super().__init__(
        key,
        units=units,
        feature_name=feature_name,
        readout_node_set=readout_node_set,
        validate=validate,
        name=name,
        label_fn=label_fn,
        label_feature_name=label_feature_name)


class NodeMulticlassClassification(_NodeClassification,
                                   _MulticlassClassification):
  """Node multiclass classification via structured readout."""

  def __init__(self,
               key: str = "seed",
               *,
               feature_name: str = tfgnn.HIDDEN_STATE,
               readout_node_set: tfgnn.NodeSetName = "_readout",
               validate: bool = True,
               num_classes: Optional[int] = None,
               class_names: Optional[Sequence[str]] = None,
               per_class_statistics: bool = False,
               name: str = "classification_logits",
               label_fn: Optional[LabelFn] = None,
               label_feature_name: Optional[str] = None):
    """Node multiclass classification via structured readout.

    Args:
      key: A string key to select between possibly multiple named readouts.
      feature_name: The name of the feature to read. If unset,
        `tfgnn.HIDDEN_STATE` will be read.
      readout_node_set: A string, defaults to `"_readout"`. This is used as the
        name for the readout node set and as a name prefix for its edge sets.
      validate: Setting this to false disables the validity checks for the
        auxiliary edge sets. This is stronlgy discouraged, unless great care is
        taken to run `tfgnn.validate_graph_tensor_for_readout()` earlier on
        structurally unchanged GraphTensors.
      num_classes: The number of classes. Exactly one of `num_classes` or
        `class_names` must be specified
      class_names: The class names. Exactly one of `num_classes` or
        `class_names` must be specified
      per_class_statistics: Whether to compute statistics per class.
      name: The classification head's layer name. To control the naming of saved
        model outputs see the runner model exporters (e.g.,
        `KerasModelExporter`).
      label_fn: A label extraction function. This function mutates the input
        `GraphTensor`. Mutually exclusive with `label_feature_name`.
      label_feature_name: A label feature name for readout from the auxiliary
        '_readout' node set. Readout does not mutate the input `GraphTensor`.
        Mutually exclusive with `label_fn`.
    """
    super().__init__(
        key,
        feature_name=feature_name,
        readout_node_set=readout_node_set,
        validate=validate,
        num_classes=num_classes,
        class_names=class_names,
        per_class_statistics=per_class_statistics,
        name=name,
        label_fn=label_fn,
        label_feature_name=label_feature_name)
